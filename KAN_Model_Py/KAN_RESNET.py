import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
from tqdm import tqdm
import albumentations as A
import torch.nn.functional as F
import math
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report
import logging
import configparser
import matplotlib.pyplot as plt
from torchvision.models import ResNet50_Weights
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


################################################################################
# Configuration & Setup
################################################################################

labels = ['PNEUMONIA', 'NORMAL']

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
config = configparser.ConfigParser()
config.read("config.yaml")

# Paths from config
train_directory = config.get("Paths", "train_directory").strip()
logger.info(f"train_directory from config {train_directory} ")
logger.info("=" * 100)
train_test = os.path.abspath(train_directory)  
logger.info(f"train_directory test from config {train_directory} ")
logger.info("=" * 100)

val_directory = config.get("Paths", "val_directory")
test_directory = config.get("Paths", "test_directory")
OUTPUT_DIRECTORY = config.get("Paths", "output_directory")

# Training hyperparameters from config
train_batch = config.getint("Training", "train_batch")
train_epochs = config.getint("Training", "train_epochs")
train_learning_rate = config.getfloat("Training", "train_learning_rate")
model_name = config.get("Prediction", "model_name")
model_description =config.get("Prediction", "description")
img_size = config.getint("Training", "img_size")

# Ensure output directory exists
model_path = os.path.join(OUTPUT_DIRECTORY, model_name)
os.makedirs(model_path, exist_ok=True)

# Check how many GPUs are being used
num_gpus = torch.cuda.device_count()
if num_gpus > 1:
    logger.info(f"Using {num_gpus} GPUs for training.")
else:
    logger.info("Using a single GPU for training.")
logger.info("=" * 100)

################################################################################
# Data Loading & Preprocessing
################################################################################

def get_training_data(data_dir):
    """
    Loads images from two folders: 'PNEUMONIA' and 'NORMAL'.
    Returns a list of (image, label).
    """
    data = []

    for label in labels:
        path = os.path.join(data_dir.strip(), label)
        logger.info(f"Processing path: {path}")
        logger.info("=" * 100)
        class_num = labels.index(label)

        for img in tqdm(os.listdir(path)):
            try:
                # Load and resize the image to (img_size, img_size)
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                resized_arr = cv2.resize(img_arr, (img_size, img_size))  # Resize
                # Add the image and label as a pair
                data.append([resized_arr, class_num])
            except Exception as e:
                logger.info(f"Error loading image {img}: {e}")
                logger.info("=" * 100)

    # Convert the list to a NumPy array
    data = np.array(data, dtype=object)  # Use dtype=object to allow image-label pairing
    return data

def normalize_images(data):
    """
    Normalizes images by dividing pixel values by 255.
    Returns separate numpy arrays for images & labels.
    """
    images = []
    labels = []
    # Optional Albumentations normalizer for advanced scaling if needed
    # normalizer = A.Normalize(mean=0.488, std=0.234, max_pixel_value=1)

    for img, label in tqdm(data):
        normalized_img = img / 255.0
        # Optionally apply advanced Albumentations normalization:
        # normalized_img = normalizer(image=normalized_img)['image']
        images.append(normalized_img)
        labels.append(label)

    images = np.array(images)
    labels = np.array(labels)
    return images, labels

################################################################################
# Model Definitions
################################################################################

class ResNet(nn.Module):
    """
    A ResNet50 model adapted for single-channel (grayscale) input
    and 2 output classes (PNEUMONIA vs NORMAL).
    """
    def __init__(self, num_classes=2, softmax=True):
        super(ResNet, self).__init__()
        self.resnet = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        num_ftrs = self.resnet.fc.out_features
        self.fc = nn.Linear(num_ftrs, num_classes)
        self.bn = nn.BatchNorm1d(num_ftrs)
        self.relu = nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1) if softmax else None
        self.change_conv1()

    def forward(self, x):
        x = self.resnet(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.fc(x)
        if self.softmax:
            x = self.softmax(x)
        return x

    def change_conv1(self):
        """
        Modify the first convolutional layer to handle a single (1) input channel
        instead of the default 3 channels (RGB).
        """
        original_conv1 = self.resnet.conv1
        # Create a new conv layer with 1 input channel
        new_conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=(original_conv1.bias is not None)
        )
        # Initialize the new conv layer's weights by averaging the old (RGB) weights
        with torch.no_grad():
            new_conv1.weight = nn.Parameter(
                original_conv1.weight.mean(dim=1, keepdim=True)
            )
        # Replace conv1
        self.resnet.conv1 = new_conv1

class KANLinear_v1(nn.Module):
    """
    Custom linear layer with KAN (Kernel-based Activation Network) approach
    for demonstration. 
    """
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3,
                 scale_noise=0.1, scale_base=1.0, scale_spline=1.0,
                 enable_standalone_scale_spline=True, base_activation=nn.SiLU,
                 grid_eps=0.02, grid_range=[-1, 1]):
        super(KANLinear_v1, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]).expand(in_features, -1).contiguous()
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(torch.Tensor(out_features, in_features))

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = ((torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 0.5)
                     * self.scale_noise / self.grid_size)
            self.spline_weight.data.copy_(
                self.scale_spline * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order], noise
                )
            )
            if self.enable_standalone_scale_spline:
                nn.init.kaiming_uniform_(
                    self.spline_scaler, 
                    a=math.sqrt(5) * self.scale_spline
                )

    def b_splines(self, x):
        """
        Compute B-spline basis functions for x using the precomputed self.grid.
        """
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = ((x - grid[:, :-(k+1)]) 
                        / (grid[:, k:-1] - grid[:, :-(k+1)]) 
                        * bases[:, :, :-1]
                     + (grid[:, k+1:] - x) 
                        / (grid[:, k+1:] - grid[:, 1:-k]) 
                        * bases[:, :, 1:])
        return bases.contiguous()

    def curve2coeff(self, x, y):
        """
        Convert curve points to spline coefficients by solving a least squares system.
        """
        A = self.b_splines(x).transpose(0, 1)
        B = y.transpose(0, 1)
        solution = torch.linalg.lstsq(A, B).solution
        return solution.permute(2, 0, 1).contiguous()

    def forward(self, x):
        """
        Forward pass: standard linear + spline-based transformation.
        """
        x = x.view(x.size(0), -1)
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.spline_weight.view(self.out_features, -1)
        )
        return base_output + spline_output

class FKAN_ResNet(nn.Module):
    """
    A ResNet50 model (with 1-channel input) followed by 3 KANLinear layers.
    """
    def __init__(self, num_classes=2, softmax=True):
        super(FKAN_ResNet, self).__init__()
        self.backbone = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.kan_layer1 = KANLinear_v1(1000, 256)
        self.kan_layer2 = KANLinear_v1(256, 128)
        self.kan_layer3 = KANLinear_v1(128, num_classes)
        self.softmax = torch.nn.Softmax(dim=1) if softmax else None
        self.change_conv1()

    def forward(self, x):
        x = self.backbone(x)       # (N, 1000)
        x = self.kan_layer1(x)     # (N, 256)
        x = self.kan_layer2(x)     # (N, 128)
        x = self.kan_layer3(x)     # (N, num_classes)
        if self.softmax:
            x = self.softmax(x)
        return x

    def change_conv1(self):
        original_conv1 = self.backbone.conv1
        new_conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=(original_conv1.bias is not None)
        )
        with torch.no_grad():
            new_conv1.weight = nn.Parameter(
                original_conv1.weight.mean(dim=1, keepdim=True)
            )
        self.backbone.conv1 = new_conv1

################################################################################
# Training Loop with Validation & Early Stopping
################################################################################

def train_model(model, 
                train_loader, 
                val_loader, 
                criterion, 
                optimizer, 
                device, 
                num_epochs=100, 
                patience=10):
    """
    Train a PyTorch model (supports multi-GPU via nn.DataParallel if available),
    tracking and plotting training/validation loss & accuracy,
    and saving the best model weights (based on lowest val loss here).
    """

    # 1) Check for Multi-GPU
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs for training with nn.DataParallel.")
        logger.info("=" * 100)
        model = nn.DataParallel(model)
    model.to(device)

    best_validation_loss = float('inf')  # Start with a very high number
    patience_counter = 0

    # Lists to store metrics per epoch
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # -------------------
        # TRAINING PHASE
        # -------------------
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        p_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training", leave=False)
        for i, (images, labels) in enumerate(p_bar):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backprop
            loss.backward()
            optimizer.step()

            # Accumulate training loss
            running_loss += loss.item()

            # Calculate training accuracy
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            avg_loss = running_loss / (i + 1)
            avg_acc = 100.0 * correct / total
            p_bar.set_postfix({'Loss': f'{avg_loss:.4f}', 'Acc': f'{avg_acc:.2f}%'})

        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_accuracy = 100.0 * correct / total

        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_accuracy)

        logger.info(f"[Epoch {epoch+1}/{num_epochs}] "
              f"Train Loss: {epoch_train_loss:.4f} | "
              f"Train Acc: {epoch_train_accuracy:.2f}%")
        

        # -------------------
        # VALIDATION PHASE
        # -------------------
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            p_bar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation", leave=False)
            for j, (images, labels) in enumerate(p_bar_val):
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item()
                
                _, preds = torch.max(outputs, 1)
                
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
                # Collect for classification metrics
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Compute average validation loss and accuracy
        epoch_val_loss = val_running_loss / len(val_loader)
        epoch_val_accuracy = 100.0 * val_correct / val_total
        
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_accuracy)
        
        # Classification report details
        class_report = classification_report(
            all_labels, 
            all_preds, 
            target_names=['Pneumonia', 'Normal'], 
            output_dict=True
        )
        validation_accuracy = class_report['accuracy']  # ~ epoch_val_accuracy / 100
        validation_f1_score = class_report['weighted avg']['f1-score']
        validation_precision = class_report['weighted avg']['precision']
        validation_recall = class_report['weighted avg']['recall']

        logger.info(f"[Epoch {epoch+1}/{num_epochs}] "
              f"Val Loss: {epoch_val_loss:.4f} | "
              f"Val Acc: {epoch_val_accuracy:.2f}% | "
              f"Precision: {validation_precision:.4f} | "
              f"Recall: {validation_recall:.4f} | "
              f"F1: {validation_f1_score:.4f}")
        logger.info("=" * 100)


        # -------------------
        # CHECK FOR BEST MODEL (Lowest Validation Loss)
        # -------------------
        if epoch_val_loss < best_validation_loss:
            best_validation_loss = epoch_val_loss
            patience_counter = 0
            best_model_path = os.path.join(model_path, 'best_model_resnet.pth')
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Best model saved with lower val loss = {epoch_val_loss:.4f}")
            logger.info("=" * 100)
        else:
            patience_counter += 1

        # -------------------
        # EARLY STOPPING
        # -------------------
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs. "
                  f"No improvement in {patience} consecutive validation checks.")
            logger.info("=" * 100)
            break

    # -------------------
    # PLOTTING
    # -------------------
    epochs_range = range(1, len(train_losses) + 1)

    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.title("Training and Validation Loss")
    plt.plot(epochs_range, train_losses, label="Train Loss")
    plt.plot(epochs_range, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    loss_plot_path = os.path.join(model_path, "loss_function.png")
    plt.savefig(loss_plot_path)
    logger.info(f"Loss plot saved to {loss_plot_path}")
    logger.info("=" * 100)
    plt.close()

    # Plot Accuracy
    plt.figure(figsize=(10, 5))
    plt.title("Training and Validation Accuracy")
    plt.plot(epochs_range, train_accuracies, label="Train Acc")
    plt.plot(epochs_range, val_accuracies, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)

    acu_plot_path = os.path.join(model_path, "acu_function.png")
    plt.savefig(acu_plot_path)
    logger.info(f"Accuracy plot saved to {acu_plot_path}")
    logger.info("=" * 100)
    plt.close()

    logger.info("Training complete.")
    logger.info(f"Best validation loss score achieved: {best_validation_loss:.4f}")
    logger.info("=" * 100)

# ------------------------------------------------------------------
# 4) EVALUATION FUNCTION (FOR VAL & TEST)
# ------------------------------------------------------------------


def evaluate_model(model, dataloader, criterion):
    """
    Evaluate the model on a given DataLoader to obtain Loss, Accuracy,
    Precision, Recall, and F1-Score.

    - Moves the model to CPU for evaluation (you can adjust this if you prefer GPU).
    - Collects predictions and labels, then computes standard metrics for binary classification.
    """
    device = torch.device("cpu")  # <-- Using CPU for evaluation (you can change this if you'd like)
    model.to(device)
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for data, targets in dataloader:
            data = data.to(torch.float32).to(device)
            targets = targets.to(torch.long).to(device)

            outputs = model(data)
            loss = criterion(outputs, targets)

            # Accumulate total loss
            total_loss += loss.item() * data.size(0)

            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

    # Ensure we only have {0,1} as labels/preds
    unique_labels = set(all_labels)
    unique_preds = set(all_preds)
    assert unique_labels <= {0, 1}, f"Invalid labels: {unique_labels}"
    assert unique_preds <= {0, 1}, f"Invalid predictions: {unique_preds}"

    # Compute average loss over the entire dataset
    avg_loss = total_loss / len(dataloader.dataset)

    # Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="binary")
    recall = recall_score(all_labels, all_preds, average="binary")
    f1 = f1_score(all_labels, all_preds, average="binary")

    return avg_loss, accuracy, precision, recall, f1


################################################################################
# Save metrics helper
################################################################################

def save_metrics_to_file(file_path, phase, loss, acc, prec, rec, f1):
    """
    Save the evaluation metrics to a text file.
    'phase' can be 'Validation', 'Test', etc.
    """
    with open(file_path, "a") as f:  # 'a' for append, 'w' for overwrite
        f.write(f"{phase} Metrics:\n")
        f.write(f"  Loss:      {loss:.4f}\n")
        f.write(f"  Accuracy:  {acc:.4f}\n")
        f.write(f"  Precision: {prec:.4f}\n")
        f.write(f"  Recall:    {rec:.4f}\n")
        f.write(f"  F1-Score:  {f1:.4f}\n\n")

################################################################################
# Main
################################################################################

def main():
    
    
    #save info about the model
    with open(os.path.join(model_path, "evaluation_metrics.txt"), "a") as f:  # append
        f.write("Information:\n")
        f.write(f"  Model Name:   {model_name} \n")
        f.write(f"  Model Name:   {model_description} \n")
        f.write(f"  Train batch:  {train_batch} \n")
        f.write(f"  Train epochs: {train_epochs} \n")
        f.write(f"  Train learning rate:  {train_learning_rate} \n")
        f.write(f"  Image size:  {img_size} \n\n")
    
    # -------------------------------
    # 1) Load the train/val/test data
    # -------------------------------
    train_data = get_training_data(train_directory)
    test_data = get_training_data(test_directory)
    val_data = get_training_data(val_directory)

    # --------------------------------
    # 2) Normalize the images
    # --------------------------------
    train_images, train_labels = normalize_images(train_data)
    val_images, val_labels = normalize_images(val_data)
    test_images, test_labels = normalize_images(test_data)

    # Check shapes
    logger.info(f"Shape of train images: {train_images.shape}")
    logger.info(f"Shape of val images:   {val_images.shape}")
    logger.info(f"Shape of test images:  {test_images.shape}")
    logger.info(f"Sample train image max={train_images[0].max()}, min={train_images[0].min()}")
    logger.info("=" * 100)

    # -------------------------------
    # 3) Define and set up the model
    # -------------------------------
    model = FKAN_ResNet(num_classes=2, softmax=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    logger.info(f"Using device: {device}")
    logger.info("=" * 100)

    # -------------------------------
    # 4) Convert numpy arrays to Tensors
    # -------------------------------
    # Expand dims for channel=1
    train_images_tensor = torch.stack([
        torch.tensor(img, dtype=torch.float) for img in train_images
    ]).unsqueeze(1)

    val_images_tensor = torch.stack([
        torch.tensor(img, dtype=torch.float) for img in val_images
    ]).unsqueeze(1)

    train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
    val_labels_tensor = torch.tensor(val_labels, dtype=torch.long)

    # (N, 1, H, W) shape
    logger.info(f"Train tensor shape: {train_images_tensor.shape}")
    logger.info(f"Val tensor shape:   {val_images_tensor.shape}")
    logger.info("=" * 100)

    # --------------------------------
    # 5) Create DataLoader
    # --------------------------------
    train_dataset = TensorDataset(train_images_tensor, train_labels_tensor)
    val_dataset = TensorDataset(val_images_tensor, val_labels_tensor)

    train_loader = DataLoader(train_dataset, batch_size=train_batch, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=train_batch, shuffle=True)

    logger.info("DataLoaders ready.")
    logger.info("=" * 100)

    # -------------------------------
    # 6) Define Loss and Optimizer
    # -------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_learning_rate, weight_decay=1e-5)

    # -------------------------------
    # 7) Train the model
    # -------------------------------
    
    train_model(
        model, 
        train_loader, 
        val_loader, 
        criterion, 
        optimizer, 
        device, 
        num_epochs=train_epochs, 
        patience=100
    )
    

    # -------------------------------
    # 8) Testing / Evaluation
    # -------------------------------
    # Load the best model saved
    best_model_weights = os.path.join(model_path, 'best_model_resnet.pth')
    checkpoint = torch.load(best_model_weights, map_location=device, weights_only=True)

    
    # Create a new dict without the "module." prefix
    new_state_dict = {}
    for k, v in checkpoint.items():
        if k.startswith("module."):
            # Remove "module." 
            k = k.replace("module.", "", 1)
        new_state_dict[k] = v
        # Now load into your single-GPU / CPU model
    model.load_state_dict(new_state_dict)
  
    
    # 5A) Evaluate on Validation Set
    val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate_model(
        model, val_loader, criterion
        )
    logger.info("Validation Metrics:")
    logger.info(f"Loss:      {val_loss:.4f}")
    logger.info(f"Accuracy:  {val_acc:.4f}")
    logger.info(f"Precision: {val_prec:.4f}")
    logger.info(f"Recall:    {val_rec:.4f}")
    logger.info(f"F1-Score:  {val_f1:.4f}")
    logger.info("=" * 100)

    # Save Validation metrics to file
    metrics_file = os.path.join(model_path, "evaluation_metrics.txt")
    save_metrics_to_file(metrics_file, "Validation", val_loss, val_acc, val_prec, val_rec, val_f1)
    

    # Convert test images
    test_images_tensor = torch.stack([
        torch.tensor(img, dtype=torch.float) for img in test_images
    ]).unsqueeze(1)

    test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)

    test_dataset = TensorDataset(test_images_tensor, test_labels_tensor)
    test_loader = DataLoader(test_dataset, batch_size=train_batch, shuffle=True)

    

  # Evaluate on the test set
    test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate_model(model, test_loader, criterion)

    logger.info("\n==== Final Test Metrics ====")
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Accuracy: {test_acc*100:.2f}%")
    logger.info(f"Precision: {test_prec:.4f}")
    logger.info(f"Recall:    {test_rec:.4f}")
    logger.info(f"F1-score:  {test_rec:.4f}")
    logger.info("=" * 100)
    
    # Save test metrics to file
    metrics_file = os.path.join(model_path, "evaluation_metrics.txt")
    save_metrics_to_file(metrics_file, "Test", test_loss, test_acc, test_prec, test_rec, test_rec)

if __name__ == "__main__":
    main()
