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
#model_name = config.get("Prediction", "model_name")
model_name ='RESNET_KANFourier'
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
    data = []
    for label in labels:
        path = os.path.join(data_dir.strip(), label)
        logger.info(f"Processing path: {path}")
        logger.info("=" * 100)
        class_num = labels.index(label)

        for img in tqdm(os.listdir(path)):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                resized_arr = cv2.resize(img_arr, (img_size, img_size))
                data.append([resized_arr, class_num])
            except Exception as e:
                logger.info(f"Error loading image {img}: {e}")
                logger.info("=" * 100)

    data = np.array(data, dtype=object)
    return data

def normalize_images(data):
    images = []
    labels = []
    for img, label in tqdm(data):
        normalized_img = img / 255.0
        images.append(normalized_img)
        labels.append(label)
    return np.array(images), np.array(labels)

################################################################################
# KAN & FourierKAN Layers
################################################################################

class NaiveFourierKANLayer(nn.Module):
    def __init__(self, inputdim, outdim, initial_gridsize, addbias=True):
        super(NaiveFourierKANLayer, self).__init__()
        self.addbias = addbias
        self.inputdim = inputdim
        self.outdim = outdim

        # Learnable gridsize parameter
        self.gridsize_param = nn.Parameter(
            torch.tensor(initial_gridsize, dtype=torch.float32)
        )

        # Fourier coefficients (Xavier init)
        self.fouriercoeffs = nn.Parameter(
            torch.empty(2, outdim, inputdim, initial_gridsize)
        )
        nn.init.xavier_uniform_(self.fouriercoeffs)

        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(1, outdim))

    def forward(self, x):
        gridsize = torch.clamp(self.gridsize_param, min=1).round().int()
        xshp = x.shape
        outshape = xshp[:-1] + (self.outdim,)

        x = torch.reshape(x, (-1, self.inputdim))  # (batch, inputdim)
        k = torch.reshape(
            torch.arange(1, gridsize + 1, device=x.device), (1, 1, 1, gridsize)
        )
        xrshp = torch.reshape(x, (x.shape[0], 1, x.shape[1], 1))  # (batch, 1, inputdim, 1)
        c = torch.cos(k * xrshp)
        s = torch.sin(k * xrshp)

        # first half is cos coeffs
        y = torch.sum(c * self.fouriercoeffs[0:1, :, :, :gridsize], dim=(-2, -1))
        # second half is sin coeffs
        y += torch.sum(s * self.fouriercoeffs[1:2, :, :, :gridsize], dim=(-2, -1))

        if self.addbias:
            y += self.bias
        y = torch.reshape(y, outshape)
        return y

class FourierKAN(nn.Module):
    def __init__(self, params_list):
        super(FourierKAN, self).__init__()
        self.layer1 = nn.Linear(params_list[0], params_list[1])
        nn.init.xavier_uniform_(self.layer1.weight)
        self.bn1 = nn.BatchNorm1d(1)
        self.relu1 = nn.ReLU()

        self.layer2 = NaiveFourierKANLayer(params_list[1], params_list[2], initial_gridsize=16)
        self.layer3 = NaiveFourierKANLayer(params_list[2], params_list[3], initial_gridsize=8)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x shape: (N, 1000) from ResNet
        x = self.layer1(x)  # (N, 256)
        x = self.bn1(x)     # BN over dimension=1
        x = self.relu1(x)   # ReLU
        x = self.layer2(x)  # (N, 128)
        x = self.layer3(x)  # (N, num_classes)
        x = torch.squeeze(x, dim=1)  # if needed, but typically (N, 2) is already correct shape
        x = self.softmax(x)
        return x

################################################################################
# FourierKANResNet
################################################################################

class FourierKANResNet(nn.Module):
    """
    Replace the old KANLinear_v1 layers with FourierKAN.
    ResNet50 -> FourierKAN -> final output
    """
    def __init__(self):
        super(FourierKANResNet, self).__init__()
        self.backbone = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.change_conv1()

        # param_list = [1000, 256, 128, num_classes]
        self.fourier_kan = FourierKAN([224*224, 128, 64, 2])

    def forward(self, x):
        # (N,1,H,W)->ResNet->(N,1000)
        x = self.backbone(x)  
        # Then pass that into FourierKAN
        x = self.fourier_kan(x)
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
            new_conv1.weight = nn.Parameter(original_conv1.weight.mean(dim=1, keepdim=True))
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
    and saving the best model weights (based on lowest val loss).
    """

    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs for training with nn.DataParallel.")
        logger.info("=" * 100)
        model = nn.DataParallel(model)
    model.to(device)

    best_validation_loss = float('inf')
    patience_counter = 0

    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # -------------------
        # TRAIN PHASE
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
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
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
                    f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_accuracy:.2f}%")

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
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        epoch_val_loss = val_running_loss / len(val_loader)
        epoch_val_accuracy = 100.0 * val_correct / val_total
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_accuracy)

        class_report = classification_report(
            all_labels, all_preds, target_names=['Pneumonia', 'Normal'], output_dict=True
        )
        validation_accuracy = class_report['accuracy']
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

        # Check best model
        if epoch_val_loss < best_validation_loss:
            best_validation_loss = epoch_val_loss
            patience_counter = 0
            best_model_path = os.path.join(model_path, 'best_model_resnet.pth')
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Best model saved with lower val loss = {epoch_val_loss:.4f}")
            logger.info("=" * 100)
        else:
            patience_counter += 1

        # Early stopping
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
    device = torch.device("cpu") 
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
            total_loss += loss.item() * data.size(0)

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

    unique_labels = set(all_labels)
    unique_preds = set(all_preds)
    assert unique_labels <= {0, 1}, f"Invalid labels: {unique_labels}"
    assert unique_preds <= {0, 1}, f"Invalid predictions: {unique_preds}"

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="binary")
    recall = recall_score(all_labels, all_preds, average="binary")
    f1 = f1_score(all_labels, all_preds, average="binary")

    return avg_loss, accuracy, precision, recall, f1

################################################################################
# Save metrics helper
################################################################################

def save_metrics_to_file(file_path, phase, loss, acc, prec, rec, f1):
    with open(file_path, "a") as f:  # append
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
        f.write(f"  Model Name: {model_name} \n")
        f.write(f"  Model Name: {model_description} \n")
        f.write(f"  Train batch: {train_batch} \n")
        f.write(f"  Train epochs: {train_epochs} \n")
        f.write(f"  Train learning rate: {train_learning_rate} \n")
        f.write(f"  Image size: {img_size} \n\n")
    
    
    # 1) Load train/val/test data
    train_data = get_training_data(train_directory)
    test_data = get_training_data(test_directory)
    val_data = get_training_data(val_directory)
  
    # Separate the images and the labels
    train_images_display = np.array([x[0] for x in train_data])  # Extract only the images
    train_labels_display = np.array([x[1] for x in train_data])  # Extract only the labels

    # Initialize counts for each label
    count_0 = 0
    count_1 = 0

    # Loop through the labels and count occurrences
    for label in train_labels_display:
        if label == 0:
            count_0 += 1
        elif label == 1:
            count_1 += 1

    # Print the results
    logger.info(f"Count of label 0: {count_0}")
    logger.info(f"Count of label 1: {count_1}")
    logger.info("=" * 100)

    val_images_display = np.array([x[0] for x in val_data])  # Extract only the images
    val_labels_display = np.array([x[1] for x in val_data])  # Extract only the labels

    count_0 = 0
    count_1 = 0

    # Loop through the labels and count occurrences
    for label in val_labels_display:
        if label == 0:
            count_0 += 1
        elif label == 1:
            count_1 += 1

    # Print the results
    logger.info(f"Count of label 0: {count_0}")
    logger.info(f"Count of label 1: {count_1}")
    logger.info("=" * 100)

    test_images_display = np.array([x[0] for x in test_data])  # Extract only the images
    test_labels_display = np.array([x[1] for x in test_data])  # Extract only the labels

    # Check the shape and an example of the dataset
    logger.info(f"Shape of train images: {train_images_display.shape}")
    logger.info(f"Shape of validation images: {val_images_display.shape}")
    logger.info("=" * 100)

    # =================

    # Normalize the images in the training dataset
    train_images, train_labels = normalize_images(train_data)
    val_images, val_labels = normalize_images(val_data)
    test_images, test_labels = normalize_images(test_data)

    # Check the shape and an example of the normalized and shuffled data
    logger.info(f"Shape of normalized and shuffled test images: {test_images.shape}")
    logger.info(f"Shape of normalized and shuffled train images: {train_images.shape}")
    logger.info(f"Shape of normalized and shuffled validation images: {val_images.shape}")
    logger.info("=" * 100)

    # ===============================

    # Convert the images and labels to PyTorch tensors

    # Apply the transformation to training and validation images
    train_images_tensor = torch.stack([torch.tensor(img, dtype=torch.float) for img in train_images]).unsqueeze(1)
    train_images_tensor = torch.flatten(train_images_tensor, start_dim=2)

    val_images_tensor = torch.stack([torch.tensor(img, dtype=torch.float) for img in val_images]).unsqueeze(1)
    val_images_tensor = torch.flatten(val_images_tensor, start_dim=2)

    test_images_tensor = torch.stack([torch.tensor(img, dtype=torch.float) for img in test_images]).unsqueeze(1)
    test_images_tensor = torch.flatten(test_images_tensor, start_dim=2)

    # Now permute them
    train_images_tensor = train_images_tensor.permute(0, 1, 2)  # (N, 1, 244 x 244)
    val_images_tensor = val_images_tensor.permute(0, 1, 2)
    test_images_tensor = test_images_tensor.permute(0, 1, 2)  # (N, 1, 244 x 244)
    logger.info(f"test data shape {test_images_tensor.shape}")
    logger.info("=" * 100)

    # The tensors are now in the shape (N, 1, 244, 244), where N is the number of images

    train_labels_tensor = torch.tensor(train_labels, dtype=torch.long)
    val_labels_tensor = torch.tensor(val_labels, dtype=torch.long)
    test_labels_tensor = torch.tensor(test_labels, dtype=torch.long)

    # Create the dataset and DataLoader
    train_dataset = TensorDataset(train_images_tensor, train_labels_tensor)
    val_dataset = TensorDataset(val_images_tensor, val_labels_tensor)
    test_dataset = TensorDataset(test_images_tensor, test_labels_tensor)

    # Define the batch size
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    logger.info(('DataLoader for train_loader and val_loader and test_loader done!'))
    logger.info("=" * 100)

    logger.info("DataLoaders ready.")
    logger.info("=" * 100)
    

    model = FourierKAN([224 * 224, 128, 64, 2])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 6) Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_learning_rate, weight_decay=1e-5)

    # 7) Train
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

    # 8) Testing/Evaluation
    best_model_weights = os.path.join(model_path, 'best_model_resnet.pth')
    checkpoint = torch.load(best_model_weights, map_location=device, weights_only=True)

    new_state_dict = {}
    for k, v in checkpoint.items():
        if k.startswith("module."):
            # Remove "module."
            k = k.replace("module.", "", 1)
        new_state_dict[k] = v
    model.load_state_dict(new_state_dict)

    # Evaluate on Validation
    val_loss, val_acc, val_prec, val_rec, val_f1 = evaluate_model(model, val_loader, criterion)
    logger.info("Validation Metrics:")
    logger.info(f"Loss:      {val_loss:.4f}")
    logger.info(f"Accuracy:  {val_acc:.4f}")
    logger.info(f"Precision: {val_prec:.4f}")
    logger.info(f"Recall:    {val_rec:.4f}")
    logger.info(f"F1-Score:  {val_f1:.4f}")
    logger.info("=" * 100)
    metrics_file = os.path.join(model_path, "evaluation_metrics.txt")
    save_metrics_to_file(metrics_file, "Validation", val_loss, val_acc, val_prec, val_rec, val_f1)


    test_loss, test_acc, test_prec, test_rec, test_f1 = evaluate_model(model, test_loader, criterion)

    logger.info("\n==== Final Test Metrics ====")
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Accuracy: {test_acc*100:.2f}%")
    logger.info(f"Precision: {test_prec:.4f}")
    logger.info(f"Recall:    {test_rec:.4f}")
    logger.info(f"F1-score:  {test_rec:.4f}")
    logger.info("=" * 100)

    save_metrics_to_file(metrics_file, "Test", test_loss, test_acc, test_prec, test_rec, test_rec)

if __name__ == "__main__":
    main()
