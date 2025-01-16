import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from convkan import ConvKAN, LayerNorm2D
from tqdm import tqdm
from sklearn.metrics import (
    precision_score,
    recall_score,
    roc_auc_score,
    accuracy_score,
    f1_score,
)
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import os
from sklearn.utils import class_weight
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KAN_CNN(nn.Module):
    def __init__(self):
        super(KAN_CNN, self).__init__()
        self.features = nn.Sequential(
            ConvKAN(1, 32, kernel_size=3, padding=1, stride=1),
            LayerNorm2D(32),
            nn.ReLU(inplace=True),
            
            ConvKAN(32, 32, kernel_size=3, padding=1, stride=2),
            LayerNorm2D(32),
            nn.ReLU(inplace=True),
            
            ConvKAN(32, 10, kernel_size=3, padding=1, stride=2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(10, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x



def main():
   
    # Paths to your datasets
    train_path = "../data/chest_xray/train"
    val_path = "../data/chest_xray/val"
    test_path = "../data/chest_xray/test"
    model_name = "model1_KAN2_Random_Erasin"
    model_folder = "saved_models"

    # Define transformations for training with augmentation
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((256, 256)),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), shear=0.2),
        # Randomly crop the 256×256 image to 224×224
        transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.2)),
        transforms.ToTensor(),
        
        
        # ---------- ADD THIS RANDOM ERASING ----------
        transforms.RandomErasing(
            p=0.5,            # Probability of erasing
            scale=(0.02, 0.33),  # Range of proportion of erased area
            ratio=(0.3, 3.3),    # Aspect ratio range of the erased area
            value=0,             # Pixel value for the erased region (0 for black)
            inplace=False
            ),
        # ---------------------------------------------
        transforms.Normalize((0.1307,), (0.3081,))
        
    ])

    # Define transformations for validation and test (Rescaling only)
    val_test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    logger.info("Define transformations done")
    logger.info("=" * 100)

    # Load datasets
    train_dataset = datasets.ImageFolder(root=train_path, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_path, transform=val_test_transform)

    # Create data loaders
    batch_size = 4
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    logger.info("Load dataset and create data loaders done")
    logger.info("=" * 100)

    # Check number of samples
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")

    """Handle class imbalance by applying class weights during training"""
    train_labels = [label for _, label in train_dataset.imgs]
    class_indices = train_dataset.class_to_idx
    logger.info(f"Class Indices: {class_indices}")

    # Compute class weights
    class_weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_labels),
        y=train_labels,
    )
    class_weights = dict(enumerate(class_weights))
    logger.info(f"Computed Class Weights: {class_weights}")

    # Initialize the model, loss function, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = KAN_CNN().to(device)

    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs")
        logger.info("=" * 100)
        model = nn.DataParallel(model)

    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    # ------------------ ADD THIS ------------------ #
    # # We use ReduceLROnPlateau to reduce LR by factor=0.5 if val_loss doesn't improve
    # # for 'patience' epochs. 
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',      # we're looking to minimize validation loss
        factor=0.5,      # LR will be halved
        patience=3,      # wait 'patience' epochs with no improvement
        verbose=True     # logs LR reduction
        )
    
    # ---------------------------------------------- #

    logger.info("Initialize the model, loss function, and optimizer done")
    logger.info("=" * 100)

    # Training the model
    num_epochs = 40
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    

    os.makedirs(model_folder, exist_ok=True)
    model_path=os.path.join(model_folder, model_name)
    os.makedirs(model_path, exist_ok=True)
    
    best_model_path = os.path.join(model_path, f"{model_name}_best.pth")

    logger.info(f"Training started with epochs {num_epochs}")
    logger.info("=" * 100)

    for epoch in range(num_epochs):
        logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        train_preds = []
        train_targets = []

        # Training Loop
        progress_bar = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}/{num_epochs}",
        )
        
        for batch_idx, (inputs, labels) in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device).float()
            optimizer.zero_grad()
            outputs = model(inputs)

            # Apply class weights during training
            weights = torch.tensor([class_weights[label.item()] for label in labels]).to(device)
            loss = criterion(outputs.squeeze(), labels)
            weighted_loss = loss * weights
            weighted_loss.mean().backward()
            optimizer.step()

            running_loss += weighted_loss.mean().item() * inputs.size(0)
            
            # Calculate metrics
            preds = (outputs.squeeze() > 0.5).float()
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)
            train_preds.extend(preds.cpu().numpy())
            train_targets.extend(labels.cpu().numpy())

            # Update progress bar
            progress_bar.set_postfix(
                {"loss": running_loss / ((batch_idx + 1) * train_loader.batch_size)}
            )

    
        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = correct_train / total_train
        epoch_precision = precision_score(train_targets, train_preds)
        epoch_recall = recall_score(train_targets, train_preds)
        
        train_losses.append(epoch_loss)
        logger.info(
        f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}, "
        f"Accuracy: {epoch_accuracy:.4f}, Precision: {epoch_precision:.4f}, Recall: {epoch_recall:.4f}"
        )

        # Validation Loop
        logger.info(f"Starting validation after epoch {epoch + 1}/{num_epochs}")
        logger.info("=" * 100)
        model.eval()
        val_running_loss = 0.0
        correct_val = 0
        total_val = 0
        val_preds = []
        val_targets = []
        
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).float()
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)
                val_running_loss += loss.item() * inputs.size(0)
                
                # Calculate metrics
                preds = (outputs.squeeze() > 0.5).float()
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        
        # Calculate epoch metrics for validation
        val_loss = val_running_loss / len(val_loader.dataset)
        val_accuracy = correct_val / total_val
        val_precision = precision_score(val_targets, val_preds)
        val_recall = recall_score(val_targets, val_preds)
        val_losses.append(val_loss)
        logger.info(
        f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}, "
        f"Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}"
        )
        logger.info("=" * 100)
        
        # ------------------ ADD THIS ------------------ #
        # # Step the scheduler with our validation loss
        scheduler.step(val_loss)
        
        # ---------------------------------------------- #

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"New best model saved with Validation Loss: {val_loss:.4f}")
            logger.info("=" * 100)

    logger.info("Training and validation complete")
    logger.info("=" * 100)

    # Plot Loss Curves
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label="Training Loss")
    plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid()
    plt.savefig(
        os.path.join(model_path, f"{model_name}_loss_curve.png")
    )
    plt.show()

    logger.info(f"Test model start it for {model_name}")
    logger.info("=" * 100)

    # Testing the model
    test_dataset = datasets.ImageFolder(root=test_path, transform=val_test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    model.load_state_dict(
        torch.load(best_model_path)
    )
    model.eval()
    test_running_loss = 0.0
    test_preds = []
    test_targets = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device).float()
            outputs = model(inputs)
            preds = (outputs.squeeze() > 0.5).float()
            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(labels.cpu().numpy())
            loss = criterion(outputs.squeeze(), labels)
            test_running_loss += loss.item() * inputs.size(0)

    test_loss = test_running_loss / len(test_loader.dataset)
    test_accuracy = accuracy_score(test_targets, test_preds)
    test_precision = precision_score(test_targets, test_preds)
    test_recall = recall_score(test_targets, test_preds)
    test_f1 = f1_score(test_targets, test_preds)
    test_auc = roc_auc_score(test_targets, test_preds)
    conf_matrix = confusion_matrix(test_targets, test_preds)
    tn, fp, fn, tp = conf_matrix.ravel()
    report_df = pd.DataFrame(
        classification_report(
            test_targets,
            test_preds,
            target_names=["NORMAL", "PNEUMONIA"],
            output_dict=True,
        )
    )

    logger.info(
        f"Test Metrics - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, AUC: {test_auc:.4f}, F1: {test_f1:.4f}"
    )
    logger.info("=" * 100)

    # Save the model results
    results_filename = os.path.join(model_path, f"{model_name}_results.txt")
    with open(results_filename, "w") as results_file:
        results_file.write("Description: This code implements a KAN+CNN with Weight Decay (L2 Regularization).\n")
        results_file.write(f"Model Name: {model_name}\n")
        results_file.write("Batch Size: 4\n")
        results_file.write("Epochs: {num_epochs}\n")
        results_file.write("\nTest Results:\n")
        results_file.write(f"Test Loss: {test_loss:.4f}\n")
        results_file.write(f"Test Accuracy: {test_accuracy * 100:.2f}%\n")
        results_file.write(f"Test Precision: {test_precision * 100:.2f}%\n")
        results_file.write(f"Test Recall: {test_recall * 100:.2f}%\n")
        results_file.write(f"Test AUC: {test_auc * 100:.2f}%\n")
        results_file.write(f"F1-Score: {test_f1 * 100:.2f}%\n")
        results_file.write("\nClassification Report:\n")
        results_file.write(report_df.to_string())
        results_file.write("\n\nConfusion Matrix:\n")
        results_file.write(f"TN: {tn}, FP: {fp}\n")
        results_file.write(f"FN: {fn}, TP: {tp}\n")
    logger.info(f"Model results saved at: {results_filename}")


if __name__ == "__main__":
    main()
