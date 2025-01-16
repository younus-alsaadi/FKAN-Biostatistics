# ### Imports and Configuration
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import seaborn as sns
import pandas as pd



# ### Environment Check
# Check for GPU availability
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU(s) available: {gpus}")
else:
    print("No GPU found. Running on CPU.")

# ### Model Definition
# Assign a name to the model
model_name = "150X150complex_cnn_model_g"

# ### Model Testing
# Load the model
model_folder = os.path.join("saved_models", model_name)
saved_model_path = os.path.join(model_folder, f"{model_name}.keras")
model = load_model(saved_model_path)
print(f"Loaded model: {model.name}")

# Test data generator
test_datagen = ImageDataGenerator(rescale=1./255)

test_path = '../data/chest_xray/test'

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(150, 150),
    batch_size=4,
    class_mode='binary',
    shuffle=False,
    color_mode='grayscale'
)

# Evaluate the model
test_loss, test_accuracy, test_precision, test_recall, test_auc = model.evaluate(test_generator)

print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
print(f"Test Precision: {test_precision}")
print(f"Test Recall: {test_recall}")
print(f"Test AUC: {test_auc}")

# ### Generate Metrics and Confusion Matrix
# Predictions
predictions = model.predict(test_generator)
binary_predictions = (predictions > 0.5).astype(int).flatten()
true_labels = test_generator.classes

# Confusion Matrix
cm = confusion_matrix(true_labels, binary_predictions)
tn, fp, fn, tp = cm.ravel()

print("Confusion Matrix:")
print(cm)
print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

# F1-Score and Classification Report
f1 = f1_score(true_labels, binary_predictions)
print(f"F1-Score: {f1}")

# Classification Report
target_names = ['NORMAL', 'PNEUMONIA']
report = classification_report(true_labels, binary_predictions, target_names=target_names, output_dict=True)

# Convert the report to DataFrame
report_df = pd.DataFrame(report).transpose()
report_df['support'] = report_df['support'].fillna(0).astype(int)

# Display the Classification Report
print("\nEvaluation Report with Macro Average and Weighted Average:")
print(report_df)

# Save the classification report as a CSV file
report_filename = os.path.join(model_folder, f"{model_name}_classification_report.csv")
report_df.to_csv(report_filename, index=True)
print(f"Classification report saved at: {report_filename}")

# ### Plot Confusion Matrix
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Paired", xticklabels=target_names, yticklabels=target_names, annot_kws={"size": 20})
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')

# Add information in the corner of the plot
plt.figtext(0.98, 0.98, "Batch Size: 4, Epochs: 100", fontsize=10,
            color="black", ha="right")
plt.figtext(0.98, 0.96, f"Test Accuracy: {test_accuracy * 100:.2f}%", fontsize=10, color="black", ha="right")
plt.figtext(0.98, 0.94, f"Test Recall: {test_recall * 100:.2f}%", fontsize=10, color="black", ha="right")
plt.figtext(0.98, 0.92, f"F1 score: {f1 * 100:.2f}%", fontsize=10, color="black", ha="right")

plt.tight_layout(rect=[0, 0, 1, 0.97])

# Save the confusion matrix plot
conf_matrix_filename = os.path.join(model_folder, "Confusion_Matrix.png")
plt.savefig(conf_matrix_filename)
print(f"Confusion Matrix plot saved at: {conf_matrix_filename}")
plt.close()

# Save the model result
print("Saving the model results...")
results_filename = os.path.join(model_folder, f"{model_name}_results_testdata2.txt")

with open(results_filename, 'w') as results_file:
    results_file.write("Description: This code implements a complex CNN.\n")
    results_file.write(f"Model Name: {model_name}\n")
    results_file.write("Batch Size: 4\n")
    results_file.write("Epochs: 100\n")
    results_file.write("\nTest Results:\n")
    results_file.write(f"Test Loss: {test_loss:.4f}\n")
    results_file.write(f"Test Accuracy: {test_accuracy * 100:.2f}%\n")
    results_file.write(f"Test Precision: {test_precision * 100:.2f}%\n")
    results_file.write(f"Test Recall: {test_recall * 100:.2f}%\n")
    results_file.write(f"Test AUC: {test_auc * 100:.2f}%\n")
    results_file.write(f"F1-Score: {f1 * 100:.2f}%\n")
    results_file.write("\nClassification Report:\n")
    results_file.write(report_df.to_string())
    results_file.write("\n\nConfusion Matrix:\n")
    results_file.write(f"TN: {tn}, FP: {fp}\n")
    results_file.write(f"FN: {fn}, TP: {tp}\n")
print(f"Model results saved at: {results_filename}")
