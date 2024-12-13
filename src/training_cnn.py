# ### Imports and Configuration
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import seaborn as sns

# ### Environment Check
# Check for GPU availability
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPU(s) available: {gpus}")
else:
    print("No GPU found. Running on CPU.")

# ### Model Definition
# Assign a name to the model
model_name = "my_cnn_model"

# Create the Sequential model
model = Sequential(name=model_name)
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3), kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.01)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
)

print(model.summary())

# ### Data Preparation
# Paths
train_path = '../data/chest_xray/train'
val_path = '../data/chest_xray/val'
test_path = '../data/chest_xray/test'

# Data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Rescaling for validation and test
val_datagen = ImageDataGenerator(rescale=1./255)

# Generators
train_generator = train_datagen.flow_from_directory(
    train_path, target_size=(224, 224), batch_size=8, shuffle=True, class_mode='binary'
)

validation_generator = val_datagen.flow_from_directory(
    val_path, target_size=(224, 224), batch_size=8, class_mode='binary'
)

# Check number of samples
print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {validation_generator.samples}")

# ### Callbacks Setup
log_dir = os.path.join("logs", "fit", model_name)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(checkpoint_dir, f"{model_name}_epoch_{{epoch:02d}}.keras"),
    monitor='val_loss', save_best_only=False, save_weights_only=False, mode='min', verbose=1
)

reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

# ### Model Training
# Train the model
print("Training started...")
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator,
    callbacks=[checkpoint_callback, tensorboard_callback, reduce_lr_callback]
)
print("Training complete.")

# Save the model inside a folder named after the model
model_folder = os.path.join("saved_models", model_name)
os.makedirs(model_folder, exist_ok=True)
saved_model_path = os.path.join(model_folder, f"{model_name}.keras")
model.save(saved_model_path)
print(f"Model saved at: {saved_model_path}")

# ### Plot Training Results
# Plot training results
plt.figure(figsize=(12, 8))

# Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()

# Save plots in the model folder
plot_filename = os.path.join(model_folder, f"{model_name}_training_plot.png")
plt.savefig(plot_filename)
plt.close()
print(f"Training plots saved at: {plot_filename}")

# ### Model Testing
# Load the model
model = load_model(saved_model_path)
print(f"Loaded model: {model.name}")

# Test data generator
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_path, target_size=(224, 224), batch_size=8, class_mode='binary', shuffle=False
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
print("\nClassification Report:")
print(classification_report(true_labels, binary_predictions, target_names=['NORMAL', 'PNEUMONIA']))

# ### Plot Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Paired", xticklabels=['NORMAL', 'PNEUMONIA'], yticklabels=['NORMAL', 'PNEUMONIA'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')

# Save the confusion matrix plot in the model folder
conf_matrix_filename = os.path.join(model_folder, "Confusion_Matrix.png")
plt.savefig(conf_matrix_filename)
print(f"Confusion Matrix plot saved at: {conf_matrix_filename}")
plt.show()

