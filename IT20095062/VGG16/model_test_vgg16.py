import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# load the testing data set
test_data_dir = 'D:\\SLIIT\\Y4S2\\SE4050 - DL\\Assignmentts\\pneumonia_classification\\Data_set\\test'           

# set image size and batch size
img_width, img_height = 224, 224
batch_size = 32

# Create a data generator for the test dataset without data augmentation
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary', 
    shuffle=False  # Disable shuffling to match predictions with true labels
)

# Load the trained model
model = tf.keras.models.load_model('D:/SLIIT/Y4S2/SE4050 - DL/Assignmentts/Deep_Learning_project/IT20095062/pneumonia_vgg16_model2.h5')

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(test_generator)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Predict the labels for the test dataset
predictions = model.predict(test_generator)
predicted_classes = (predictions > 0.5).astype(int)  # Convert probabilities to binary predictions

# Get the true labels
true_labels = test_generator.classes

# Generate Classification Report
classification_rep = classification_report(true_labels, predicted_classes, target_names=['Normal', 'Pneumonia'])
print("\nClassification Report:")
print(classification_rep)

# Generate Confusion Matrix
confusion = confusion_matrix(true_labels, predicted_classes)

# Plot the Confusion Matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Pneumonia'], yticklabels=['Normal', 'Pneumonia'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()



# Load training history 
with open('D:/SLIIT/Y4S2/SE4050 - DL/Assignmentts/Deep_Learning_project/IT20095062/pneumonia_vgg16_history2.pkl', 'rb') as history_file:
    history = pickle.load(history_file)

# Plot training and validation accuracy
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

# Plot training and validation loss
plt.subplot(1, 2, 2)
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.tight_layout()
plt.show()