import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the data directory
#data_dir = 'D:\\SLIIT\\Y4S2\SE4050 - DL\\Assignmentts\\pneumonia_classification\\Data_set\\train'
data_dir = 'D:\\SLIIT\\Y4S2\SE4050 - DL\\Assignmentts\\Deep_Learning_project\\IT20095062\\Data_set\\train'

# Define image size and batch size
img_width, img_height = 224, 224
batch_size = 32

# Create data generators with data augmentation for training
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Split data into training and validation
)

train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',  # For binary classification (pneumonia or normal)
    subset='training'  # Specify this as the training set
)

validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation'  # Specify this as the validation set
)

# Load the VGG16 model with pre-trained weights (excluding the top layer)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Freeze the layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers for binary classification
x = Flatten()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)

# Create the custom model
model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 10  # Adjust as needed
model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator
)

# Save the trained model
model.save('pneumonia_vgg16_model.h5')
