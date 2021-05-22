import numpy as np
import os
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pydoc import classname
import tensorflow_addons as tfa

# Set parameters
batch_size = 32
img_height = 224
img_width = 224
IMG_SIZE = (img_height,img_width)
initial_epochs = 12
fine_tune_epochs = 12

# Read and store datasets
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    'D:\\ZJU_graduate\\Graduation Design\\Graduation Project\\1_Classification Standard\\dataset-resized',
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    'D:\\ZJU_graduate\\Graduation Design\\Graduation Project\\1_Classification Standard\\dataset-resized',
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_dataset.class_names
print(class_names)

#Show sample pictures
#plt.figure(figsize=(10, 10))
#for images, labels in train_dataset.take(1):
#    for i in range(9):
#        ax = plt.subplot(3, 3, i + 1)
#        plt.imshow(images[i].numpy().astype("uint8"))
#        plt.title(class_names[labels[i]])
#        plt.axis("off")
    
val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 5)
validation_dataset = validation_dataset.skip(val_batches // 5)
number_train_batches = tf.data.experimental.cardinality(train_dataset).numpy()
print('Number of training batches:', number_train_batches)
print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_dataset))
print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

#Perform data augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal_and_vertical'),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])

# Show the data augmentation example
example_train_image, label = next(iter(train_dataset))

plt.figure(figsize=(10, 10))
for i in range(9):
    augmented_image = data_augmentation(example_train_image)
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(augmented_image[0]/255)
    plt.axis("off")
plt.show()

# Create a base model from ResNet101V2
preprocess_input = tf.keras.applications.resnet_v2.preprocess_input
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.ResNet101V2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
image_batch, label_batch = next(iter(train_dataset))
feature_batch = base_model(image_batch)
print(feature_batch.shape)

base_model.trainable = False
base_model.summary()

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)
prediction_layer = tf.keras.layers.Dense(6)
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)

#Save base model
base_model.save('D:\\ZJU_graduate\\Graduation Design\\Graduation Project\\3_Model Optimization\\model_ResNet101V2_base')

#Add the classification head
inputs = tf.keras.Input(shape=(img_height, img_width, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

# Set cyclic learning rate
k_clr = 2
cyclic_learning_rate_base = tfa.optimizers.CyclicalLearningRate(
    initial_learning_rate=1e-8,
    maximal_learning_rate=1e-3,
    step_size = k_clr*number_train_batches,
    scale_fn=lambda x: 1 / (2.0 ** (x - 1)),
    scale_mode='cycle')
base_learning_rate = 0.0001 # base learning rate

# Compile model with the classification head
optimizer_base = tf.keras.optimizers.Adam(learning_rate = cyclic_learning_rate_base)
loss_base = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer = optimizer_base,
              loss = loss_base,
              metrics=['accuracy'])
model.summary()

print("Number of layers in the base model: ", len(base_model.layers))
print("Number of layers in the model: ", len(model.layers))

#Train the model
loss0, accuracy0 = model.evaluate(validation_dataset)
print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))
history = model.fit(train_dataset,
                    epochs=initial_epochs,
                    validation_data=validation_dataset)

# Plot loss and accuracy
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,2.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

#Save model before fine-tuning
model.save('D:\\ZJU_graduate\\Graduation Design\\Graduation Project\\3_Model Optimization\\model_ResNet101V2_noFT')

# Perform fine-tuning
base_model.trainable = True
print("Number of layers in the base model: ", len(base_model.layers))

# Fine-tune from this layer
fine_tune_at = 364

# Freeze all the layers before the fine_tune_at layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable =  False

# Compile the new model
cyclic_learning_rate_finetune = tfa.optimizers.CyclicalLearningRate(
    initial_learning_rate=1e-9,
    maximal_learning_rate=1e-4,
    step_size = k_clr*number_train_batches,
    scale_fn=lambda x: 1 / (2.0 ** (x - 1)),
    scale_mode='cycle')

optimizer_fine = tf.keras.optimizers.RMSprop(learning_rate=cyclic_learning_rate_finetune)
loss_fine = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer = optimizer_fine,
              loss = loss_fine,
              metrics=['accuracy'])
model.summary()


#model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#              optimizer = tf.keras.optimizers.RMSprop(learning_rate=cyclic_learning_rate_finetune),
#              metrics=['accuracy'])
#model.summary()
len(model.trainable_variables)

total_epochs =  initial_epochs + fine_tune_epochs

# Train the fine-tuning model
history_fine = model.fit(train_dataset,
                         epochs=total_epochs,
                         initial_epoch=history.epoch[-1],
                         validation_data=validation_dataset)

acc += history_fine.history['accuracy']
val_acc += history_fine.history['val_accuracy']

loss += history_fine.history['loss']
val_loss += history_fine.history['val_loss']

# Plot loss and accuracy
plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.ylim([0.5, 1])
plt.plot([initial_epochs-1,initial_epochs-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.ylim([0, 2.0])
plt.plot([initial_epochs-1,initial_epochs-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

#Test the model
loss, accuracy = model.evaluate(test_dataset)
print('Test accuracy :', accuracy)

# Save the model
model.save('D:\\ZJU_graduate\\Graduation Design\\Graduation Project\\3_Model Optimization\\model_ResNet101V2_FTat364')

# predict a single image
file_name = 'D:\\ZJU_graduate\\Graduation Design\\Graduation Project\\1_Classification Standard\\dataset-resized\\metal\\metal5.jpg'

image_jpg = tf.io.read_file(file_name)
image_encoded = tf.image.decode_jpeg(image_jpg)
image_decoded = tf.image.convert_image_dtype(image_encoded, tf.uint8)
image_tensor = tf.image.resize(image_decoded, [img_height, img_width])
image_tensor1 = image_tensor[None,:,:,:]

single_prediction = model(image_tensor1)
single_score = tf.nn.softmax(single_prediction)
single_predicted_id = np.argmax(single_score)

plt.figure()
plt.imshow(mpimg.imread(file_name))
plt.title(class_names[single_predicted_id])

# Show prediction examples
plt.figure(figsize=(10, 10))
#test_images = test_dataset.take(2)
#predictions = model.predict(test_images)

image_batch, label_batch = test_dataset.as_numpy_iterator().next()
predictions = model.predict_on_batch(image_batch)

print('Labels:\n', label_batch)
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    score = tf.nn.softmax(predictions[i]);
    predicted_id = np.argmax(score)
    plt.imshow(image_batch[i].astype("uint8"))
    plt.title(class_names[predicted_id])
    print(predicted_id)
    print(label_batch[i])
    print(score)
    plt.axis("off")
plt.show()

