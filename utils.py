import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
np.random.seed(98383822)
tf.random.set_seed(278732344)
import keras
from tensorflow.keras import layers
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
import logging
import sklearn
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

def get_model(input_shape, data_augmentation, normalization_layer, base_model, num_classes):

    # define the model geometry
    inputs = tf.keras.Input(shape=input_shape)
    x = data_augmentation(inputs)
    x = normalization_layer(x)
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    predictions = layers.Dense(num_classes, activation= 'softmax')(x)
    model = tf.keras.Model(inputs = base_model.input, outputs = predictions)

    logging.info(model.summary())
    
    # tell model how to learn (compile model)
    model.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'])
    
    return model

def evaluate_model(model, option, ds):
    loss, accuracy = model.evaluate(ds, verbose=2)
    logging.info(f"For {option}: loss={loss:.3f}, accuracy={accuracy:.3f}")

def predict_for_a_batch(model, list_class_name, option, ds, index_batch = 1, output_folder_name = "./output"):
    # predict for some images taken from the validation dataset
    for batch_image, batch_label in ds.take(index_batch):
        # check the images from this first batch
        for i in range(len(batch_image)):
            img_array = batch_image[i].numpy().astype("uint8")
            img_array_2 = tf.expand_dims(img_array, 0)  # Create batch axis
            predictions = model.predict(img_array_2)
            list_prediction = predictions[0]
            logging.debug(img_array.shape)
            fig, ax = plt.subplots(figsize=(8, 6))
            plt.imshow(img_array)
            index = np.argmax(list_prediction)
            predicted_label = list_class_name[index]
            true_label = list_class_name[batch_label[i]]
            title = f"true={true_label}, predicted={predicted_label}"
            # title += f"pred_values={[f'{prediction:.2f}' for prediction in list_prediction]}"
            plt.title(title, fontsize = 12)
            plt.axis("off")
            plt.savefig(f"{output_folder_name}/plot_{option}_image_{str(i).zfill(3)}.png")
            plt.close()

def predict_for_an_image(model, image_file_name, list_class_name, image_size):
    img = keras.preprocessing.image.load_img(image_file_name, target_size=image_size)
    img_array = keras.preprocessing.image.img_to_array(img)
    logging.info(img_array.shape)
    img_array = tf.expand_dims(img_array, 0)  # create batch axis
    logging.info(img_array.shape)
    predictions = model.predict(img_array)
    list_prediction = predictions[0]
    logging.info(f"Sum of fractions for predictions is indeed 1.0 (within rounding errors), as sum = {sum(list_prediction)}")
    [logging.info("Class name: {:20} prediction: {:.3f}".format(*pair)) for pair in zip(list_class_name, list_prediction)]
    predicted_class = list_class_name[np.argmax(list_prediction)]
    logging.info(f"Predicted class is {predicted_class}.")

def predict(model, option, ds, batch_size):
    logging.info(f"Running inference on the entire dataset of type {option}. This may take a few minutes ...")
    logging.info("Printing out the labels true and predicted for the first 5 images.")
    logging.info(f"The {option} dataset has {len(ds)} batches of images, each with a batch_size={batch_size}, for a total of {len(ds)*batch_size} images.")
    for i, batch in enumerate(ds):
        labels_true_current = batch[1].numpy()
        labels_pred_0 = model.predict(batch[0])
        labels_pred_current = np.argmax(labels_pred_0, axis = 1)
        if i%1000 == 0:
            logging.info(f"i={i}")
        if i < 5:
            logging.info(f"{i}, true, shape={labels_true_current.shape}, {labels_true_current}")
            logging.info(f"{i}, pred, shape={labels_pred_current.shape}, {labels_pred_current}")
            logging.debug(f"{i}, predicted_0, shape={labels_pred_0.shape}, {labels_pred_0}")
        if i == 0:
            labels_true = labels_true_current
            labels_pred = labels_pred_current
        else:
            labels_true = np.concatenate((labels_true, labels_true_current), axis = 0)
            labels_pred = np.concatenate((labels_pred, labels_pred_current), axis = 0)
    return (labels_true, labels_pred)

def evaluate_confusion_matrix(option, labels_true, labels_pred, output_folder_name = "./output"):
    logging.info(f"Confusion matrix for {option}:")
    confusion_matrix_1 = confusion_matrix(labels_true, labels_pred)
    logging.info(f"\n{confusion_matrix_1}")
    data = {
        'labels_true': labels_true,
        'labels_pred': labels_pred,
        }
    df = pd.DataFrame(data, columns=['labels_true','labels_pred'])
    confusion_matrix_2 = pd.crosstab(df['labels_true'], df['labels_pred'], rownames=['True'], colnames=['Predicted'], margins = False)
    sn.heatmap(confusion_matrix_2, annot=True)
    plt.title(f"Confusion matrix for {option}")
    plt.savefig(f"{output_folder_name}/plot_{option}_confusion_matrix.png")
    plt.close()
