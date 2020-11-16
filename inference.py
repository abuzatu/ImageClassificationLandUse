from utils import *

INPUT_FOLDER_NAME = "./data/2750"
OUTPUT_FOLDER_NAME = "./output"
IMAGE_HEIGHT = 64 # number of x rows, or N, in a matrix of NxM
IMAGE_WIDTH = 64 # number of y rows, or M, in a matrix of NxM
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
NB_COLOR = 3 # 3 for RGB
SHUFFLE = True
MODEL_FILE_NAME = f"{OUTPUT_FOLDER_NAME}/model.h5"

image_size = (IMAGE_HEIGHT, IMAGE_WIDTH)
input_shape = (IMAGE_HEIGHT, IMAGE_WIDTH, NB_COLOR)

# logging level: NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# load dataset
valid_ds = tf.keras.preprocessing.image_dataset_from_directory(
    INPUT_FOLDER_NAME, labels='inferred', label_mode='int', class_names=None,
    color_mode='rgb', batch_size=BATCH_SIZE, image_size=image_size, shuffle=SHUFFLE, seed=42,
    validation_split=VALIDATION_SPLIT, subset="validation", interpolation='bilinear', follow_links=False
)

# name and number of classes
list_class_name = valid_ds.class_names
num_classes = len(list_class_name)

# standardize the dataset
normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)

# configure dataset for performance
AUTOTUNE = tf.data.experimental.AUTOTUNE
valid_ds = valid_ds.cache().prefetch(buffer_size=AUTOTUNE)

# data augmentation
data_augmentation = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
  tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])

# base model taken from Res-Net50 already pre-trained
base_model = ResNet50(input_shape=input_shape, include_top=False, weights="imagenet")
for layer in base_model.layers:
    layer.trainable=False

# create an empty model
model = get_model(input_shape, data_augmentation, normalization_layer, base_model, num_classes)

# load the weights that we already trained
model.load_weights(MODEL_FILE_NAME)

option = "validation"

# evaluate loss and accuracy
evaluate_model(model, option, valid_ds)

# predict for the first batch of the validation dataset
predict_for_a_batch(model, list_class_name, option, valid_ds, index_batch = 1, output_folder_name = OUTPUT_FOLDER_NAME)

# predict on a particular image
image_file_name = f"{INPUT_FOLDER_NAME}/Highway/Highway_624.jpg"
predict_for_an_image(model, image_file_name, list_class_name, image_size)

# evaluate the confusion matrix on all the data in the validation dataset
labels_true_valid, labels_pred_valid = predict(model, option, valid_ds, BATCH_SIZE)
evaluate_confusion_matrix(option, labels_true_valid, labels_pred_valid, OUTPUT_FOLDER_NAME)
