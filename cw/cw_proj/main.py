import cv2 as cv
import numpy as np
import random
import os
import fnmatch
import re
import time
import skimage
import tensorflow as tf
from pathlib import Path

CAD60_DIR = "" #os.environ["CAD_DIR"]
NTU_DIR = os.environ["NTU_DIR"]


def load_cad60labels():
    labels = {}
    with open("%s/activityLabel.txt" % CAD60_DIR) as fh:
        index = 0
        for line in fh.readlines():
            if line == "END\n":
                continue
            (label, label_name, _) = line.split(",")
            print("[INFO] Index: %s" % index)
            labels[label] = (label_name, index)
            index += 1
    return labels


def load_CAD60_dataset_OLD():
    folders = {}
    for folder in [x[0] for x in os.walk(CAD60_DIR)][1:]:
        print("[INFO] Folder: %s" % folder)
        folder_name = folder.split("/")[-1]
        folders[folder_name] = []
        #  lambda x: int(re.search("Depth_(.*).png", x)
        for image in sorted(
            os.listdir(folder),
            key=lambda x: int(re.search("[A-z]*_(.*).png", x).group(1))
        ):
            if fnmatch.fnmatch(image, "Depth_*.png"):
                print("[INFO] Loading image %s" % image)
                past_time = time.time()
                img = None
                img = cv.imread("%s/%s" % (folder, image))
                folders[folder_name].append(img)

    # concatonate arrays
    for folder in folders:
        print("[INFO] Flattening: %s" % folder)
        folders[folder] = np.array(folders[folder]).flatten()


    print("[INFO] Keys: %s" % folders.keys())
    labels = load_cad60labels()

    class_ids_to_arrays = []
    label_names = {}

    # Read labels
    for label in labels:
        (label_name, index) = labels[label]
        class_ids_to_arrays.append(folders[label])
        label_names[index] = label_name
        index += 1

    return np.array(class_ids_to_arrays), label_names

def lable_to_index(labels):
    index = 0
    label_index = {}
    for label in labels:
        label_index[label] = index
        index += 1

    return label_index

def load_CAD60_dataset(num_samples=1):
    labels = load_cad60labels()
    label_to_index = lable_to_index(labels)
    imgs = []
    targets = []
    index = 0
    for sample in range(1, num_samples):
        for label in labels:
            img = cv.imread("%s/%s/Depth_%s.png" % (CAD60_DIR, label, sample))
            data = np.array(img, dtype=np.float32)
            # print("[INFO] Img Shape: {}".format(data.shape))
            imgs.append(data.flatten())
            targets.append(label_to_index[label])
            index += 1
    return imgs, targets


def load_NTU_labels(dataset_dir):
    lines = open(Path("{}/ntu_labels.txt".format(dataset_dir)), "r").readlines()
    labels = {}
    count = 1
    for line in lines:
        labels[count] = line
        count += 1
    return labels


def load_NTU_dataset(num_samples=1):
    NTU_masked = os.environ["NTU_DIR_masked"]
    NTU_depth = os.environ["NTU_DIR_depth"]
    labels = load_NTU_labels(NTU_masked)

    id2masks = {}
    id2depth = {}

    print("[INFO] Loading dataset")

    def map2fullpath(path):
        return list(
            map(
                lambda x: "{}/{}".format(path, x),
                os.listdir(path)
            )
        )

    for lbl_id in labels:
        # zfill - adds padding zeros.
        # IE: str(1).zfill(3) == 001... or .zfill(2) == 01... and so on.
        masked_dir = Path(
            "{}/S001C001P001R001A{}".format(
                NTU_masked,
                str(lbl_id).zfill(3)
            )
        )

        masked_imgs = map2fullpath(masked_dir)

        id2masks[lbl_id] = masked_imgs
        depth_dir = Path(
            "{}/S001C001P001R001A{}".format(
                NTU_depth,
                str(lbl_id).zfill(3)
            )
        )

        depth_imgs = map2fullpath(depth_dir)
        id2depth[lbl_id] = depth_imgs

        import json
        json.dump({
            "id2depth": id2depth,
            "id2mask": id2masks
        }, open("dataset_ntu_dump.json", "w"))


    return id2masks, id2depth


def get_hog_features(
    img,
    orient,
    pix_per_cell,
    cell_per_block,
    vis=False,
    feature_vec=True
):
    features, hog_image = skimage.hog(
        img,
        orientientations=orient,
        pixels_per_cell=(pix_per_cell, pix_per_cell),
        cells_per_block=(cell_per_block, cell_per_block),
        transform_sqrt=True,
        visualise=vis,
        feature_vector=feature_vec
    )
    if vis:
        return features, hog_image
    else:
        return features

def bin_spatial(img, size=(32, 32)):
    color1 = cv.resize(img[:, :, 0], size).ravel()
    color2 = cv.resize(ing[:, :, 1], size).ravel()
    color3 = cv.resize(ing[:, :, 2], size).ravel()
    return np.hstack((color1, color2, color3))

def color_hist(img, nbins=32, bins_range=(0, 256)):
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)

    return np.concatenate((channel1_hist, channel2_hist, channel3_hist))

def convert_color(img, color_space="RGB"):
    if color_space == "HSV":
        img = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    elif color_space == "LUV":
        img = cv.cvtColor(img, cv.COLOR_RGB2LUV)
    elif color_space == "HLS":
        img = cv.cvtColor(img, cv.COLOR_RGB2HLS)
    elif color_space == "YUV":
        img = cv.cvtColor(img, cv.COLOR_RGB2YUV)
    elif color_space == "YCrCb":
        img = cv.cvtColor(img, cv.COLOR_RGB2YCrCb)
    return img




# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(image, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    file_features = []
    # apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        feature_image = convert_color(image, color_space)
    else: feature_image = np.copy(image)

    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        file_features.append(spatial_features)
    if hist_feat == True:
        # Apply color_hist()
        hist_features = color_hist(feature_image, nbins=hist_bins)
        file_features.append(hist_features)
    if hog_feat == True:
    # Call get_hog_features() with vis=False, feature_vec=True
        pass
    if hog_channel == 'ALL':
        hog_features = []
        for channel in range(feature_image.shape[2]):
            hog_features.append(get_hog_features(feature_image[:,:,channel],
                                orient, pix_per_cell, cell_per_block,
                                vis=False, feature_vec=True))
        hog_features = np.ravel(hog_features)
    else:
        hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                    pix_per_cell, cell_per_block, vis=False, feature_vec=True)
    # Append the new feature vector to the features list
    file_features.append(hog_features)
    return file_features


def create_model(input_shape, ltsm_layers, num_classes, conv_layers_num):
    #https://machinelearningmastery.com/cnn-long-short-term-memory-networks/
    cnn = tf.keras.Sequential()
    lstm = tf.keras.Sequential()


    cnn.add(tf.keras.layers.Conv2D(3, (5, 5), activation="relu", input_shape=(input_shape, input_shape, 1)))
    cnn.add(tf.keras.layers.MaxPooling2D((5, 5)))
    cnn.add(tf.keras.layers.Conv2D(3, (5, 5), activation="relu", input_shape=(input_shape, input_shape, 1)))
    cnn.add(tf.keras.layers.MaxPooling2D((5, 5)))
    cnn.add(tf.keras.layers.Flatten())
    lstm.add(tf.keras.layers.TimeDistributed(cnn))
    lstm.add(tf.keras.layers.LSTM(64))
    lstm.add(tf.keras.layers.Dense(64))
    lstm.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    return lstm



def dataset2imgs(img_dict, num_samples, img_shape):
    imgs = []
    loaded_classes = []
    class_ids = list(img_dict.keys())
    random.shuffle(class_ids)
    print("ClassIDS: {}".format(class_ids))
    count = 0
    for classid in class_ids:
        print("Loaded classes: {} Class id".format(count, classid))
        for sample in range(num_samples):
            img = cv.imread(img_dict[classid][sample], cv.IMREAD_GRAYSCALE)
            img = cv.resize(
                img,
                (img_shape, img_shape)
            ).reshape(img_shape, img_shape, 3).astype("float32")
            imgs.append(img)
            loaded_classes.append(classid)
        count += 1
    return np.array(imgs), loaded_classes


def create_model_v2(img_size, num_classes):
    # Last night's parameters Conv2D(32.... ), MaxPooling2D(....pool_size=(2, 2))
    # https://medium.com/nybles/create-your-first-image-recognition-classifier-using-cnn-keras-and-tensorflow-backend-6eaab98d14dd
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(50, (3, 3), input_shape=(img_size, img_size, 3), activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(100, (3, 3), activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Conv2D(50, (3, 3), activation="relu"))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dense(128, activation="relu"))
    model.add(tf.keras.layers.Dense(128, activation="relu"))

    model.add(tf.keras.layers.Dense(num_classes, activation="sigmoid"))

    return model

def testv2():

    model = create_model_v2(256, 60)
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    training_set = train_datagen.flow_from_directory(
        "{}/{}".format(os.environ["NTU_DIR"], "S001C001P006R001AN"),
        target_size=(256, 256),
        batch_size=32,
        class_mode="categorical"
    )


    test_set = train_datagen.flow_from_directory(
        "{}/{}".format(os.environ["NTU_DIR"], "S001C001P003R001AN"),
        target_size=(256, 256),
        batch_size=32,
        class_mode="categorical"
    )

    from IPython.display import display
    from PIL import Image
    model.compile(
        optimizer=tf.optimizers.RMSprop(),
        loss=tf.losses.CategoricalCrossentropy(),
        metrics=["accuracy"]
    )
    model.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=10,
        validation_data=test_set,
        validation_steps=800
    )
    import datetime
    model.save("run:{}".format(datetime.datetime.now()))



def rnn_train():
    # tick when done - ✓
    # create ltsm ✓
    # compile ✓

    num_samples = 50
    depth_images = load_NTU_dataset(num_samples)[0]
    image_shape =  256 # cv.imread(depth_images[1][0]).shape[:2]
    num_classes = len(depth_images.keys())

    model = create_model(image_shape, 64, num_classes)
    model.compile(
        optimizer=tf.optimizers.Adam(),
        loss=tf.losses.MeanAbsolutePercentageError()
    )

    imgs, class_ids = dataset2imgs(depth_images, num_samples, image_shape)

    # fit ✓ - working on it.

    # save



def cnn_train():
    pass


def cnn_predict():
    pass


def rnn_predict():
    pass


def read_train():
    pass

def read_labels(label_fh):
    pass

def get_files():
    pass

def train():
    num_samples = 100
    images, targets = load_CAD60_dataset(num_samples)


    svm = cv.ml.SVM_create()
    svm.setType(cv.ml.SVM_C_SVC)
    svm.setKernel(cv.ml.SVM_LINEAR)
    svm.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
    # print("[INFO] Keys: %s" % label_to_img.())
    svm.train(np.array(images), cv.ml.ROW_SAMPLE, np.array(targets))

    svm.save("trained_/svm")

    # svm.train(trainingData, cv.ml.ROW_SAMPLE, labels)
    pass



if __name__ == "__main__":
    train()