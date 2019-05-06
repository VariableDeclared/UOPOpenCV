from tensorflow.keras.layers import Dense, LSTM, Dropout
import tensorflow as tf
import pprint
import os
import datetime
import main
import numpy as np
import cv2 as cv

def modelv1(num_frames):
    model = tf.keras.Sequential()
    model.add(LSTM(100, input_shape=(num_frames, 1000)))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(60, activation="softmax"))
    model.compile(
        loss=tf.losses.CategoricalCrossentropy(),
        optimizer=tf.optimizers.Adam(),
        metrics=['accuracy']
    )

    return model

def modelv2_dyn(*args, **kwargs):
    model = tf.keras.Sequential()
    model.add(LSTM(kwargs["ltsm_units"], input_shape=(kwargs["num_frames"], 1000)))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(60, activation="softmax"))
    model.compile(
        loss=tf.losses.CategoricalCrossentropy(),
        optimizer=tf.optimizers.Adam(),
        metrics=['accuracy']
    )
    return model

# ltsm_units, dense_units=50, learning_rate=0.001, num_frames=38
def modelv3_dyn(*args, **kwargs):
    model = tf.keras.Sequential()
    model.add(LSTM(kwargs["ltsm_units"], input_shape=(kwargs["num_frames"], 1000)))
    model.add(Dropout(0.5))
    model.add(Dense(kwargs["dense_units"]))
    model.add(Dense(kwargs["dense_units"], activation="relu"))
    model.add(Dense(60, activation="softmax"))
    model.compile(
        loss=tf.losses.CategoricalCrossentropy(),
        optimizer=tf.optimizers.Adam(learning_rate=kwargs["learning_rate"]),
        metrics=['accuracy']
    )
    return model


from sklearn.metrics import accuracy_score
def eval_svm(train, val, *args, **kwargs):
    print("[DEBUG] Train Shape: {} Val shape: {}".format(train[0].shape, train[1].shape))

    if train is None or val is None:
        train, val = main.testv3()
    kernel = kwargs["svm_kernel"] if kwargs.get("svm_kernel") else cv.ml.SVM_LINEAR
    c = kwargs["C"] if kwargs.get("C") else 0
    svm = cv.ml.SVM_create()
    svm.setType(cv.ml.SVM_NU_SVC)
    svm.setKernel(kernel)
    svm.setC(c)
    svm.setTermCriteria((cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
    # print("[INFO] Keys: %s" % label_to_img.())
    svm.train(train[0].astype("float32"), cv.ml.ROW_SAMPLE, train[1].astype("int32"))

    # svm.save("trained_/svm")

    predicted = svm.predict(val[0])

    # https://stackoverflow.com/questions/19629331/python-how-to-find-accuracy-result-in-svm-text-classifier-algorithm-for-multil
    return accuracy_score(val[1], predicted)


def perform_modelv2_run(train=None, validation=None, info=None):
    epochs = int(os.environ.get("EPOCHS")) if os.environ.get("EPOCHS") else 15

    results = {}
    train_tuple = None
    val_tuple = None
    if train is None or validation is None:
        train_tuple, val_tuple = main.testv3()
    else:
        train_tuple = train
        val_tuple = validation

    num_frames = 38
    dense_units = 100
    #unit_size, 100, 0.005, num_frames
    #(unit_size, 100, 0.010, num_frames)
    models = {
        "modelv2": {
            "fn": modelv2_dyn,
            "opt_args": None
        },
        "modelv3": {
            "fn": modelv3_dyn,
            "opt_args": 0.005
        },
        "modelv4": {
            "fn": modelv3_dyn,
            "opt_args": 0.010
        }#(unit_size, 100, 0.010, num_frames)
    }

    model_count = 1
    for model in models:
        dir_prefix = "./runs/{}.{}".format(
            model,
            datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        )
        model_index = "Model:{}".format(model)
        for unit_size in [32, 64, 256, 512]:
            _model = models[model]["fn"](
                ltsm_units=unit_size,
                dense_units=dense_units,
                learning_rate=models[model]["opt_args"],
                num_frames=num_frames
            )
            results[unit_size] = {}
            results[unit_size][model_index] = {}

            with open("model_summaries/{}.{}".format(model_index, unit_size), "w") as fh:
                _model.summary(print_fn=lambda x: fh.write(x + "\n"))
            runs = {}
            for run in range(0,10):
                accuracy = main.evaluate_model(
                    _model,
                    train_tuple[0],
                    train_tuple[1],
                    val_tuple[0],
                    val_tuple[1],
                    num_frames,
                    epochs
                )
                print("[DEBUG] Accuracy: {}".format(accuracy[1]))
                runs[run] = float(accuracy[1])
            results[unit_size][model_index]["runs"] = runs
            model_count += 1
        results.update({
            "epochs": epochs,
            "train_folder_len": train_tuple[0].shape[0],
            "val_folder_len": val_tuple[0].shape[0],
            "info": info,
            "model": model,
            "model_params": models
        })
        save_results(dir_prefix, results)
    return train_tuple, val_tuple






def print_results_dict(dictionary):
    pp = pprint.PrettyPrinter()
    pp.pprint(dictionary)


def print_results(results):
    for run in results:
        print("{}".format(run[1]*100))


def load_data():
    DATA_DIR = os.environ["SAVED_TRAIN_DATA"] if os.environ.get("SAVED_TRAIN_DATA") else "./data"

    x_train = np.load("{}/x_train.npy".format(DATA_DIR))
    x_val = np.load("{}/x_val.npy".format(DATA_DIR))
    y_train = np.load("{}/y_train.npy".format(DATA_DIR))
    y_val = np.load("{}/y_val.npy".format(DATA_DIR))

    return (x_train, y_train), (x_val, y_val)


def save_results(prefix="run", results={}):
    import json
    from pathlib import Path
    path = Path("./{}".format(prefix))

    if not os.path.isdir(path):
        os.mkdir(path)

    # https://stackoverflow.com/questions/7999935/python-datetime-to-string-without-microsecond-component
    file_name = Path(
            "./{}/run:{}.json".format(
            prefix,
            datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        )
    )
    json.dump(results, open(file_name, "w"))
    print_results_dict(results)
    print("[INFO]: Value Saved to {}".format(file_name))

def run_svm(train, val, info, **kwargs):
    svm_results = {}
    for run in range(1, 10):
        accuracy = eval_svm(train, val, kwargs)
        svm_results[run] = accuracy
    svm_results.update({
        "info": info,
    })
    save_results("SVM", svm_results)


def they_hate_when_then_come_tru(train=None, val=None):
    K_FOLD = int(os.environ.get("K_FOLD")) if os.environ.get("K_FOLD") else 10
    if train is None or val is None:
        train_tuple, val_tuple = main.testv3()
        train, val = train_tuple, val_tuple
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=K_FOLD)
    kf.get_n_splits(train)

    info = {
        "KFold": K_FOLD
    }
    for train_index, test_index in kf.split(train[1]):
        X_train, X_test = train[0][train_index], train[0][test_index]
        y_train, y_test = train[1][train_index], train[1][test_index]

        # run_epochs = ["60", "120", "240", "480"]
        # run_epochs = ["120", "240", "480", "920"]
        run_epochs = ["240"]
        for kernel in [cv.ml.SVM_LINEAR, cv.ml.SVM_INTER, cv.ml.SVM_CHI2]:
            for c in [0, 0.001, 0.01]:
                i = 0
                info.update({
                    "C": c,
                    "kernel": kernel
                })
                while i < train[0].shape[0]:
                    run_svm(
                        (X_train[i], np.full((X_train[i].shape[0], 60), y_train[i])),
                        (X_test[1], np.full((X_test[1].shape[0], 60), y_test[i])),
                        info,
                        C=c,
                        kernel_type=kernel
                    )
                    i += 1
        for epoch_count in run_epochs:
            os.environ["EPOCHS"] = epoch_count
            print("[INFO] Epoch Count: {}".format(epoch_count))
            perform_modelv2_run(
                (X_train, y_train),
                (X_test, y_test),
                info
            )

    return train, val



