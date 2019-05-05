from tensorflow.keras.layers import Dense, LSTM, Dropout
import tensorflow as tf
import pprint
import os
import datetime


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

def modelv2_dyn(ltsm_units, num_frames):
    model = tf.keras.Sequential()
    model.add(LSTM(ltsm_units, input_shape=(num_frames, 1000)))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(60, activation="softmax"))
    model.compile(
        loss=tf.losses.CategoricalCrossentropy(),
        optimizer=tf.optimizers.Adam(),
        metrics=['accuracy']
    )
    return model


def modelv3_dyn(ltsm_units, dense_units=50, num_frames=38):
    model = tf.keras.Sequential()
    model.add(LSTM(ltsm_units, input_shape=(num_frames, 1000)))
    model.add(Dropout(0.5))
    model.add(Dense(dense_units))
    model.add(Dense(dense_units, activation="relu"))
    model.add(Dense(60, activation="softmax"))
    model.compile(
        loss=tf.losses.CategoricalCrossentropy(),
        optimizer=tf.optimizers.Adam(),
        metrics=['accuracy']
    )
    return model
import main
def perform_modelv2_run(train=None, validation=None):
    epochs = int(os.environ["EPOCHS"]) if os.environ["EPOCHS"] else 15
    dir_prefix = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    results = {}
    train_tuple = None
    val_tuple = None
    if train is None or validation is None:
        train_tuple, val_tuple = main.testv3()
    else:
        train_tuple = train
        val_tuple = validation

    for unit_size in [32, 64, 256, 512]:
        results[unit_size] = {}
        runs = {}
        for run in range(0,10):
            model, accuracy = main.evaluate_model(modelv2_dyn(unit_size, 38), train_tuple[0], train_tuple[1], val_tuple[0], val_tuple[1], 38, epochs)
            runs[run] = float(accuracy[1])
        results[unit_size] = runs
    results.update({
        "epochs": epochs,
        "train_folder_len": train_tuple[0].shape[0],
        "val_folder_len": val_tuple[0].shape[0]
    })
    save_results(dir_prefix, results)
    return train_tuple, val_tuple


def print_results_dict(dictionary):
    pp = pprint.PrettyPrinter()
    pp.pprint(dictionary)


def print_results(results):
    for run in result:
        print("{}".format(run[1]*100))

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



def they_hate_when_then_come_tru(train, val):
    for epoch_count in ["120", "240", "480", "920"]:
        os.environ["EPOCHS"] = epoch_count
        print("[INFO] Epoch Count: {}".format(epoch_count))
        perform_modelv2_run(train, val)

