from tensorflow.keras.layers import Dense, LSTM, Dropout


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
    model.add(LSTM(100, input_shape=(num_frames, 1000)))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(60, activation="softmax"))
    model.compile(
        loss=tf.losses.CategoricalCrossentropy(),
        optimizer=tf.optimizers.Adam(),
        metrics=['accuracy']
    )

import main
def perform_modelv2_run():
    results = {}
    train_tuple, val_tuple = main.testv3()
    for unit_size in ["32", "64", "256", "512"]:
        results[unit_size] = {}
        runs = {}
        for run in range(0,10):
            model, accuracy = main.evaluate_model(train_tuple[0], train_tuple[1], val_tuple[0], val_tuple[1], 38)
            runs[run] = accuracy[1]
        results[unit_size] = runs

    save_results(results)
    return train_tuple, val_tuple



def print_results(results):
    for run in results:
        print("{}".format(run[1]*100))

def save_results(results):
    import json
    import datetime

    json.dump(results, open("run:{}".format(datetime.datetime.now())))
