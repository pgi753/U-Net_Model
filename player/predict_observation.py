from player.player import Player
from typing import Dict, List, Optional
import numpy as np
import tensorflow as tf
from collections import deque
import random
import pandas as pd
import os
import shutil
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from PIL import Image

np.set_printoptions(precision=4)


def save_csv(result: list, path, model_num: int):
    path += ('result_%d.csv' % model_num)
    sc = pd.DataFrame(result, columns=['result_percentage', 'predict_one', 'predict_zero', 'obs_one', 'obs_zero'])
    sc.to_csv(path, index=False)


def save_graph(train_loss: list, train_acc: list, path: str, model_num: int,
               val_x: list, val_loss: list, val_acc: list):
    path += 'result_graph_%d' % model_num
    plt.figure(figsize=(15, 9))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, 'r', label='train loss')
    plt.plot(val_x, val_loss, 'y', label='val loss')
    plt.xlabel('batch')
    plt.ylabel('loss')
    plt.legend(loc='upper right')

    plt.subplot(1, 2, 2)
    plt.plot(train_acc, 'b', label='train acc')
    plt.plot(val_x, val_acc, 'g', label='val acc')
    plt.xlabel('batch')
    plt.ylabel('accuracy')
    plt.legend(loc='lower right')
    plt.show()
    plt.savefig(path)


def save_txt(path: str, test_result: list, channel_num: int, data_size: int, epoch: int):
    path += 'test_result.txt'
    f = open(path, 'w')
    f.write('Loss: %.4f, Accuracy: %.4f \n' % (test_result[0], test_result[1]))
    f.write('Channel num: %d \nData set size: %d \nepoch: %d' % (channel_num, data_size, epoch))
    f.close()


class predictObservation(Player):
    def __init__(self, identifier: str, observation_history_length: int, dnn_learning_rate: float,
                 scenario: int, modelNumber: int):
        super().__init__(identifier)
        self._freq_channel_list: List[int] = []
        self._num_freq_channel = 0
        self._freq_channel_combination = []
        self._num_freq_channel_combination = 0
        self._num_action = 0
        self._observation_history_length = observation_history_length
        self._observation_history = np.empty(0)
        self._cca_thresh = -70
        self._data_set = deque()
        self._observation_history_list = deque()
        self._real_output_list = deque()
        self._latest_observation_dict = None
        self._main_dnn: Optional[tf.keras.Model] = None
        self._dnn_learning_rate = dnn_learning_rate
        self._scenario = scenario
        self._modelNumber = modelNumber
        self._result = []
        self._saveCSV = 0
        self._model_path = 'savedModel/scenario_%d/model_%d/' % (self._scenario, self._modelNumber)

        if not os.path.isdir(self._model_path):
            os.mkdir(self._model_path)

    @staticmethod
    def downsample(filters, size, apply_batchnorm=True):
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        result.add(tf.keras.layers.Conv2D(filters, size, strides=(2, 1), padding='same',
                                          kernel_initializer=initializer, use_bias=False))
        if apply_batchnorm:
            result.add(tf.keras.layers.BatchNormalization())
        result.add(tf.keras.layers.LeakyReLU())

        return result

    @staticmethod
    def upsample(filters, size, strides=(2, 1), apply_dropout=False):
        initializer = tf.random_normal_initializer(0., 0.02)
        result = tf.keras.Sequential()
        result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=strides, padding='same',
                                                   kernel_initializer=initializer, use_bias=False))
        if apply_dropout:
            result.add(tf.keras.layers.Dropout(0.5))
        result.add(tf.keras.layers.ReLU())

        return result

    def Generator(self, shape):
        inputs = tf.keras.layers.Input(shape=shape)

        down_stack = [
            self.downsample(64, (32, 1), apply_batchnorm=False),
            self.downsample(128, (32, 1)),
            self.downsample(256, (32, 1)),
            self.downsample(512, (32, 1)),
            self.downsample(512, (32, 1)),
            self.downsample(512, (32, 1)),
            self.downsample(512, (32, 1)),
            self.downsample(512, (32, 1)),
        ]

        up_stack = [
            self.upsample(512, (32, 1), apply_dropout=True),
            self.upsample(512, (32, 1), apply_dropout=True),
            self.upsample(512, (32, 1), apply_dropout=True),
            self.upsample(512, (32, 1)),
            self.upsample(256, (32, 1)),
            self.upsample(128, (32, 1)),
            self.upsample(64, (32, 1)),
        ]

        initializer = tf.random_normal_initializer(0., 0.02)
        last = tf.keras.layers.Conv2DTranspose(1, (32, 1), strides=(2, 1), padding='same',
                                               kernel_initializer=initializer, activation='sigmoid')
        x = inputs
        skips = []
        for down in down_stack:
            x = down(x)
            skips.append(x)
            print(x.shape)

        skips = reversed(skips[:-1])

        for up, skip in zip(up_stack, skips):
            x = up(x)
            x = tf.keras.layers.Concatenate()([x, skip])
            print(x.shape)
        # x = inputs
        # for down in down_stack:
        #     x = down(x)
        #
        # for up in up_stack:
        #     x = up(x)
        #
        x = last(x)
        return tf.keras.Model(inputs=inputs, outputs=x)

    def connect_to_server(self, server_address: str, server_port: int):
        super().connect_to_server(server_address, server_port)
        self._freq_channel_list = self.operator_info['freq channel list']
        self._num_freq_channel = len(self._freq_channel_list)
        self._observation_history = np.zeros((self._observation_history_length, self._num_freq_channel, 1))
        initial_action = {'type': 'sensing'}
        self._latest_observation_dict = self.step(initial_action)
        self.update_observation_history(self._latest_observation_dict)
        self._main_dnn = self.Generator(shape=[self._observation_history_length, self._num_freq_channel, 1])
        self._main_dnn.compile(optimizer=Adam(learning_rate=self._dnn_learning_rate), loss="binary_crossentropy",
                               metrics=['accuracy'])

    def train_dnn(self, data_set_size: int, mini_batch_size: int, dnn_epochs: int, progress_report: bool):
        self.accumulate_data_set(data_set_size, progress_report)

        x = np.stack([x for x in self._observation_history_list], axis=0)
        y = np.stack([x for x in self._real_output_list], axis=0)

        custom_train_val_history = Custom_train_val_History()

        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6, random_state=1, shuffle=True)
        x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=1, shuffle=True)

        self._main_dnn.fit(x_train, y_train, batch_size=mini_batch_size, epochs=dnn_epochs,
                           validation_data=(x_val, y_val), callbacks=[custom_train_val_history])

        print('<<<<<Test Result >>>>>')
        loss_and_metrics = self._main_dnn.evaluate(x_test, y_test, batch_size=mini_batch_size)
        # print('<Test result> loss: %0.4f, acc: %0.4f' % (loss_and_metrics[0], loss_and_metrics[1]))

        self.model_save(csv=True)
        save_graph(train_loss=custom_train_val_history.train_loss, train_acc=custom_train_val_history.train_acc,
                   path=self._model_path, model_num=self._modelNumber, val_x=custom_train_val_history.val_x,
                   val_loss=custom_train_val_history.val_loss_y, val_acc=custom_train_val_history.val_acc_y)
        save_txt(path=self._model_path, test_result=loss_and_metrics, channel_num=self._num_freq_channel,
                 data_size=data_set_size, epoch=dnn_epochs)
        self.test_run(50)

    def accumulate_data_set(self, data_set_size: int, progress_report: bool):
        self._data_set.clear()
        path = self._model_path+'Data set/'
        if not os.path.isdir(path):
            os.mkdir(path)

        action_dict = {'type': 'sensing'}
        self._observation_history_list = deque(maxlen=data_set_size)
        self._real_output_list = deque(maxlen=data_set_size)
        for i in range(data_set_size):
            if progress_report:
                print(f"data set sample: {i+1}/{data_set_size}\r", end='')
            self._observation_history_list.append(self._observation_history)
            self._real_output_list.append(self._observation_history)
            observation_dict = self.step(action_dict)
            self._latest_observation_dict = observation_dict
            self.update_observation_history(observation_dict)
        for i in range(self._observation_history_length):
            self._real_output_list.append(self._observation_history)
            observation_dict = self.step(action_dict)
            self._latest_observation_dict = observation_dict
            self.update_observation_history(observation_dict)
        for i in range(data_set_size):
            self._data_set.append((self._observation_history_list[i], self._real_output_list[i]))
        if progress_report:
            print()

    def test_run(self, length: int):
        path = self._model_path + 'image/'
        if not os.path.isdir(path):
            os.mkdir(path)
        else:
            shutil.rmtree(path)
            os.mkdir(path)

        for ind in range(length):
            print(f"\n------Test run: {ind+1}/{length}------")
            observation_history = self._observation_history
            observation_history = np.reshape(observation_history, [1, self._observation_history_length, self._num_freq_channel, 1])
            prediction = self._main_dnn.predict(observation_history, batch_size=1)
            action_dict = {'type': 'sensing'}
            for i in range(self._observation_history_length):
                observation_dict = self.step(action_dict)
                self._latest_observation_dict = observation_dict
                self.update_observation_history(observation_dict)

            ground_truth = np.reshape(self._observation_history, [1, self._observation_history_length, self._num_freq_channel, 1])

            evaluation_result = self._main_dnn.evaluate(observation_history, ground_truth, verbose=0)
            print('>> loss: %0.4f, acc: %0.4f' % (evaluation_result[0], evaluation_result[1]))

            # prediction = np.round(prediction)
            comparison = np.equal(self._observation_history, prediction)
            result_percentage = np.count_nonzero(comparison)/(self._observation_history_length*self._num_freq_channel)
            predict_one = np.count_nonzero(prediction == 1)
            predict_zero = np.count_nonzero(prediction == 0)
            obs_one = np.count_nonzero(self._observation_history == 1)
            obs_zero = np.count_nonzero(self._observation_history == 0)
            print(f" prediction and real_output match percentage:{result_percentage}\n"
                  f" prediction count 1: {predict_one}   0: {predict_zero}  observation count 1: {obs_one}   0: {obs_zero}")

            ground_truth = self._observation_history.flatten().reshape((self._num_freq_channel, 256), order='F')[:]*255
            prediction = prediction.flatten().reshape((self._num_freq_channel, 256), order='F')[:]*255
            print(np.round(ground_truth[:]/255, 3))
            print(np.round(prediction[:]/255, 3))

            ba = bytearray(ground_truth.astype(np.uint8))
            im = Image.frombuffer('I', (256, self._num_freq_channel), ba, 'raw', 'P', 0, 1)
            im = im.resize((2000, 250))
            im.save(path + '%d Ground_truth - %0.4f.png' % (ind, evaluation_result[1]))

            ba = bytearray(prediction.astype(np.uint8))
            im = Image.frombuffer('I', (256, self._num_freq_channel), ba, 'raw', 'P', 0, 1)
            im = im.resize((2000, 250))
            im.save(path + '%d Prediction - %0.4f.png' % (ind, evaluation_result[1]))
            self._result.append([result_percentage, predict_one, predict_zero, obs_one, obs_zero])

    def get_mini_batch(self, batch_size: int):
        samples = random.sample(self._data_set, batch_size)
        observation = np.stack([x[0] for x in samples], axis=0)
        real_output = np.stack([x[1] for x in samples], axis=0)
        return observation, real_output

    def update_observation_history(self, observation: Dict):
        observation_type = observation['type']
        new_observation = np.zeros((self._num_freq_channel, 1))
        new_observation_length = 1
        if observation_type == 'sensing':
            sensed_power = observation['sensed_power']
            occupied_channel_list = [int(freq_channel) for freq_channel in sensed_power
                                     if sensed_power[freq_channel] > self._cca_thresh]
            new_observation[occupied_channel_list] = 1
            new_observation_length = 1

        new_observation = np.broadcast_to(new_observation, (new_observation_length, self._num_freq_channel, 1))
        self._observation_history = np.concatenate((new_observation, self._observation_history),
                                                   axis=0)[:self._observation_history_length, ...]

    def model_save(self, csv: bool = False):
        self._main_dnn.save(self._model_path)
        if csv:
            save_csv(self._result, self._model_path, self._modelNumber)


class Custom_train_val_History(keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.train_loss: List[float] = []
        self.train_acc: List[float] = []

        self.val_loss_y: List[float] = [0]
        self.val_acc_y: List[float] = [0]
        self.val_x: List[float] = [0]

        self.batch_cnt: int = 0

    def on_train_batch_end(self, batch, logs=None):
        # keys = list(logs.keys())
        # print("  End batch {} of training; got log keys: {}".format(batch, keys))
        # print("///////////training")
        self.train_loss.append(logs.get('loss'))
        self.train_acc.append(logs.get('accuracy'))
        self.batch_cnt += 1

    # def on_test_batch_end(self, batch, logs=None):
    #     # keys = list(logs.keys())
    #     # print(" validation End batch {} of training; got log keys: {}".format(batch, keys))
    #     # print("For batch {}, loss is {:7.2f}.".format(batch, logs["loss"]))
    #     # print("///////////validation")
    #     self.val_loss.append(logs.get('loss'))
    #     self.val_acc.append(logs.get('accuracy'))

    def on_test_end(self, logs=None):
        self.val_loss_y.append(logs.get('loss'))
        self.val_acc_y.append(logs.get('accuracy'))
        self.val_x.append(self.batch_cnt)


