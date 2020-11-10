import numpy as np
import matplotlib.pyplot as plt
import h5py
from sklearn import metrics
import seaborn as sn
import sys

#Activation Function
class sigmoid:
    @staticmethod
    def activation(x):
        y = 1 / (1 + np.exp(-x))
        return y

    @staticmethod
    def prime(x):
        y = sigmoid.activation(x) * (1 - sigmoid.activation(x))
        return y

class relu:
    @staticmethod
    def activation(x):
        y = np.maximum(0, x)
        return y

    @staticmethod
    def prime(x):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x

class Cost:

    @staticmethod
    def call(m, y, output):
        return (-1) * (1 / m) * (np.sum((y * np.log(output)) + ((1 - y) * (np.log(1 - output)))))

    @staticmethod
    def prime(m, y1, y2):
        return (-1 / m) * ((y1 / y2) + (y1 - 1) * (1 / (1 - y2)))

def str_to_class(str):
    return getattr(sys.modules[__name__], str)

def flatten(x):
    return x.reshape(x.shape[0], -1).T

class NeuralNetwork:

    def __init__(self):
        self.Loss_list = []
        self.epochs_list = []
        self.accuracy = []
        self.neurons = []
        self.activations = {}
        self.layers = 0

    def add_layer(self, neurons, activation):
        self.neurons.append(neurons)
        self.layers += 1

        self.activations[self.layers] = activation

    def num_layers(self):
        return print('Total number of layers: ' + str(self.layers))

    def get_layers_list(self):
        return self.layers

    def get_neurons_list(self):
        return self.neurons

    def get_neurons(self, layer):
        return self.neurons[layer-1]

    def get_layer_info(self, num):
        a = self.neurons[num-1]
        b = self.activations[num - 1]
        return a, b

    def weight_init(self, X_train):
        self.input = X_train
        self.w = {}


        self.w[1] = np.random.randn(self.input.shape[0], model.get_neurons(1)) * np.sqrt(2 / model.get_neurons(1))

        for i in range(self.layers-1):
            self.w[i+2] = np.random.randn(model.get_neurons(i+1), model.get_neurons(i+2)) * np.sqrt(2 / model.get_neurons(i+2))

        return self.w

    def bias_init(self, X_train):
        self.input = X_train
        self.b = {}

        for i in range(self.layers):
            self.b[i+1] = np.zeros((model.get_neurons(i+1), 1))

        return self.b

    def update_w_b(self, index, dw, db):

        self.w[index] -= self.learning_rate * dw
        self.b[index] -= self.learning_rate * db

    def feedforward(self, i):

        self.z = {}
        self.a = {}

        self.a = {0: self.input}

        if i == 0:
            self.w = self.weight_init(X_train)
            self.b = self.bias_init(X_train)

        for i in range(0, self.layers):
            self.z[i+1] = np.dot(self.w[i+1].T, self.a[i]) + self.b[i+1]
            self.a[i+1] = eval(self.activations[i+1]).activation(self.z[i+1])

        self.output = self.a[self.layers]

        self.cost = Cost.call(self.m, self.y, self.output)

        return self.a, self.output, self.cost

    def backpropogation(self):

        delta = Cost.prime(self.m, self.y, self.output) * eval(self.activations[self.layers]).prime(self.z[self.layers])

        dw = (1/self.m) * np.dot(delta, self.a[self.layers-1].T).T

        db = (1/self.m) * np.sum(delta)

        update_params = {
            self.layers: (dw, db)
        }

        for i in reversed(range(1, self.layers)):

            delta = np.dot(self.w[i+1].T.T, delta) * eval(self.activations[i]).prime(self.z[i])
            dw = (1/self.m) * np.dot(delta, self.a[i-1].T).T

            db = (1/self.m) * np.sum(delta)

            update_params[i] = (dw, db)

        for i, j in update_params.items():
            self.update_w_b(i, j[0], j[1])

        return update_params

    def propogation(self, i):
        self.a, self.output, self.cost = self.feedforward(i)
        self.update_params = self.backpropogation()
        return self.a, self.output, self.cost, self.update_params

    def fit(self, X_train, y_train, epochs, learning_rate):

        self.input = X_train
        self.y = y_train
        self.m = X_train.shape[1]
        self.learning_rate = learning_rate
        self.epochs = epochs

        print("Training........")
        for i in range(self.epochs):

            self.a, self.output, self.cost, self.update_params = self.propogation(i)

            print("epochs:" + str(i) + " | "
                  "Loss:" + str(self.cost) + " | "
                  "Accuracy: {} %".format(100 - np.mean(np.abs(self.output - self.y)) * 100))

            if i % 100 == 0:

                self.accuracy.append(100 - np.mean(np.abs(self.output - self.y)) * 100)
                self.Loss_list.append(self.cost)
                self.epochs_list.append(i)

        #Plotting accuracy
        accuracy = np.array(self.accuracy)
        accuracy = accuracy.reshape(-1, 1)

        #Plotting Loss
        Loss_array = np.array(self.Loss_list)
        y_loss = Loss_array.reshape(-1, 1)
        x_epochs = np.array(self.epochs_list).reshape(-1, 1)

        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        plt.plot(x_epochs, accuracy)
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.title('epochs_vs_accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(x_epochs, y_loss)
        plt.xlabel('epochs')
        plt.ylabel('Loss')
        plt.title('epochs_vs_loss')

        fig1 = plt.gcf()
        plt.show()
        fig1.savefig('Results.png')

        print("Training accuracy: {} %".format(100 - np.mean(np.abs(self.output - self.y)) * 100))

        return self.a, self.output, self.cost, self.update_params

    def predict(self, x, threshold):

        self.input = x
        self.z = {}
        self.a = {}

        self.a = {0: self.input}

        for i in range(0, self.layers):
            self.z[i + 1] = np.dot(self.w[i + 1].T, self.a[i]) + self.b[i + 1]
            self.a[i + 1] = eval(self.activations[i + 1]).activation(self.z[i + 1])

        self.output = self.a[self.layers]

        probablity = self.output

        probablity[probablity <= threshold] = 0
        probablity[probablity > threshold] = 1

        y_predicted = probablity.astype(int)

        return y_predicted

    def confusion_matrix(self, y_test, y_predicted):

        y_predicted = np.squeeze(y_predicted)
        y_test = np.squeeze(y_test)
        cm = metrics.confusion_matrix(y_test, y_predicted)

        plt.figure(figsize=(10, 7))
        sn.heatmap(cm, annot=True, fmt='d')
        plt.xlabel("Predicted")
        plt.ylabel("Truth")

        fig1 = plt.gcf()
        plt.show()
        fig1.savefig('cm.png')

    def evaluate(self, y_test, y_predicted):

        y_predicted = np.squeeze(y_predicted)
        y_test = np.squeeze(y_test)
        cm = metrics.confusion_matrix(y_test, y_predicted)

        TP = cm[0, 0]
        FP = cm[0, 1]
        FN = cm[1, 0]
        TN = cm[1, 1]

        accuracy = (TP + TN) / float(TP + TN + FP + FN)
        print('Testing accuracy: {} %'.format(accuracy*100))

        precision = TP / (TP + FP)
        print('Precision: {} %'.format(precision*100))

        recall = TP / (TP + FN)
        print('Recall: {} %'.format(recall*100))

        f1_score = 2 * ((precision * recall)/(precision + recall))
        print('F1_score: {} %'.format(f1_score * 100))

def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

if __name__ == '__main__':

    X_train, y_train, X_test, y_test, classes = load_dataset()

    X_train_flatten = flatten(X_train)
    X_test_flatten = flatten(X_test)

    X_train = X_train_flatten / 255
    X_test = X_test_flatten / 255

    model = NeuralNetwork()
    model.add_layer(100, activation="sigmoid")
    model.add_layer(50, activation="sigmoid")
    model.add_layer(1, activation="sigmoid")

    model.num_layers()
    model.fit(X_train, y_train, epochs=15000, learning_rate=0.5)

    y_predicted = model.predict(X_test, threshold=0.3)

    model.confusion_matrix(y_test, y_predicted)

    model.evaluate(y_test, y_predicted)