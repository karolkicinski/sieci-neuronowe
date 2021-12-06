from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import pickle
import matplotlib.pyplot as plt
import csv

from neural_network_approach.gd_neural_network import GDNeuralNetwork
from neural_network_approach.layer import Layer

FILE = "mnist.txt"


def start():
    print("Start!")

    x, y = load_mnist()
    x = (x / 255).astype('float32')
    y = to_categorical(y)

    x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.0425, test_size=0.0075, random_state=42)

    network = GDNeuralNetwork(
        input_data_size=784,
        mini_batch=16,
        epochs=10,
        layers=[
            Layer(layer_size=50, activation_function='relu'),
            Layer(layer_size=10, activation_function='softmax')
        ],
    )
    network.compile(
        weights_init_method='standard',
        strategy='adadelta',
        learning_factor=2,
    )
    print(f"Acc beginning: {network.evaluate(x_val, y_val)}!")
    print(f"=========================================")
    epoch, acc, list_epochs, list_errors = network.fit((x_train, y_train), (x_val, y_val))
    print(f"=========================================")
    print(f"Epochs: {epoch}, acc: {acc}!")

    plt.plot(list_epochs, list_errors)
    plt.ylabel('Error')
    plt.xlabel('Epoch')
    plt.title(f"Epochs: {epoch}, acc: {acc}")
    plt.show()

    print("End!")

def badania():
    print("Start!")

    x, y = load_mnist()
    x = (x / 255).astype('float32')
    y = to_categorical(y)

    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.0075, random_state=42)

    gamma = [0.3]
    iters = 10
    list_epochs = []
    list_errors = []
    final_list_acc = []
    final_list_param = []

    with open('axx.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['b2', 'epochs', 'accuracy'])

        for g in gamma:
            print(f"====================> ADAM: {g}")
            results_acc = []
            results_ep = []

            act = 'sigmoid'

            for i in range(iters):
                network = GDNeuralNetwork(
                    input_data_size=784,
                    mini_batch=16,
                    epochs=9999,
                    layers=[
                        Layer(layer_size=220, activation_function=act),
                        Layer(layer_size=150, activation_function=act),
                        Layer(layer_size=120, activation_function=act),
                        Layer(layer_size=80, activation_function=act),
                        Layer(layer_size=75, activation_function=act),
                        Layer(layer_size=56, activation_function=act),
                        Layer(layer_size=33, activation_function=act),
                        Layer(layer_size=21, activation_function=act),
                        Layer(layer_size=10, activation_function='softmax')
                    ],
                )
                network.compile(
                    weights_init_method='he',
                    strategy='adam',
                    learning_factor=0.01,
                )

                epoch, acc, list_epochs, list_errors = network.fit((x_train, y_train), (x_val, y_val))
                results_acc.append(acc)
                results_ep.append(epoch)

            final_list_acc.append(sum(results_acc)/len(results_acc))
            final_list_param.append(g)
            writer.writerow([g, round(sum(results_ep)/len(results_ep)), round(sum(results_acc)/len(results_acc), 4)])

        # plt.plot(list_epochs, list_errors)
        # plt.ylabel('Error')
        # plt.xlabel('Epoch')
        # plt.title(f"Adam - single case")
        # plt.show()

        # plt.plot(final_list_param, final_list_acc)
        # plt.ylabel('Accuracy')
        # plt.xlabel('b2')
        # plt.title(f"Adam - testing")
        # plt.show()

        print(f"Liczba epok: {round(sum(results_ep)/len(results_ep))}, acc={round(sum(results_acc)/len(results_acc), 4)}")

        print("End!")


def save_mnist(x, y):
    with open(FILE, mode="wb") as file:
        pickle.dump((x, y), file)


def load_mnist():
    with open(FILE, mode="rb") as file:
        (x, y) = pickle.load(file)
        return x, y

def pl():
    plt.plot(
        [0.3, 0.5, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0],
        [0.3305, 0.421, 0.4933, 0.5152, 0.561, 0.6381, 0.6924, 0.8838]
    )
    plt.ylabel('Accuracy')
    plt.xlabel('Momentum')
    plt.title(f"Momentum - testing")
    plt.show()


if __name__ == '__main__':
    # start()
    badania()
    # pl()