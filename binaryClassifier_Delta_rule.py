import os
import random
import numpy as np
from PIL import Image
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


# Sigmoid function (activation function).
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Derivative function of the sigmoid.
def sigmoid_deriv(x):
    value = sigmoid(x)
    return value * (1 - value)


# Trains the network.
def train(data, stop, error_stop):
    N, n = data.shape
    learning_rate = 0.8
    global_error = 0

    weights = np.random.randn(1, n - 1)
    err_x_grad = []
    new_data = []

    for epoch in range(0, stop):
        global_error = 0
        for case in data:
            new_data = []
            err_x_grad = []
            case_data = case[:-1]
            input_x_weights = np.matmul(weights, case[:-1])
            input_x_weights = input_x_weights[0]
            output = sigmoid(input_x_weights)
            error = output - case[-1]
            global_error += (error ** 2)
            err_x_grad.append(2 * error * sigmoid_deriv(input_x_weights))
            new_data.append(case_data)

        new_data = np.asarray(new_data)
        gr = new_data * np.asarray(err_x_grad)

        weights = weights - learning_rate * gr

        #print(global_error)
        if global_error < error_stop:
            break

    print(global_error)
    return weights


# Evaluates the network.
def evaluate(data, weights, print_matrix=True):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    output_arr = []
    real_values = []
    for case in data:
        case_data = case[:-1]
        input_x_weights = np.matmul(weights, case[:-1])
        input_x_weights = input_x_weights[0]
        output = sigmoid(input_x_weights)
        real_values.append(output)
        output = 0 if output < 0.5 else 1
        if output <= 0 and case[-1] == 0:
            TN += 1
        if output <= 0 and case[-1] == 1:
            FN += 1
        if output > 0 and case[-1] == 0:
            FP += 1
        if output > 0 and case[-1] == 1:
            TP += 1

        output_arr.append(output)

    if print_matrix:
        df_cm = pd.DataFrame([[TN, FN], [FP, TP]])
        sn.set(font_scale=1.4)
        colour_list = ['lightblue', 'darkblue']
        sn.heatmap(df_cm,
                   annot=True,
                   annot_kws={"size": 16},
                   cmap=colour_list,
                   cbar=False)
        plt.title('Confusion Matrix')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.show()

    return output_arr, real_values


# Loads and converts the images to grayscale. Returns an array with arrays.
def load_images(path, label):
    data = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        for filename in filenames:
            im = Image.open(dirpath + '\\' + filename, 'r').convert('LA').resize((64, 64))
            # returns a tuple with the intensity and 255, so we will scoop out the first number
            pixel_values = [x[0] / 255.0 for x in im.getdata()]

            pixel_values.append(1)  # bias
            pixel_values.append(label)  # label

            data.append(np.array(pixel_values))

    return np.asarray(data)

# def cross_entropy(x, label):
#     return -np.log(x) if label == 1 else -np.log(1-x)


# Plots some sample image and their classifications provided by the neural network.
def plot_images(image, prediction, real_values):
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 2, 1)
    im = Image.open(image[0], 'r').resize((64, 64))
    imgplot = plt.imshow(im)
    plt.xticks([])
    plt.yticks([])
    title = 'Stegosaurus' if prediction[0] == 1 else 'Crocodile'
    ax1.set_title(title)
    ax2 = fig.add_subplot(2, 2, 2)
    im = Image.open(image[1], 'r').resize((64, 64))
    imgplot = plt.imshow(im)
    plt.xticks([])
    plt.yticks([])
    title = 'Stegosaurus' if prediction[1] == 1 else 'Crocodile'
    ax2.set_title(title)
    ax3 = fig.add_subplot(2, 2, 3)
    im = Image.open(image[2], 'r').resize((64, 64))
    imgplot = plt.imshow(im)
    plt.xticks([])
    plt.yticks([])
    title = 'Stegosaurus' if prediction[2] == 1 else 'Crocodile'
    ax3.set_title(title)
    ax4 = fig.add_subplot(2, 2, 4)
    im = Image.open(image[3], 'r').resize((64, 64))
    imgplot = plt.imshow(im)
    plt.xticks([])
    plt.yticks([])
    title = 'Stegosaurus' if prediction[3] == 1 else 'Crocodile'
    ax4.set_title(title)
    plt.xticks([])
    plt.yticks([])
    plt.show()


# Main
def main():
    path1 = <<path--to--folder1>>
    path2 = <<path--to--folder2>>
    test_data_num = 10
    train_data_num = 700
    
    data1 = load_images(path1, 0)
    train_data = data1[0:train_data_num]  # [)
    test_data = data1[train_data_num:]

    data2 = load_images(path2, 1)
    train_data2 = data2[0:train_data_num]
    test_data2 = data2[train_data_num:]

    train_data = np.concatenate((train_data, train_data2))
    np.random.shuffle(train_data)

    stop = 10000
    error_stop = 2

    # Training our network
    weights = train(train_data, stop, error_stop)

    # Recognising test dataset
    test_data = np.concatenate((test_data, test_data2))
    np.random.shuffle(test_data)

    print("Training is complete")
    evaluate(test_data, weights)

    # Plotting images and their classifications
    image1 = path1 + "\\" + 'image_0001.jpg'
    image2 = path1 + "\\" + 'image_0004.jpg'
    image3 = path2 + "\\" + 'image_0001.jpg'
    image4 = path2 + "\\" + 'image_0025.jpg'

    images = [image1, image2, image3, image4]

    predictions = []
    for i in range(0, 4):
        im = Image.open(images[i]).convert('LA').resize((64, 64))
        # returns a tuple with the intensity and 255, so we will scoop out the first number
        pixel_values = [x[0] / 255.0 for x in im.getdata()]
        pixel_values.append(1)  # bias
        if i > 1:
            pixel_values.append(1)  # label
        else:
            pixel_values.append(0)  # label

        predictions.append(np.array(pixel_values))

    predictions, real_values = evaluate(predictions, weights, False)
    plot_images([image1, image2, image3, image4], predictions, real_values)


if __name__ == '__main__':
    main()
