import pickle
import random
import numpy as np
import cv2
import argparse
from imutils import paths
from tqdm import tqdm

# Constants
BIAS = 1  					# Dummy Feature for use in setting constant factor in Training.
TRAIN_TEST_RATIO = .6  		# Default Ratio of data to be used in Training vs. Testing.
OUTPUT_PATH = 'output/'


class MultiClassPerceptron:
    # initialize values
    accuracy = 0

    """
    :param  classes           List of categories/classes (match tags in tagged data).
    :param  feature_data      Feature Data, in format specified in README, usually imported from feature_data module.
    :param  iterations        Number of iterations to run training data through. Set to 100 by default.
    :param  lr                Set learning rate for the model
    :param  hidden            [Optional] set the number of hidden neurons for multilayer perceptron network.
    :param  train_test_ratio  Ratio of data to be used in training vs. testing. Set to 75% by default.
    """

    def __init__(self, classes, feature_data, iterations, lr, hidden=None, train_test_ratio=TRAIN_TEST_RATIO):
        self.lr = lr
        self.classes = classes
        self.feature_data = feature_data
        self.ratio = train_test_ratio
        self.iterations = iterations
        # Split feature data into train set, and test set
        random.shuffle(self.feature_data)
        self.train_set = self.feature_data[:int(len(self.feature_data) * self.ratio)]
        self.test_set = self.feature_data[int(len(self.feature_data) * self.ratio):]
        # Initialize empty weight vectors, with extra BIAS term.
        self.hidden_exists = False
        if type(hidden) is list:
            self.hidden = hidden
            self.hidden_exists = True
            self.weight_vectors = {c: np.array([0.0 for _ in range(900 + 1)]) for c in self.hidden}
            self.weight_hidd_out = {p: np.array([0.0 for _ in range(len(self.hidden))]) for p in self.classes}
        else:
            self.weight_vectors = {c: np.array([0.0 for _ in range(900 + 1)]) for c in self.classes}

    @staticmethod
    def __reLu(x):
        return max(0, x)

    def train(self):
        """
        Training is implemented in the following way:

        During each iteration of training, the data (formatted as a feature vector) is read in, and the dot
        product is taken with each unique weight vector (which are all initially set to 0). The class that
        yields the highest product is the class to which the data belongs(modified ReLU function). In the case this class is the
        correct value (matches with the actual category to which the data belongs), nothing happens, and the
        next data point is read in. However, in the case that the predicted value is wrong, the weight vectors a
        re corrected as follows: The feature vector is subtracted from the predicted weight vector, and added to
        the actual (correct) weight vector. This makes sense, as we want to reject the wrong answer, and accept
        the correct one.
        """
        if not self.hidden_exists:
            print('Training datasets ', end='')
        else:
            print('Training datasets with hidden layer ', end='')
        print('for', self.iterations, 'epochs...')
        pbar = tqdm(range(self.iterations))
        for _ in pbar:
            pbar.set_description('Epoch %d' % (_ + 1), refresh=False)
            # print('Epoch = ', (_ + 1))
            for category, feature_dict in self.train_set:
                # Format feature values as a vector, with extra BIAS term.
                img = cv2.imread(feature_dict['path'])
                dim = (30, 30)
                if img is None:
                    print('Image is none')
                resized = cv2.resize(img, dim, cv2.INTER_AREA)

                input_array = []
                for i in range(resized.shape[0]):
                    for j in range(resized.shape[1]):
                        input_array.append((np.mean(resized[i][j]) / 255))

                input_array.append(BIAS)
                input_vector = np.array(input_array)
                # Initialize arg_max value, predicted class.
                arg_max, predicted_class = 0, self.classes[0]
                if not self.hidden_exists:
                    # Multi-Class Decision Rule:
                    for c in self.classes:
                        current_activation = np.dot(input_vector, self.weight_vectors[c])
                        if current_activation >= arg_max:
                            arg_max, predicted_class = current_activation, c
                    # Update Rule:
                    if not (category == predicted_class):
                        self.weight_vectors[category] += [i * self.lr for i in input_vector]
                        self.weight_vectors[predicted_class] -= [i * self.lr for i in input_vector]
                else:
                    # This is the Input for Hidden Layers
                    input_array_hidden = []

                    # We Make MultiClass Decision Rule Based on the above conditions
                    for index in self.hidden:
                        current_activation = np.dot(input_vector, self.weight_vectors[index])
                        current_activation = MultiClassPerceptron.__reLu(current_activation)

                        input_array_hidden.append(current_activation)
                        # if current_activation >= arg_max:
                        # arg_max,predicted_class = current_activation,index

                    for index in self.classes:
                        current_activation = np.dot(input_array_hidden, self.weight_hidd_out[index])
                        if current_activation >= arg_max:
                            arg_max, predicted_class = current_activation, index

                    # Updation of Weights And Biases
                    # that is y!=t Update weights and biases
                    # If y!=t we reduce its probability
                    if not (category == predicted_class):
                        for index in self.hidden:
                            # target = self.classes.index(category)+1
                            self.weight_vectors[int(index)] += [weight * self.lr for weight in input_vector]
                            self.weight_vectors[int(index)] -= [weight * self.lr for weight in input_vector]

    def predict(self, feature_dict):
        """
        Categorize an unseen data point based on the existing collected data.
        :param  feature_dict        Dictionary of the same form as the training feature data.
        :return                     Return the predicted category for the data point.
        """
        img = cv2.imread(feature_dict['path'])
        dim = (30, 30)
        resized = cv2.resize(img, dim, cv2.INTER_AREA)

        input_array = [np.mean(resized[i][j] / 255) for i in range(resized.shape[0]) for j in range(resized.shape[1])]
        input_array.append(BIAS)
        feature_vector = np.array(input_array)

        arg_max, predicted_class = 0, self.classes[0]
        if not self.hidden_exists:
            # Multi-Class Decision Rule:
            for c in self.classes:
                current_activation = np.dot(feature_vector, self.weight_vectors[c])
                if current_activation >= arg_max:
                    arg_max, predicted_class = current_activation, c
        else:
            # This holds the output activations of hidden layer neurons so that it acts as input to output layer
            input_array_hidden = []

            # Computing hidden layer activations
            for index in self.hidden:
                current_activation = np.dot(feature_vector, self.weight_vectors[index])
                current_activation = MultiClassPerceptron.__reLu(current_activation)
                input_array_hidden.append(current_activation)

            # predicting the class
            for index in self.classes:
                current_activation = np.dot(input_array_hidden, self.weight_hidd_out[index])
                if current_activation >= arg_max:
                    arg_max, predicted_class = current_activation, index

        return predicted_class

    def test_image(self, path):
        curr_class = path.split('/')[-2]
        feature = {'path': path}
        item = (curr_class, feature)
        print("Actual class: " + item[0])
        pred_class = self.predict(item[1])
        print("Predicted class: " + pred_class)

    def calculate_accuracy(self):
        """
        Calculates the accuracy of the classifier by running algorithm against test set and comparing
        the output to the actual categorization.
        """
        correct, incorrect = 0, 0
        random.shuffle(self.feature_data)
        self.test_set = self.feature_data[int(len(self.feature_data) * self.ratio):]
        for feature_dict in self.test_set:
            actual_class = feature_dict[0]
            predicted_class = self.predict(feature_dict[1])

            if actual_class == predicted_class:
                correct += 1
            else:
                incorrect += 1

        # print("ACCURACY:")
        print("Model Accuracy:", (correct * 1.0) / ((correct + incorrect) * 1.0))

    def save(self, classifier_name):
        """
        Saves classifier as a .pickle file to the output directory.
        :param  classifier_name  Name under which to save the classifier model.
        """
        classifier_name = OUTPUT_PATH + classifier_name
        with open(classifier_name + ".pik", 'wb') as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_classifier(classifier_name):
        """
        Unpickle the classifier, returns the MultiClassPerceptron object.
        :param  classifier_name  Name the classifier was saved under.
        :return                  Return instance of MultiClassPerceptron.
        """
        with open(classifier_name + ".pik", 'rb') as f:
            return pickle.load(f)


def fetch_data(data):
    image_paths = list(paths.list_images(data))
    classes = list(set([name.split('/')[1] for name in image_paths]))
    feature_data = []
    for path in image_paths:
        curr_class = path.split('/')[1]
        feature = {'path': path}
        feature_data.append((curr_class, feature))
    # print(feature_data)
    return classes, feature_data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True,
                    help="Path to input dataset")
    ap.add_argument("-o", "--output", required=True,
                    help="Name under which to save the classifier model")
    ap.add_argument("-hn", "--hidden", type=int,
                    help="set the number of hidden layer neurons")
    ap.add_argument('-e', "--epochs", type=int, default=10,
                    help="Train the model with specified number of epochs")
    ap.add_argument("-lr", "--learningrate", type=float, default=0.1,
                    help="Set the learning rate for the algorithm")
    args = vars(ap.parse_args())

    classes, feature_data = fetch_data(args['dataset'])
    lr = args['learningrate']
    epochs = args['epochs']
    classifier_name = args['output']
    hidden = args['hidden']

    if type(hidden) is int:
        if hidden > 3 and hidden < 201:
            hidden_neuron = [random.randrange(1,10) for i in range(hidden)]
            classifier = MultiClassPerceptron(classes, feature_data, epochs, lr, hidden_neuron)
        else:
            print('Invalid number of hidden neurons. It must lie between 4 and 200.')
            exit()
    else:
        classifier = MultiClassPerceptron(classes, feature_data, epochs, lr)

    classifier.train()
    classifier.calculate_accuracy()
    classifier.save(classifier_name)


if __name__ == "__main__":
    main()
