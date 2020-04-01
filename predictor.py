import argparse
from trainer import MultiClassPerceptron


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--test_image",
                    help="path to input directory of images")
    ap.add_argument('-m', "--model", default='output/bike_classifier',
                    help="Train the model with specified number of epochs")
    args = vars(ap.parse_args())
    classifier_path = args['model']
    classifier = MultiClassPerceptron.load_classifier(classifier_path)
    if args['test_image']:
        test_image_path = args['test_image']
        classifier.test_image(test_image_path)
    else:
        classifier.calculate_accuracy()


if __name__ == "__main__":
    main()
