import argparse
from trainer import MultiClassPerceptron

OUTPUT_PATH = 'output/'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--test_image",
                    help="path to input directory of images")
    ap.add_argument('-m', "--model", required=True,
                    help="Train the model with specified number of epochs")
    args = vars(ap.parse_args())
    classifier_name = args['model']
    classifier_path = OUTPUT_PATH + classifier_name
    classifier = MultiClassPerceptron.load_classifier(classifier_path)
    if args['test_image']:
        test_image_path = args['test_image']
        classifier.test_image(test_image_path)
    else:
        classifier.calculate_accuracy()


if __name__ == "__main__":
    main()
