# multiclass-perceptron


**Training is implemented in the following way:**

During each iteration of training, the data (formatted as a feature vector) is read in, and the dot
product is taken with each unique weight vector (which are all initially set to 0). The class that
yields the highest product is the class to which the data belongs(modified ReLU function). In the case this class is the
correct value (matches with the actual category to which the data belongs), nothing happens, and the
next data point is read in. However, in the case that the predicted value is wrong, the weight vectors a
re corrected as follows: The feature vector is subtracted from the predicted weight vector, and added to
the actual (correct) weight vector.

### Usage :

##### general:

`python3 trainer.py --dataset  bike_dataset --output bike_classifier`

`python3 predictor.py --model bike`

##### optional:

###### Set number of epochs:

`python3 trainer.py --dataset  bike_dataset --output bike_classifier --epochs 20 `

###### Set number of hidden layer neurons if you want to train using hidden layer:

`python3 trainer.py --dataset  bike_dataset --output bike_classifier --hidden 10 `

###### Test an arbitrary image with the model:
`python3 predictor.py --model bike --test_image bike_dataset/sports/3.png`

###### to see other customizations:

`python3 trainer.py --help`

`python3 predictor.py --help`

