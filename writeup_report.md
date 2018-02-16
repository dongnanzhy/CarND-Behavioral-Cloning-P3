# **Behavioral Cloning**

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/model-architecture.png "Model Visualization"
[image2]: ./examples/center.jpg "Grayscaling"
[image3]: ./examples/recovery.jpg "Recovery Image"
[image4]: ./examples/train_history.png "train history"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results
* run_speed_9.mp4 video recording autonomous drive at speed = 9
* run_speed_30.mp4 video recording autonomous drive at speed = 30

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model architecture uses Nvidia's end-to-end deep learning self-driving car architecture. The convolutional layers are designed to perform feature extraction. I use strided convolutions in the first three convolutional layers with a 2×2 stride and a 5×5 kernel, and a non-strided convolution with a 3×3 kernel size in the final two convolutional layers (model.py lines 84-95)

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 82).

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 90-92).

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 122). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, with a mse loss. I did not tune learning rate manually (model.py line 97).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the right side of the road, used left and right camera as well as flipping.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to make the model complicated enough for almost half a million training images, as well as keep the model's generation ability.

My first step was to duplicate my previous work on LeNet architecture. I thought this model might be appropriate because it has 3 convolutional layers with pooling, followed by 2 FC layers. However, after I trained with 5 epochs, the model converge with no significant training loss decreasing, but when I applied the model on simulator, my car got crashed out of road at the last curve.

Then I tried with a more complicated architecture, which is Nvidia's end-to-end deep learning architecture for self-driving. However, compared to number of training data then use, my data is quite limited. In addition to data augmentation work which I will talk later, I also add dropout at FC layers to combat overfitting.

Then I applied my model on simulator to drive autonomously. At first with default speed equals to 9, the car can drive pretty smooth. However, when I increase the speed to 30, the car got oscillating sharply, even at straight road. I think the reason for this is because at higher speed, low MSE will be magnified, so I double checked the train-validation loss history, found that with complicated architecture, 5 epochs seems not enough for convergence. So I increase number of training epochs and retrain the model.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 81-95) consisted of a cropping layer to crop top 70 and bottom 25 pixels with limited context information, followed by a image normalization layer implemented with Keras Lambda layer. Then I added 5 convolution layers, with first three convolutional layers (a 2×2 stride and a 5×5 kernel), and final two non-strided convolutional layers (a 1×1 stride and a 3×3 kernel), all convolutional layers apply a relu activation. Followed by convolutional layers, I flattened the feature may and applied four fully connected layers, with layer sizes 100, 50, 10 and 1. Then I used the output to calculate MSE loss and used Adam optimizer for training.

Here is a visualization of the architecture.

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, in addition to sample training data provided by Udacity, I recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering mostly from the right side of the road back to center so that the vehicle would learn to correct itself when crossing out of roads.

![alt text][image3]

To augment the dataset, I used left and right camera with a small value of correction applied on angles. I also flipped images and angles thinking that this would give the model more generalization ability, to drive clockwise and counter-clockwise robustly. In this way, one image in training log will be augmented to 6 images.

After the collection process, I had 69,918 number of data points. I then preprocessed this data by cropping out top and bottom parts and normalizing pixel values to -0.5 to 0.5

I finally randomly shuffled the data set and put 10% of the data into a validation set. The reason I did not use a 20% validation set is because I think I have generated enough data so that even with 10%, validation data already has 7,000 images.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 13 as evidenced by the training log shown below. I used an adam optimizer so that manually training the learning rate wasn't necessary.

![alt text][image4]

### Findings
1. Higher speed require even lower MSE. At first I run simulator with default speed equals to 9, the car can drive pretty smooth. However, when I increase the speed to 30, the car got oscillating sharply, even at straight road. I think the reason for this is because at higher speed, low MSE will be magnified. This makes me think in real situation, we have to make sure our model has a low enough error to overcome possibly high speed.
2. In this project, I found data collection and augmentation has equal importance as model architecture. Tuning the model architecture from LeNet to Nvidia's net helps me a lot, and utilizing flipping and side camera also gives me big boost on performance. In future, to further improve the model, I will collect training data with driving cars counter-clockwise, which I think will definitely help the model
3. When I run the challenging task, my car did not drive successfully. Although I did not collect training data from track 2, but I think the model should be able to generalize. So there are more complicated architecture, more robust data and other techniques needed.
