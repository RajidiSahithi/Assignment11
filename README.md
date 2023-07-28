# APPLYING GRAD-CAM ON MISCLASSIFIED IMAGES TAKEN FROM A TRAINED MODEL

## TARGET :
* Train resnet18 for 20 epochs on the CIFAR10 dataset
* Show loss curves for test and train datasets
* Show a gallery of 10 misclassified images
* Show gradcamLinks to an external site. output on 10 misclassified images. 
* Apply these transforms while training:RandomCrop(32, padding=4),CutOut(16x16)

## CONTENTS :
- [DATASET](#dataset)
- [IMPORTING_LIBRARIES](#importing_libraries)
- [SET_THE_ALBUMENTATIONS](#set_the_albumentations)
- [DATA_AUGMENTATIONS](#data_augmentations)
- [SET_DATA_LOADER](#Set_Data_Loader)
- [CNN_MODEL](#cnn_model)
- [TRAINING_THE_MODEL](training_the_model)
- [LR_SCHEDULAR](lr_schedular)
- [MISCLASSIFIED_IMAGES](misclassified_images)
- [GRAD_CAM](grad_cam)
- [RESULTS](results)

## DATASET 
### CIFAR DATASET
CIFAR-10 is an established computer-vision dataset used for object recognition. It is a subset of the 80 million tiny images dataset and consists of 60,000 32x32 color (RGB) images containing one of 10 object classes, with 6000 images per class.

## IMPORTING_LIBRARIES
Import the required libraries. 
* NumPy is used for for numerical operations.The torch library is used to import Pytorch.
* Pytorch has an nn component that is used for the abstraction of machine learning operations. 
* The torchvision library is used so that to import the CIFAR-10 dataset. This library has many image datasets and is widely used for research. The transforms can be imported to resize the image to equal size for all the images. 
* The optim is used train the neural Networks.
* MATLAB libraries are imported to plot the graphs and arrange the figures with labelling
* Albumenations are imported for Middle Man's Data Augmentation Strategy
* cv2 is imported 
* torch_lr_finder is imported for finding the maximum and minimum learning rate
* imageio is imported for reading and writing images
* pytorch_grad_cam is imported to visualize grad-cam images

## SET_THE_ALBUMENTATIONS
* cv2.setNumThreads(0) sets the number of threads used by OpenCV to 0. This is done to avoid a deadlock when using OpenCV’s resize method with PyTorch’s dataloader1.

* cv2.ocl.setUseOpenCL(False) disables the usage of OpenCL in OpenCV2 and is used when you want to disable the usage of OpenCL.

* The  class is inherited from torchvision.datasets.CIFAR10. It overrides the __init__ and __getitem__ methods of the parent class. The __getitem__ method returns an image and its label after applying a transformation to the image3. (This is to be done while using Albumenations)


## DATA_AUGMENTATIONS
For this import albumentations as A

Middle-Class Man's Data Augmentation Strategy is used. Like
### Normalize
<pre>
Syntax:
     A.Normalize(
            mean = (0.4914, 0.4822, 0.4465),
            std = (0.2470, 0.2435, 0.2616), always_apply = True)

Normalization is a common technique used in deep learning to scale the pixel values of an image to a standard range. This is done to ensure that the input features have similar ranges and are centered around zero. 
Normalization is done with respect to mean and standard Deviation.
For CIFAR10 (RGB) will have 3 means and 3 standard devivation that is equal to 
(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
Normalize all the iamges
Applied to training and test data
</pre>
### HorizontalFlip :
<pre>
Syntax : A.HorizontalFlip()
Flip the input horizontally around the y-axis.
Args:     
p (float): probability of applying the transform. Default: 0.5.
Applied only to Training data
</pre>
### PadIfNeeded
<pre>
Syntax:
    A.PadIfNeeded(min_height=36, min_width=36, p=1.0),
PadIfNeeded is an image augmentation technique that pads the input image on all four sides if the side is less than the desired number. The desired number is specified by the min_height and min_width parameters. In this case padding is equal to 4.

</pre>
### RandomCrop
<pre>
Syntax:
  A.RandomCrop(height=32, width=32, always_apply = False,p=1.0),

RandomCrop is an image augmentation technique that crops a random part of the input and rescales it to some size without loss of bounding boxes. The height and width parameters specify the size of the crop. In this case iamge is cropped to size 32 X 32
</pre>
### Cutout
<pre>
Syntax:
 A.CoarseDropout(max_holes = 1, max_height=16, max_width=16, min_holes = 1, min_height=16, min_width=16,
                        fill_value=(0.4914, 0.4822, 0.4465), always_apply = True)
 It is similar to cutout

    Args:
        max_holes(int): The maximum number of rectangular regions to be masked. (for CIFAR10 Dataset its 32X32)
        max_height(int): The maximum height of the rectangular regions. 
        max_width(int): The maximum width of the rectangular regions.
        min_holes(int): The minimum number of rectangular regions to be masked.
        min_height(int): The minimum height of the rectangular regions.
        min_width(int): The minimum width of the rectangular regions.
        fill_value(float): The value to be filled in the masked region. It can be a tuple or a single value . 
            It is usually equal to the mean of dataset for CIFAR10 its (0.4914, 0.4822, 0.4465)
        always_apply = True - Applies to all the images
       
Applied only to Training data 
</pre>


### ToTensorV2
<pre>
Syntax:
    ToTensorV2()

To make this function work we need to ToTensorV2 from albumentations.pytorch.transforms
It is a class in the PyTorch library that converts an image to a PyTorch tensor. It is part of the torchvision.transforms module and is used to preprocess images before feeding them into a neural network. 

Applied to training and test data
</pre>
 #### PRINTED TRAIN_TRANSFORMS and TEST_TRANSFORMS 
<pre>

Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to ../data/cifar-10-python.tar.gz
100%|██████████| 170498071/170498071 [00:12<00:00, 14130374.69it/s]
Extracting ../data/cifar-10-python.tar.gz to ../data
Files already downloaded and verified
Compose([
  Normalize(always_apply=True, p=1.0, mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.2435, 0.2616), max_pixel_value=255.0),
  PadIfNeeded(always_apply=True, p=1.0, min_height=36, min_width=36, pad_height_divisor=None, pad_width_divisor=None, border_mode=4, value=None, mask_value=None),
  RandomCrop(always_apply=True, p=1.0, height=32, width=32),
  HorizontalFlip(always_apply=False, p=0.5),
  CoarseDropout(always_apply=True, p=0.5, max_holes=1, max_height=16, max_width=16, min_holes=1, min_height=16, min_width=16, fill_value=(0.4914, 0.4822, 0.4465), mask_fill_value=None),
  ToTensorV2(always_apply=True, p=1.0, transpose_mask=False),
], p=1.0, bbox_params=None, keypoint_params=None, additional_targets={})
Compose([
  Normalize(always_apply=True, p=1.0, mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.201), max_pixel_value=255.0),
  ToTensorV2(always_apply=True, p=1.0, transpose_mask=False),
], p=1.0, bbox_params=None, keypoint_params=None, additional_targets={})
</pre>

## SET_DATA_LOADER
* Batch Size = 512
* Number of Workers = 2
* CUDA is used

#### PRINTED TRAIN and TEST LOADER:
<pre>
<torch.utils.data.dataloader.DataLoader object at 0x7bfb181c6a70>
length of train_loader 98
<torch.utils.data.dataloader.DataLoader object at 0x7bfb181c54e0>
length of test_loader 20

</pre>
#### SAMPLE IMAGES IN TRAIN LOADER
![alt text]sample images train loader

## CNN_MODEL

#### MODEL
RESNET18 MODEL
<pre>
ResNet(
  (conv1): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (layer1): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (shortcut): Sequential()
    )
    (1): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (shortcut): Sequential()
    )
  )
  (layer2): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (shortcut): Sequential(
        (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (shortcut): Sequential()
    )
  )
  (layer3): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (shortcut): Sequential(
        (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (shortcut): Sequential()
    )
  )
  (layer4): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (shortcut): Sequential(
        (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (shortcut): Sequential()
    )
  )
  (linear): Linear(in_features=512, out_features=10, bias=True)
)
</pre>
### Summary of Training model
<pre>
Requirement already satisfied: torchsummary in /usr/local/lib/python3.10/dist-packages (1.5.1)
cpu
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 32, 32]           1,728
       BatchNorm2d-2           [-1, 64, 32, 32]             128
            Conv2d-3           [-1, 64, 32, 32]          36,864
       BatchNorm2d-4           [-1, 64, 32, 32]             128
            Conv2d-5           [-1, 64, 32, 32]          36,864
       BatchNorm2d-6           [-1, 64, 32, 32]             128
        BasicBlock-7           [-1, 64, 32, 32]               0
            Conv2d-8           [-1, 64, 32, 32]          36,864
       BatchNorm2d-9           [-1, 64, 32, 32]             128
           Conv2d-10           [-1, 64, 32, 32]          36,864
      BatchNorm2d-11           [-1, 64, 32, 32]             128
       BasicBlock-12           [-1, 64, 32, 32]               0
           Conv2d-13          [-1, 128, 16, 16]          73,728
      BatchNorm2d-14          [-1, 128, 16, 16]             256
           Conv2d-15          [-1, 128, 16, 16]         147,456
      BatchNorm2d-16          [-1, 128, 16, 16]             256
           Conv2d-17          [-1, 128, 16, 16]           8,192
      BatchNorm2d-18          [-1, 128, 16, 16]             256
       BasicBlock-19          [-1, 128, 16, 16]               0
           Conv2d-20          [-1, 128, 16, 16]         147,456
      BatchNorm2d-21          [-1, 128, 16, 16]             256
           Conv2d-22          [-1, 128, 16, 16]         147,456
      BatchNorm2d-23          [-1, 128, 16, 16]             256
       BasicBlock-24          [-1, 128, 16, 16]               0
           Conv2d-25            [-1, 256, 8, 8]         294,912
      BatchNorm2d-26            [-1, 256, 8, 8]             512
           Conv2d-27            [-1, 256, 8, 8]         589,824
      BatchNorm2d-28            [-1, 256, 8, 8]             512
           Conv2d-29            [-1, 256, 8, 8]          32,768
      BatchNorm2d-30            [-1, 256, 8, 8]             512
       BasicBlock-31            [-1, 256, 8, 8]               0
           Conv2d-32            [-1, 256, 8, 8]         589,824
      BatchNorm2d-33            [-1, 256, 8, 8]             512
           Conv2d-34            [-1, 256, 8, 8]         589,824
      BatchNorm2d-35            [-1, 256, 8, 8]             512
       BasicBlock-36            [-1, 256, 8, 8]               0
           Conv2d-37            [-1, 512, 4, 4]       1,179,648
      BatchNorm2d-38            [-1, 512, 4, 4]           1,024
           Conv2d-39            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-40            [-1, 512, 4, 4]           1,024
           Conv2d-41            [-1, 512, 4, 4]         131,072
      BatchNorm2d-42            [-1, 512, 4, 4]           1,024
       BasicBlock-43            [-1, 512, 4, 4]               0
           Conv2d-44            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-45            [-1, 512, 4, 4]           1,024
           Conv2d-46            [-1, 512, 4, 4]       2,359,296
      BatchNorm2d-47            [-1, 512, 4, 4]           1,024
       BasicBlock-48            [-1, 512, 4, 4]               0
           Linear-49                   [-1, 10]           5,130
================================================================
Total params: 11,173,962
Trainable params: 11,173,962
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 11.25
Params size (MB): 42.63
Estimated Total Size (MB): 53.89
----------------------------------------------------------------
</pre>




## TRAINING_THE_MODEL
The train function takes the model, device, train_loader, optimizer, and epoch as inputs. It performs the following steps:

* Sets the model to train mode, which enables some layers and operations that are only used during training, such as dropout and batch normalization.
* Creates a progress bar object from the train_loader, which is an iterator that yields batches of data and labels from the training set.
* Initializes two variables to keep track of the number of correct predictions and the number of processed samples.
* Loops over the batches of data and labels, and performs the following steps for each batch:
* Moves the data and labels to the device, which can be either a CPU or a GPU, depending on what is available.
* Calls optimizer.zero_grad() to reset the gradients of the model parameters to zero, because PyTorch accumulates them on subsequent backward passes.
* Passes the data through the model and obtains the predictions (y_pred).
* Calculates the loss between the predictions and the labels using Cross Entropy.
* Appends the loss to the train_losses list for later analysis.
* Performs backpropagation by calling loss.backward(), which computes the gradients of the loss with respect to the model parameters.
* Performs optimization by calling optimizer.step(), which updates the model parameters using the gradients and the chosen optimization algorithm (such as SGD or Adam).
* Updates the progress bar with the current loss, batch index, and accuracy. The accuracy is computed by comparing the predicted class (the index of the max log-probability) with the true class, and summing up the correct predictions and processed samples.
* Appends the accuracy to the train_acc list for later analysis.

The test function takes the model, device, and test_loader as inputs. It performs the following steps:

* Sets the model to eval mode, which disables some layers and operations that are only used during training, such as dropout and batch normalization.
* Initializes two variables to keep track of the total test_loss and the number of correct predictions.
* Uses a torch.no_grad() context manager to disable gradient computation, because we don’t need it during testing and it saves memory and time.
* Loops over the batches of data and labels from the test set, and performs the following steps for each batch:
* Moves the data and labels to the device, which can be either a CPU or a GPU, depending on what is available.
* Passes the data through the model and obtains the output (predictions).
* Adds up the batch loss to the total test loss using the negative log-likelihood loss function (F.nll_loss) with reduction=‘sum’, which means it returns a scalar instead of a vector.
* Compares the predicted class (the index of the max log-probability) with the true class, and sums up the correct predictions.
* Divides the total test loss by the number of samples in the test set to get the average test loss, and appends it to the test_losses list for later analysis.

* creates an instance of the Adam optimizer, which is a popular algorithm that adapts the learning rate for each parameter based on the gradient history and the current gradient. You pass the model parameters, the initial learning rate (lr), and some other hyperparameters to the optimizer constructor. 
* creates an instance of the OneCycleLR scheduler, which is a learning rate policy that cycles the learning rate between two boundaries with a constant frequency. You pass the optimizer, the maximum learning rate (0.01), the number of epochs (30), and the number of steps per epoch (len(train_loader)) to the scheduler constructor.
* Defines a constant for the number of epochs = 30, which is the number of times you iterate over the entire training set.
* Prints out a summary of the average test loss, accuracy, and number of samples in the test set. 

## [LR_SCHEDULAR]
The learning rate finder is a method to discover a good learning rate for most gradient based optimizers. The LRFinder method can be applied on top of every variant of the stochastic gradient descent, and most types of networks.
* LR FINDER SYNTAX:
<pre>
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
lr_finder.range_test(train_loader, end_lr=10, num_iter=200, step_mode="exp")
lr_finder.plot() # to inspect the loss-learning rate graph

min_loss = min(lr_finder.history["loss"])
max_lr = lr_finder.history["lr"][np.argmin(lr_finder.history["loss"], axis=0)]

print("Min Loss = {}, Max LR = {}".format(min_loss, max_lr))
</pre>
* Always starting with maximum learning rates.
* If Learning Rate is negative either increase the number of epochs or decrease learning rate. By decreasing the learning rate Iam not able to get 90% of accuracy. So I incresed the number of epoches in lr schedular.
* The following LR Schedular is used the program
<pre>
scheduler = OneCycleLR(
        optimizer,
        max_lr=1.74E-03,
        steps_per_epoch=1,
        epochs=26,
        pct_start=5/EPOCHS,
        div_factor=100,
        three_phase=False,
        final_div_factor=100,
        anneal_strategy='linear'
    )
</pre>

## [MISCLASSIFIED_IMAGES]
* Creates a dictionary to map class indices to their corresponding labels.
* Compares predicted labels (y_pred) with actual labels (y_actual) to identify misclassified instances.
* Stores the batch ID and image number for each misclassified instance.
* The plotmis() function displays each image with its predicted and actual labels.

#### Benefits of analyzing misclassified images:
* Visual representation helps identify patterns or common characteristics behind misclassifications.
* Gain insights into the performance of your image classification model.
* Refine training data, adjust model parameters, or consider alternative approaches to enhance accuracy.

## [GRAD_CAM]
* Implements gradCam of the given images and specified layer of the model.(Last Convolutional Layer)

## [RESULTS]
<pre>
Collecting albumentations==0.4.6
  Downloading albumentations-0.4.6.tar.gz (117 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 117.2/117.2 kB 2.6 MB/s eta 0:00:00
  Preparing metadata (setup.py) ... done
Requirement already satisfied: numpy>=1.11.1 in /usr/local/lib/python3.10/dist-packages (from albumentations==0.4.6) (1.22.4)
Requirement already satisfied: scipy in /usr/local/lib/python3.10/dist-packages (from albumentations==0.4.6) (1.10.1)
Requirement already satisfied: imgaug>=0.4.0 in /usr/local/lib/python3.10/dist-packages (from albumentations==0.4.6) (0.4.0)
Requirement already satisfied: PyYAML in /usr/local/lib/python3.10/dist-packages (from albumentations==0.4.6) (6.0.1)
Requirement already satisfied: opencv-python>=4.1.1 in /usr/local/lib/python3.10/dist-packages (from albumentations==0.4.6) (4.7.0.72)
Requirement already satisfied: six in /usr/local/lib/python3.10/dist-packages (from imgaug>=0.4.0->albumentations==0.4.6) (1.16.0)
Requirement already satisfied: Pillow in /usr/local/lib/python3.10/dist-packages (from imgaug>=0.4.0->albumentations==0.4.6) (9.4.0)
Requirement already satisfied: matplotlib in /usr/local/lib/python3.10/dist-packages (from imgaug>=0.4.0->albumentations==0.4.6) (3.7.1)
Requirement already satisfied: scikit-image>=0.14.2 in /usr/local/lib/python3.10/dist-packages (from imgaug>=0.4.0->albumentations==0.4.6) (0.19.3)
Requirement already satisfied: imageio in /usr/local/lib/python3.10/dist-packages (from imgaug>=0.4.0->albumentations==0.4.6) (2.25.1)
Requirement already satisfied: Shapely in /usr/local/lib/python3.10/dist-packages (from imgaug>=0.4.0->albumentations==0.4.6) (2.0.1)
Requirement already satisfied: networkx>=2.2 in /usr/local/lib/python3.10/dist-packages (from scikit-image>=0.14.2->imgaug>=0.4.0->albumentations==0.4.6) (3.1)
Requirement already satisfied: tifffile>=2019.7.26 in /usr/local/lib/python3.10/dist-packages (from scikit-image>=0.14.2->imgaug>=0.4.0->albumentations==0.4.6) (2023.7.18)
Requirement already satisfied: PyWavelets>=1.1.1 in /usr/local/lib/python3.10/dist-packages (from scikit-image>=0.14.2->imgaug>=0.4.0->albumentations==0.4.6) (1.4.1)
Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.10/dist-packages (from scikit-image>=0.14.2->imgaug>=0.4.0->albumentations==0.4.6) (23.1)
Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib->imgaug>=0.4.0->albumentations==0.4.6) (1.1.0)
Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.10/dist-packages (from matplotlib->imgaug>=0.4.0->albumentations==0.4.6) (0.11.0)
Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.10/dist-packages (from matplotlib->imgaug>=0.4.0->albumentations==0.4.6) (4.41.1)
Requirement already satisfied: kiwisolver>=1.0.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib->imgaug>=0.4.0->albumentations==0.4.6) (1.4.4)
Requirement already satisfied: pyparsing>=2.3.1 in /usr/local/lib/python3.10/dist-packages (from matplotlib->imgaug>=0.4.0->albumentations==0.4.6) (3.1.0)
Requirement already satisfied: python-dateutil>=2.7 in /usr/local/lib/python3.10/dist-packages (from matplotlib->imgaug>=0.4.0->albumentations==0.4.6) (2.8.2)
...
Requirement already satisfied: cmake in /usr/local/lib/python3.10/dist-packages (from triton==2.0.0->torch) (3.25.2)
Requirement already satisfied: lit in /usr/local/lib/python3.10/dist-packages (from triton==2.0.0->torch) (16.0.6)
Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.10/dist-packages (from jinja2->torch) (2.1.3)
Requirement already satisfied: mpmath>=0.19 in /usr/local/lib/python3.10/dist-packages (from sympy->torch) (1.3.0)
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
Files already downloaded and verified
Files already downloaded and verified
Compose([
  Normalize(always_apply=True, p=1.0, mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.2435, 0.2616), max_pixel_value=255.0),
  PadIfNeeded(always_apply=True, p=1.0, min_height=36, min_width=36, border_mode=4, value=None, mask_value=None),
  RandomCrop(always_apply=True, p=1.0, height=32, width=32),
  HorizontalFlip(always_apply=False, p=0.5),
  CoarseDropout(always_apply=True, p=0.5, max_holes=1, max_height=16, max_width=16, min_holes=1, min_height=16, min_width=16),
  ToTensorV2(always_apply=True, p=1.0),
], p=1.0, bbox_params=None, keypoint_params=None, additional_targets={})
Compose([
  Normalize(always_apply=True, p=1.0, mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.201), max_pixel_value=255.0),
  ToTensorV2(always_apply=True, p=1.0),
], p=1.0, bbox_params=None, keypoint_params=None, additional_targets={})
<torch.utils.data.dataloader.DataLoader object at 0x79064c065f00>
length of train_loader 98
<torch.utils.data.dataloader.DataLoader object at 0x79064c065de0>
length of test_loader 20
WARNING:matplotlib.image:Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
WARNING:matplotlib.image:Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
WARNING:matplotlib.image:Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
WARNING:matplotlib.image:Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
WARNING:matplotlib.image:Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
WARNING:matplotlib.image:Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
WARNING:matplotlib.image:Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
WARNING:matplotlib.image:Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
WARNING:matplotlib.image:Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
WARNING:matplotlib.image:Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
WARNING:matplotlib.image:Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
WARNING:matplotlib.image:Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
WARNING:matplotlib.image:Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
WARNING:matplotlib.image:Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
WARNING:matplotlib.image:Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).
WARNING:matplotlib.image:Clipping input data to the valid range for imshow with RGB data ([0..1] for floats or [0..255] for integers).

==> Building model..
torch.Size([1, 10])
cuda
Requirement already satisfied: torchsummary in /usr/local/lib/python3.10/dist-packages (1.5.1)
cuda
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 32, 32]           1,728
       BatchNorm2d-2           [-1, 64, 32, 32]             128
            Conv2d-3           [-1, 64, 32, 32]          36,864
       BatchNorm2d-4           [-1, 64, 32, 32]             128
            Conv2d-5           [-1, 64, 32, 32]          36,864
       BatchNorm2d-6           [-1, 64, 32, 32]             128
        BasicBlock-7           [-1, 64, 32, 32]               0
            Conv2d-8           [-1, 64, 32, 32]          36,864
       BatchNorm2d-9           [-1, 64, 32, 32]             128
           Conv2d-10           [-1, 64, 32, 32]          36,864
      BatchNorm2d-11           [-1, 64, 32, 32]             128
       BasicBlock-12           [-1, 64, 32, 32]               0
           Conv2d-13          [-1, 128, 16, 16]          73,728
      BatchNorm2d-14          [-1, 128, 16, 16]             256
           Conv2d-15          [-1, 128, 16, 16]         147,456
      BatchNorm2d-16          [-1, 128, 16, 16]             256
           Conv2d-17          [-1, 128, 16, 16]           8,192
      BatchNorm2d-18          [-1, 128, 16, 16]             256
       BasicBlock-19          [-1, 128, 16, 16]               0
           Conv2d-20          [-1, 128, 16, 16]         147,456
...
Forward/backward pass size (MB): 11.25
Params size (MB): 42.63
Estimated Total Size (MB): 53.89
----------------------------------------------------------------
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
ResNet(
  (conv1): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (layer1): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (shortcut): Sequential()
    )
    (1): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (shortcut): Sequential()
    )
  )
  (layer2): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
...
      (shortcut): Sequential()
    )
  )
  (linear): Linear(in_features=512, out_features=10, bias=True)
)
Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...
/usr/local/lib/python3.10/dist-packages/torch_lr_finder/lr_finder.py:5: TqdmExperimentalWarning: Using `tqdm.autonotebook.tqdm` in notebook mode. Use `tqdm.tqdm` instead to force console mode (e.g. in jupyter console)
  from tqdm.autonotebook import tqdm
Could not render content for "application/vnd.jupyter.widget-view+json"
{"version_major":2,"version_minor":0,"model_id":"c2a499bdb4d0419ba6583e223093160a"}
Stopping early, the loss has diverged
Learning rate search finished. See the graph with {finder_name}.plot()
LR suggestion: steepest gradient
Suggested LR: 1.59E-03

Min Loss = 1.720862924424122, Max LR = 0.16257556664437933
/usr/local/lib/python3.10/dist-packages/torch/optim/lr_scheduler.py:139: UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`. In PyTorch 1.1.0 and later, you should call them in the opposite order: `optimizer.step()` before `lr_scheduler.step()`.  Failure to do this will result in PyTorch skipping the first value of the learning rate schedule. See more details at https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
  warnings.warn("Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
EPOCH: 0
lr=  0.0003657
Loss=1.2841397523880005 Batch_id=97 Accuracy=43.78: 100%|██████████| 98/98 [00:59<00:00,  1.66it/s]

Test set:  Accuracy: 4355/10000 (43.55%)

EPOCH: 1
lr=  0.0007155
Loss=1.047451138496399 Batch_id=97 Accuracy=56.60: 100%|██████████| 98/98 [00:57<00:00,  1.70it/s]

Test set:  Accuracy: 5594/10000 (55.94%)

EPOCH: 2
lr=  0.0010653
Loss=1.0053366422653198 Batch_id=97 Accuracy=63.55: 100%|██████████| 98/98 [01:06<00:00,  1.48it/s]

Test set:  Accuracy: 6504/10000 (65.04%)

EPOCH: 3
lr=  0.0014150999999999999
Loss=0.829108715057373 Batch_id=97 Accuracy=67.49: 100%|██████████| 98/98 [00:56<00:00,  1.75it/s]

Test set:  Accuracy: 6818/10000 (68.18%)

EPOCH: 4
lr=  0.001541823
Loss=0.8069862127304077 Batch_id=97 Accuracy=70.57: 100%|██████████| 98/98 [00:57<00:00,  1.70it/s]

Test set:  Accuracy: 5819/10000 (58.19%)

EPOCH: 5
lr=  0.001445469
Loss=0.7664628028869629 Batch_id=97 Accuracy=73.36: 100%|██████████| 98/98 [00:55<00:00,  1.75it/s]

Test set:  Accuracy: 6962/10000 (69.62%)

EPOCH: 6
lr=  0.0013491150000000001
Loss=0.6739451289176941 Batch_id=97 Accuracy=75.72: 100%|██████████| 98/98 [00:55<00:00,  1.75it/s]

Test set:  Accuracy: 7752/10000 (77.52%)

EPOCH: 7
lr=  0.001252761
Loss=0.6944218873977661 Batch_id=97 Accuracy=77.06: 100%|██████████| 98/98 [00:56<00:00,  1.75it/s]

Test set:  Accuracy: 7631/10000 (76.31%)

EPOCH: 8
lr=  0.0011564070000000001
Loss=0.6590081453323364 Batch_id=97 Accuracy=78.85: 100%|██████████| 98/98 [00:55<00:00,  1.75it/s]

Test set:  Accuracy: 8052/10000 (80.52%)

EPOCH: 9
lr=  0.001060053
Loss=0.6748124361038208 Batch_id=97 Accuracy=79.87: 100%|██████████| 98/98 [00:55<00:00,  1.77it/s]

Test set:  Accuracy: 7979/10000 (79.79%)

EPOCH: 10
lr=  0.000963699
Loss=0.6375299692153931 Batch_id=97 Accuracy=81.00: 100%|██████████| 98/98 [00:55<00:00,  1.77it/s]

Test set:  Accuracy: 8381/10000 (83.81%)

EPOCH: 11
lr=  0.000867345
Loss=0.4822651147842407 Batch_id=97 Accuracy=82.33: 100%|██████████| 98/98 [00:55<00:00,  1.76it/s]

Test set:  Accuracy: 8151/10000 (81.51%)

EPOCH: 12
lr=  0.0007709910000000001
Loss=0.4360412359237671 Batch_id=97 Accuracy=83.09: 100%|██████████| 98/98 [00:55<00:00,  1.76it/s]

Test set:  Accuracy: 8463/10000 (84.63%)

EPOCH: 13
lr=  0.000674637
Loss=0.40959280729293823 Batch_id=97 Accuracy=84.01: 100%|██████████| 98/98 [00:55<00:00,  1.76it/s]

Test set:  Accuracy: 8395/10000 (83.95%)

EPOCH: 14
lr=  0.000578283
Loss=0.4574733376502991 Batch_id=97 Accuracy=84.99: 100%|██████████| 98/98 [00:55<00:00,  1.77it/s]

Test set:  Accuracy: 8260/10000 (82.60%)

EPOCH: 15
lr=  0.00048192899999999986
Loss=0.45749631524086 Batch_id=97 Accuracy=85.96: 100%|██████████| 98/98 [00:55<00:00,  1.77it/s]

Test set:  Accuracy: 8633/10000 (86.33%)

EPOCH: 16
lr=  0.000385575
Loss=0.3288038969039917 Batch_id=97 Accuracy=86.51: 100%|██████████| 98/98 [00:55<00:00,  1.76it/s]

Test set:  Accuracy: 8819/10000 (88.19%)

EPOCH: 17
lr=  0.0002892209999999999
Loss=0.39501675963401794 Batch_id=97 Accuracy=87.91: 100%|██████████| 98/98 [00:56<00:00,  1.74it/s]

Test set:  Accuracy: 8960/10000 (89.60%)

EPOCH: 18
lr=  0.000192867
Loss=0.3057962954044342 Batch_id=97 Accuracy=88.44: 100%|██████████| 98/98 [00:55<00:00,  1.76it/s]

Test set:  Accuracy: 9010/10000 (90.10%)

EPOCH: 19
lr=  9.65129999999999e-05
Loss=0.28667116165161133 Batch_id=97 Accuracy=89.90: 100%|██████████| 98/98 [00:55<00:00,  1.77it/s]

Test set:  Accuracy: 9049/10000 (90.49%)
</pre>