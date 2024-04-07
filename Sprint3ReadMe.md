# Data-Challenge-1: Diagnosing Thoracic X-Rays 
The following project must be supplied with images of chest x-rays, which then are used to
build a CNN which attempts to identify one of 5 diseases, or No Finding. It can serve as a guiding tool for practicioners to take informed
decisions, however it is not intended to be a replacement.

## Functionalities
Setting Calibration = True in main.py ensures the model will performing Calibration
after the model has been trained. To perform Calibration, the Validation set must also
be set to True. The code has been pre-set to these conditions.

## Saved model
At the end of each run, the weights of two models are saved:
1. The weights of the final epoch (FINAL_WEIGHTS.pth)
2. The weights of the epoch with the Cross Entropy Loss (BEST_WEIGHTS.pth)

The model weights are saved in the folder dc1, and are then at the disposal of the user,
however it must be noted that the weight files are overwritten at each run, unless
saved under alternative names.

## Output, Saved Metrics and Saved Plots 
After each epoch, the following are output (For both the Training and Test sets):
1. Cross Entropy Loss
2. Accuracy, Recall, Precision
3. Class Accuracy
4. AUC for ROC plot
5. Plot of the Cross Entropy loss, of thus far performed epochs
6. The number of images per class used for training and testing the data


At the end of the last epoch, the Expected Calibration Error (ECE) and Average
Brier score are supplied before and after Calibration. The Temperature is also provided.

Additionally, multiple plots are saved after running (and can be found in the artifacts folder):
1. Train/Test Loss per epoch + Train/Test Accuracy per epoch
2. ROC AUC per Class over epochs

## Environment setup instructions
We recommend to set up a virtual Python environment to install the package and its dependencies. To install the package, we recommend to execute `pip install -r requirements.txt.` in the command line. This will install it in editable mode, meaning there is no need to reinstall after making changes. If you are using PyCharm, it should offer you the option to create a virtual environment from the requirements file on startup. Note that also in this case, it will still be necessary to run the pip command described above.

