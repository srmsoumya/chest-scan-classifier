**Classification of Common Thorax Diseases**

NIH.gov released the largest database of chest x-ray images. We tried to build a Deep Learning model on top of the Images to predict if a patient has any of the common thorax diseases.

The data was collected and labelled in an approximate way using NLP with 90% accuracy. They tried running object detection on top of it and were able to achieve 60-70% accuracy in predicting the thorax diseases. You can look at the results in 1705.02315.pdf.

We took a sample from the entire collection available at https://nihcc.app.box.com/v/ChestXray-NIHCC.

**Dataset Features**
 - 10000 images
 - 15 Labels
 - Some Images have multiple labels
 - Gray scaled Images
 - Class Imbalance in the data

We are using Keras with Tensorflow as the backend, the model training process was:

 - Used Transfer Learning with Resnet50 Architecture
 - Removed the last few layers
 - Added 2 dense layers and a softmax layer
 - Trained the model for 30 epochs freezing the resnet layers
 - Trained the model for more 30 epochs unfreezing the resnet layers

Why Resnet50 ? We tried using VGG16 and Inception as well, but Resnet seemed to work better.
With a small subset of the entire dataset and hyperparameter tuning we were able to achieve the same results as the nih guys.

Next Steps:
Since this was a 24 hour hackathon and we needed to create our own structure in the data, clean it and do pre-processing, we could only do Image classification. The next steps for this would be:
- Try Object detection and then run classification on top of it
- Run it in distributed environment like spark for training the entire dataset.

Scripts to run
==============
**Preferably use Anaconda and install keras and TF**

Run the ipython notebooks in the following order:
Data_Explorer.ipynb
Inspecting_Images.ipynb
Split_Data.ipynb

Train the model using:
python scan_classifier.py --train
