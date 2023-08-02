
# ML-Flow Experiments





## ⌛Installation

Kaggle environment Installation

```bash
  ! pip install kaggle

  from google.colab import files
  !ls -lha kaggle.json
  !pip install -q kaggle
  !mkdir -p ~/.kaggle
  !cp kaggle.json ~/.kaggle/
  !pwd

  !chmod 600 ~/.kaggle/kaggel.json
  !kaggle datasets list

  !kaggle competitions download -c dogs-vs-cats
```
    
## API Reference

#### Get all items
## [Numpy](https://numpy.org/doc/stable/reference/)
## [Pandas](https://pandas.pydata.org/docs/reference/index.html)
## [Sciket-learn](https://scikit-learn.org/stable/modules/classes.html)
## [Keras](https://keras.io/api/)


# Steps fro model creation

1.Dataset Collection: The first step is to gather a labeled dataset of images containing people with and without face masks. This dataset should cover various scenarios, lighting conditions, and angles. It's important to have a balanced representation of both masked and unmasked faces to train a reliable model.

2.Data Preprocessing: Once you have collected the dataset, it needs to be preprocessed to ensure compatibility with the CNN model. This step involves resizing the images to a consistent resolution, normalizing the pixel values, and splitting the data into training and testing sets.

3.Model Architecture: CNNs are well-suited for image classification tasks, including face mask detection. The architecture typically consists of multiple convolutional layers for feature extraction, followed by fully connected layers for classification. You can choose from existing CNN architectures like VGGNet, ResNet, or custom designs depending on the complexity of your project.

4.Model Training: The next step is to train the CNN model using the preprocessed dataset. During training, the model learns to extract relevant features from the input images and classify them as either "with mask" or "without mask." The training process involves feeding batches of images to the model, computing the loss (difference between predicted and actual labels), and adjusting the model's parameters through backpropagation to minimize the loss.

5.Model Evaluation: Once the model is trained, it's important to evaluate its performance using the testing dataset. This helps assess how well the model generalizes to unseen data and provides insights into its accuracy, precision, recall, and F1 score. Adjustments to the model or training process can be made based on the evaluation results to improve performance.

6.Deployment: After achieving satisfactory performance, the model can be deployed in real-world scenarios. This may involve integrating the model into a software application, web service, or edge device capable of capturing or processing images. The model takes an input image, processes it through the CNN, and provides a prediction (masked or unmasked) based on the detected features.

7.Fine-tuning and Continuous Improvement: Face mask detection models can be further fine-tuned or improved by using techniques like transfer learning or by collecting additional data to address specific challenges or variations in real-world scenarios. Continuous monitoring and updating of the model based on feedback and new data can help maintain its accuracy and effectiveness over time.
## 📝Description

* Thise projects based on Keras.sequential algorithm.
* In thise projects i have using convolutional neural network (CNN)
* model = keras.Sequential()


## 📊Datasets
## [Download Datasets](https://drive.google.com/drive/folders/19R7Bo7LPMNfxO3FCDtmAbhgO7BAaQhnp)
* Download the datasets for costom training


## 🎯Inference Demo

## Building a convolutional Neural Network (CNN)
```bash
num_of_classes =2
model = keras.Sequential()

model.add(keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu',input_shape=(128,128,3)))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dropout(0.5))


model.add(keras.layers.Dense(num_of_classes, activation='sigmoid'))
```

## 🕮 Please go through [Face_Mask_Detection.docx]() more info.
