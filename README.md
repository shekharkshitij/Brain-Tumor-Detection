# Brain Tumor Detection Using CNN

This project uses convolutional neural networks (CNNs) to detect brain tumors from MRI scans. The dataset consists of MRI images labeled as "glioma," "meningioma," "notumor," and "pituitary." The goal is to build a model that can accurately classify these images to aid in early diagnosis and treatment planning.

## Project Overview

The project involves the following steps:
1. Data Preprocessing: Load, preprocess, and augment the MRI images.
2. Model Architecture: Build and compile a CNN model to classify the images.
3. Model Training: Train the model using the training dataset.
4. Evaluation: Evaluate the model on the test dataset and generate various performance metrics.
5. Visualization: Plot various graphs to visualize the training process, model performance, and sample predictions.
6. Save the Model: Save the trained model for future use.

## Dataset

The dataset is divided into two main directories:

- **Training Data**: Used to train the model. It contains 5,712 images.
- **Testing Data**: Used to evaluate the model. It contains 1,311 images.

Each directory contains four subdirectories representing the categories:
- **Glioma**
- **Meningioma**
- **Notumor**
- **Pituitary**

## Prerequisites

Ensure you have the following libraries installed:

- Python 3.x
- OpenCV
- NumPy
- TensorFlow
- Matplotlib
- Seaborn
- Pandas

You can install these libraries using pip:

```sh
pip install opencv-python-headless numpy tensorflow matplotlib seaborn pandas
```

## Data Preprocessing

- Images are loaded and preprocessed to ensure they are of the same size (150x150 pixels).
- Data augmentation is applied to enhance the dataset and make the model more robust. Augmentations include rescaling, rotations, shifts, shear, zoom, and flips.

## Model Architecture

A Convolutional Neural Network (CNN) is used for classification, with the following layers:

- 4 Convolutional layers with ReLU activation and MaxPooling layers.
- Flattening layer to convert 2D matrix to a vector.
- Dense (fully connected) layer with 512 units and ReLU activation.
- Dropout layer with a rate of 0.5 to prevent overfitting.
- Output Dense layer with softmax activation for multi-class classification.

## Training

The model is compiled using the Adam optimizer and categorical cross-entropy loss function. It is trained for 50 epochs with a batch size of 32. The training process includes monitoring both training and validation accuracy and loss.

## Evaluation

- The model's performance is evaluated on the test dataset.
- Metrics such as accuracy, precision, recall, and F1-score are calculated for each class.
- A confusion matrix is generated to visualize the model's performance.

## Visualization

- The distribution of tumor types in the dataset is visualized using a bar plot.
- Sample images from each category are displayed.
- Training and validation accuracy and loss are plotted over epochs.
- A confusion matrix heatmap is generated to analyze model predictions.
- Sample images with their predicted and true labels are shown to illustrate the model's performance.

## Results

- The model achieved an overall test accuracy of 93.75%.
- Precision, recall, and F1-score are calculated for each class to assess model performance:
  - **Glioma**: Precision: 0.97, Recall: 0.95, F1-Score: 0.96
  - **Meningioma**: Precision: 0.95, Recall: 0.82, F1-Score: 0.88
  - **Notumor**: Precision: 0.89, Recall: 1.0, F1-Score: 0.94
  - **Pituitary**: Precision: 0.97, Recall: 0.96, F1-Score: 0.97

## Usage

Run the following command to train the model:

```python
python brain_tumor_detection.py
```

## Save the Model

The trained model is saved as `brain_tumor_detection_model.h5` for future use.

## Conclusion

The CNN model demonstrates good performance in classifying brain tumor images from MRI scans. With further tuning and optimization, the model can potentially be used for assisting radiologists in the diagnosis of brain tumors.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.

## Acknowledgments

- The dataset used in this project is from the "Brain Tumor MRI Dataset" available on Kaggle.

---
