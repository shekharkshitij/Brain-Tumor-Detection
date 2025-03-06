# 🧠 Brain Tumor Detection Using CNN

## 📌 Introduction
Brain tumors are abnormal growths of cells in the brain that can be **benign (non-cancerous) or malignant (cancerous)**. Early detection of brain tumors is **critical for effective treatment** and improving patient survival rates. **Magnetic Resonance Imaging (MRI)** is widely used in diagnosing brain tumors due to its high-resolution imaging of soft tissues.

This project employs **Convolutional Neural Networks (CNNs)** to **automate brain tumor detection** from MRI scans. CNNs have proven to be highly effective in **medical image analysis**, enabling **automated feature extraction** and **high-accuracy classification**. The model is designed to classify MRI images into **four categories**:
- **Glioma**
- **Meningioma**
- **No Tumor**
- **Pituitary**

This deep learning approach aims to **enhance diagnostic accuracy** and assist radiologists in making **informed clinical decisions**.

---

## 📊 Project Overview
The project follows a structured pipeline to develop an **efficient and accurate** brain tumor classification model. The main stages of the project are:

1. **Data Preprocessing**:
   - Load, resize, and normalize MRI images.
   - Apply **data augmentation** to improve generalization.
   
2. **Model Architecture**:
   - Design a **CNN-based deep learning model** for tumor classification.
   - Include **regularization techniques** to prevent overfitting.
   
3. **Model Training**:
   - Train the CNN using **MRI scan images** with appropriate **hyperparameters**.
   - Monitor performance through **loss functions and accuracy metrics**.
   
4. **Evaluation & Performance Analysis**:
   - Assess the model using **accuracy, precision, recall, and F1-score**.
   - Use **confusion matrices** and **visualization techniques** to interpret predictions.
   
5. **Visualization**:
   - Graphically analyze dataset distribution.
   - Display training performance, confusion matrices, and sample predictions.

---

## 📂 Dataset
The dataset consists of **MRI scans** categorized into four distinct types of brain tumors. The dataset is structured into two directories:

- **Training Data**: Contains **5,712** images used for model training.
- **Testing Data**: Contains **1,311** images used for model evaluation.

Each directory has the following four subdirectories representing tumor types:

| Tumor Type    | Description |
|--------------|-------------|
| **Glioma**   | A tumor that arises from the **glial cells**, which support neurons. |
| **Meningioma** | A tumor that forms on the **meninges**, the protective layers of the brain. |
| **No Tumor** | MRI scans with no visible tumor presence. |
| **Pituitary** | A tumor located in the **pituitary gland**, affecting hormone production. |

MRI scans provide **detailed soft-tissue contrast**, making them highly effective for brain tumor diagnosis. However, **manual analysis** of MRI scans is **time-consuming** and requires **specialized expertise**. Deep learning techniques, particularly **CNNs**, provide an automated, scalable solution for this problem.

### 🔗 **Dataset Source**
The dataset used in this project is from the **Brain Tumor MRI Dataset** available on [Kaggle](https://www.kaggle.com).

---

## 🛠 Prerequisites
Ensure that you have the following **dependencies installed** to run the model:

- **Python 3.x**
- **TensorFlow**
- **Keras**
- **NumPy**
- **OpenCV**
- **Matplotlib**
- **Seaborn**
- **Pandas**

You can install the required packages using:
```bash
pip install tensorflow keras numpy opencv-python-headless matplotlib seaborn pandas
```

## 🔄 Data Preprocessing
Before training the CNN model, the MRI images undergo **preprocessing steps** to enhance feature extraction:

### **1️⃣ Image Resizing**
- All images are resized to **150×150 pixels** to maintain **uniform input dimensions**.

### **2️⃣ Data Augmentation**
To improve the model’s generalization capability, the following augmentation techniques are applied:
- **Rotation:** Randomly rotates images to make the model **rotation invariant**.
- **Shifting:** Applies **horizontal and vertical shifts** to account for variations in MRI scans.
- **Shearing:** Shearing transformations to **reduce overfitting**.
- **Zooming:** Random zoom transformations to improve **robustness**.
- **Flipping:** **Horizontal and vertical flips** to introduce additional **data variability**.

### **3️⃣ Normalization**
- Pixel values are **scaled between 0 and 1** to **speed up training** and improve **model convergence**.

---

## 🏗 Model Architecture
A **deep CNN architecture** is designed to **capture spatial patterns** in MRI scans effectively. The architecture consists of:

### **1️⃣ Convolutional Layers**
- Extract **important features** such as **edges, textures, and patterns**.
- Use **ReLU activation** to introduce **non-linearity**.

### **2️⃣ Max-Pooling Layers**
- Reduces **spatial dimensions** while retaining the **most significant features**.

### **3️⃣ Flattening Layer**
- Converts the **feature maps** into a **single vector** for classification.

### **4️⃣ Fully Connected Layers**
- Introduces **dense layers** for **high-level feature abstraction**.
- Uses **ReLU activation** to **increase model complexity**.

### **5️⃣ Dropout Layers**
- **Prevents overfitting** by randomly dropping neurons during training.

### **6️⃣ Softmax Output Layer**
- Computes **probabilities for each tumor class**, enabling **multi-class classification**.

---

## 📊 Evaluation & Performance Metrics
After training, the model is **evaluated on the test dataset** using **key performance metrics**:

- **Accuracy:** Measures overall **correct classification**.
- **Precision:** Evaluates **correct positive predictions**.
- **Recall:** Assesses the model’s ability to **detect actual tumors**.
- **F1-Score:** Harmonic mean of **precision and recall**.

---

## ⚙ Installation
### **1️⃣ Clone the Repository**
To set up the project, run the following commands:
```bash
git clone https://github.com/shekharkshitij/Brain_Tumor_Detection.git
```

## 🔮 Future Work
- ✅ **Implement additional explainability techniques**, such as **SHAP (SHapley Additive Explanations)**, to gain deeper insights into **model decision-making** and interpretability.
- ✅ **Improve model generalization** by integrating **more diverse MRI datasets** and **advanced data augmentation techniques** to enhance **robustness**.
- ✅ **Expand the model to classify additional brain tumor subtypes**, enabling more **granular diagnosis** beyond the current four categories.
- ✅ **Deploy the model as a Web Application using Streamlit**, providing **real-time inference, interactive visualization, and user-friendly accessibility**.

---

## 📚 References
1. [Brain Tumor MRI Dataset - Kaggle](https://www.kaggle.com)  
2. **Deep Learning for Medical Imaging - IEEE**  
3. **Convolutional Neural Networks - Research Papers**  
4. **SHAP for Explainability in AI Models**  

---
