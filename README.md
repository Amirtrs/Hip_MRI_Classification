
# **Hip MRI Classification with ResNet101**

This project focuses on classifying Hip MRI images into two categories: `normal` and `osseous_lesion` using deep learning techniques. It employs ResNet101 for feature extraction and fine-tunes the model for binary classification. The project includes the following key steps: data preparation, model training, evaluation, and performance visualization.

## **Project Overview**

The goal of this project is to develop a deep learning model capable of classifying Hip MRI images. The dataset used contains images labeled as either `normal` or `osseous_lesion`, and the task is to classify these images using a Convolutional Neural Network (CNN) based on the ResNet101 architecture.

### **Key Steps:**

1. **Data Preprocessing**:  
   - The dataset is stored in a zip file, which is extracted from Google Drive and split into training, validation, and test sets.
   - The images are preprocessed using CLAHE (Contrast Limited Adaptive Histogram Equalization) to improve image contrast before being resized to the target size of 224x224.
   - The images are then saved into corresponding folders for `train`, `val`, and `test`.

2. **Model Architecture**:  
   - **ResNet101**: The pre-trained ResNet101 model is used as a base model with the top classification layer removed.
   - **Additional Layers**: Custom layers are added to adapt the network for binary classification.
     - A Global Average Pooling layer.
     - A Dense layer with 1024 units and ReLU activation.
     - A Dense output layer with a sigmoid activation function for binary classification.

3. **Training**:  
   - The model is trained using a binary cross-entropy loss function and the Adam optimizer.
   - The model is fine-tuned by freezing the weights of the pre-trained ResNet101 layers.
   - Data augmentation techniques, such as horizontal flipping, are applied to the training data to improve generalization.

4. **Evaluation**:  
   - The trained model is evaluated using a test dataset that is processed similarly to the training data.
   - The performance metrics include accuracy, confusion matrix, and various visualizations like ROC and Precision-Recall curves.

## **Project Setup**

### **1. Environment Setup**
   - The project requires Google Colab or a local environment with TensorFlow and Keras installed. 
   - Required libraries:
     - TensorFlow
     - Keras
     - OpenCV
     - Scikit-learn
     - Matplotlib
     - Seaborn

### **2. Data**
   - The dataset, `RadImageNet - Hip MRI.zip`, contains two classes:
     - `normal`
     - `osseous_lesion`
   - The data is split into training, validation, and test sets with the following ratios:
     - Training: 70%
     - Validation: 15%
     - Testing: 15%

### **3. Preprocessing**
   - **CLAHE** is applied to enhance the contrast of the MRI images before resizing them to 224x224 pixels.
   - The images are saved into directories: `/train`, `/val`, and `/test`.

### **4. Model Configuration**
   - **ResNet101** is used as the base model with pre-trained weights from ImageNet.
   - The final layer is replaced with custom dense layers to fit the binary classification task.

### **5. Training**
   - Model training is conducted for 30 epochs with a batch size of 150.
   - The Adam optimizer is used with a learning rate of `0.0001`.
   - The training process is visualized with training and validation accuracy/loss plots.

### **6. Evaluation**
   - The final evaluation uses the test set to determine the model's accuracy.
   - A confusion matrix is plotted to visualize the classification performance.
   - A ROC curve and Precision-Recall curve are plotted to further assess model performance.

## **File Structure**

The project directory structure is as follows:

```
/content/
    RadImageNet - Hip MRI.zip          # Raw dataset file
    DATA/
        train/
            normal/
            osseous_lesion/
        val/
            normal/
            osseous_lesion/
        test/
            normal/
            osseous_lesion/
```

### **Files**
- `hip.ipynb`: The Jupyter notebook containing the entire project workflow, including data processing, model definition, training, and evaluation.

## **Results**

The trained model's performance is visualized using:
- **Confusion Matrix**: Shows the true vs. predicted labels.
- **ROC Curve**: Evaluates the model’s ability to distinguish between classes.
- **Precision-Recall Curve**: Assesses the trade-off between precision and recall for the positive class.

## **Dependencies**
Install the required libraries using pip:

```bash
!pip install tensorflow keras opencv-python scikit-learn matplotlib seaborn
```

## **License**
This project is licensed under the MIT License - see the LICENSE file for details.
