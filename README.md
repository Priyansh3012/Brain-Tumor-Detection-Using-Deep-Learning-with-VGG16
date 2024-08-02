# Brain-Tumor-Detection-Using-Deep-Learning-with-VGG16
This project involves developing a deep learning model to classify MRI brain scan images into two categories: tumor and no tumor. By leveraging the VGG16 architecture, a pre-trained convolutional neural network, this project aims to automate the detection of brain tumors, providing an effective tool for aiding medical professionals in diagnosis.

![image](https://github.com/user-attachments/assets/0a7bc61a-f183-4afb-a129-cf350a6b6ce0)   

## Tools and Libraries Used:
**Programming Language:** 
   * Python
     
**Libraries:**
   * TensorFlow
   * Keras
   * OpenCV
   * Matplotlib
   * Numpy
   * Scikit-learn
   * Pandas

## Dataset Description:

  * The dataset consists of MRI images classified into two categories:
  * Yes: Images containing brain tumors.
  * No: Images without brain tumors.
  * Each image is grayscale but preprocessed to fit a 3-channel input to suit the VGG16 model, resulting in an image size of 224x224x3.

## Project Steps:

   1. **Data Loading and Preprocessing:**
    
        * **Image Loading:** Images are loaded from the directory using OpenCV and resized to 224x224.
        * **Normalization:** Pixel values are normalized to a range of [0, 1].
        * **Label Encoding:** The labels ('yes' and 'no') are encoded to numerical values for model training.

   2. **Data Augmentation and Cropping:**

        * **Augmentation:** Images are augmented using techniques such as rotation, width and height shifts, and horizontal flips to increase dataset variability.
        * **Cropping:** Images are cropped to focus on regions of interest (ROIs) that potentially contain tumor information.
     
  3. **Data Splitting:**
     
       * The dataset is shuffled and split into training and testing sets using an 90/10 ratio to ensure robust evaluation of model performance.

  4. **Model Architecture:**
     
       * **VGG16 Model:** A pre-trained VGG16 model is fine-tuned for binary classification. The top layers are replaced with a global average pooling layer, followed by fully connected 
                          layers, dropout for regularization, and a final sigmoid activation function for binary output.
         
  5. **Model Training:**
     
      * **Compilation:** The model is compiled with the Adam optimizer and binary cross-entropy loss.
      * **Training:** The model is trained on the augmented dataset, including original, augmented, and cropped images.

  6. **Evaluation:**

      * **Validation Accuracy and Loss:** The model achieved a validation accuracy of 98.69%, indicating a high level of precision in predicting the presence or absence of brain tumors. 
                                      Validation loss and training loss were also monitored to track the model's performance and convergence during training.

      * **Training Accuracy and Loss:** The training accuracy and loss were calculated to assess how well the model fit the training data. These metrics help identify potential 
                                        overfitting, where the model performs well on training data but not on unseen data.

      * **Overfitting Analysis:** Graphs of training and validation accuracy and loss were plotted to visualize overfitting. A small gap between training and validation curves suggests the model generalizes well.
    
        ![image](https://github.com/user-attachments/assets/f26debdb-8712-4763-a32b-c2cc1eb6c979)


      * The trained model is evaluated on the test set to determine its accuracy, precision, recall, and F1-score.
        
      * **Confusion Matrix:** Confusion matrices were plotted to provide a detailed breakdown of true positives, true negatives, false positives, and false negatives. This visualization 
                              offers insights into the types of errors the model makes and helps ensure it is reliable in practical applications. 

  7. **Visualization:**
      
      * Sample images are displayed with their original, augmented, and cropped versions using subplots to provide insight into preprocessing effectiveness.

      ![image](https://github.com/user-attachments/assets/0a7bc61a-f183-4afb-a129-cf350a6b6ce0)
               

  8. **Model Prediction on New Data:**
      
      * The model is used to predict new MRI scans, providing class labels and confidence scores for brain tumor detection.

## Conclusion: 

   * This project demonstrates the application of transfer learning in medical image analysis, showcasing how pre-trained models like VGG16 can be adapted for specialized tasks such as 
   brain tumor detection. The model provides a foundation for developing more advanced and accurate diagnostic tools, highlighting the potential of deep learning in healthcare.

## Future Work:

   * **Data Expansion:** Acquire more diverse datasets to improve model generalization.
   * **Advanced Architectures:** Explore more advanced architectures like ResNet or EfficientNet for improved accuracy.
   * **Real-Time Deployment:** Implement the model in a real-time application for clinical use. 

     
        
