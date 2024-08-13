
# Gender Classification from Handwriting

## Overview
This project classifies gender based on handwriting using computer vision techniques. Given a dataset of handwriting samples, the code determines whether the handwriting belongs to a male or female.

## Author
**Safaa Azbarqa**  
Email: [safaa8721@gmail.com](mailto:safaa8721@gmail.com)

## How to Run
1. **Prerequisites:**  
   Ensure you have Python installed along with the following libraries:
   - OpenCV
   - NumPy
   - scikit-image
   - scikit-learn

2. **Running the Classifier:**  
   Execute the following command in the terminal:
   ```
   python classifier.py <path_train> <path_val> <path_test>
   ```
   - `classifier.py`: Contains the main code.
   - `<path_train>`, `<path_val>`, `<path_test>`: Directories containing training, validation, and test images respectively.

3. **Output:**  
   The results will be saved in a file named `results.txt`, which includes:
   - The model used
   - Confusion matrix
   - Local Binary Pattern (LBP) parameters
   
   *Example of `results.txt`:*
   ```
   SVM with kernel: linear
   Number of points: 8
   Radius: 1
   Accuracy: 66.67%

   Confusion Matrix:
           Male   Female
   Male     26       14
   Female    9       21
   ```

## Date
8th February 2022
