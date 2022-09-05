# Gender-Classification-Computer-Vision-
Gender Classification from Handwriting

the author:

	safaa azbarqa  -- email: "safaa8721@gmail.com"

With a given file, the code cross over all the pictures and determine which of them are in a female handwriting 
and which are in the handwriting of a male.

To run the code:
use any development environment with python installation and install these libraries:
- opencv
- numpy
- skimage
- sklearn

then write this row in the terminal:
                  > python classifier.py path_train path_val path_test
                  
classifier.py --- python file that contains the code

path_train,path_val,path_test------ Are the folder names of pictures are in train, valid and test.

As an output you will receive the file "results.txt" that contains: 

what model do we use, confusion matrix,
the parameters that we in LBP.

*Example for the results file:

svm with kernal: linear

number points: 8

radius: 1

Accuracy: 66.67%

 	 male 	 female 
   
 male 	 26     	 14 
 
 female 	 9    	 21
 
 ---------------
 8.2.2022
