import numpy as np
import cv2
import matplotlib.image as pltimage
import matplotlib.pyplot as plt
from scipy.linalg import hadamard
import scipy
import os

#Success Percentage Calculation
"""
percent_error_array = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
success_array = []

for q in range(len(percent_error_array)):
    if percent_error_array[q] <= 5:
        success_array.append(percent_error_array[q])
print("Number of successful image recoveries (On the basis that the acceptable percent error is 5% and under) \n" + str(len(success_array)))


success_failure_array = []
for p in range(len(percent_error_array)):
    if percent_error_array[p] <= 5:
        success_failure = "success"
    else:
        success_failure = "failure"
    success_failure_array.append(success_failure)

Success_Failure_URL = "C:/Users/Dylan Luo/Documents/MMF data/MMF data/Recovered Images/Image Accuracies.txt"
Success_Failure_URL_Absolute_Path = os.path.abspath(Success_Failure_URL)
Success_Failure_Write = open(Success_Failure_URL_Absolute_Path, "a")
Success_Failure_Write.write("\n \n" + str(success_array))
Success_Failure_Write.write("\n" + str(len(success_array)))
Success_Failure_Write.write("\n \n" + str(success_failure_array))
Success_Failure_Write.close()
"""

"""
percent_error_array_threshed = [100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0]
success_array_threshed = []

for q in range(len(percent_error_array_threshed)):
    if percent_error_array_threshed[q] <= 5:
        success_array_threshed.append(percent_error_array_threshed[q])
print("Number of successful image recoveries (On the basis that the acceptable percent error is 5% and under) \n" + str(len(success_array)))


success_failure_array_threshed = []
for p in range(len(percent_error_array_threshed)):
    if percent_error_array_threshed[p] <= 5:
        success_failure_threshed = "success"
    else:
        success_failure_threshed = "failure"
    success_failure_array_threshed.append(success_failure_threshed)

Success_Failure_URL_Threshed = "C:/Users/Dylan Luo/Documents/MMF data/MMF data/Recovered Images - Threshed/Image Accuracies (Threshed).txt"
Success_Failure_URL_Absolute_Path_Threshed = os.path.abspath(Success_Failure_URL_Threshed)
Success_Failure_Write_Threshed = open(Success_Failure_URL_Absolute_Path_Threshed, "a")
Success_Failure_Write_Threshed.write("\n \n" + str(success_array_threshed))
Success_Failure_Write_Threshed.write("\n" + str(len(success_array_threshed)))
Success_Failure_Write_Threshed.write("\n \n" + str(success_failure_array_threshed))
Success_Failure_Write_Threshed.close()
"""

#Prove TM * Hln = Sln

S_3_URL = "C:/Users/Dylan Luo/Documents/MMF data/MMF data/calibdata/000003.tif"
S_3_URL_Absolute_Path = os.path.abspath(S_3_URL)
S_3 = cv2.imread(S_3_URL_Absolute_Path, cv2.IMREAD_UNCHANGED)
S_3 = S_3.astype("float64")
S_3_Resize = cv2.resize(S_3, (100, 100), interpolation = cv2.INTER_LINEAR)
S_3_Vector = np.reshape(S_3_Resize, (10000, 1))
print("Speckle Pattern 3 \n", S_3_Vector)
print("Shape of Speckle Pattern 3 \n", S_3_Vector.shape)

S_4_URL = "C:/Users/Dylan Luo/Documents/MMF data/MMF data/calibdata/000004.tif"
S_4_URL_Absolute_Path = os.path.abspath(S_4_URL)
S_4 = cv2.imread(S_4_URL_Absolute_Path, cv2.IMREAD_UNCHANGED)
S_4 = S_4.astype("float64")
S_4_Resize = cv2.resize(S_4, (100, 100), interpolation = cv2.INTER_LINEAR)
S_4_Vector = np.reshape(S_4_Resize, (10000, 1))
print("Speckle Pattern 4 \n", S_4_Vector)
print("Shape of Speckle Pattern 4 \n", S_4_Vector.shape)

S_Difference_Test = np.subtract(S_3, S_4)
S_Difference_Test_Resize = cv2.resize(S_Difference_Test, (100, 100), interpolation = cv2.INTER_LINEAR)
S_Difference_Test_Vector = np.reshape(S_Difference_Test_Resize, (10000, 1))
print("Difference of Speckle Pattern 3 and Speckle Pattern 4 \n", S_Difference_Test_Vector)
print("Shape of Difference of Speckle Pattern 3 and Speckle Pattern 4 \n", S_Difference_Test_Vector.shape)

#Load all speckle patterns, turn them into vectors, then append them all together to create a giant matrix with dimensions 10000 rows x 256 columns

S_Matrix = []
increment_for_loop_1 = 0

MMF_Data_URL = "C:/Users/Dylan Luo/Documents/MMF data/MMF data/calibdata/"
MMF_Data_URL_Absolute_Path = os.path.abspath(MMF_Data_URL)
MMF_Data_List = os.listdir(MMF_Data_URL_Absolute_Path)
print("MMF List Data \n", MMF_Data_List)

for x in range(256):
    x = x + increment_for_loop_1
    increment_for_loop_1 = increment_for_loop_1 + 1
    y = x + 1
    S_Negative_URL = "C:/Users/Dylan Luo/Documents/MMF data/MMF data/calibdata/" + MMF_Data_List[x]
    S_Negative_URL_Absolute_Path = os.path.abspath(S_Negative_URL)
    S_Negative = cv2.imread(S_Negative_URL_Absolute_Path, cv2.IMREAD_UNCHANGED)
    S_Negative = S_Negative.astype("float64")

    S_Positive_URL = "C:/Users/Dylan Luo/Documents/MMF data/MMF data/calibdata/" + MMF_Data_List[y]
    S_Positive_URL_Absolute_Path = os.path.abspath(S_Positive_URL)
    S_Positive = cv2.imread(S_Positive_URL_Absolute_Path, cv2.IMREAD_UNCHANGED)
    S_Positive = S_Positive.astype("float64")

    #S_Negative_Resize = cv2.resize(S_Negative, (100, 100), interpolation = cv2.INTER_LINEAR)
    #S_Positive_Resize = cv2.resize(S_Positive, (100, 100), interpolation = cv2.INTER_LINEAR)
    #S_Difference = np.subtract(S_Positive_Resize, S_Negative_Resize)
    #S_Vector = np.reshape(S_Difference, (10000, 1))
    S_Difference = np.subtract(S_Positive, S_Negative)
    S_Resize = cv2.resize(S_Difference, (100, 100), interpolation = cv2.INTER_LINEAR)
    S_Vector = np.reshape(S_Resize, (10000, 1))
    if x == 0:
        S_Matrix = S_Vector
    else:
        S_Matrix = np.column_stack((S_Matrix, S_Vector))

S_New = np.array(S_Matrix)
print("All of the Speckle Patterns Turned into Vectors and Appended Together \n", S_New)
print("Shape of All of the Speckle Patterns Appended Together \n", S_New.shape)

#Generating a Hadamard Matrix

H_New = hadamard(256)

print("Hadamard Matrix of Order 256 \n", H_New)
print("Shape of Order 256 Hadamard Matrix \n", H_New.shape)

#Finding the Transmission Matrix

# H_New.transpose()/16 is the same as np.linalg.inv(H_New).
# This is why Hadamard Matrices are useful for finding the Transmission Matrix
TM = np.matmul(S_New, (H_New.transpose()/256))
print("Transmission Matrix \n", TM)
print("Shape of Transmission Matrix \n", TM.shape)

#Image Recovery Examples

#Finding the psuedo-inverse of TM then multiplying it with a speckle pattern is faster than dividing the transmission matrix from the speckle pattern.
#The different algorithm run times for np.matmul and np.divide are proof of the statement above.
S_Recover_1_URL = "C:/Users/Dylan Luo/Documents/MMF data/MMF data/calibdata/000025.tif"
S_Recover_1_URL_Absolute_Path = os.path.abspath(S_Recover_1_URL)
S_Recover_1 = cv2.imread(S_Recover_1_URL_Absolute_Path, cv2.IMREAD_UNCHANGED)
S_Recover_1 = S_Recover_1.astype("float64")
S_Recover_1_Resize = cv2.resize(S_Recover_1, (100, 100), interpolation = cv2.INTER_LINEAR)
S_Recover_1_Vector = np.reshape(S_Recover_1_Resize, (10000, 1))
np.seterr(invalid="ignore")
H_Recover_1 = np.matmul(np.linalg.pinv(TM), S_Recover_1_Vector)
H_Recover_1_Reshape = np.reshape(H_Recover_1, (16, 16))
H_Recover_1_Reshape[H_Recover_1_Reshape >= 0.50] = 1
H_Recover_1_Reshape[H_Recover_1_Reshape < 0.50] = 0
print("Example Image Recovery 1 \n", H_Recover_1_Reshape)
print("Shape of Recovered Image 1 \n", H_Recover_1_Reshape.shape)

S_Recover_2_URL = "C:/Users/Dylan Luo/Documents/MMF data/MMF data/calibdata/000026.tif"
S_Recover_2_URL_Absolute_Path = os.path.abspath(S_Recover_2_URL)
S_Recover_2 = cv2.imread(S_Recover_2_URL_Absolute_Path, cv2.IMREAD_UNCHANGED)
S_Recover_2 = S_Recover_2.astype("float64")
S_Recover_2_Resize = cv2.resize(S_Recover_2, (100, 100), interpolation = cv2.INTER_LINEAR)
S_Recover_2_Vector = np.reshape(S_Recover_2_Resize, (10000, 1))
np.seterr(invalid="ignore")
H_Recover_2 = np.matmul(np.linalg.pinv(TM), S_Recover_2_Vector)
H_Recover_2_Reshape = np.reshape(H_Recover_2, (16, 16))
H_Recover_2_Reshape[H_Recover_2_Reshape >= 0.50] = 1
H_Recover_2_Reshape[H_Recover_2_Reshape < 0.50] = 0
print("Example Image Recovery 2 \n", H_Recover_2_Reshape)
print("Shape of Recovered Image 2 \n", H_Recover_2_Reshape.shape)

S_Recover_Difference_1 = np.subtract(S_Recover_2, S_Recover_1)
S_Recover_Difference_1_Resize = cv2.resize(S_Recover_Difference_1, (100, 100), interpolation = cv2.INTER_LINEAR)
S_Recover_Difference_1_Vector = np.reshape(S_Recover_Difference_1_Resize, (10000, 1))
np.seterr(invalid="ignore")
H_Recover_Difference_1 = np.matmul(np.linalg.pinv(TM), S_Recover_Difference_1_Vector)
H_Recover_Difference_1_Reshape = np.reshape(H_Recover_Difference_1, (16, 16))
print("Example Difference of Two Image Recoveries 1 \n", H_Recover_Difference_1_Reshape)
print("Shape of Difference of Two Recovered Images 1 \n", H_Recover_Difference_1_Reshape.shape)

#Finding an ideal image matrix input
H_New_Split_Test = np.hsplit(H_New, 256)
H_New_Split_Test_1 = H_New_Split_Test[12]
H_New_Test_1_Positive_Ideal = (H_New_Split_Test_1 + 1)/2
H_New_Test_1_Positive_Ideal_Reshape = np.reshape(H_New_Test_1_Positive_Ideal, (16, 16))
H_New_Test_1_Negative_Ideal = (-H_New_Split_Test_1 + 1)/2
H_New_Test_1_Negative_Ideal_Reshape = np.reshape(H_New_Test_1_Negative_Ideal, (16, 16))

print("Positive Ideal Hadamard Matrix \n", H_New_Test_1_Positive_Ideal_Reshape)
print("Shape of Positive Ideal Hadamard Matrix \n", H_New_Test_1_Positive_Ideal_Reshape.shape)

#Plot the recovered image and the ideal hadamard matrix input sample using matplot

#Matplot allows the output to be more user-friendly
plt.subplot(121)
plt.title("Recovered Test Image 1")
plt.imshow(H_Recover_1_Reshape)
plt.subplot(122)
plt.title("Ideal Negative Image Input")
plt.imshow(H_New_Test_1_Negative_Ideal_Reshape)
plt.show()

plt.subplot(121)
plt.title("Recovered Test Image 2")
plt.imshow(H_Recover_2_Reshape)
plt.subplot(122)
plt.title("Ideal Positive Image Input")
plt.imshow(H_New_Test_1_Positive_Ideal_Reshape)
plt.show()

# Image Recovery for all the files in the MMF Data Sample Folder

increment_for_loop_3 = 0
accuracy_matrix_threshed = []

for w in range(256):
    H_New_Split_Threshed = np.hsplit(H_New, 256)
    H_New_Split_Column_Threshed = H_New_Split_Threshed[w]
    H_New_Positive_Ideal_Threshed = (H_New_Split_Column_Threshed + 1)/2
    H_New_Negative_Ideal_Threshed = (-H_New_Split_Column_Threshed + 1)/2
    H_New_Positive_Ideal_Reshape_Threshed = np.reshape(H_New_Positive_Ideal_Threshed, (16, 16))
    H_New_Negative_Ideal_Reshape_Threshed = np.reshape(H_New_Negative_Ideal_Threshed, (16, 16))
    w = w + increment_for_loop_3
    increment_for_loop_3 = increment_for_loop_3 + 1
    z = w + 1
    S_Recover_Negative_URL_Threshed = "C:/Users/Dylan Luo/Documents/MMF data/MMF data/calibdata/" + MMF_Data_List[w]
    S_Recover_Negative_URL_Absolute_Path_Threshed = os.path.abspath(S_Recover_Negative_URL_Threshed)
    S_Recover_Negative_Threshed = cv2.imread(S_Recover_Negative_URL_Absolute_Path_Threshed, cv2.IMREAD_UNCHANGED)
    S_Recover_Negative_Threshed = S_Recover_Negative_Threshed.astype("float64")
    S_Recover_Negative_Resize_Threshed = cv2.resize(S_Recover_Negative_Threshed, (100, 100), interpolation = cv2.INTER_LINEAR)
    S_Recover_Negative_Vector_Threshed = np.reshape(S_Recover_Negative_Resize_Threshed, (10000, 1))

    H_Recover_Negative_Threshed = np.matmul(np.linalg.pinv(TM), S_Recover_Negative_Vector_Threshed)
    np.seterr(invalid="ignore")
    H_Recover_Negative_Reshape_Threshed = np.reshape(H_Recover_Negative_Threshed, (16, 16))
    H_Recover_Negative_Reshape_Threshed[H_Recover_Negative_Reshape_Threshed >= 0.50] = 1
    H_Recover_Negative_Reshape_Threshed[H_Recover_Negative_Reshape_Threshed < 0.50] = 0

    accuracy_count_negative_threshed = 0

    for u in range(len(H_Recover_Negative_Reshape_Threshed)):
        for v in range(len(H_Recover_Negative_Reshape_Threshed[u])):
            if H_Recover_Negative_Reshape_Threshed[u][v] == H_New_Negative_Ideal_Reshape_Threshed[u][v]:
                accuracy_count_negative = accuracy_count_negative_threshed + 1
    accuracy_matrix_threshed.append(accuracy_count_negative_threshed)

    S_Recover_Positive_URL_Threshed = "C:/Users/Dylan Luo/Documents/MMF data/MMF data/calibdata/" + MMF_Data_List[z]
    S_Recover_Positive_URL_Absolute_Path_Threshed = os.path.abspath(S_Recover_Positive_URL_Threshed)
    S_Recover_Positive_Threshed = cv2.imread(S_Recover_Positive_URL_Absolute_Path_Threshed, cv2.IMREAD_UNCHANGED)
    S_Recover_Positive_Threshed = S_Recover_Positive_Threshed.astype("float64")
    S_Recover_Positive_Resize_Threshed = cv2.resize(S_Recover_Positive_Threshed, (100, 100), interpolation = cv2.INTER_LINEAR)
    S_Recover_Positive_Vector_Threshed = np.reshape(S_Recover_Positive_Resize_Threshed, (10000, 1))

    H_Recover_Positive_Threshed = np.matmul(np.linalg.pinv(TM), S_Recover_Positive_Vector_Threshed)
    np.seterr(invalid="ignore")
    H_Recover_Positive_Reshape_Threshed = np.reshape(H_Recover_Positive_Threshed, (16, 16))
    H_Recover_Positive_Reshape_Threshed[H_Recover_Positive_Reshape_Threshed >= 0.50] = 1
    H_Recover_Positive_Reshape_Threshed[H_Recover_Positive_Reshape_Threshed < 0.50] = 0

    accuracy_count_positive_threshed = 0

    for u in range(len(H_Recover_Positive_Reshape_Threshed)):
        for v in range(len(H_Recover_Positive_Reshape_Threshed[u])):
            if H_Recover_Positive_Reshape_Threshed[u][v] == H_New_Positive_Ideal_Reshape_Threshed[u][v]:
                accuracy_count_positive = accuracy_count_positive_threshed + 1
    accuracy_matrix_threshed.append(accuracy_count_positive_threshed)

    Save_File_Negative_URL_Threshed = "C:/Users/Dylan Luo/Documents/MMF data/MMF data/Recovered Images - Threshed/Recovered Image " + str(w + 1) + ".png"
    Save_File_Negative_URl_Absolute_Path_Threshed = os.path.abspath(Save_File_Negative_URL_Threshed)
    Save_File_Positive_URL_Threshed = "C:/Users/Dylan Luo/Documents/MMF data/MMF data/Recovered Images - Threshed/Recovered Image " + str(z + 1) + ".png"
    Save_File_Positive_URl_Absolute_Path_Threshed = os.path.abspath(Save_File_Positive_URL_Threshed)
    plt.subplot(121)
    plt.title("Recovered Image (Negative)")
    plt.imshow(H_Recover_Negative_Reshape_Threshed)
    plt.subplot(122)
    plt.title("Ideal Negative Image Input")
    plt.imshow(H_New_Negative_Ideal_Reshape_Threshed)
    plt.savefig(Save_File_Negative_URl_Absolute_Path_Threshed)

    plt.subplot(121)
    plt.title("Recovered Image (Positive)")
    plt.imshow(H_Recover_Positive_Reshape_Threshed)
    plt.subplot(122)
    plt.title("Ideal Positive Image Input")
    plt.imshow(H_New_Positive_Ideal_Reshape_Threshed)
    plt.savefig(Save_File_Positive_URl_Absolute_Path_Threshed)

accuracy_percentage_matrix_threshed = []

for j in range(len(accuracy_matrix_threshed)):
    percent_error = abs((int(accuracy_matrix_threshed[j]) - 256)/256) * 100
    accuracy_percentage_matrix_threshed.append(percent_error)

Save_Accuracy_URL_Threshed = "C:/Users/Dylan Luo/Documents/MMF data/MMF data/Recovered Images - Threshed/Image Accuracies (Threshed).txt"
Save_Accuracy_URL_Absolute_Path_Threshed = os.path.abspath(Save_Accuracy_URL_Threshed)
Accuracy_Write = open(Save_Accuracy_URL_Absolute_Path_Threshed, "a")
Accuracy_Write.write(str(accuracy_matrix_threshed))
Accuracy_Write.write("\n \n" + str(accuracy_percentage_matrix_threshed))
Accuracy_Write.close()

increment_for_loop_2 = 0
accuracy_matrix = []

for w in range(256):
    H_New_Split = np.hsplit(H_New, 256)
    H_New_Split_Column = H_New_Split[w]
    H_New_Positive_Ideal = (H_New_Split_Column + 1)/2
    H_New_Negative_Ideal = (-H_New_Split_Column + 1)/2
    H_New_Positive_Ideal_Reshape = np.reshape(H_New_Positive_Ideal, (16, 16))
    H_New_Negative_Ideal_Reshape = np.reshape(H_New_Negative_Ideal, (16, 16))
    w = w + increment_for_loop_2
    increment_for_loop_2 = increment_for_loop_2 + 1
    z = w + 1
    S_Recover_Negative_URL = "C:/Users/Dylan Luo/Documents/MMF data/MMF data/calibdata/" + MMF_Data_List[w]
    S_Recover_Negative_URL_Absolute_Path = os.path.abspath(S_Recover_Negative_URL)
    S_Recover_Negative = cv2.imread(S_Recover_Negative_URL_Absolute_Path, cv2.IMREAD_UNCHANGED)
    S_Recover_Negative = S_Recover_Negative.astype("float64")
    S_Recover_Negative_Resize = cv2.resize(S_Recover_Negative, (100, 100), interpolation = cv2.INTER_LINEAR)
    S_Recover_Negative_Vector = np.reshape(S_Recover_Negative_Resize, (10000, 1))

    H_Recover_Negative = np.matmul(np.linalg.pinv(TM), S_Recover_Negative_Vector)
    np.seterr(invalid="ignore")
    H_Recover_Negative_Reshape = np.reshape(H_Recover_Negative, (16, 16))

    accuracy_count_negative = 0

    for u in range(len(H_Recover_Negative_Reshape)):
        for v in range(len(H_Recover_Negative_Reshape[u])):
            if H_Recover_Negative_Reshape[u][v] == H_New_Negative_Ideal_Reshape[u][v]:
                accuracy_count_negative = accuracy_count_negative + 1
    accuracy_matrix.append(accuracy_count_negative)

    S_Recover_Positive_URL = "C:/Users/Dylan Luo/Documents/MMF data/MMF data/calibdata/" + MMF_Data_List[z]
    S_Recover_Positive_URL_Absolute_Path = os.path.abspath(S_Recover_Positive_URL)
    S_Recover_Positive = cv2.imread(S_Recover_Positive_URL_Absolute_Path, cv2.IMREAD_UNCHANGED)
    S_Recover_Positive = S_Recover_Positive.astype("float64")
    S_Recover_Positive_Resize = cv2.resize(S_Recover_Positive, (100, 100), interpolation = cv2.INTER_LINEAR)
    S_Recover_Positive_Vector = np.reshape(S_Recover_Positive_Resize, (10000, 1))

    H_Recover_Positive = np.matmul(np.linalg.pinv(TM), S_Recover_Positive_Vector)
    np.seterr(invalid="ignore")
    H_Recover_Positive_Reshape = np.reshape(H_Recover_Positive, (16, 16))

    accuracy_count_positive = 0

    for u in range(len(H_Recover_Positive_Reshape)):
        for v in range(len(H_Recover_Positive_Reshape[u])):
            if H_Recover_Positive_Reshape[u][v] == H_New_Positive_Ideal_Reshape[u][v]:
                accuracy_count_positive = accuracy_count_positive + 1
    accuracy_matrix.append(accuracy_count_positive)

    Save_File_Negative_URL = "C:/Users/Dylan Luo/Documents/MMF data/MMF data/Recovered Images/Recovered Image " + str(w + 1) + ".png"
    Save_File_Negative_URl_Absolute_Path = os.path.abspath(Save_File_Negative_URL)
    Save_File_Positive_URL = "C:/Users/Dylan Luo/Documents/MMF data/MMF data/Recovered Images/Recovered Image " + str(z + 1) + ".png"
    Save_File_Positive_URl_Absolute_Path = os.path.abspath(Save_File_Positive_URL)
    plt.subplot(121)
    plt.title("Recovered Image (Negative)")
    plt.imshow(H_Recover_Negative_Reshape)
    plt.subplot(122)
    plt.title("Ideal Negative Image Input")
    plt.imshow(H_New_Negative_Ideal_Reshape)
    plt.savefig(Save_File_Negative_URl_Absolute_Path)

    plt.subplot(121)
    plt.title("Recovered Image (Positive)")
    plt.imshow(H_Recover_Positive_Reshape)
    plt.subplot(122)
    plt.title("Ideal Positive Image Input")
    plt.imshow(H_New_Positive_Ideal_Reshape)
    plt.savefig(Save_File_Positive_URl_Absolute_Path)

accuracy_percentage_matrix = []

for j in range(len(accuracy_matrix)):
    percent_error = abs((int(accuracy_matrix[j]) - 256)/256) * 100
    accuracy_percentage_matrix.append(percent_error)

Save_Accuracy_URL = "C:/Users/Dylan Luo/Documents/MMF data/MMF data/Recovered Images/Image Accuracies.txt"
Save_Accuracy_URL_Absolute_Path = os.path.abspath(Save_Accuracy_URL)
Accuracy_Write = open(Save_Accuracy_URL_Absolute_Path, "a")
Accuracy_Write.write(str(accuracy_matrix))
Accuracy_Write.write("\n \n" + str(accuracy_percentage_matrix))
Accuracy_Write.close()

#cv2 imshow function to show the speckle pattern

#cv2.imshow("image", S_Recover_1)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

