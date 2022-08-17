import numpy as np
import cv2
import matplotlib.image as pltimage
import matplotlib.pyplot as plt
from scipy.linalg import hadamard
import os

#Prove TM * Hln = Sln

S_3_URL = "C:/Users/weili/Downloads/MMF data/calibdata/000003.tif"
S_3_URL_Absolute_Path = os.path.abspath(S_3_URL)
S_3 = cv2.imread(S_3_URL_Absolute_Path, cv2.IMREAD_UNCHANGED)
S_3_Resize = cv2.resize(S_3, (100, 100), interpolation = cv2.INTER_LINEAR)
S_3_Vector = np.reshape(S_3_Resize, (10000, 1))
print("Speckle Pattern 3 \n", S_3_Vector)
print("Shape of Speckle Pattern 3 \n", S_3_Vector.shape)

S_4_URL = "C:/Users/weili/Downloads/MMF data/calibdata/000004.tif"
S_4_URL_Absolute_Path = os.path.abspath(S_4_URL)
S_4 = cv2.imread(S_4_URL_Absolute_Path, cv2.IMREAD_UNCHANGED)
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

MMF_Data_URL = "C:/Users/weili/Downloads/MMF data/calibdata/"
MMF_Data_URL_Absolute_Path = os.path.abspath(MMF_Data_URL)
MMF_Data_List = os.listdir(MMF_Data_URL_Absolute_Path)
print("MMF List Data \n", MMF_Data_List)

for x in range(256):
    x = x + increment_for_loop_1
    increment_for_loop_1 = increment_for_loop_1 + 1
    y = x + 1
    S_Positive_URL = "C:/Users/weili/Downloads/MMF data/calibdata/" + MMF_Data_List[x]
    S_Positive_URL_Absolute_Path = os.path.abspath(S_Positive_URL)
    S_Positive = cv2.imread(S_Positive_URL_Absolute_Path, cv2.IMREAD_UNCHANGED)
    S_Negative_URL = "C:/Users/weili/Downloads/MMF data/calibdata/" + MMF_Data_List[y]
    S_Negative_URL_Absolute_Path = os.path.abspath(S_Negative_URL)
    S_Negative = cv2.imread(S_Negative_URL_Absolute_Path, cv2.IMREAD_UNCHANGED)
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
TM = np.matmul(S_New, H_New.transpose()/256)
print("Transmission Matrix \n", TM)
print("Shape of Transmission Matrix \n", TM.shape)

#Image Recovery

#Finding the psuedo-inverse of TM then multiplying it with a speckle pattern is faster than dividing the transmission matrix from the speckle pattern.
#The different algorithm run times for np.matmul and np.divide are proof of the statement above.
S_Recover_1_URL = "C:/Users/weili/Downloads/MMF data/calibdata/000007.tif"
S_Recover_1_URL_Absolute_Path = os.path.abspath(S_Recover_1_URL)
S_Recover_1 = cv2.imread(S_Recover_1_URL_Absolute_Path, cv2.IMREAD_UNCHANGED)
S_Recover_1_Resize = cv2.resize(S_Recover_1, (100, 100), interpolation = cv2.INTER_LINEAR)
S_Recover_1_Vector = np.reshape(S_Recover_1_Resize, (10000, 1))
np.seterr(invalid="ignore")
H_Recover_1 = np.matmul(np.linalg.pinv(TM), S_Recover_1_Vector)
H_Recover_1_Reshape = np.reshape(H_Recover_1, (16, 16))
print("Example Image Recovery 1 \n", H_Recover_1_Reshape)
print("Shape of Recovered Image 1 \n", H_Recover_1_Reshape.shape)

S_Recover_2_URL = "C:/Users/weili/Downloads/MMF data/calibdata/000008.tif"
S_Recover_2_URL_Absolute_Path = os.path.abspath(S_Recover_2_URL)
S_Recover_2 = cv2.imread(S_Recover_2_URL_Absolute_Path, cv2.IMREAD_UNCHANGED)
S_Recover_2_Resize = cv2.resize(S_Recover_2, (100, 100), interpolation = cv2.INTER_LINEAR)
S_Recover_2_Vector = np.reshape(S_Recover_2_Resize, (10000, 1))
np.seterr(invalid="ignore")
H_Recover_2 = np.matmul(np.linalg.pinv(TM), S_Recover_2_Vector)
H_Recover_2_Reshape = np.reshape(H_Recover_2, (16, 16))
print("Example Image Recovery 2 \n", H_Recover_2_Reshape)
print("Shape of Recovered Image 2 \n", H_Recover_2_Reshape.shape)

S_Recover_Difference_1 = np.subtract(S_Recover_1, S_Recover_2)
S_Recover_Difference_1_Resize = cv2.resize(S_Recover_Difference_1, (100, 100), interpolation = cv2.INTER_LINEAR)
S_Recover_Difference_1_Vector = np.reshape(S_Recover_Difference_1_Resize, (10000, 1))
np.seterr(invalid="ignore")
H_Recover_Difference_1 = np.matmul(np.linalg.pinv(TM), S_Recover_Difference_1_Vector)
H_Recover_Difference_1_Reshape = np.reshape(H_Recover_Difference_1, (16, 16))
print("Example Difference of Two Image Recoveries 1 \n", H_Recover_Difference_1_Reshape)
print("Shape of Difference of Two Recovered Images 1 \n", H_Recover_Difference_1_Reshape.shape)

#Finding an ideal image matrix input

H_New_Split = np.hsplit(H_New, 256)
H_New_Split_4 = H_New_Split[4]
H_New_4_Positive_Ideal = (H_New_Split_4 + 1)/2
H_New_4_Positive_Ideal_Reshape = np.reshape(H_New_4_Positive_Ideal, (16, 16))

print("Positive Ideal Hadamard Matrix \n", H_New_4_Positive_Ideal_Reshape)
print("Shape of Positive Ideal Hadamard Matrix \n", H_New_4_Positive_Ideal_Reshape.shape)

#Plot the recovered image and the ideal hadamard matrix input sample using matplot

#Matplot allows the output to be more user friendly
plt.subplot(121)
plt.title("Recovered Image \n from Speckle Pattern 7 \n Subtracted by Speckle Pattern 8")
plt.imshow(H_Recover_Difference_1_Reshape)
plt.subplot(122)
plt.title("Ideal Image Input")
plt.imshow(H_New_4_Positive_Ideal_Reshape)
plt.show()

#cv2 imshow function to show the speckle pattern

#cv2.imshow("image", S_Recover_1)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
