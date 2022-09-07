
import numpy as np
import cv2
import matplotlib.image as pltimage
import matplotlib.pyplot as plt
from scipy.linalg import hadamard
import scipy
import os

#Success Percentage Calculation
"""
percent_error_array = [100.0, 100.0, 5.46875, 6.640625, 4.6875, 4.296875, 3.125, 6.640625, 8.203125, 8.203125, 3.125, 5.078125, 8.59375, 7.421875, 7.03125, 5.078125, 7.421875, 8.984375, 5.46875, 7.03125, 5.078125, 6.25, 5.078125, 5.078125, 10.9375, 8.203125, 7.421875, 6.640625, 7.03125, 6.640625, 6.25, 5.078125, 10.546875, 5.46875, 3.90625, 6.640625, 4.6875, 5.46875, 6.640625, 6.640625, 5.46875, 4.6875, 3.515625, 3.125, 2.34375, 7.03125, 5.859375, 4.6875, 5.859375, 8.203125, 4.296875, 5.859375, 5.46875, 5.859375, 4.296875, 6.640625, 7.03125, 4.6875, 2.34375, 4.6875, 4.6875, 6.25, 4.296875, 7.421875, 6.25, 4.6875, 2.34375, 4.296875, 4.296875, 7.03125, 5.078125, 6.25, 7.421875, 6.25, 5.078125, 5.46875, 5.46875, 6.640625, 5.859375, 6.640625, 8.203125, 7.03125, 4.6875, 3.90625, 4.296875, 6.25, 6.640625, 5.46875, 7.03125, 7.421875, 5.078125, 4.296875, 5.46875, 7.03125, 6.640625, 5.859375, 3.515625, 7.03125, 4.296875, 5.078125, 5.859375, 5.859375, 6.640625, 4.6875, 3.90625, 8.984375, 3.90625, 5.078125, 3.515625, 8.59375, 5.078125, 6.25, 7.8125, 5.46875, 4.6875, 8.59375, 7.421875, 5.46875, 4.296875, 4.6875, 5.46875, 6.640625, 3.515625, 3.90625, 7.8125, 7.421875, 5.859375, 3.90625, 10.546875, 3.90625, 5.46875, 5.859375, 7.8125, 7.03125, 5.859375, 6.640625, 5.078125, 3.90625, 3.90625, 5.078125, 8.203125, 5.078125, 5.078125, 5.078125, 9.765625, 7.8125, 4.6875, 7.03125, 5.078125, 7.421875, 4.6875, 5.859375, 4.6875, 10.546875, 3.125, 5.46875, 6.25, 7.03125, 1.953125, 7.421875, 6.640625, 7.03125, 3.125, 4.6875, 2.734375, 4.296875, 4.6875, 5.859375, 5.078125, 5.46875, 7.03125, 4.296875, 5.078125, 6.25, 5.46875, 4.6875, 7.8125, 7.03125, 3.125, 5.46875, 4.6875, 5.859375, 4.296875, 5.46875, 5.078125, 5.46875, 4.6875, 7.03125, 4.296875, 5.46875, 2.734375, 5.078125, 9.765625, 6.25, 5.859375, 5.859375, 5.859375, 8.203125, 5.078125, 6.640625, 8.203125, 5.859375, 5.46875, 5.46875, 7.8125, 6.640625, 3.515625, 5.078125, 5.078125, 7.421875, 3.90625, 4.296875, 9.765625, 6.640625, 4.6875, 6.25, 8.203125, 4.6875, 3.515625, 5.859375, 5.859375, 7.421875, 5.859375, 5.46875, 5.46875, 5.078125, 4.296875, 4.296875, 5.078125, 6.640625, 4.296875, 7.8125, 6.640625, 8.59375, 5.078125, 6.25, 7.03125, 5.859375, 8.203125, 5.859375, 4.296875, 6.25, 3.515625, 3.90625, 7.421875, 8.203125, 5.46875, 5.859375, 6.640625, 3.125, 4.296875, 4.6875, 5.078125, 5.46875, 4.296875, 5.078125, 10.15625, 5.078125, 5.46875, 6.25, 6.25, 5.46875, 5.46875, 6.25, 6.25, 6.25, 5.859375, 8.203125, 3.515625, 9.375, 3.90625, 5.078125, 20.3125, 13.28125, 4.296875, 5.859375, 5.46875, 5.078125, 4.6875, 7.8125, 8.59375, 10.15625, 4.6875, 9.765625, 6.640625, 8.984375, 5.46875, 5.859375, 5.46875, 7.03125, 1.953125, 1.953125, 4.6875, 7.421875, 5.859375, 6.25, 4.6875, 4.296875, 4.6875, 3.515625, 5.46875, 5.859375, 2.734375, 5.46875, 6.25, 6.25, 2.734375, 4.6875, 3.515625, 4.6875, 6.25, 5.46875, 7.03125, 6.25, 5.859375, 6.25, 5.46875, 5.859375, 3.515625, 4.6875, 15.234375, 7.421875, 3.90625, 4.296875, 5.859375, 5.46875, 3.515625, 6.25, 6.25, 7.8125, 3.125, 6.25, 4.296875, 7.421875, 3.515625, 5.859375, 5.859375, 7.8125, 4.6875, 5.859375, 2.734375, 6.640625, 3.515625, 5.859375, 3.515625, 8.203125, 3.515625, 5.078125, 3.90625, 7.8125, 4.6875, 5.46875, 7.03125, 4.6875, 5.078125, 5.46875, 5.46875, 7.8125, 5.078125, 6.25, 5.078125, 5.078125, 4.6875, 4.6875, 4.6875, 5.859375, 6.640625, 6.640625, 6.25, 7.03125, 3.90625, 5.46875, 5.078125, 8.203125, 5.078125, 6.640625, 8.203125, 5.078125, 3.90625, 5.078125, 5.078125, 7.8125, 3.125, 6.25, 23.046875, 20.3125, 3.90625, 4.6875, 3.90625, 4.296875, 5.46875, 6.640625, 9.765625, 5.46875, 6.25, 7.421875, 8.203125, 6.25, 3.515625, 5.078125, 7.03125, 7.03125, 4.296875, 5.078125, 3.90625, 7.03125, 4.296875, 4.6875, 5.46875, 6.640625, 5.46875, 5.078125, 5.46875, 7.03125, 3.90625, 5.859375, 8.59375, 7.421875, 2.34375, 6.25, 5.46875, 8.984375, 2.34375, 6.640625, 6.640625, 6.25, 3.125, 5.078125, 5.859375, 10.15625, 3.90625, 4.6875, 3.515625, 7.03125, 5.46875, 5.078125, 6.640625, 6.640625, 7.03125, 7.421875, 6.640625, 7.03125, 4.6875, 4.6875, 5.46875, 6.640625, 5.078125, 4.6875, 7.8125, 8.984375, 4.296875, 6.25, 6.25, 3.90625, 4.6875, 5.078125, 7.421875, 7.03125, 6.25, 5.46875, 5.46875, 8.203125, 7.421875, 5.859375, 5.46875, 6.25, 5.859375, 6.25, 5.859375, 5.859375, 7.421875, 5.46875, 7.8125, 8.203125, 3.90625, 5.46875, 7.03125, 8.203125, 5.859375, 5.46875, 5.46875, 6.25, 2.734375, 5.078125, 5.078125, 7.421875, 4.6875, 5.078125, 4.296875, 6.25, 4.296875, 7.8125, 3.515625, 5.46875, 4.296875, 6.25, 7.8125, 7.03125, 3.515625, 5.078125, 3.515625, 7.421875, 6.25, 5.46875, 4.6875, 5.859375, 4.296875, 5.859375, 6.25, 5.859375, 6.25, 5.46875]
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

#Prove TM * Hln = Sln

# S_3_URL = "C:/Users/Dylan Luo/Documents/MMF data/MMF data/calibdata/000003.tif"
# S_3_URL_Absolute_Path = os.path.abspath(S_3_URL)
# S_3 = cv2.imread(S_3_URL_Absolute_Path, cv2.IMREAD_UNCHANGED)
# S_3_Resize = cv2.resize(S_3, (100, 100), interpolation = cv2.INTER_LINEAR)
# S_3_Vector = np.reshape(S_3_Resize, (10000, 1))
# print("Speckle Pattern 3 \n", S_3_Vector)
# print("Shape of Speckle Pattern 3 \n", S_3_Vector.shape)

# S_4_URL = "C:/Users/Dylan Luo/Documents/MMF data/MMF data/calibdata/000004.tif"
# S_4_URL_Absolute_Path = os.path.abspath(S_4_URL)
# S_4 = cv2.imread(S_4_URL_Absolute_Path, cv2.IMREAD_UNCHANGED)
# S_4_Resize = cv2.resize(S_4, (100, 100), interpolation = cv2.INTER_LINEAR)
# S_4_Vector = np.reshape(S_4_Resize, (10000, 1))
# print("Speckle Pattern 4 \n", S_4_Vector)
# print("Shape of Speckle Pattern 4 \n", S_4_Vector.shape)

# S_Difference_Test = np.subtract(S_3, S_4)
# S_Difference_Test_Resize = cv2.resize(S_Difference_Test, (100, 100), interpolation = cv2.INTER_LINEAR)
# S_Difference_Test_Vector = np.reshape(S_Difference_Test_Resize, (10000, 1))
# print("Difference of Speckle Pattern 3 and Speckle Pattern 4 \n", S_Difference_Test_Vector)
# print("Shape of Difference of Speckle Pattern 3 and Speckle Pattern 4 \n", S_Difference_Test_Vector.shape)

#Load all speckle patterns, turn them into vectors, then append them all together to create a giant matrix with dimensions 10000 rows x 256 columns

S_Matrix = []
increment_for_loop_1 = 0

# MMF_Data_URL = "C:/Users/Dylan Luo/Documents/MMF data/MMF data/calibdata/"
MMF_Data_URL = "D:\Code\PythonCodes\MMF\Data\SI_100_0.22_100m\MMF data\calibdata"
MMF_Data_URL_Absolute_Path = os.path.abspath(MMF_Data_URL)
MMF_Data_List = os.listdir(MMF_Data_URL_Absolute_Path)
print("MMF List Data \n", MMF_Data_List[:5])

for x in range(256):
    # x = x + increment_for_loop_1
    # increment_for_loop_1 = increment_for_loop_1 + 1
    # y = x + 1
    # S_Negative_URL = "C:/Users/Dylan Luo/Documents/MMF data/MMF data/calibdata/" + MMF_Data_List[x]
    # S_Negative_URL_Absolute_Path = os.path.abspath(S_Negative_URL)
    S_Negative_URL_Absolute_Path = os.path.join(MMF_Data_URL,  MMF_Data_List[x*2])
    S_Negative = cv2.imread(S_Negative_URL_Absolute_Path, cv2.IMREAD_UNCHANGED)
    S_Negative = S_Negative.astype(np.float)
    # S_Positive_URL = "C:/Users/Dylan Luo/Documents/MMF data/MMF data/calibdata/" + MMF_Data_List[y]
    # S_Positive_URL_Absolute_Path = os.path.abspath(S_Positive_URL)
    S_Positive_URL_Absolute_Path = os.path.join(MMF_Data_URL,  MMF_Data_List[x*2+1])
    S_Positive = cv2.imread(S_Positive_URL_Absolute_Path, cv2.IMREAD_UNCHANGED)
    S_Positive = S_Positive.astype(np.float)

    #S_Negative_Resize = cv2.resize(S_Negative, (100, 100), interpolation = cv2.INTER_LINEAR)
    #S_Positive_Resize = cv2.resize(S_Positive, (100, 100), interpolation = cv2.INTER_LINEAR)
    #S_Difference = np.subtract(S_Positive_Resize, S_Negative_Resize)
    #S_Vector = np.reshape(S_Difference, (10000, 1))
    S_Difference = np.subtract(S_Positive, S_Negative)
    S_Resize = cv2.resize(S_Difference, (100, 100), interpolation = cv2.INTER_LINEAR)
    if x == 3:
        plt.imshow(S_Resize)
        plt.show()
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

#Image Recovery Examples

#Finding the psuedo-inverse of TM then multiplying it with a speckle pattern is faster than dividing the transmission matrix from the speckle pattern.
#The different algorithm run times for np.matmul and np.divide are proof of the statement above.
# S_Recover_1_URL = "C:/Users/Dylan Luo/Documents/MMF data/MMF data/calibdata/000001.tif"
# S_Recover_1_URL_Absolute_Path = os.path.abspath(S_Recover_1_URL)
# S_Recover_1 = cv2.imread(S_Recover_1_URL_Absolute_Path, cv2.IMREAD_UNCHANGED)
# S_Recover_1_Resize = cv2.resize(S_Recover_1, (100, 100), interpolation = cv2.INTER_LINEAR)
# S_Recover_1_Vector = np.reshape(S_Recover_1_Resize, (10000, 1))
# np.seterr(invalid="ignore")
# H_Recover_1 = np.matmul(np.linalg.pinv(TM), S_Recover_1_Vector)
# H_Recover_1_Reshape = np.reshape(H_Recover_1, (16, 16))
# H_Recover_1_Reshape[H_Recover_1_Reshape >= 0.28] = 1
# H_Recover_1_Reshape[H_Recover_1_Reshape < 0.28] = 0
# print("Example Image Recovery 1 \n", H_Recover_1_Reshape)
# print("Shape of Recovered Image 1 \n", H_Recover_1_Reshape.shape)

# S_Recover_2_URL = "C:/Users/Dylan Luo/Documents/MMF data/MMF data/calibdata/000002.tif"
# S_Recover_2_URL_Absolute_Path = os.path.abspath(S_Recover_2_URL)
# S_Recover_2 = cv2.imread(S_Recover_2_URL_Absolute_Path, cv2.IMREAD_UNCHANGED)
# S_Recover_2_Resize = cv2.resize(S_Recover_2, (100, 100), interpolation = cv2.INTER_LINEAR)
# S_Recover_2_Vector = np.reshape(S_Recover_2_Resize, (10000, 1))
# np.seterr(invalid="ignore")
# H_Recover_2 = np.matmul(np.linalg.pinv(TM), S_Recover_2_Vector)
# H_Recover_2_Reshape = np.reshape(H_Recover_2, (16, 16))
# H_Recover_2_Reshape[H_Recover_2_Reshape >= 0.28] = 1
# H_Recover_2_Reshape[H_Recover_2_Reshape < 0.28] = 0
# print("Example Image Recovery 2 \n", H_Recover_2_Reshape)
# print("Shape of Recovered Image 2 \n", H_Recover_2_Reshape.shape)

# S_Recover_Difference_1 = np.subtract(S_Recover_2, S_Recover_1)
# S_Recover_Difference_1_Resize = cv2.resize(S_Recover_Difference_1, (100, 100), interpolation = cv2.INTER_LINEAR)
# S_Recover_Difference_1_Vector = np.reshape(S_Recover_Difference_1_Resize, (10000, 1))
# np.seterr(invalid="ignore")
# H_Recover_Difference_1 = np.matmul(np.linalg.pinv(TM), S_Recover_Difference_1_Vector)
# H_Recover_Difference_1_Reshape = np.reshape(H_Recover_Difference_1, (16, 16))
# print("Example Difference of Two Image Recoveries 1 \n", H_Recover_Difference_1_Reshape)
# print("Shape of Difference of Two Recovered Images 1 \n", H_Recover_Difference_1_Reshape.shape)

# #Finding an ideal image matrix input
# H_New_Split_Test = np.hsplit(H_New, 256)
# H_New_Split_Test_1 = H_New_Split_Test[0]
# #H_New_Test_1_Positive_Ideal = (H_New_Split_Test_1 + 1)/2
# H_New_Test_1_Positive_Ideal = (H_New_Split_Test_1 + 1)/2
# H_New_Test_1_Positive_Ideal_Reshape = np.reshape(H_New_Test_1_Positive_Ideal, (16, 16))
# H_New_Test_1_Negative_Ideal = (-H_New_Split_Test_1 + 1)/2
# #H_New_Test_1_Negative_Ideal = (-H_New_Split_Test_1 + 1)/2
# H_New_Test_1_Negative_Ideal_Reshape = np.reshape(H_New_Test_1_Negative_Ideal, (16, 16))

# print("Positive Ideal Hadamard Matrix \n", H_New_Test_1_Positive_Ideal_Reshape)
# print("Shape of Positive Ideal Hadamard Matrix \n", H_New_Test_1_Positive_Ideal_Reshape.shape)

# #Plot the recovered image and the ideal hadamard matrix input sample using matplot

# #Matplot allows the output to be more user-friendly
# plt.subplot(121)
# plt.title("Recovered Test Image 1")
# plt.imshow(H_Recover_1_Reshape)
# plt.subplot(122)
# plt.title("Ideal Negative Image Input")
# plt.imshow(H_New_Test_1_Negative_Ideal_Reshape)
# plt.show()

# plt.subplot(121)
# plt.title("Recovered Test Image 2")
# plt.imshow(H_Recover_2_Reshape)
# plt.subplot(122)
# plt.title("Ideal Positive Image Input")
# plt.imshow(H_New_Test_1_Positive_Ideal_Reshape)
# plt.show()

invTM = np.linalg.pinv(TM)
# Image Recovery for all the files in the MMF Data Sample Folder

# increment_for_loop_2 = 0
accuracy_matrix = []
# H_New_Split = np.hsplit(H_New, 256)
for w in range(256):
    H_New_Split_Column = H_New[w]
    H_New_Positive_Ideal = (H_New_Split_Column + 1)/2
    H_New_Negative_Ideal = (-H_New_Split_Column + 1)/2
    # Strange phenomenon that is happening with the ideal matrices: The negative algorithm seems to fit the positive ideal hadamard matrix, while the positive algorithm seems to fit the negative ideal hadamard matrix.
    # H_New_Positive_Ideal = (-H_New_Split_Column + 1)/2
    # H_New_Negative_Ideal = (H_New_Split_Column + 1)/2
    H_New_Positive_Ideal_Reshape = np.reshape(H_New_Positive_Ideal, (16, 16))
    H_New_Negative_Ideal_Reshape = np.reshape(H_New_Negative_Ideal, (16, 16))
    # w = w + increment_for_loop_2
    # increment_for_loop_2 = increment_for_loop_2 + 1
    # z = w + 1
    S_Recover_Negative_URL = os.path.join(MMF_Data_URL, MMF_Data_List[w*2]) 
    S_Recover_Negative_URL_Absolute_Path = os.path.abspath(S_Recover_Negative_URL)
    S_Recover_Negative = cv2.imread(S_Recover_Negative_URL_Absolute_Path, cv2.IMREAD_UNCHANGED)
    S_Recover_Negative_Resize = cv2.resize(S_Recover_Negative, (100, 100), interpolation = cv2.INTER_LINEAR)
    S_Recover_Negative_Vector = np.reshape(S_Recover_Negative_Resize, (10000, 1))

    H_Recover_Negative = np.matmul(invTM, S_Recover_Negative_Vector)
    np.seterr(invalid="ignore")
    H_Recover_Negative_Reshape = np.reshape(H_Recover_Negative, (16, 16))
    H_Recover_Negative_Reshape_binary = np.array(H_Recover_Negative_Reshape)
    H_Recover_Negative_Reshape_binary[H_Recover_Negative_Reshape >= 0.5] = 1
    H_Recover_Negative_Reshape_binary[H_Recover_Negative_Reshape < 0.5] = 0

    accuracy_count_negative = np.sum(np.abs(H_Recover_Negative_Reshape_binary - H_New_Negative_Ideal_Reshape))
    # accuracy_count_negative = 0
    # for u in range(len(H_Recover_Negative_Reshape)):
    #     for v in range(len(H_Recover_Negative_Reshape[u])):
    #         if H_Recover_Negative_Reshape[u][v] == H_New_Negative_Ideal_Reshape[u][v]:
    #             accuracy_count_negative = accuracy_count_negative + 1
    accuracy_matrix.append(accuracy_count_negative)

    S_Recover_Positive_URL = os.path.join(MMF_Data_URL, MMF_Data_List[w*2+1])
    S_Recover_Positive_URL_Absolute_Path = os.path.abspath(S_Recover_Positive_URL)
    S_Recover_Positive = cv2.imread(S_Recover_Positive_URL_Absolute_Path, cv2.IMREAD_UNCHANGED)
    S_Recover_Positive_Resize = cv2.resize(S_Recover_Positive, (100, 100), interpolation = cv2.INTER_LINEAR)
    S_Recover_Positive_Vector = np.reshape(S_Recover_Positive_Resize, (10000, 1))

    H_Recover_Positive = np.matmul(invTM, S_Recover_Positive_Vector)
    np.seterr(invalid="ignore")
    H_Recover_Positive_Reshape = np.reshape(H_Recover_Positive, (16, 16))
    print(H_Recover_Positive_Reshape.dtype)
    print(H_Recover_Positive_Reshape)
    H_Recover_Positive_Reshape_binary = np.array(H_Recover_Positive_Reshape)
    H_Recover_Positive_Reshape_binary[H_Recover_Positive_Reshape >= 0.5] = 1
    H_Recover_Positive_Reshape_binary[H_Recover_Positive_Reshape < 0.5] = 0

    accuracy_count_positive = np.sum(np.abs(H_Recover_Positive_Reshape_binary - H_New_Positive_Ideal_Reshape))
    # accuracy_count_positive = 0
    # for u in range(len(H_Recover_Positive_Reshape)):
    #     for v in range(len(H_Recover_Positive_Reshape[u])):
    #         if H_Recover_Positive_Reshape[u][v] == H_New_Positive_Ideal_Reshape[u][v]:
    #             accuracy_count_positive = accuracy_count_positive + 1
    accuracy_matrix.append(accuracy_count_positive)
    Save_File_Negative_URL = os.path.join(MMF_Data_URL, "Recovered Images", "Recovered Image " + str(w*2) + ".png")
    Save_File_Negative_URl_Absolute_Path = os.path.abspath(Save_File_Negative_URL)
    Save_File_Positive_URL = os.path.join(MMF_Data_URL, "Recovered Images", "Recovered Image " + str(w*2+1) + ".png")
    Save_File_Positive_URl_Absolute_Path = os.path.abspath(Save_File_Positive_URL)
    plt.subplot(131)
    plt.title("Recovered Image (Negative)")
    plt.imshow(H_Recover_Negative_Reshape, vmin=0, vmax=1)
    plt.subplot(132)
    plt.title("Recovered Binary Image (Negative)")
    plt.imshow(H_Recover_Negative_Reshape_binary, vmin=0, vmax=1)
    plt.subplot(133)
    plt.title("Ideal Negative Image Input")
    plt.imshow(H_New_Negative_Ideal_Reshape, vmin=0, vmax=1)
    plt.savefig(Save_File_Negative_URl_Absolute_Path)
    plt.show()

    plt.subplot(131)
    plt.title("Recovered Image (Positive)")
    plt.imshow(H_Recover_Positive_Reshape, vmin=0, vmax=1)
    plt.subplot(132)
    plt.title("Recovered Binary Image (Positive)")
    plt.imshow(H_Recover_Positive_Reshape_binary, vmin=0, vmax=1)
    plt.subplot(133)
    plt.title("Ideal Positive Image Input")
    plt.imshow(H_New_Positive_Ideal_Reshape, vmin=0, vmax=1)
    plt.savefig(Save_File_Positive_URl_Absolute_Path)

    #Save_Accuracy_File_Negative_URL = "C:/Users/Dylan Luo/Documents/MMF data/MMF data/Image Accuracy/Accuracy of Image " + str(w + 1) + ".txt"
    #Save_Accuracy_File_Negative_URL_Absolute_Path = os.path.abspath(Save_Accuracy_File_Negative_URL)
    #Save_Accuracy_File_Positive_URL = "C:/Users/Dylan Luo/Documents/MMF data/MMF data/Image Accuracy/Accuracy of Image " + str(z + 1) + ".txt"
    #Save_Accuracy_File_Positive_URL_Absolute_Path = os.path.abspath(Save_Accuracy_File_Positive_URL)

    #Accuracy_Path_Negative = open(Save_Accuracy_File_Negative_URL_Absolute_Path, "a")
    #Accuracy_Path_Negative.write(str(accuracy_count_negative))
    #Accuracy_Path_Negative.close()

    #Accuracy_Path_Positive = open(Save_Accuracy_File_Positive_URL_Absolute_Path, "a")
    #Accuracy_Path_Positive.write(str(accuracy_count_positive))
    #Accuracy_Path_Positive.close()

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

