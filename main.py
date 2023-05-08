import os
import json
import re
import math
import cv2
import random as rd
from tqdm import tqdm
import numpy as np  
import matplotlib.pyplot as plt
import time

# 1. Remove all TODOs
# 2. Remove all print statements
# 3. Try with ORB and SIFT
# 4. Try with different features

WIDTH_IDEAL = 256
HEIGHT_IDEAL = 256
TOP_MATCHES_FOR_FEATURES=50

def retrieve_sift_features(img1, img2):
    # TODO: add source https://stackoverflow.com/questions/36172913/opencv-depth-map-from-uncalibrated-stereo-system 
    orb = cv2.ORB_create(nfeatures=10000)
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1,des2)
    final_matches = sorted(matches, key = lambda x:x.distance)[:TOP_MATCHES_FOR_FEATURES]
    img_with_keypoints = cv2.drawMatches(img1,kp1,img2,kp2,final_matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # Getting x,y coordinates of the matches
    list_kp1 = [list(kp1[mat.queryIdx].pt) for mat in final_matches] 
    list_kp2 = [list(kp2[mat.trainIdx].pt) for mat in final_matches]
    left_points = np.float32([kp1[m.queryIdx].pt for m in matches])
    right_points = np.float32([kp2[m.trainIdx].pt for m in matches])
    return list_kp1, list_kp2, left_points, right_points, img_with_keypoints




def calculate_F_matrix(list_kp1, list_kp2):
    """This function is used to calculate the F matrix from a set of 8 points using SVD.
        Furthermore, the rank of F matrix is reduced from 3 to 2 to make the epilines converge."""

    A = np.zeros(shape=(len(list_kp1), 9))

    for i in range(len(list_kp1)):
        x1, y1 = list_kp1[i][0], list_kp1[i][1]
        x2, y2 = list_kp2[i][0], list_kp2[i][1]
        A[i] = np.array([x1*x2, x1*y2, x1, y1*x2, y1*y2, y1, x2, y2, 1])

    U, s, Vt = np.linalg.svd(A)
    F = Vt[-1,:]
    F = F.reshape(3,3)
   
    # Downgrading the rank of F matrix from 3 to 2
    Uf, Df, Vft = np.linalg.svd(F)
    Df[2] = 0
    s = np.zeros((3,3))
    for i in range(3):
        s[i][i] = Df[i]

    F = np.dot(Uf, np.dot(s, Vft))
    return F

def RANSAC_F_matrix(list_of_cood_list):
    """This method is used to shortlist the best F matrix using RANSAC based on the number of inliers."""
    
    list_kp1 = list_of_cood_list[0]
    list_kp2 = list_of_cood_list[1]
    pairs = list(zip(list_kp1, list_kp2))  
    max_inliers = 20
    threshold = 0.05  # Tune this value
  
    for i in range(1000):
        pairs = rd.sample(pairs, 8)  
        rd_list_kp1, rd_list_kp2 = zip(*pairs) 
        F = calculate_F_matrix(rd_list_kp1, rd_list_kp2)
        
        tmp_inliers_img1 = []
        tmp_inliers_img2 = []

        for i in range(len(list_kp1)):
            img1_x = np.array([list_kp1[i][0], list_kp1[i][1], 1])
            img2_x = np.array([list_kp2[i][0], list_kp2[i][1], 1])
            distance = abs(np.dot(img2_x.T, np.dot(F,img1_x)))
            # print(distance)

            if distance < threshold:
                tmp_inliers_img1.append(list_kp1[i])
                tmp_inliers_img2.append(list_kp2[i])

        num_of_inliers = len(tmp_inliers_img1)

        if num_of_inliers > max_inliers:
            print("Number of inliers", num_of_inliers)
            max_inliers = num_of_inliers
            Best_F = F
            inliers_img1 = tmp_inliers_img1
            inliers_img2 = tmp_inliers_img2
    return Best_F

def drawlines(img1src, img2src, lines, pts1src, pts2src):
    """This fucntion is used to visualize the epilines on the images
        img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines """
    r, c = img1src.shape
    img1color = cv2.cvtColor(img1src, cv2.COLOR_GRAY2BGR)
    img2color = cv2.cvtColor(img2src, cv2.COLOR_GRAY2BGR)
    # Edit: use the same random seed so that two images are comparable!
    np.random.seed(0)
    for r, pt1, pt2 in zip(lines, pts1src, pts2src):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1color = cv2.line(img1color, (x0, y0), (x1, y1), color, 1)
        img1color = cv2.circle(img1color, tuple(pt1), 5, color, -1)
        img2color = cv2.circle(img2color, tuple(pt2), 5, color, -1)
    
    return img1color, img2color


# Its all about resolution - Its a trade off between resolution and time of computation
def rectification(img1, img2, pts1, pts2, F):
    """This function is used to rectify the images to make camera pose's parallel and thus make epiplines as horizontal.
        Since camera distortion parameters are not given we will use cv2.stereoRectifyUncalibrated(), instead of stereoRectify().
    """

    # Stereo rectification
    _, H1, H2 = cv2.stereoRectifyUncalibrated(np.float32(pts1), np.float32(pts2), F, imgSize=(WIDTH_IDEAL, HEIGHT_IDEAL))
    # print("H1",H1)
    # print("H2",H2)
    rectified_pts1 = np.zeros((pts1.shape), dtype=int)
    rectified_pts2 = np.zeros((pts2.shape), dtype=int)

    # Rectify the feature points
    for i in range(pts1.shape[0]):
        source1 = np.array([pts1[i][0], pts1[i][1], 1])
        new_point1 = np.dot(H1, source1)
        new_point1[0] = int(new_point1[0]/new_point1[2])
        new_point1[1] = int(new_point1[1]/new_point1[2])
        new_point1 = np.delete(new_point1, 2)
        rectified_pts1[i] = new_point1

        source2 = np.array([pts2[i][0], pts2[i][1], 1])
        new_point2 = np.dot(H2, source2)
        new_point2[0] = int(new_point2[0]/new_point2[2])
        new_point2[1] = int(new_point2[1]/new_point2[2])
        new_point2 = np.delete(new_point2, 2)
        rectified_pts2[i] = new_point2

    # Rectify the images and save them
    img1_rectified = cv2.warpPerspective(img1, H1, (WIDTH_IDEAL, HEIGHT_IDEAL))
    img2_rectified = cv2.warpPerspective(img2, H2, (WIDTH_IDEAL, HEIGHT_IDEAL))
    
    f_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "intermediate_steps", "rectified_1.png")
    cv2.imwrite(f_path, img1_rectified)
    f_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "intermediate_steps", "rectified_2.png")
    cv2.imwrite(f_path, img2_rectified)
    
    return rectified_pts1, rectified_pts2, img1_rectified, img2_rectified





def main():
    # Get directory in which file is located
    dir_path = os.path.dirname(os.path.realpath(__file__))

    img1 = cv2.imread(os.path.join(dir_path, "data", "img1.png"), 0)
    img2 = cv2.imread(os.path.join(dir_path, "data", "img2.png"), 0)
    # print("SIDHU", img2)
    plt.imshow(img2, cmap='hot')

    img1 = cv2.resize(img1, (WIDTH_IDEAL, HEIGHT_IDEAL), interpolation = cv2.INTER_AREA)
    img2 = cv2.resize(img2, (WIDTH_IDEAL, HEIGHT_IDEAL), interpolation = cv2.INTER_AREA)
    
    variables = {}
    with open(os.path.join(dir_path, "data", "calib.txt"), "r") as file:
        for line in file:
            if line.startswith("cam0"):
                cam1 = []
                for row in line.strip().split('=')[1].strip('[]').split(';'):
                    cam1.append([float(num) for num in row.strip().split()])
                variables["K1"] = np.array(cam1)
                K1 = np.array(cam1)
            elif line.startswith("cam1"):
                cam2 = []
                for row in line.strip().split('=')[1].strip('[]').split(';'):
                    cam2.append([float(num) for num in row.strip().split()])
                variables["K2"] = np.array(cam2)
                K2 = np.array(cam2 )
            elif line.startswith("baseline"):
                variables["baseline"] = float(line.strip().split('=')[1])
                baseline = float(line.strip().split('=')[1])
            elif line.startswith("width"):
                variables["width"] = int(line.strip().split('=')[1])
            elif line.startswith("height"):
                variables["height"] = int(line.strip().split('=')[1])
    
    f = K1[0][0]

    count=  0
    t = time.time()
    for i in tqdm(range(1000)):
        try:
            count+=1
            list_kp1, list_kp2, left_points, right_points, img_withcharted_features = retrieve_sift_features(img1, img2)
            f_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "intermediate_steps", "img_withcharted_features.png")
            cv2.imwrite(f_path, img_withcharted_features)
            # fundamental matrix 
            F = RANSAC_F_matrix([list_kp1, list_kp2])
            # essential matrix
            E = np.dot(F,K1)
            E = np.dot(K2.T, E)
            try:
                _, R, t, _ = cv2.recoverPose(E, left_points, right_points, K1, K2)
                f_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "intermediate_steps", "rotation_matrix.json")
                json.dump(R.tolist(), open(f_path, "w+"))
                f_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "intermediate_steps", "translation_matrix.json")
                json.dump(t.tolist(), open(f_path, "w+"))
            except Exception as e:
                print("Error in saving rotation and translation matrix")
                print(e)
            try:
                rectified_pts1, rectified_pts2, img1_rectified, img2_rectified = rectification(img1, img2, np.int32(list_kp1), np.int32(list_kp2), F)
            except Exception as e:
                print("Error in rectification")
                print(e)
            print("Times ran to get R and T matrix", count)
            break
        except Exception as e:
            continue
    print("Time taken to get R and T matrix", time.time()-t, "s")
    t = time.time()

if __name__ == "__main__":
    main()
