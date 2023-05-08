import os
import re
import math
import cv2
import random as rd
import numpy as np  
import matplotlib.pyplot as plt

# 1. Remove all TODOs
# 2. Remove all print statements
# 3. Try with ORB and SIFT
# 4. MAKSIM: DO SIFT

WIDTH_IDEAL = 256
HEIGHT_IDEAL = 256

def main():
    # Get directory in which file is located
    dir_path = os.path.dirname(os.path.realpath(__file__))

    img1 = cv2.imread(os.path.join(dir_path, "data", "img1.png"), 0)
    img2 = cv2.imread(os.path.join(dir_path, "data", "img2.png"), 0)
    print("SIDHU", img2)
    plt.imshow(img2, cmap='hot')

    img1 = cv2.resize(img1, (WIDTH_IDEAL, HEIGHT_IDEAL), interpolation = cv2.INTER_AREA)
    img2 = cv2.resize(img2, (WIDTH_IDEAL, HEIGHT_IDEAL), interpolation = cv2.INTER_AREA)
    
    variables = {}
    with open(os.path.join(dir_path, "data", "calib.txt"), "r") as file:
        for line in file:
            if line.startswith("cam1"):
                cam1 = []
                for row in line.strip().split('=')[1].strip('[]').split(';'):
                    cam1.append([float(num) for num in row.strip().split()])
                variables["K1"] = cam1
                K1 = cam1
            elif line.startswith("cam2"):
                cam2 = []
                for row in line.strip().split('=')[1].strip('[]').split(';'):
                    cam2.append([float(num) for num in row.strip().split()])
                variables["K2"] = cam2
                K2 = cam2 #TODO: USE AS DICT
            elif line.startswith("baseline"):
                variables["baseline"] = float(line.strip().split('=')[1])
                baseline = float(line.strip().split('=')[1])
            elif line.startswith("width"):
                variables["width"] = int(line.strip().split('=')[1])
            elif line.startswith("height"):
                variables["height"] = int(line.strip().split('=')[1])
    
    f = K1[0][0]

if __name__ == "__main__":
    main()
    
