import os
import json
import re
import math
import cv2
import random as rd
from tqdm import tqdm
import numpy as np  
from cv2 import ORB_create as SIFT_create
import matplotlib.pyplot as plt
import time
WIDTH_IDEAL = int(384/2)
HEIGHT_IDEAL = int(256/2)
TOP_MATCHES_FOR_FEATURES=50
max_inliers = 20
threshold = 0.05  # Tune this value
