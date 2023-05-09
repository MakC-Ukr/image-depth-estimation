from global_imports import * 

COEFF = 450
WIDTH_IDEAL = int(COEFF*1.77)
HEIGHT_IDEAL = int(COEFF)
TOP_MATCHES_FOR_FEATURES=60

def retrieve_sift_features(img1, img2):
    # TODO: add source https://stackoverflow.com/questions/36172913/opencv-depth-map-from-uncalibrated-stereo-system 
    sift = SIFT_create(nfeatures=10000)
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1,des2)
    final_matches = sorted(matches, key = lambda x:x.distance)[:TOP_MATCHES_FOR_FEATURES]
    img_with_keypoints = cv2.drawMatches(img1,kp1,img2,kp2,final_matches,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    left_feature_points = [list(kp1[mat.queryIdx].pt) for mat in final_matches] 
    right_feature_points = [list(kp2[mat.trainIdx].pt) for mat in final_matches]
    left_points = np.float32([kp1[m.queryIdx].pt for m in matches])
    right_points = np.float32([kp2[m.trainIdx].pt for m in matches])
    return left_feature_points, right_feature_points, left_points, right_points, img_with_keypoints


def calculate_F_matrix(left_points, right_points):
    A = np.zeros(shape=(len(left_points), 9))

    for i in range(len(left_points)):
        x1, y1 = left_points[i][0], left_points[i][1]
        x2, y2 = right_points[i][0], right_points[i][1]
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

def calc_F_M(list_of_cood_list):
    left_feature_points = list_of_cood_list[0]
    right_feature_points = list_of_cood_list[1]
    pairs = list(zip(left_feature_points, right_feature_points))  
  
    for i in range(10):
        pairs = rd.sample(pairs, 8)  
        rd_left_feature_points, rd_right_feature_points = zip(*pairs) 
        F = calculate_F_matrix(rd_left_feature_points, rd_right_feature_points)
        
        tmp_inliers_img1 = []
        tmp_inliers_img2 = []

        for i in range(len(left_feature_points)):
            img1_x = np.array([left_feature_points[i][0], left_feature_points[i][1], 1])
            img2_x = np.array([right_feature_points[i][0], right_feature_points[i][1], 1])
            distance = abs(np.dot(img2_x.T, np.dot(F,img1_x)))
            if distance < threshold:
                tmp_inliers_img1.append(left_feature_points[i])
                tmp_inliers_img2.append(right_feature_points[i])

        num_of_inliers = len(tmp_inliers_img1)
        if num_of_inliers > 25:
            max_inliers = num_of_inliers
            Best_F = F
            inliers_img1 = tmp_inliers_img1
            inliers_img2 = tmp_inliers_img2
    return Best_F

def rectify_images(img1, img2, pts1, pts2, F):
    # TODO: add ref https://www.andreasjakl.com/understand-and-apply-stereo-rectification-for-depth-maps-part-2/
    retBool ,homography_mat1, homography_mat2 = cv2.stereoRectifyUncalibrated(np.float32(pts1), np.float32(pts2), F, imgSize=(WIDTH_IDEAL, HEIGHT_IDEAL))
    left_rectified = np.zeros((pts1.shape), dtype=int)
    right_rectified = np.zeros((pts2.shape), dtype=int)

    for i in range(pts1.shape[0]):
        temp_location = np.dot(homography_mat1, np.array([pts1[i][0], pts1[i][1], 1]))
        temp_location = temp_location/temp_location[2]
        temp_location = temp_location[:2]
        left_rectified[i] = temp_location
    
    for i in range(pts1.shape[0]):
        temp_location2 = np.dot(homography_mat2, np.array([pts2[i][0], pts2[i][1], 1]))
        temp_location2 = temp_location2/temp_location2[2]
        temp_location2 = temp_location2[:2]
        right_rectified[i] = temp_location2
    img1_rectified = cv2.warpPerspective(img1, homography_mat1, (WIDTH_IDEAL, HEIGHT_IDEAL))
    img2_rectified = cv2.warpPerspective(img2, homography_mat2, (WIDTH_IDEAL, HEIGHT_IDEAL))
    return left_rectified, right_rectified, img1_rectified, img2_rectified

def closest_index(y, x, block_left, right_array, block_size, x_search_block_size, y_search_block_size):
    # TODO: Add ref https://pramod-atre.medium.com/disparity-map-computation-in-python-and-c-c8113c63d701 
    x_min = max(0, x - x_search_block_size)
    x_max = min(right_array.shape[1], x + x_search_block_size)
    y_min = max(0, y - y_search_block_size)
    y_max = min(right_array.shape[0], y + y_search_block_size)
    
    first = True
    min_ssd = None
    min_index = None

    for y in range(y_min, y_max):
        for x in range(x_min, x_max):
            block_right = right_array[y: y+block_size, x: x+block_size]
            if block_left.shape != block_right.shape:
                ssd =  -1
            elif block_left.shape == block_right.shape: 
                ssd_values = (block_left - block_right)**2
                ssd = np.sum(ssd_values)

            if first:
                min_ssd = ssd
                min_index = (y, x)
                first = False
            else:
                if ssd < min_ssd:
                    min_ssd = ssd
                    min_index = (y, x)

    return min_index


def find_disparity_map(img1, img2):
    block_size = 15 
    x_search_block_size = 50 
    y_search_block_size = 1
    disparity_map = np.zeros((HEIGHT_IDEAL, WIDTH_IDEAL))

    for y in tqdm(range(block_size, HEIGHT_IDEAL-block_size)):
        for x in range(block_size, WIDTH_IDEAL-block_size):
            block_left = img1[y:y + block_size, x:x + block_size]
            index = closest_index(y, x, block_left, img2, block_size, x_search_block_size, y_search_block_size)
            disparity_map[y, x] = abs(index[1] - x)
    
    disparity_map_unscaled = disparity_map.copy()
    disparity_map = (disparity_map * 255) / (np.max(disparity_map) - np.min(disparity_map))
    disparity_map = disparity_map.astype(int)
    
    disparity_map_scaled = disparity_map
    return disparity_map_unscaled, disparity_map_scaled


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
    
    f = K1[0][0]

    count=  0
    t = time.time()
    for i in tqdm(range(1000)):
        try:
            count+=1
            left_feature_points, right_feature_points, left_points, right_points, img_withcharted_features = retrieve_sift_features(img1, img2)
            f_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "intermediate_steps", "img_withcharted_features.png")
            cv2.imwrite(f_path, img_withcharted_features)
            # fundamental matrix 
            F = calc_F_M([left_feature_points, right_feature_points])
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
                left_rectified, right_rectified, img1_rectified, img2_rectified = rectify_images(img1, img2, np.int32(left_feature_points), np.int32(right_feature_points), F)
                try:
                    f_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "intermediate_steps", "rectified_1.png")
                    cv2.imwrite(f_path, img1_rectified)
                    f_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "intermediate_steps", "rectified_2.png")
                    cv2.imwrite(f_path, img2_rectified)
                except Exception as e:
                    print("Error in saving rectified images")
                    print(e)
            except Exception as e:
                print("Error in rectification")
                print(e)
            print("Times ran to get R and T matrix", count)
            break
        except Exception as e:
            continue
    print("Time taken to get R and T matrix", time.time()-t, "s")
    t = time.time()
    disparity_map_unscaled, disparity_map_scaled = find_disparity_map(img1_rectified, img2_rectified)
    print("Time taken to get disparity map", time.time()-t, "s")
    t = time.time()
    disparity_map_unscaled_mean = disparity_map_unscaled[disparity_map_unscaled != 0].mean()
    depth_map = np.zeros((HEIGHT_IDEAL, WIDTH_IDEAL))
    depth_array = np.zeros((HEIGHT_IDEAL, WIDTH_IDEAL))
    disparity_map_unscaled[disparity_map_unscaled == 0] = disparity_map_unscaled_mean  
    for i in range(HEIGHT_IDEAL):
        for j in range(WIDTH_IDEAL):
            depth_array[i][j] = baseline*f/disparity_map_unscaled[i][j]

    plt.title('Result-Depth Array')
    plt.imshow(depth_array, cmap='gray')
    f_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data", "depth_array.png")
    plt.savefig(f_path, bbox_inches='tight')
    plt.close()
    print("Time taken to get disparity map", time.time()-t, "s")

if __name__ == "__main__":
    main()
