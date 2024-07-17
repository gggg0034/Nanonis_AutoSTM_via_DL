import atexit
import math
import os
import random
import time

import cv2
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import lstsq


# def a function to change the Hexadecimal color to BGR color, Hexadecimal color input is a string, BGR color output is a tuple
def Hex_to_BGR(Hex):
    Hex = Hex.strip('#')
    RGB = tuple(int(Hex[i:i+2], 16) for i in (0, 2, 4))
    BGR = (RGB[2], RGB[1], RGB[0]) # cv2 use BGR
    return BGR

def BGR_to_Hex(rgb_color):
    """ Convert RGB to hexadecimal. """
    return f"#{rgb_color[2]:02x}{rgb_color[1]:02x}{rgb_color[0]:02x}"

def interpolate_colors(color_a, color_b, t):
    """ Interpolate between two hexadecimal colors with a ratio t from 0 to 1, using BGR format. """
    if not (0 <= t <= 1):
        raise ValueError("The interpolation parameter t must be between 0 and 1.")

    bgr_a = Hex_to_BGR(color_a)
    bgr_b = Hex_to_BGR(color_b)

    # Calculate the interpolated BGR values
    interpolated_bgr = tuple(int(a + (b - a) * t) for a, b in zip(bgr_a, bgr_b))

    # Convert the BGR back to hexadecimal
    return interpolated_bgr

# def a function to transform the center egde to the left top point and right bottom point of the square
def center_to_square(center, edge):
    x, y = center
    x1 = round(x - 0.5*edge)
    y1 = round(y - 0.5*edge)
    x2 = round(x + 0.5*edge)
    y2 = round(y + 0.5*edge)
    return (x1, y1), (x2, y2)

# input: circle_i and circle_j are two tuple, (x, y, r), x, y is the center of the circle, r is the radius of the circle
def circle_intersection(circle_i, circle_j):
    
    if len(circle_i) == 4 or len(circle_j) == 4:
        x0, y0, r0, _ = circle_i
        x1, y1, r1, _ = circle_j
    elif len(circle_i) == 3 or len(circle_j) == 3:
        x0, y0, r0 = circle_i
        x1, y1, r1 = circle_j
    else:
        raise ValueError("circle_i and circle_j must be tuple of length 3 or 4")
    d = math.sqrt((x1 - x0)**2 + (y1 - y0)**2)
    if d > r0 + r1:
        return None
    elif d < abs(r0 - r1):
        return None
    elif d == 0 and r0 == r1:
        return None
    else:
        a = (r0**2 - r1**2 + d**2) / (2 * d)
        h = math.sqrt(r0**2 - a**2)
        x2 = x0 + a * (x1 - x0) / d
        y2 = y0 + a * (y1 - y0) / d
        x3 = x2 + h * (y1 - y0) / d
        y3 = y2 - h * (x1 - x0) / d
        x4 = x2 - h * (y1 - y0) / d
        y4 = y2 + h * (x1 - x0) / d
        return (x3, y3), (x4, y4)

def closest_points_np(all_points, reference_point, X):
    # 将点列表和参考点转换为NumPy数组
    points_array = np.array(all_points)
    ref_point_array = np.array(reference_point)

    # 计算每个点与参考点的欧氏距离
    distances = np.sqrt(np.sum((points_array - ref_point_array) ** 2, axis=1))

    # 获取最近的X个点的索引
    closest_indices = np.argsort(distances)[:X]

    # 返回最近的X个点
    return points_array[closest_indices]

# def a function to check the point that is outside the circle in the circle_list
def point_out_circles(point, circles):
  (Xi, Yi) = point

  for x, y, r, _ in circles:
    distance = math.hypot(Xi - x, Yi - y)
    if distance <= r-1:
      return False
  return True

# def a function to check the point that is inside the circle in the 
def point_in_circles(point, circles):
  (Xi, Yi) = point

  for x, y, r, _ in circles:
    distance = math.hypot(Xi - x, Yi - y)
    if distance <= r-1:
      return True
  return False

def get_unique_coords(A, B):
    result = []
    for a in A:
        if not any(np.array_equal(a, b) for b in B):
            result.append(a)
    return np.array(result)
    # def a state function, Determine the Preselected circles at time tn+1 by virtue of its scan range at time t1, t2.....tn
    # input: circle_list is a list of tuple, (x, y, r), x, y is the center of the circle, r is the radius of the circle
    # output: Next point of tuple, (x, y), x, y is the center of the Next circle

def Next_inter(circle_list, plane_size=2000):
    # check if the circle_list is empty
    inter_closest = (round(plane_size/2), round(plane_size/2))
    X1, Y1 = inter_closest
    if not circle_list:
        return inter_closest
    # check if the circle_list has only one circle
    if len(circle_list) == 1:
        random_0to1 = random.random()
        X2 = round(plane_size/2 + circle_list[0][2] * math.cos(random_0to1 * 2 * math.pi))
        Y2 = round(plane_size/2 + circle_list[0][2] * math.sin(random_0to1 * 2 * math.pi))
        inter_closest = (X2, Y2)
        return inter_closest
    # check if the circle_list has more than two circles
    if len(circle_list) >= 2:
        # get the last circle in the circle_list
        circle_last = circle_list[-1]
        # get the lagrest radius in the circle_list
        R_now_max = max(circle_list, key=lambda x: x[2])[2]
        # get the list cicrle that there center is in the range of the +R_now_max pix -R_now_max pix of the last circle
        Preselected_circle_list = []
        for i in range(len(circle_list[:-1])):
            circle = circle_list[i]
            if circle_last[0] -circle[0] < 2*R_now_max < circle_last[0] + circle[0] and circle_last[1] -circle[1] < 2*R_now_max < circle_last[1] + circle[1]:
                Preselected_circle_list.append(circle)
        # cuclulate all the intersection point of the last circle and the circle in the Preselected_center_list
        intersection_list = []
        for i in range(len(Preselected_circle_list)):
            try:
                inter_1st, inter_2nd = circle_intersection(circle_last, Preselected_circle_list[i])
                # intersection_list.append(inter_1st)
                # intersection_list.append(inter_2nd)
                if point_out_circles(inter_1st, circle_list[:-1]):
                    intersection_list.append(inter_1st)
                if point_out_circles(inter_2nd, circle_list[:-1]):
                    intersection_list.append(inter_2nd)
            except:
                pass
        # keep the point that is outside the circle in the circle_list
        # get the last circle center
        (Xi_1 , Yi_1) = (circle_list[-1][0], circle_list[-1][1])        

        # if inter_list is not empty. 
        if intersection_list:
            # select the intersection point that is closest to the center of the image
            inter_closest = min(intersection_list, key=lambda x: (x[0]-X1)**2 + (x[1]-Y1)**2)
        else:
            #select the intersection point that most outside the last circle if intersection_list is empty
            last_distance =  math.sqrt((Xi_1-X1)**2 + (Yi_1-Y1)**2)
            R_last = circle_list[-1][2]
            inter_closest = (Xi_1 + ((Xi_1-X1)*R_last)/last_distance , Yi_1 + ((Yi_1-Y1)*R_last)/last_distance)
        return inter_closest

def Next_inter_line(circle_list, plane_size=2000, step=20, real_scan_factor = 0.8):
    # check if the circle_list is empty
    down = plane_size/2 * (1 - real_scan_factor)
    up = plane_size/2 * (1 + real_scan_factor)
    left = plane_size/2 * (1 - real_scan_factor)
    right = plane_size/2 * (1 + real_scan_factor)

    inter_closest = (round(plane_size/2), round(plane_size/2))
    X1, Y1 = inter_closest
    if not circle_list:
        return inter_closest
    # check if the circle_list has only one circle
    if len(circle_list) >= 1:
        my_number_X = random.choice((0, 1, 2, 3))
        if my_number_X == 0:
            if circle_list[-1][0] + step >= plane_size/2 * (1 + real_scan_factor): # if the next point is out of the plane, then the next point is the go back step
                X2 = circle_list[-1][0] - step
            else:
                X2 = round(circle_list[-1][0] + step)   
            Y2 = round(circle_list[-1][1])

        elif my_number_X == 1:
            if circle_list[-1][0] - step <= plane_size/2 * (1 - real_scan_factor):
                X2 = circle_list[-1][0] + step
            else:
                X2 = round(circle_list[-1][0] - step)
            Y2 = round(circle_list[-1][1])

        elif my_number_X == 2:
            if circle_list[-1][1] + step >= plane_size/2 * (1 + real_scan_factor):
                Y2 = circle_list[-1][1] - step
            else:
                Y2 = round(circle_list[-1][1] + step)
            X2 = round(circle_list[-1][0])

        elif my_number_X == 3:
            if circle_list[-1][1] - step <= plane_size/2 * (1 - real_scan_factor):
                Y2 = circle_list[-1][1] + step
            else:
                Y2 = round(circle_list[-1][1]- step)
            X2 = round(circle_list[-1][0])            
        inter_closest = (X2, Y2)
        return inter_closest


def initial_net_point(plane_size=2000, step=30, real_scan_factor = 0.8):
    inter_closest = (round(plane_size/2), round(plane_size/2))
    X1, Y1 = inter_closest
    net_x = np.arange(plane_size/2, X1 - plane_size/2 * real_scan_factor, -step)[::-1]
    net_x = np.append(net_x, np.arange(plane_size/2, X1 + plane_size/2 * real_scan_factor, step))
    net_y = np.arange(plane_size/2, Y1 - plane_size/2 * real_scan_factor, -step)[::-1]
    net_y = np.append(net_y, np.arange(plane_size/2, Y1 + plane_size/2 * real_scan_factor, step))
    # use the net_x as the x and net_y as y to create a net of points
    net = np.array(np.meshgrid(net_x, net_y)).T.reshape(-1, 2)

    return np.unique(net, axis=0)

def Next_inter_net(circle_list, net, plane_size=2000):

    # initialize the inter_closest
    if len(circle_list) == 0:
        return (round(plane_size/2), round(plane_size/2)), net

    # if circle_list is not empty
    (X1, Y1) = (round(plane_size/2), round(plane_size/2))
    point_in_circle = []
    # traversal every point in the net_outofcircle to find how many point that is in the circle_list
    for point in net:
        for circle in circle_list:
            if point_in_circles(point, [circle]):
                new_coordinate = np.array([point[0], point[1]])
                point_in_circle.append(new_coordinate)

    point_in_circle = np.array(point_in_circle)
            
    # print(len(point_in_circle))
    # remove the point_in_circle in net
    point_in_circle = np.unique(point_in_circle, axis=0)
    # print(point_in_circle)
    net_str = np.array([f"{x},{y}" for x, y in net])
    point_in_circle_str = np.array([f"{x},{y}" for x, y in point_in_circle])
    diff_str = np.setdiff1d(net_str, point_in_circle_str)
    net = np.array([list(map(float, item.split(','))) for item in diff_str])
    print('net_long',len(net))
    # net = np.delete(net, point_in_circle, axis=0)
    
    # print(net)
    # find the 5 most closest point near the last circle
    if len(net) <= 1:
        return (round(plane_size/2), round(plane_size/2)), []
    inter_closest_list = closest_points_np(net, circle_list[-1][0:2], 5)
    # print('next point list: ', inter_closest_list)
    # find the closest point in the (X1, Y1)
    inter_closest = min(inter_closest_list, key=lambda x: (x[0]-X1)**2 + (x[1]-Y1)**2)
    # print('next point: ', inter_closest)
    return inter_closest, net

    






#def a function to increase the radius of the circle. input: 1.if the circle radius should be increased. 2.the last radius 3. the initial radius. 4. the max radius. 5. the step of the radius
def increase_radius(scan_qulity, R_last, R_init, R_max, R_step):
    if not scan_qulity:
        if R_last < R_max:
            R = R_last + R_step
        else:
            R = R_max
    else:
        R = R_init
    return R

# def a function to convert the pix point to nanomite coordinate: example: (0, 0)-> (-1e-6, 1e-6), (2000, 2000) -> (1e-6, -1e-6), (1000, 1000) -> (0.0, 0.0)
def pix_to_nanocoordinate(pix_point, plane_size=2000):
    (X, Y) = pix_point
    X =     round(X - plane_size/2)* 1e-9 * (2000/plane_size)
    Y = (-1)*round(Y - plane_size/2)* 1e-9 * (2000/plane_size)              # pix_point is matrix, but the nanomite coordinate is nomal coordinate, so the Y should be -Y
    return (X, Y)

# convert the pix in the image to nanomite coordinate in surface
def pix_to_nanomite(pix, plane_size=2000):
    nanomite =  pix * 1e-9 * (2000/plane_size)
    return nanomite

def normalize_2darray(arr):

    # Find min and max values across the entire 2D array
    min_val = np.min(arr)  
    max_val = np.max(arr)

    # Rescale all values in the 2D array 
    normalized = 255 * (arr - min_val) / (max_val - min_val)

    return normalized

def normalize_to_image(arr):

    # Normalize the 2D array 
    min_val = np.min(arr)
    max_val = np.max(arr)
    normalized = 255 * (arr - min_val) / (max_val - min_val)

    # Convert the normalized array to uint8 type
    normalized = normalized.astype(np.uint8)

    # Convert the normalized array to an image
    img = cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)

    return normalized

def linear_whole(matrix):
    rows, cols = matrix.shape
    Y, X = np.indices((rows, cols))  # Y is row indices, X is column indices
    X = X.ravel()  # Flatten X to a 1D array
    Y = Y.ravel()  # Flatten Y to a 1D array
    data = matrix.ravel()  # Flatten the matrix data to a 1D array

    # Prepare matrix A for linear fitting
    A = np.column_stack([X, Y, np.ones(rows * cols)])
    c, _, _, _ = lstsq(A, data)  # Perform linear regression

    # Calculate the fitted plane over the entire matrix
    fitted_plane = c[0] * X + c[1] * Y + c[2]
    fitted_plane = fitted_plane.reshape(rows, cols)  # Reshape to the original matrix shape

    # Subtract the fitted plane
    processed_matrix = matrix - fitted_plane

    return processed_matrix

def linear_normalize_whole(matrix):
    rows, cols = matrix.shape
    Y, X = np.indices((rows, cols))  # Y is row indices, X is column indices
    X = X.ravel()  # Flatten X to a 1D array
    Y = Y.ravel()  # Flatten Y to a 1D array
    data = matrix.ravel()  # Flatten the matrix data to a 1D array

    # Prepare matrix A for linear fitting
    A = np.column_stack([X, Y, np.ones(rows * cols)])
    c, _, _, _ = lstsq(A, data)  # Perform linear regression

    # Calculate the fitted plane over the entire matrix
    fitted_plane = c[0] * X + c[1] * Y + c[2]
    fitted_plane = fitted_plane.reshape(rows, cols)  # Reshape to the original matrix shape

    # Subtract the fitted plane
    processed_matrix = matrix - fitted_plane

    # Normalize to 0-255
    min_val = processed_matrix.min()
    max_val = processed_matrix.max()
    if min_val == max_val:  # if the matrix is a constant matrix
        return processed_matrix
    
    normalized_matrix = 255 * (processed_matrix - min_val) / (max_val - min_val)
    normalized_matrix = np.array(normalized_matrix, dtype=np.uint8)

    return normalized_matrix


def linear_background_and_normalize(matrix):
    # matrix = np.mat(matrix)
    rows, cols = matrix.shape
    X = np.arange(cols)

    # 初始化一个与原矩阵同样大小的矩阵来存储处理后的数据
    processed_matrix = np.zeros_like(matrix, dtype=float)

    for i in range(rows):
        # 对每一行进行线性拟合
        A = np.column_stack([X, np.ones(cols)])
        b = matrix[i, :]
        c, _, _, _ = lstsq(A, b)

        # 计算拟合出的线性趋势
        line_fit = c[0] * X + c[1]

        # 扣除线性背底
        processed_matrix[i, :] = matrix[i, :] - line_fit

    # 归一化到0-255
    min_val = processed_matrix.min()
    max_val = processed_matrix.max()
    if min_val == max_val: # if the matrix is a constant matrix
        return processed_matrix
    
    normalized_matrix = 255 * (processed_matrix - min_val) / (max_val - min_val)

    normalized_matrix = np.array(normalized_matrix, dtype=np.uint8)
    
    return normalized_matrix

def get_latest_checkpoint(parent_folder, checkpoint_name="checkpoint.json"):
    """Returns the path to the most recently created subfolder in the specified parent folder."""
    # Ensure the path is absolute
    parent_folder = os.path.abspath(parent_folder)
    
    # Get all entries in the directory that are directories themselves
    subfolders = [os.path.join(parent_folder, d) for d in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, d))]
    
    # Find the subfolder with the latest creation time
    if not subfolders:
        return None  # Return None if there are no subfolders
    
    latest_subfolder = max(subfolders, key=os.path.getctime)
    
    return os.path.join(latest_subfolder,checkpoint_name)

def get_latest_filelist(parent_folder):
    """Returns the path to the most recently created subfolder in the specified parent folder."""
    # Ensure the path is absolute
    parent_folder = os.path.abspath(parent_folder)
    
    # Get all entries in the directory that are directories themselves
    subfolders = [os.path.join(parent_folder, d) for d in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, d))]
    
    # Find the subfolder with the latest creation time
    if not subfolders:
        return None  # Return None if there are no subfolders
    #sort the subfolders by the time
    subfolders.sort(key=lambda x: os.path.getctime(x))
    
    return subfolders

def time_trajectory_list(parent_path, file_extension='.json'):
    """ Return a list of '.json' file paths sorted by creation time. """
    npy_files = []
    # Walk through all directories and files in the parent path
    for root, dirs, files in os.walk(parent_path):
        for file in files:
            if file.endswith(file_extension):
                file_path = os.path.join(root, file)
                creation_time = os.path.getctime(file_path)
                npy_files.append((file_path, creation_time))

    # Sort files based on creation time
    npy_files_sorted = sorted(npy_files, key=lambda x: x[1])
    
    # Return only the file paths, sorted by creation time
    return [file[0] for file in npy_files_sorted]

def subtract_plane(arr):
    # Fit plane to points
    A = np.c_[arr[:,0], arr[:,1], np.ones(arr.shape[0])]
    C, _, _, _ = lstsq(A, arr[:,2])   
    
    # Evaluate plane on grid
    ny, nx = arr.shape[:2]
    x = np.arange(nx)
    y = np.arange(ny)
    xv, yv = np.meshgrid(x, y)
    z = C[0]*xv + C[1]*yv + C[2]
    
    # Subtract plane from original array
    return arr - z

def tip_in_boundary(inter_closest, plane_size, real_scan_factor):
    if inter_closest[0] <= plane_size/2 * (1 + real_scan_factor) and inter_closest[0] >= plane_size/2 * (1-real_scan_factor) and inter_closest[1] <= plane_size/2 * (1 + real_scan_factor) and inter_closest[1] >= plane_size/2 * (1-real_scan_factor):
        return True
    else:
        return False

def images_equalization(image, alpha=0.5):
    # Normalize the image
    norm_image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    # Fully equalize the image
    equalized_image = cv2.equalizeHist(norm_image)
    
    # Blend the original normalized image and the equalized image based on alpha
    adjusted_image = cv2.addWeighted(norm_image, 1-alpha, equalized_image, alpha, 0)
    
    return adjusted_image

if __name__ == '__main__':

    # set the initial scan square edge length                           30pix  ==>>  30nm
    scan_square_edge = 30

    # plane size repersent the area of the Scanable surface  2um*2um    2000pix  ==>>  2um
    plane_size = 2000
    
    # set the initial radius of the first circle                       
    # R_init = round(scan_square_edge*(math.sqrt(2)))
    R_init = scan_square_edge  + 200
    R_max = R_init*3
    R_step = int(0.5*R_init)

    R = R_init
    



    # circle color hexadecimal
    circle_color_hex = "#F2F2F2"
    circle_egde_color_hex = "#515151"

    square_color_hex = "#BABABA"
    square_bad_color_hex = "#FE5E5E"

    circle_bed_color_hex = "#F8CBAD"
    circle_bed_egde_color_hex = "#EF8D4B"

    line_color_hex = "#C7C7C7" 

    cv2.namedWindow('img', cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow('img', 800, 800)
    
    # change the circle color from hexadecimal to RGB

    
    circle_color = Hex_to_BGR(circle_color_hex)
    circle_egde_color = Hex_to_BGR(circle_egde_color_hex)
    circle_bed_color = Hex_to_BGR(circle_bed_color_hex)
    circle_bed_egde_color = Hex_to_BGR(circle_bed_egde_color_hex)
    square_good_color = Hex_to_BGR(square_color_hex)
    square_bad_color = Hex_to_BGR(square_bad_color_hex)
    line_color = Hex_to_BGR(line_color_hex)

    #plane center
    (X1, Y1) = (int(plane_size/2), int(plane_size/2))

    # creat a 400*400 pix white image by numpy array
    img = np.ones((plane_size, plane_size, 3), np.uint8) * 255
    
    
    # draw a circle on the center of the image with radius R_init
    cv2.circle(img, (X1, Y1), R-1, circle_color, -1)
    cv2.circle(img, (X1, Y1), R-1, circle_egde_color, 1)

    # draw a square on the (X1, Y1) and edge of the square with scan_square_edge
    left_top, right_bottom = center_to_square((X1, Y1), scan_square_edge)
    cv2.rectangle(img, left_top, right_bottom, square_good_color, -1)


    # save all the circle center and radius
    circle_list = []
    circle_list.append((X1, Y1, R, 1))

    cv2.imshow("img", img)
    cv2.waitKey(100)

    # draw a circle on the edge of the first circle with radius 20, random angle
    # get a random number between 0 and 1
    random_0to1 = random.random()
    X2 = X1 + round(R * math.cos(random_0to1 * 2 * math.pi))
    Y2 = Y1 + round(R * math.sin(random_0to1 * 2 * math.pi))

    cv2.circle(img, (X2, Y2), R, circle_color, -1)
    cv2.circle(img, (X2, Y2), R, circle_egde_color, 1)

    # draw a square on the (X1, Y1) and edge of the square with scan_square_edge
    left_top, right_bottom = center_to_square((X2, Y2), scan_square_edge)
    cv2.rectangle(img, left_top, right_bottom, square_good_color, -1)
    # draw a line between the center of the first circle and the center of the second circle
    cv2.line(img, (X1, Y1), (X2, Y2), line_color, 2)
    circle_list.append((X2, Y2, R, 1))

    cv2.imshow("img", img)
    cv2.waitKey(100)
    
    # initialize the buffer augument
    bad_times = 0
    good_times = 0
    inter_closest = (X1, Y1)

    net = initial_net_point(plane_size=2000, step=30, real_scan_factor = 0.8)
    
    # for i in net:
    #     # draw the net point on the image
    #     cv2.circle(img, (round(i[0]), round(i[1])), 2, (0, 0, 255), -1)


    # while inter_closest[0] + 2*R <= plane_size and inter_closest[0] - 2*R >= 0 and inter_closest[1] + 2*R <= plane_size and inter_closest[1] - 2*R >= 0:
    while len(net) != 0:
        # inter_closest = Next_inter(circle_list)
        inter_closest, net = Next_inter_net(circle_list, net)
        for i in net:
            # draw the net point on the image
            cv2.circle(img, (round(i[0]), round(i[1])), 2, (0, 0, 255), -1)

        # int the inter_closest point
        (Xi, Yi) = (round(inter_closest[0]), round(inter_closest[1]))
        
        # draw the point (Xi, Yi) on the image
        cv2.circle(img, (Xi, Yi), 3, (255, 0, 255), -1)
        cv2.imshow("img", img)
        cv2.waitKey(100)

        # judge whether radius should be increased or not, and draw a new circle       
        #########################################################################################
        if random.random() > 0.99:       # !!!Changing this judgment statement can change the conditions for expanding the scanning range!!!
        #########################################################################################
            increase = True
            bad_times += 1
            square_color = square_bad_color
        else:
            increase = False
            good_times += 1
            square_color = square_good_color
        
        R = increase_radius(increase, circle_list[-1][2], R_init, R_max, R_step)
            
            # cv2.circle(img, (Xi, Yi), R_init, circle_color, -1)
            # cv2.circle(img, (Xi, Yi), R_init, circle_egde_color, 1)    
            # draw a square on the (Xi, Yi) and edge of the square with scan_square_edge
        
        left_top, right_bottom = center_to_square((Xi, Yi), scan_square_edge) 
        # draw circle in 0.2 transparency
        cv2.circle(img, (Xi, Yi), R, circle_bed_color, -1)
        cv2.circle(img, (Xi, Yi), R, circle_bed_egde_color, 1)

        cv2.rectangle(img, left_top, right_bottom, square_color, -1)
        cv2.line(img, (circle_list[-1][0], circle_list[-1][1]), (Xi, Yi), line_color, 2)
        # add inter_closest point to the circle_list
        circle_list.append((Xi, Yi, R, 1))
        # print((Xi, Yi, R))

        #show the image
        cv2.imshow("img", img)
        cv2.waitKey(100)

    

    # save the circle_list in a txt file
    time_now =  time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    circle_list_name = "./log/circle_list" + "_good_" + str(good_times) + "_bad_" + str(bad_times) + "_total_" + str(good_times + bad_times)+"_frame"+str(scan_square_edge)+ "time" + time_now + ".txt"
    with open(circle_list_name, 'w') as f:
        for item in circle_list:
            f.write("%s\n" % str(item))
    # save the image, including the time, good_times, bad_times, total_times
    log_name = "../log/Scan log"  + "_good_" + str(good_times) + "_bad_" + str(bad_times) + "_total_" + str(good_times + bad_times)+"_frame"+str(scan_square_edge)+ "time"+ time_now + ".jpg"
    cv2.imwrite(log_name, img)
    
    #print the result
    print("The scanning range is out of the boundary! \n Scaning finished!")
    print("good_times: ", good_times)
    print("bad_times: ", bad_times)
    print("total_times: ", good_times + bad_times)
    print("time: ", time_now)

    # show the last image
    cv2.imshow("img", img)
    cv2.waitKey(0)
