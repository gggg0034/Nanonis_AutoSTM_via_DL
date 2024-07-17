import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import lstsq
from scipy.optimize import curve_fit
from scipy.signal import correlate

prefixes = {'a':-18, 'f':-15, 'p':-13, 'n': -9, 'μ': -6, 'm': -3, '': 0, 'k': 3, 'M': 6, 'G': 9}

def sci_to_unit(value):
    """Convert a value to a string with units"""
    exp = int(np.floor(np.log10(abs(value))))
    prefix = next((k for k,v in prefixes.items() if v <= exp), '')

    scaled = value * 10 ** (-exp)
    return f"{scaled:.7g}{prefix}"

def unit_to_sci(string):
    """Convert a string with units to a float value"""
    for prefix, exp in prefixes.items():
        if string.endswith(prefix):
            number = float(string[:-len(prefix)])
            return number * 10 ** exp
    
    return float(string)


# #a function to subtract a fitted plane from a 2D array
# def subtract_plane(arr):
#     # Fit plane to points
#     A = np.c_[arr[:,0], arr[:,1], np.ones(arr.shape[0])]
#     C, _, _, _ = lstsq(A, arr[:,2])   
    
#     # Evaluate plane on grid
#     ny, nx = arr.shape[:2]
#     x = np.arange(nx)
#     y = np.arange(ny)
#     xv, yv = np.meshgrid(x, y)
#     z = C[0]*xv + C[1]*yv + C[2]
    
#     # Subtract plane from original array
#     return arr - z




#def a function to fit the line scan data array as a line and return a array that subtract the line
def fit_line(data):
    # Fit a linear model y = m*x + b 
    # m = slope, b = intercept
    m, b = np.polyfit(np.arange(len(data)), data, 1)
    
    # Generate the fitted line
    fit = m*np.arange(len(data)) + b
    
    # Compute residuals
    residuals = data - fit
    
    return residuals


# Calculate similarity between two vectors
def vector_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    cos_sim = dot_product / (norm_vec1 * norm_vec2)
    
    return cos_sim


# Calculate the gap in a 1D array
def linescan_max_min_check(arr):
    arr = np.array(arr)
    max_val = np.max(arr)
    min_val = np.min(arr)
    return max_val - min_val

# 

def linescan_similarity_check(vec1, vec2, threshold=0.9):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    similarity = vector_similarity(vec1, vec2)
    # print('linescan_similarity' + str(similarity))
    if similarity < threshold:
        # print('linescan_similarity' + str(similarity)+'bad')
        return 0    # 0 means the similarity is too low
    else:
        return 1    # 1 means the similarity is acceptable

# Calculate the phase difference between two signals   
def find_phase_diff(sig1, sig2):
    corr = correlate(sig1, sig2)
    max_index = np.argmax(corr)
    # print('phase_diff' + str(max_index - len(sig1) + 1))
    return max_index - len(sig1) + 1 

# def a function to calculate the angle between two points vector and the x axis, 90 degree is -Y axis 180 degree is -X axis
def angle_calculate(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    
    dx = x2 - x1
    dy = y2 - y1
    
    angle = -1*(np.math.atan2(dy, dx) * 180 / np.pi)
    
    if angle > 90:
        angle -= 180
    elif angle < -90:
        angle += 180
        
    return angle

# test angle_calculate


if __name__ == '__main__':
# test angle_calculate

    p1 = (2, 2)
    p2 = (1, 1)
    print(angle_calculate(p1, p2))
    pass