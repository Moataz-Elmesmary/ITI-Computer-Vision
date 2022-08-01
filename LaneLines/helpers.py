

import numpy as np
import cv2
import math


def helpers_edges(gray):
    kernel_size = 5 #kernel size for Gaussian smoothing / blurring
    blur_gray= cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)#Gaussian smoothing / blurring

    ##
    # finding edges - Canny Edge detection (strong gradient between adjacent pixels)
    threshold1 = 50 # 50
    threshold2 = 150 #  180
    edges = cv2.Canny(blur_gray, threshold1, threshold2)
    return(edges)


def helpers_masked_edges(edges):
    mask = np.zeros_like(edges)
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(edges.shape) > 2:
        channel_count = edges.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    imshape = edges.shape        
    vertices = np.array([[((1/6*imshape[1]),imshape[0]),((5/12*imshape[1]), (3/5*imshape[0])), ((7/12*imshape[1]), (3/5*imshape[0])), ((9/10*imshape[1]),imshape[0])]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_edges = cv2.bitwise_and(edges, mask)
    return(masked_edges)

def helpers_formulate_lanes(lines, img, color=[255, 0, 0], thickness=2):    
    #algorithm logic:
    #aim is to find x_min, y_min, x_max, y_max , slope and intercept for both lanes lines.
    #for each line returned from the hough lines function:
    #   calculate slope
    #   calculate intercept
    #   store positive and negative slope and intercept values separately in arrays.
    #   y_min is the minimum of all the y coordinates.
    #   y_max is the bottom of the image from where the lane lines start.
    #   slope and intercept values for both lines are just the averages of all values stored previously.
    #  x_min and x_max can now be calculated by fitting all the lines in the equation x = (y - intercept)/slope.
    
    #LINE DISPLAY PARAMETERS
    color = [243, 105, 14]
    thickness = 12
    
    #LINE PARAMETERS
    SLOPE_THRESHOLD = 0.3
    Y_MIN_ADJUST = 15
    
    positive_slopes = []
    negative_slopes = []
    
    positive_intercepts = []
    negative_intercepts = []
    
    #named as y_max despte being at the bottom corner of the image due to y axis in reverse direction
    y_max = img.shape[0]
    y_min = img.shape[0]
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            
            #calculate slope for the line
            slope = (y2-y1)/(x2-x1)
            intercept = y2 - (slope*x2)
            
            #for negative slope
            if slope < 0.0 and slope > -math.inf and abs(slope) > SLOPE_THRESHOLD:
                #print('negative slope')
                negative_slopes.append(slope)
                negative_intercepts.append(intercept)
                
            #for positive slope
            elif slope > 0.0 and slope < math.inf and abs(slope) > SLOPE_THRESHOLD:
                #print('positive slope')
                positive_slopes.append(slope)
                positive_intercepts.append(intercept)
            
            y_min = min(y_min, y1, y2)
            #cv2.line(img, (x1, y1), (x2, y2), color, thickness)
    
    y_min+=Y_MIN_ADJUST
    
    #get averages for positive and negative slopes
    positive_slope_mean = np.mean(positive_slopes)
    negative_slope_mean = np.mean(negative_slopes)

    #get averages for potitive and negative intercepts
    positive_intercept_mean = np.mean(positive_intercepts)
    negative_intercept_mean = np.mean(negative_intercepts)
    lst = [
        [[0, 0, 0, 0]],
        [[0, 0, 0, 0]]
    ]

    #calculation of coordinates for lane for positive slopes
    if len(positive_slopes) > 0:
        x_max = int((y_max - positive_intercept_mean)/positive_slope_mean)
        x_min = int((y_min - positive_intercept_mean)/positive_slope_mean)
        #cv2.line(img, (x_min, y_min), (x_max, y_max), color, thickness)
        lst[0][0] = [x_min, y_min, x_max, y_max]
    
    #calculation of coordinates for lane for negative slopes
    if len(negative_slopes) > 0:
        x_max = int((y_max - negative_intercept_mean)/negative_slope_mean)
        x_min = int((y_min - negative_intercept_mean)/negative_slope_mean)
        #cv2.line(img, (x_min, y_min), (x_max, y_max), color, thickness)
        lst[1][0] = [x_min, y_min, x_max, y_max]
        #lst.append([x_min, y_min, y_min, y_max])
    return(np.array(lst))

def helpers_draw_lines(lines, masked_edges):
    color = [243, 105, 14]
    thickness = 12
    lines_image = np.zeros((masked_edges.shape[0], masked_edges.shape[1], 3), dtype=np.uint8)
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(lines_image, (x1, y1), (x2, y2), color, thickness)
    return(lines_image)

#HOUGH LINES PARAMETERS
RHO = 3.5 
THETA = np.pi/180
MIN_VOTES = 30    
MIN_LINE_LEN = 5 
MAX_LINE_GAP= 25  

def helpers_hough_lines(img):
    """
    `img` should be the output of a Canny transform.
        
    Returns the lines from hough transform
    """
    lines = cv2.HoughLinesP(img, RHO, THETA, MIN_VOTES, np.array([]), minLineLength=MIN_LINE_LEN, maxLineGap=MAX_LINE_GAP)
    return lines

