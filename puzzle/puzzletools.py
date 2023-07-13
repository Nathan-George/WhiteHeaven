import numpy as np
import cv2

# we need to access the points of the contour in a circular manner
def get_circular(contour, index):
    """Gets the point at the index of the contour, 
    but wraps around if the index is out of bounds

    Args:
        contour (np.array(len, 2)): array of points
        index (int): index of the point to get

    Returns:
        np.array(2): the point
    """
    # make sure index is positive
    index = index % len(contour) + len(contour)
    return contour[index % len(contour)]

def get_slice_circular(contour, start, end):
    """Gets the slice of the contour from start to end, but
    wraps around if the index is out of bounds

    Args:
        contour (np.array(len, 2)): array of points
        start (int): index of the start of the slice, may be negative
        end (int): index of the end of the slice, may be negative

    Returns:
        np.array(end-start, 2): array of points that make up the slice
    """
    
    # make sure start and end are positive
    start = start % len(contour)
    end = end % len(contour)
    if start < 0:
        start += len(contour)
    if end < 0:
        end += len(contour)
    
    # make sure end is larger than start
    if end < start:
        end += len(contour)
    
    if end > len(contour):
        return np.concatenate((contour[start:], contour[:end - len(contour)]))
    
    return contour[start:end]

def get_contour(puzzle_piece_img):
    """gets the contour of the puzzle piece

    Args:
        puzzle_piece_img (Mat): image of the puzzle piece

    Returns:
        np.array(len, 2): list of points that make up the contour
    """
    img_gray = cv2.cvtColor(puzzle_piece_img, cv2.COLOR_BGR2GRAY)
    # Threshold the image
    _, img_thresh = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    return contours[0][:, 0, :]

# corner indicators
# angle indicator
def get_angle_indicators(contour, window_size=5):
    """Gets the angle indicators for the puzzle piece. The angle indicators are the
    angles between vectors along the contour of the puzzle piece.

    Returns:
        np.array(len(contour)): array of angles
    """
    
    angle_indicators = np.zeros(len(contour))
    for i in range(len(contour)):
        point = get_circular(contour, i)
        start  = get_circular(contour, i - window_size)
        end = get_circular(contour, i + window_size)
        
        vec_start = point - start
        vec_end = end - point
        angle_indicators[i] = np.arctan2(np.cross(vec_end, vec_start), np.dot(vec_end, vec_start))
    
    return angle_indicators

# center indicator
def get_center_indicators(contour, window_size=5):
    """Gets the center indicators for the puzzle piece. The center indicators indicate how
    well a corner contains the center of the puzzle piece. The center is defined as the point (200, 200)

    Returns:
        np.array(len(contour)): array of distances
    """
    
    center = np.array([200, 200])
    center_indicators = np.zeros(len(contour))
    for i in range(len(contour)):
        point = get_circular(contour, i)
        start  = get_circular(contour, i - window_size)
        end = get_circular(contour, i + window_size)
        
        vec_start = point - start
        vec_end = end - point
        vec_center = center - point
        
        # rotate vec_start by 135 degrees clockwise and vec_end by 45 degrees clockwise
        theta = -np.pi * 3 / 4
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        vec_start = np.dot(rot, vec_start)
        theta = -np.pi / 4
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        vec_end = np.dot(rot, vec_end)
        
        assert np.linalg.norm(vec_start) > 0
        assert np.linalg.norm(vec_end) > 0
        assert np.linalg.norm(vec_center) > 0
        
        center_indicators[i] = (np.dot(vec_start, vec_center) / np.linalg.norm(vec_start) + np.dot(vec_end, vec_center) / np.linalg.norm(vec_end)) / np.linalg.norm(vec_center)

    return center_indicators

# distance indicator
def get_distance_indicators(contour, window_size=5):
    """Gets the distance indicators for a contour. Tge distance indicators indicate how
    much further away the corner is from the center of the puzzle piece compared to its
    neighbors. The center is assumed to be the point (200, 200)

    Args:
        contour (_type_): _description_
        window_size (int, optional): _description_. Defaults to 5.
    """
    
    center = np.array([200, 200])
    distance_indicators = np.zeros(len(contour))
    for i in range(len(contour)):
        point = get_circular(contour, i)
        start  = get_circular(contour, i - window_size)
        end = get_circular(contour, i + window_size)
        
        distance = np.linalg.norm(point - center)
        
        distance_indicators[i] = distance - (np.linalg.norm(start - center) + np.linalg.norm(end - center))/2
    
    return distance_indicators

def get_corner_indicators(contour, w1=1.0, w2=1.0, w3=0.125, window_size=5):
    """Gets the corner indicators for a contour. The corner indicators are a weighted sum
    of the angle, center, and distance indicators

    Args:
        contour : _description_
        w1 (float): weight for angle indicators
        w2 (float): weight for center indicators
        w3 (float): weight for distance indicators
        window_size (int, optional): _description_. Defaults to 5.
    """
    
    angle_indicators = get_angle_indicators(contour, window_size)
    center_indicators = get_center_indicators(contour, window_size)
    distance_indicators = get_distance_indicators(contour, window_size)
    
    return w1 * angle_indicators + w2 * center_indicators + w3 * distance_indicators

# uses the corner indicators to find the corners
def find_corners(contour, corner_size=40):
    """Finds the indices of the corners of the puzzle piece"""
    
    indicators = get_corner_indicators(contour, 1.0, 1.0, 0.125)
    
    p_corners = [0]
    for i in range(len(indicators)):
        if i - p_corners[-1] < corner_size:
            
            if i - p_corners[0] > len(indicators) - corner_size:
                # point is too close to first and last corner
                if indicators[i] > indicators[p_corners[-1]] and indicators[i] > indicators[p_corners[0]]:
                    # remove first corner
                    p_corners = p_corners[1:]
                    
                    p_corners[-1] = i
            
            # point is too close to the previous corner
            elif indicators[i] > indicators[p_corners[-1]]:
                p_corners[-1] = i
        
        elif i - p_corners[0] > len(indicators) - corner_size:
            
            # point is too close to the first corner
            if indicators[i] > indicators[p_corners[0]]:
                # remove first corner
                p_corners = p_corners[1:]
                
                p_corners.append(i)
        else:
            p_corners.append(i)
    
    # find the 4 best corners
    p_corners = np.array(p_corners)
    return np.sort(p_corners[np.argsort(indicators[p_corners])[:-5:-1]])