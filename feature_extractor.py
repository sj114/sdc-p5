import matplotlib.image as mpimg
import numpy as np
import cv2
import random
from skimage.feature import hog
import matplotlib.pyplot as plt


# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features

# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

# Define a function to compute color histogram features 
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        plot_images = []
        cls = []
        plot_images.append(image)
        cls.append('RGB')
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)      

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                plot_images.append(feature_image)
                cls.append('HLS')
                for channel in range(feature_image.shape[2]):
                    #plot_images.append(feature_image[:,:,channel])
                    #cls.append('HLS channel{}'.format(channel))
                    #hog_g, hog_i = get_hog_features(feature_image[:,:,channel], 
                    #                    orient, pix_per_cell, cell_per_block, 
                    #                    vis=True, feature_vec=True)
                    #plot_images.append(hog_i)
                    #cls.append('HOG channel{}'.format(channel))

                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
        
    # Return list of feature vectors
    return features
    
# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap

def avg_heat(current_heatmap, avg_heatmap, N):
    avg_heatmap = (avg_heatmap*N + current_heatmap)/(N+1)

    # Return updated heatmap
    return avg_heatmap

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def get_window_distance(window1, window2):
    # Compute distance between centroids of the two windows
    centroid1 = np.mean(window1, axis=0)
    centroid2 = np.mean(window2, axis=0)
    dist = np.linalg.norm(centroid1 - centroid2)
    #print(window1, window2, centroid1, centroid2, dist)
    return dist

def draw_labeled_bboxes(img, labels, prev_bboxes):
    new_bboxes = []

    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        mean = np.mean(nonzero, axis=1)
        x1 = np.min(nonzerox)
        y1 = np.min(nonzeroy)
        x2 = np.max(nonzerox)
        y2 = np.max(nonzeroy)

        # Define a bounding box based on min/max x and y
        bbox = ((x1, y1), (x2, y2))

        # Check if this existed in the previous frame
        if len(prev_bboxes):
            for i, prev_bbox in enumerate(prev_bboxes):
                if get_window_distance(bbox, prev_bbox[1]) < 96:
                    # Likely same car as in previous frame
                    prev_x1 = prev_bbox[1][0][0]
                    prev_x2 = prev_bbox[1][1][0]
                    prev_y1 = prev_bbox[1][0][1]
                    prev_y2 = prev_bbox[1][1][1]
                    x1 = np.int((x1 + prev_x1)/2) 
                    x2 = np.int((x2 + prev_x2)/2) 
                    y1 = np.int((y1 + prev_y1)/2) 
                    y2 = np.int((y2 + prev_y2)/2) 
                    bbox = ((x1, y1), (x2, y2))
                    del prev_bboxes[i]
                    break

        # Filter out tiny boxes (unlikely to be cars)
        if abs(x2 - x1) >= 40 and abs(y2 - y1) >= 40:
            # Draw the box on the image
            if mean[0] >= 200:
                cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 2)
                cv2.drawMarker(img, (np.int(mean[1]),np.int(mean[0])), 
                                 (0,255,255), markerType=cv2.MARKER_STAR, 
                                   markerSize=10, thickness=2, line_type=cv2.LINE_AA)
            new_bboxes.append((mean, bbox, 0))

    # Return the image
    return img, new_bboxes

''' Function to plot images for data visualization '''
def plot_images_fxn(images, labels, title, nx, ny):
    
    # Create figure with nx x ny sub-plots.
    fig, axes = plt.subplots(nx, ny)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    
    fig.suptitle(title)
    for i, ax in enumerate(axes.flat):
        if i >= len(images):
            break
            
        # Plot image.
        ax.imshow(images[i])

        xlabel = "True: {0}".format(labels[i])

        # Show the angles as the label on the x-axis.
        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    fig.savefig('{}.png'.format(title), bbox_inches='tight')
    plt.close()
