import argparse
import sys
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
import pickle
import collections
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from feature_extractor import *
from scipy.ndimage.measurements import label
from scipy.ndimage import generate_binary_structure
from sklearn.model_selection import train_test_split
from skimage.measure import block_reduce

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip

# Global parameters 
g_debug_plot = False 
color_space = 'HLS' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 8  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off


# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, window_hog, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):    
    #1) Define an empty list to receive features
    img_features = []
    #2) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(img, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #3) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(img, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #4) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(img.shape[2]):
                hog_features.extend(window_hog[channel])
        else:
            hog_features = window_hog
        #8) Append features to list
        img_features.append(hog_features)

    #5) Return concatenated array of features
    return np.concatenate(img_features)

# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, window_hog_list, clf, scaler, bboxes, 
                    color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window, window_hog in zip(windows, window_hog_list):
        # Check if this window is in the same region as prev. frames' bboxes
        is_in_vicinity = False
        for bbox in bboxes:
            if get_window_distance(bbox[1], window) < 128:
                is_in_vicinity = True
                break
                
        if is_in_vicinity == False and len(bboxes):
            continue

        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, window_hog, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
        if clf.decision_function(test_features) > 0.6:
            #print(clf.decision_function(test_features), abs(window[0][0]-window[1][0]))
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows
    
# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, hog_features, x_start_stop=[None, None], y_start_stop=[None, None], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_windows = np.int(xspan/nx_pix_per_step) - (np.int(xy_window[0]/nx_pix_per_step) - 1)
    ny_windows = np.int(yspan/ny_pix_per_step) - (np.int(xy_window[1]/ny_pix_per_step) - 1)
    # Initialize a list to append window positions to
    window_list = []
    window_hog_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    num_ch = img.shape[2]
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))

            # Grab HOG features specific to this window from the overall
            # feature matrix
            magic_number = np.int(xy_window[0]/8-1)
            block_size = np.int(xy_window[0]/64)
            window_hog = np.zeros((num_ch,magic_number,magic_number,cell_per_block,cell_per_block,orient))
            window_hog_scaled = np.zeros((num_ch,7,7,cell_per_block,cell_per_block,orient))
            for ch in range(0, num_ch):
                a = (hog_features[ch][2*block_size*ys:2*block_size*ys+magic_number, \
                                        2*block_size*xs:2*block_size*xs+magic_number, :, :, :])
                window_hog_scaled[ch] = block_reduce(a, block_size=(block_size,block_size,1,1,1), \
                                                                       func=np.mean)[0:7,0:7,:,:,:]

            window_hog_list.append(((np.ravel(window_hog_scaled[0])), \
                                     (np.ravel(window_hog_scaled[1])), \
                                      (np.ravel(window_hog_scaled[2]))))

    # Return the list of windows
    return window_list, window_hog_list

def get_train_data():
    # Read in cars and notcars
    cars = []
    images = glob.glob('./vehicles/*/*.png')
    for image in images:
        cars.append(image)

    ### Get a random set of images from the training-set and plot them
    images = []
    cls_true = []
    for i in range(0, 9):
        index = random.randint(0, len(cars)-1)
        images.append(cv2.imread(cars[index]))
        
        # Get the true classes for those images.
        cls_true.append('1')

    # Plot the images and labels using our helper-function above.
    plot_images_fxn(images, cls_true, "vehicles", 3, 3)

    notcars = []
    images = glob.glob('./non-vehicles/*/*.png')
    for image in images:
        notcars.append(image)

    ### Get a random set of images from the training-set and plot them
    images = []
    cls_true = []
    for i in range(0, 9):
        index = random.randint(0, len(notcars)-1)
        images.append(cv2.imread(notcars[index]))
        
        # Get the true classes for those images.
        cls_true.append('0')

    # Plot the images and labels using our helper-function above.
    plot_images_fxn(images, cls_true, "non-vehicles", 3, 3)

    car_features = extract_features(cars, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = extract_features(notcars, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)

    X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    # Save the data for easy access
    pickle_file = 'data.p'
    print('Saving data to pickle file...')
    try:
        with open('data.p', 'wb') as pfile:
            pickle.dump(
                {
                    'train_dataset': X_train,
                    'train_labels': y_train,
                    'test_dataset': X_test,
                    'test_labels': y_test,
                    'full_dataset': X,
                },
                pfile, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print('Unable to save data to', pickle_file, ':', e)
        raise
    print('Data cached in pickle file.')

    return X_train, y_train, X_test, y_test, X_scaler

 
class VehicleDetector(object):

    to_plot = g_debug_plot
    X_scaler = None

    def __init__(self):
        # Use a linear SVC 
        self.svc = LinearSVC()
        self.avg_N = 100 #arbitrary large number
        self.avg_heatmap = []
        self.heatmaps = collections.deque(maxlen=8)
        self.t_start = 0
        self.t_stop = 0
        self.prev_bboxes = []

    def __plot(self, img, cmap=None, interpolation=None, 
                force_plot=False, grid=False, title=None):
        if self.to_plot or force_plot:
            plt.imshow(img, cmap=cmap, interpolation=interpolation)
            if grid:
                plt.grid(True)
                plt.grid(b=True, which='major', color='g', linestyle='-')
                plt.grid(b=True, which='minor', color='r', linestyle='--')
                plt.minorticks_on()
            if title:
                plt.savefig(title)
                plt.close()
            else:
                plt.show()

    def __timestamp(self, marker, operation, force_record=False):
        if force_record:
            if marker == 'start':
                self.t_start = time.time()
            else:
                self.t_stop = time.time()
                print(round(self.t_stop-self.t_start, 2), 'Seconds to ', operation)

    def clear_metadata(self):
        self.avg_N = 100 #arbitrary large number
        self.avg_heatmap = []
        self.heatmaps = collections.deque(maxlen=8)
        self.prev_bboxes = []

    def train_svm(self, X_train, y_train, X_test, y_test):
        # Check the training time for the SVC
        self.__timestamp('start', 'train SVC', True)
        self.svc.fit(X_train, y_train)
        self.__timestamp('stop', 'train SVC', True)

        # Check the score of the SVC
        print('Test Accuracy of SVC = ', round(self.svc.score(X_test, y_test), 4))


    ''' Full pipeline to take in a camera image and return the vehicle-detected
        image
    '''
    def pipeline(self, image):
        draw_image = np.copy(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Input image is scaled 0 to 255; bring it down to 0-1 to match
        # training input
        image = image.astype(np.float32)/255

        heatmap = np.zeros_like(image[:,:,0]).astype(np.float)
        all_windows = []
        img_height_half = int(image.shape[0]/2) + 20

        if (self.avg_N > 10):
            self.avg_N = 0
            self.avg_heatmap = np.zeros_like(image[:,:,0]).astype(np.float)
            prev_bboxes = [] 
        else:
            self.avg_N += 1
            prev_bboxes = self.prev_bboxes

        # Convert to relevant colorspace
        feature_image = convert_colorspace(image, color_space)

        # Get HOG over the entire region of interest (bottom half of image)
        img_ch_half = feature_image[img_height_half:image.shape[0]][:]
        hog_features = []
        self.__timestamp('start', 'get HOG')
        for channel in range(feature_image.shape[2]):
            if self.to_plot:
                hog_feature, hog_image = get_hog_features(img_ch_half[:,:,channel], orient, 
                                pix_per_cell, cell_per_block, vis=True, feature_vec=False)
                hog_features.append(hog_feature)
                self.__plot(hog_image, title='HOG_pipeline_image.png')
            else:
                hog_features.append(get_hog_features(img_ch_half[:,:,channel], orient, 
                                pix_per_cell, cell_per_block, vis=False, feature_vec=False))
        self.__timestamp('stop', 'get HOG')
        np.set_printoptions(threshold=np.inf)

        # Slide windows of varying sizes (multiples of 64) across the bottom
        # half of the image and search the sliding windows for vehicle matches
        for window_size in np.arange(64, img_height_half-96, 64):
            self.__timestamp('start', 'get sliding windows')
            windows, window_hog_list = slide_window(image, hog_features, x_start_stop=[None, None], 
                                   y_start_stop=[img_height_half,min(img_height_half+window_size+100,image.shape[0]-50)],
                                       xy_window=(window_size, window_size), xy_overlap=(0.75, 0.75))
            self.__timestamp('stop', 'get sliding windows')

            if self.to_plot:
                for window in windows:
                    cv2.rectangle(draw_image, (window[0][0],window[0][1]), \
                            (window[1][0], window[1][1]), (0,255,204), 1)

            self.__timestamp('start', 'get hot windows')
            hot_windows = search_windows(feature_image, windows, window_hog_list, self.svc, 
                                    self.X_scaler, prev_bboxes, color_space=color_space, 
                                    spatial_size=spatial_size, hist_bins=hist_bins, 
                                    orient=orient, pix_per_cell=pix_per_cell, 
                                    cell_per_block=cell_per_block, 
                                    hog_channel=hog_channel, spatial_feat=spatial_feat, 
                                    hist_feat=hist_feat, hog_feat=hog_feat)                       
            self.__timestamp('stop', 'get hot windows')

            all_windows += hot_windows
        
        # Averaged and labeled heatmaps 
        # Heatmap:
        self.__timestamp('start', 'draw/heat windows')
        heatmap = add_heat(heatmap, all_windows)
        heatmap = avg_heat(heatmap, self.avg_heatmap, self.avg_N)
        self.__plot(heatmap, cmap='hot', interpolation='nearest', force_plot=False)

        # Labels:
        threshold = 1.5 #video stream
        self.avg_heatmap = apply_threshold(heatmap, threshold)
        label_struct = generate_binary_structure(2, 2)
        labels = label(self.avg_heatmap, structure=label_struct)
        self.__plot(labels[0], cmap='gray', force_plot=False)

        # Draw bounding boxes on a copy of the image
        draw_img, self.prev_bboxes = draw_labeled_bboxes(draw_image, labels, self.prev_bboxes)
        self.__plot(draw_img)

        self.__timestamp('stop', 'draw/heat windows')
        return draw_img


def run_pipeline(is_train, test_type):

    vehicle_detector = VehicleDetector()
    
    if is_train:
        print('Extracting features from training images...')
        X_train, y_train, X_test, y_test, vehicle_detector.X_scaler = get_train_data()
    else:
        # Reload the data
        pickle_file = 'data.p'
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
            X_train = pickle_data['train_dataset']
            y_train = pickle_data['train_labels']
            X_test = pickle_data['test_dataset']
            y_test = pickle_data['test_labels']
            vehicle_detector.X_scaler = StandardScaler().fit(pickle_data['full_dataset'])
            
            del pickle_data  # Free up memory

        print('Data and modules loaded.')
    print('Feature vector length:', len(X_train[0]))

    # Train SVM
    vehicle_detector.train_svm(X_train, y_train, X_test, y_test)

    # Execute pipeline
    if test_type == 0: #Images
        images = glob.glob('./test_images/*.jpg')
        for fname in images:
            image = cv2.imread(fname)
            #vehicle_detector.__timestamp('start', 'process 1 frame')
            output_image = vehicle_detector.pipeline(image)
            #vehicle_detector.__timestamp('stop', 'process 1 frame')
            cv2.imwrite('output_images/output_{}'.format(fname.split("/")[2]), output_image)
            vehicle_detector.clear_metadata()

    elif test_type == 1: #FullVideo
        clips=[]
        for i in range(10):
            video_output = 'video_output{}.mp4'.format(i)
            clip1 = VideoFileClip("project_video.mp4").subclip(i*5,(i+1)*5)
            video_clip = clip1.fl_image(vehicle_detector.pipeline) 
            video_clip.write_videofile(video_output, audio=False)
            clips.append(clip1)
        final_clip = concatenate_videoclips(clips)
        final_clip.write_videofile("output_video.mp4")

    else: #TestVideos
        video_output = 'test_output.mp4'
        #clip1 = VideoFileClip("test_video.mp4")
        clip1 = VideoFileClip("project_video.mp4").subclip(40,45)
        video_clip = clip1.fl_image(vehicle_detector.pipeline) 
        video_clip.write_videofile(video_output, audio=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--init", help="Extracting features from scratch",
                        action="store_true")
    parser.add_argument("-t", "--testtype", type=int, default=0, help='0 for images, 1 for video')
    args = parser.parse_args()

    if not args.init:
        print ("Using pickled features")

    run_pipeline(args.init, args.testtype)

if __name__ == '__main__': main()

