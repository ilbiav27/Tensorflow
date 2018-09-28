import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from random import sample

#stretgh image from borders
def stretch_image(img, major_value, WH):
    #get a shape of images
    height, width, channels = img.shape

    #compare width with height
    top, bottom, left, right = (0,0,0,0)
    
    #find a longer one
    longest = max(height, width)

    if height < longest:
        dh = longest - height
        top = dh // 2
        bottom = dh - top
    elif width < longest:
        dw = longest - width
        left = dw // 2
        right = dw - left
    else:
        pass
    
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value = major_value)
    img = cv2.resize(img, (WH,WH),interpolation=cv2.INTER_CUBIC)
    return img

#???
def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    hist, _ = np.histogram(clt.labels_, bins=numLabels)
    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()
    # return the histogram
    return hist

#Get major color RGB in borders -> ( , , )
def get_major_color_at_borders(image):
    num_clusters = 3
    border_len = 3
    image_border = []
    
    height, width, channels = image.shape
    
    for i in range(height):
        for j in range(width):
            if i < border_len or j < border_len or i >= height - border_len or j >= width - border_len:
                image_border.append(image[i][j])
    image_border = np.array(image_border)

    clt = KMeans(n_clusters=num_clusters)
    clt.fit(image_border)

    hist = centroid_histogram(clt)
    result_sorted = sorted(zip(hist, clt.cluster_centers_), reverse=True)

    hist_sorted = []
    cluster_centers_sorted = []
    for i in range(len(result_sorted)):
        hist_sorted.append(result_sorted[i][0])
        cluster_centers_sorted.append(result_sorted[i][1])

    major_color = cluster_centers_sorted[0]
    return major_color

#like main It returns lists of imgs and their names
def get_image(path, num, WH):
    #test data directory
    img = np.ndarray((num,WH,WH,3), (np.uint8))
    #All test images name
    images = os.listdir(path)

    #rand for sampling test images
    img_name = sample(images, num)
    
    for i in range(num):
        #make path of image;
        path_img = os.path.join(path, img_name[i])
        
        #read image
        img_ = cv2.imread(path_img)
        major_color = get_major_color_at_borders(img_)
        stretched = stretch_image(img_, major_color, WH)
        img[i] = stretched
    return img, img_name
