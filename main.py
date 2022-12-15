from sklearn.cluster import KMeans
import cv2
from skimage.io import imread
import os

from utils import *


ImgPath = 'input/sample.jpg'

def main():

    oimg = imread(ImgPath)

    if not os.path.exists('output'):
        os.makedirs('output')

    preprocessedOimg = preprocess(oimg)
    cv2.imwrite('output/preprocessedOimg.jpg', preprocessedOimg)
    


if __name__ == '__main__':
    main()