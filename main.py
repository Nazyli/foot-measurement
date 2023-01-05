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
    
    clusteredImg = kMeans_cluster(preprocessedOimg)
    cv2.imwrite('output/clusteredImg.jpg', clusteredImg)

    edgedImg = edgeDetection(clusteredImg)
    cv2.imwrite('output/edgedImg.jpg', edgedImg)

    boundRect, contours, contours_poly, img = getBoundingBox(edgedImg)
    pdraw = drawCnt(boundRect[1], contours, contours_poly, img)
    cv2.imwrite('output/pdraw.jpg', pdraw)


if __name__ == '__main__':
    main()