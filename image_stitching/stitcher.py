import logging

import cv2
import numpy

from .combine import combine_images
from .combine import compute_matches

__doc__ = '''ImageStitcher class for combining all images together'''


class ImageStitcher:
    __doc__ = '''ImageStitcher class for combining all images together'''

    def __init__(self, min_num: int = 10, lowe: float = 0.7, knn_clusters: int = 2):
        '''constructor that initialises the SIFT class and Flann matcher'''
        self.min_num = min_num
        self.lowe = lowe
        self.knn_clusters = knn_clusters

        self.flann = cv2.FlannBasedMatcher({'algorithm': 0, 'trees': 5}, {'checks': 50})
        self.sift = cv2.SIFT_create()

        self.result_image = None
        self.result_image_gray = None

    def add_image(self, image: numpy.ndarray):
        '''
        this adds a new image to the stitched image by
        running feature extraction and matching them
        '''
        assert image.ndim == 3, 'must be an image!'
        assert image.shape[-1] == 3, 'must be BGR!'
        assert image.dtype == numpy.uint8, 'must be a uint8'

        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        if self.result_image is None:
            self.result_image = image
            self.result_image_gray = image_gray
            return

        # todo(will.brennan) - stop computing features on the results image each time!
        result_features = self.sift.detectAndCompute(self.result_image_gray, None)
        image_features = self.sift.detectAndCompute(image_gray, None)

        matches_src, matches_dst, n_matches = compute_matches(result_features,
                                                              image_features,
                                                              matcher=self.flann,
                                                              knn=self.knn_clusters,
                                                              lowe=self.lowe)

        if n_matches < self.min_num:
            logging.warning('too few correspondences to add image to stitched image')
            return

        logging.debug('computing homography between accumulated and new images')
        homography, _ = cv2.findHomography(matches_src, matches_dst, cv2.RANSAC, 5.0)

        logging.debug('stitching images together')
        self.result_image = combine_images(image, self.result_image, homography)
        self.result_image_gray = cv2.cvtColor(self.result_image, cv2.COLOR_RGB2GRAY)

    def image(self):
        '''class for fetching the stitched image'''
        return self.result_image
