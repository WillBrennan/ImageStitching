import logging

import cv2
import numpy

__doc__ = '''helper functions for combining images, only to be used in the stitcher class'''


def compute_matches(features0, features1, matcher, knn=5, lowe=0.7):
    '''
    this applies lowe-ratio feature matching between feature0 an dfeature 1 using flann
    '''
    keypoints0, descriptors0 = features0
    keypoints1, descriptors1 = features1

    logging.debug('finding correspondence')

    matches = matcher.knnMatch(descriptors0, descriptors1, k=knn)

    logging.debug("filtering matches with lowe test")

    positive = []
    for match0, match1 in matches:
        if match0.distance < lowe * match1.distance:
            positive.append(match0)

    src_pts = numpy.array([keypoints0[good_match.queryIdx].pt for good_match in positive],
                          dtype=numpy.float32)
    src_pts = src_pts.reshape((-1, 1, 2))
    dst_pts = numpy.array([keypoints1[good_match.trainIdx].pt for good_match in positive],
                          dtype=numpy.float32)
    dst_pts = dst_pts.reshape((-1, 1, 2))

    return src_pts, dst_pts, len(positive)


def combine_images(img0, img1, h_matrix):
    '''
    this takes two images and the homography matrix from 0 to 1 and combines the images together!
    the logic is convoluted here and needs to be simplified!
    '''
    logging.debug('combining images... ')

    points0 = numpy.array(
        [[0, 0], [0, img0.shape[0]], [img0.shape[1], img0.shape[0]], [img0.shape[1], 0]],
        dtype=numpy.float32)
    points0 = points0.reshape((-1, 1, 2))
    points1 = numpy.array(
        [[0, 0], [0, img1.shape[0]], [img1.shape[1], img1.shape[0]], [img1.shape[1], 0]],
        dtype=numpy.float32)
    points1 = points1.reshape((-1, 1, 2))

    points2 = cv2.perspectiveTransform(points1, h_matrix)
    points = numpy.concatenate((points0, points2), axis=0)

    [x_min, y_min] = (points.min(axis=0).ravel() - 0.5).astype(numpy.int32)
    [x_max, y_max] = (points.max(axis=0).ravel() + 0.5).astype(numpy.int32)

    h_translation = numpy.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])

    logging.debug('warping previous image...')
    output_img = cv2.warpPerspective(img1, h_translation.dot(h_matrix),
                                     (x_max - x_min, y_max - y_min))
    output_img[-y_min:img0.shape[0] - y_min, -x_min:img0.shape[1] - x_min] = img0
    return output_img
