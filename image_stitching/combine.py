#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Will Brennan'

# Built-in Modules
import logging
# Standard Modules
import cv2
import numpy
# Custom Modules

logger = logging.getLogger('main')


def combine_images(img0, img1, h_matrix):
    logger.debug('combining images... ')
    points0 = numpy.array(
        [[0, 0], [0, img0.shape[0]], [img0.shape[1], img0.shape[0]], [img0.shape[1], 0]], dtype=numpy.float32)
    points0 = points0.reshape((-1, 1, 2))
    points1 = numpy.array(
        [[0, 0], [0, img1.shape[0]], [img1.shape[1], img0.shape[0]], [img1.shape[1], 0]], dtype=numpy.float32)
    points1 = points1.reshape((-1, 1, 2))
    points2 = cv2.perspectiveTransform(points1, h_matrix)
    points = numpy.concatenate((points0, points2), axis=0)
    [x_min, y_min] = numpy.int32(points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = numpy.int32(points.max(axis=0).ravel() + 0.5)
    H_translation = numpy.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    logger.debug('warping previous image...')
    output_img = cv2.warpPerspective(img1, H_translation.dot(h_matrix), (x_max - x_min, y_max - y_min))
    output_img[-y_min:img0.shape[0] - y_min, -x_min:img0.shape[1] - x_min] = img0
    return output_img
