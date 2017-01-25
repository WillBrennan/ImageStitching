#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Will Brennan'

import os
import cv2
import numpy
import logging

logger = logging.getLogger("main")


def is_cv2():
    major, minor, increment = cv2.__version__.split(".")
    return major == "2"


def is_cv3():
    major, minor, increment = cv2.__version__.split(".")
    return major == "3"


def display(title, img, max_size=500000):
    assert isinstance(img, numpy.ndarray), 'img must be a numpy array'
    assert isinstance(title, str), 'title must be a string'
    scale = numpy.sqrt(min(1.0, float(max_size) / (img.shape[0] * img.shape[1])))
    shape = (int(scale * img.shape[1]), int(scale * img.shape[0]))
    img = cv2.resize(img, shape)
    cv2.imshow(title, img)


def save_image(path, result):
    name, ext = os.path.splitext(path)
    img_path = '{0}.png'.format(name)
    logger.debug('writing image to {0}'.format(img_path))
    cv2.imwrite(img_path, result)
    logger.debug('writing complete')
