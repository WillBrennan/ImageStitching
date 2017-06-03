#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Will Brennan'

import logging

import cv2
import numpy

logger = logging.getLogger("main")


def compute_matches(features0, features1, matcher, knn=5, lowe=0.7):
    keypoints0, descriptors0 = features0
    keypoints1, descriptors1 = features1

    logger.debug('finding correspondence')

    matches = matcher.knnMatch(descriptors0, descriptors1, k=knn)

    logger.debug("filtering matches with lowe test")

    positive = []
    for match0, match1 in matches:
        if match0.distance < lowe * match1.distance:
            positive.append(match0)

    src_pts = numpy.array([keypoints0[good_match.queryIdx].pt for good_match in positive], dtype=numpy.float32)
    src_pts = src_pts.reshape((-1, 1, 2))
    dst_pts = numpy.array([keypoints1[good_match.trainIdx].pt for good_match in positive], dtype=numpy.float32)
    dst_pts = dst_pts.reshape((-1, 1, 2))

    return src_pts, dst_pts, len(positive)
