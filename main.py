#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Will Brennan'


# Built-in Modules
import os
import argparse
import logging
# Standard Modules
import cv2
import numpy
# Custom Modules
import scripts
import combine

logger = logging.getLogger('main')


def main(args):
    assert isinstance(args, argparse.Namespace), 'args must be of type argparse.Namespace not {0}'.format(type(args))
    sift = cv2.SIFT()
    flann = cv2.FlannBasedMatcher({'algorithm': 0, 'trees': 5}, {'checks': 50})
    result, result_gry = None, None
    for path in args.paths:
        try:
            assert os.path.exists(path), '{0} is not a valid path'.format(path)
            logger.info('processing {0}'.format(path))
            cam = cv2.VideoCapture(path)
            logger.debug('opened video')
            count = 0
            while True:
                logger.debug('reading frame {0}'.format(count))
                ret, frame = cam.read()
                if ret:
                    logger.debug('frame read correctly')
                    count += 1
                    if result is not None and result_gry is not None:
                        frame_gry = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                        logger.debug('computing sift features')
                        keypoints0, descriptors0 = sift.detectAndCompute(result_gry, None)
                        keypoints1, descriptors1 = sift.detectAndCompute(frame_gry, None)
                        logger.debug('finding correspondence')
                        matches = flann.knnMatch(descriptors0, descriptors1, k=args.knn)
                        positive = []
                        for match0, match1 in matches:
                            if match0.distance < args.lowe*match1.distance:
                                positive.append(match0)
                        if len(positive) > args.min_correspondence:
                            src_pts = numpy.array([keypoints0[good_match.queryIdx].pt for good_match in positive], dtype=numpy.float32)
                            src_pts = src_pts.reshape((-1, 1, 2))
                            dst_pts = numpy.array([keypoints1[good_match.trainIdx].pt for good_match in positive], dtype=numpy.float32)
                            dst_pts = dst_pts.reshape((-1, 1, 2))
                            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                            result = combine.combine_images(frame, result, M)
                            if args.display and not args.quiet:
                                scripts.display('result', result)
                                if cv2.waitKey(25) & 0xFF == ord('q'):
                                    break
                        else:
                            logger.warning('too few correspondence points')
                    else:
                        result = frame
                    result_gry = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
                else:
                    break
            logger.debug('{0} is completed'.format(path))
            cam.release()
            cv2.destroyAllWindows()
            if args.save:
                scripts.save_image(path, result)
        except Exception as error:
            logger.warning('Failed to process {0}'.format(path))
            logger.debug('Error msg: {0}'.format(error))
    return result


if __name__ == '__main__':
    args = scripts.get_args()
    logger = scripts.get_logger(quiet=args.quiet, debug=args.debug)
    result = main(args)
    scripts.display('final result', result)
    cv2.waitKey(0)