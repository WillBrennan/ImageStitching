#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Will Brennan'

# Built-in Modules
import argparse
import logging

import cv2

import image_stitching

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image_paths', type=str, nargs='+', help="paths to one or more images or image directories")
    parser.add_argument('-b', '--debug', dest='debug', action='store_true', help='enable debug logging')
    parser.add_argument('-q', '--quiet', dest='quiet', action='store_true', help='disable all logging')
    parser.add_argument('-d', '--display', dest='display', action='store_true', help="display result")
    parser.add_argument('-s', '--save', dest='save', action='store_true', help="save result to file")
    parser.add_argument("--save_path", dest='save_path', default="stitched.png", type=str, help="path to save result")
    parser.add_argument('-k', '--knn', dest='knn', default=2, type=int, help="Knn cluster value")
    parser.add_argument('-l', '--lowe', dest='lowe', default=0.7, type=float, help='acceptable distance between points')
    parser.add_argument('-m', '--min', dest='min_correspondence', default=10, type=int, help='min correspondences')
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("main")

    logging.info("beginning sequential matching")

    if image_stitching.helpers.is_cv2():
        sift = cv2.SIFT()
    elif image_stitching.helpers.is_cv3():
        sift = cv2.xfeatures2d.SIFT_create()
    else:
        raise RuntimeError("error! unknown version of python!")

    result = None
    result_gry = None

    flann = cv2.FlannBasedMatcher({'algorithm': 0, 'trees': 5}, {'checks': 50})

    cap = cv2.VideoCapture(args.video_path)

    while True:
        ret, frame = cap.read()
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if not ret:
            break

        if not result:
            result = frame
        else:
            features0 = sift.detectAndCompute(result_gry, None)
            features1 = sift.detectAndCompute(frame_gray, None)

            matches_src, matches_dst, n_matches = image_stitching.compute_matches(
                features0, features1, flann, knn=args.knn)

            if n_matches < args.min_correspondence:
                logger.error("error! too few correspondences")
                continue

            H, mask = cv2.findHomography(matches_src, matches_dst, cv2.RANSAC, 5.0)
            result = image_stitching.combine_images(frame, result, H)

            if args.display and not args.quiet:
                image_stitching.helpers.display('result', result)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

        result_gry = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    logger.info('{0} is completed'.format(args.video_path))
    cap.release()
    cv2.destroyAllWindows()
    if args.save:
        image_stitching.helpers.save_image(args.video_path, result)
