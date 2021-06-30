import logging
import pathlib

from typing import List
from typing import Generator

import cv2
import numpy

__doc__ = '''helper functions for loading frames and displaying them'''


def display(title, img, max_size=500000):
    '''
    resizes the image before it displays it,
    this stops large stitches from going over the screen!
    '''
    assert isinstance(img, numpy.ndarray), 'img must be a numpy array'
    assert isinstance(title, str), 'title must be a string'
    scale = numpy.sqrt(min(1.0, float(max_size) / (img.shape[0] * img.shape[1])))
    shape = (int(scale * img.shape[1]), int(scale * img.shape[0]))
    img = cv2.resize(img, shape)
    cv2.imshow(title, img)


def read_video(video_path: pathlib.Path):
    '''read video is a generator class yielding frames'''
    cap = cv2.VideoCapture(str(video_path))

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        yield frame


def load_frames(paths: List[str]) -> Generator[numpy.ndarray, None, None]:
    '''
    load_frames takes in a list of paths to image,
    video files, or directories and yields them
    '''
    for path in paths:
        path = pathlib.Path(path)

        if path.is_dir():
            yield from load_frames(path.rglob('*'))
        elif path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            yield cv2.imread(str(path))
        elif path.suffix.lower() in ['.avi', '.mp4', '.mov']:
            yield from read_video(path)
        else:
            logging.warning(f'skipping {path.name}...')
