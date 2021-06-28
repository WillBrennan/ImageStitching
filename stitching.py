import argparse
import logging

import cv2

from image_stitching import ImageStitcher
from image_stitching import load_frames
from image_stitching import display

__doc__ = '''This script lets us stich images together and display or save the results'''


def parse_args():
    '''parses the command line arguments'''
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paths',
                        type=str,
                        nargs='+',
                        help="paths to images, directories, or videos")
    parser.add_argument('--debug', action='store_true', help='enable debug logging')

    parser.add_argument('--display', action='store_true', help="display result")
    parser.add_argument('--save', action='store_true', help="save result to file")
    parser.add_argument("--save_path", default="stitched.png", type=str, help="path to save result")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level)

    stitcher = ImageStitcher()

    for idx, frame in enumerate(load_frames(args.paths)):
        stitcher.add_image(frame)

        result = stitcher.image()

        if args.display:
            logging.info(f'displaying image {idx}')
            display('result', result)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        if args.save:
            image_name = f'result_{idx}.jpg'
            logging.info(f'saving result image on {image_name}')

            cv2.imwrite(image_name, result)

    logging.info('finished stitching images together')

    if args.save:
        logging.info(f'saving final image to {args.save_path}')
        cv2.imwrite(args.save_path, result)
