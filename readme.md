# Image and Video Stitching
This algorithm runs through a video file, or a set of images, and stitches them together to form a single image. It can be
used for scanning in large documents where the resolution from a single photo may not be sufficient. Currently this doesnt
take into account image blurring, evaluating whether an incoming frame has a better quality than the previous one, or
lens undistortion.

## Quick Start
Getting the app to run is pretty easy, just follow the script below!
In case your python environment does not treat image_stitching/combine.py, matching.py, helpers.py as modules, paste the files in root directory
```bash
# Clone the repo
git clone https://github.com/WillBrennan/VideoStitcher && cd VideoStitcher

# install deps
make install

# Run the algorithm!
python video_stitching.py <path to video file> --display --save
python image_stitching.py <path to image directory> --display --save
```

## Demonstration
![Demo on Video](https://raw.githubusercontent.com/WillBrennan/ImageStitching/master/examples/display.png "Demonstration")

## References
[Automatic Panoramic Image Stitching using Invariant Features](https://www.cs.bath.ac.uk/brown/papers/ijcv2007.pdf)
