# Image and Video Stitching
This algorithm runs through a video file, or a set of images, and stitches them together to form a single image. It can be
used for scanning in large documents where the resolution from a single photo may not be sufficient. Currently this doesnt
take into account image blurring, evaluating whether an incoming frame has a better quality than the previous one, or
lens distortion.

## Quick Start
Getting the app running is pretty simple; clone, install the requirements, and run!

```bash
# Clone the repo
git clone https://github.com/WillBrennan/ImageStitching && cd ImageStitching

# install deps
pip install -r requirements.txt

# Run the stitching!
python stitching.py <path to image directory or video files> --display --save
```

## Demonstration
![Demo on Video](https://raw.githubusercontent.com/WillBrennan/ImageStitching/master/examples/display.png "Demonstration")

## References
[Automatic Panoramic Image Stitching using Invariant Features](https://www.cs.bath.ac.uk/brown/papers/ijcv2007.pdf)