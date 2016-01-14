# VideoSticher
This algorithm runs through a video file and stitches together all the frames in the video to form a canvas. It can be
used for scanning in large documents where the resolution from a single photo may not be sufficient. Currently this doesnt
take into account image blurring, evaluating whether an incoming frame has a better quality than the previous one, or
lens undistortion.


## Quick Start
Getting the app to run is pretty easy, just follow the script below! This script will not
[install OpenCV](http://docs.opencv.org/doc/tutorials/introduction/linux_install/linux_install.html) or
[Numpy](http://docs.scipy.org/doc/numpy/user/install.html)

```bash
# Clone the repo
git clone https://github.com/WillBrennan/VideoStitcher && cd VideoStitcher
# Run the algorithm!
python main.py <path to video file> --display --save
```

## Demonstration
![Demo on Video](https://raw.githubusercontent.com/WillBrennan/ImageStitching/master/examples/display.png "Demonstration")

## References
[Automatic Panoramic Image Stitching using Invariant Features](https://www.cs.bath.ac.uk/brown/papers/ijcv2007.pdf)