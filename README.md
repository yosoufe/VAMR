# 🚧 Under Construction 🚧

# Vision Algorithms for Mobile Robotics

This repository contains all the exercises of the course "Vision Algorithm for Mobile Robotics" in C++.

If there are any questions, feel free to open an issue in this github repository.

## Links

[The Course Website](http://rpg.ifi.uzh.ch/teaching2020.html).

## requirements

- Developed on Ubuntu 20.04
- Nvidia GPU
- latest Nvidia driver compatible with cuda 11.4 (tested with driver 470)
- docker
- [Nvidia docker container runtime](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#)
- Python 3 and pip
- **NO** MATLAB:
  - The official course is doing all the exercises in MATLAB and since I do not have MATLAB License, I am doing it in C++. The second reason is, **all the jobs in robotics require strong C++ knowledge**, so I am practicing C++.
    The followings are inside the docker image:
    - Using Eigen library for linear algebra.
    - Using OpenCV and VTK only for reading, writing of image and video files and visualization. The rest are coded from scratch.

## Usage

We are using docker, We are installing all the requirements inside the docker and run everything inside the docker.

The container has [terminator](https://terminator-gtk3.readthedocs.io/en/latest/) inside which you can open multiple terminals
as tabs or split windows. Read the documentation of the [terminator](https://terminator-gtk3.readthedocs.io/en/latest/) to learn short keys.

`cli.py` is a helper command line tool to build and run docker image.

```sh
# to install the cli.py's dependencies
python -m pip install -r requirements.txt

# to build the docker image
python cli.py build

# or to pull the image from the dockerhub
# instead of building it,
# It can be faster than building,
# Compiling opencv can take a long time.
python cli.py pull

# to run the docker image and have a terminal inside the docker image, we compile everything in the container
# It will pull my image if you do not build it yourself.
python cli.py run
```

## Compile

```bash
cv VAMR
mkdir -p output/ex{01..09}
python cli.py run
cd exercises
mkdir build
cd build
cmake ..
make -j`nproc`
```

# Exercises

## Exercise 1 - Augmented Reality Wireframe Cube

This is about camera and distortion models.

- Problem statement: `exercises/statements/Exercise 1 - Augmented Reality Wireframe Cube/statement.pdf`.
- solution is in `exercises/exercise01.cpp`.
- Output Videos:
  - https://youtu.be/RD8uO2pETIE
  - https://youtu.be/Ba9SmGKgBmU

![Output](exercises/statements/outputs/ex01.gif)

## Exercise 2 - PnP Problem

This exercise is about the PnP (Perspective-n-Point) problem. We basically find the position and orientation of a calibrated camera based on known points in world and their known correspondences in the image frame.

- Problem statement: `exercises/statements/Exercise 2 - PnP/statement.pdf`.
- Solution: `exercises/exercise02.cpp`.
- Output Video:
  - https://youtu.be/nbFseP4vRTU

The following video shows the calculated pose and orientation of the camera relative to the pattern of April Tags.

![Output](exercises/statements/outputs/ex02.gif)

## Exercise 3 - Simple Keypoint Tracker

- Problem statement: `exercises/statements/Exercise 3 - Simple Keypoint Tracker/statement.pdf`.
- Solution: `exercises/exercise03.cpp`.
- Output Videos:
  - https://youtu.be/8O97v3q7bC4
  - https://youtu.be/T8WX1ktlg8E

Tracking:

![Output](exercises/statements/outputs/ex03-tracking.gif)

The following image shows the Harris and Shi-Tomasi scores, key points and descriptors for the first frame of the dataset.

![Output](exercises/statements/outputs/ex03-harris_shitomasi.png)

## Exercise 4 - Simple SIFT Keypoint Detection and Matching

- Problem statement: `exercises/statements/Exercise 4 - simple SIFT/statement.pdf`.
- Solution: `exercises/exercise04.cpp`.

  - :warning: **I guess there are still some bugs in my code** :warning:, but because of
    lack of time and relatively good results, I would go to the next exercise for now. I also skipped the
    optional part of the exercise. I might come back to it later. The descriptor matching could be optimized later.

  ![Output](exercises/statements/outputs/ex04-simple_sift.png)

## Exercise 5 - Stereo Dense Reconstruction

- Problem statement: `exercises/statements/Exercise 5 - Stereo Dense Reconstruction`.
- Solution: `exercises/exercise05.cpp`.
  - left first image
    ![Disparity image for first frame](https://user-images.githubusercontent.com/7648675/143288423-132e50ef-0a4b-48f0-9532-4c5ccce54b9b.png)
  - Disparity image from left and right images
    ![Disparity image for first frame](exercises/statements/outputs/ex05-disparity-filtered.png)
  - Rough Point Cloud from Disparity

    https://user-images.githubusercontent.com/7648675/144170690-082ce219-87d4-4637-81cf-84824a2f44d5.mp4

  - Point Cloud from Disparity with sub-pixel accuracy

    https://user-images.githubusercontent.com/7648675/144175521-0e370d34-662c-4df6-9901-5bec0fb630fa.mp4

  - complete point cloud from all of the pair of frames (better quality video in `exercises/statements/outputs/ex05-complete_point_cloud.mp4`)

    https://user-images.githubusercontent.com/7648675/144346394-fddc5b30-3640-41db-ae77-8c951ad19c4e.mp4

## Exercise 6 - Two-view Geometry

- Problem statement: `exercises/statements/Exercise 6 - Two-view Geometry`.
- Solution: `exercises/exercise06.cpp`.
- I developed unit tests using Google test framework, similar to the matlab test scripts provided by the exercise in `exercises/tests/test_two_view_geometry.cpp`.
   To execute them after the compilation in the build directory:
    ```bash
    ./tests/two_view_geometry_tests --gtest_filter=Two_View_Geometry.linear_triangulation
    ./tests/two_view_geometry_tests --gtest_filter=Two_View_Geometry.eight_point
    # or the following to run all of the tests for exercise 06.
    ./tests/two_view_geometry_tests
    ```
  - Point cloud from perfect feature matches
     ![Point Cloud](exercises/statements/outputs/ex06-8_point_sfm.png)


## Useful Commands

```bash
# convert to gif
ffmpeg -ss 0 -t 5 -i input.mp4 -vf "fps=10,scale=320:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 output.gif

# reduce the size and quality
ffmpeg -i input.mp4 -vcodec libx264 -crf 28 output.mp4
```

### cv::Viz3d Key commands
cv::Viz3d used for 3d visualizations and point cloud visualization. These shortkeys are useful to navigate the view.

```
| Help:
-------
          p, P   : switch to a point-based representation
          w, W   : switch to a wireframe-based representation (where available)
          s, S   : switch to a surface-based representation (where available)

          j, J   : take a .PNG snapshot of the current window view
          k, K   : export scene to Wavefront .obj format
    ALT + k, K   : export scene to VRML format
          c, C   : display current camera/window parameters
          F5     : enable/disable fly mode (changes control style)

          e, E   : exit the interactor
          q, Q   : stop and call VTK's TerminateApp

           +/-   : increment/decrement overall point size
     +/- [+ ALT] : zoom in/out

    r, R [+ ALT] : reset camera [to viewpoint = {0, 0, 0} -> center_{x, y, z}]

    ALT + s, S   : turn stereo mode on/off
    ALT + f, F   : switch between maximized window mode and original size
```
