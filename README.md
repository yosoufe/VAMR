# Vision Algorithm for Mobile Robotics (Fall 2020)

## Links

[The Course Website](http://rpg.ifi.uzh.ch/teaching2020.html).

## requirements

- Developed on Ubuntu 20.04
- Nvidia GPU
- latest Nvidia driver compatible with cuda 11.4 (tested with driver 470)
- docker
- [Nvidia docker container runtime](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#)
- Python 3 and pip

## Usage

We are using docker, We are installing all the requirements inside the docker and run everything inside the docker.

`cli.py` is a helper command line tool to build and run docker images.

```sh
# to install the cli.py's dependencies
python -m pip install -r requirements.txt

# to build the docker image
python cli.py build

# to run the docker image and have a terminal inside the docker image, compile everything in the image
python cli.py run
```

## compile

```bash
cd exercise
mkdir build
cd build
cmake ..
make -j`nproc`
```

## Exercises

### Exercise 1 - Augmented Reality Wireframe Cube

To learn about camera models and Distortion.

![Output](exercises/statements/Exercise%201%20-%20Augmented%20Reality%20Wireframe%20Cube/output.gif)

solution is in `exercises/exercise01.cpp`.

## useful command

Convert to gif

```bash
ffmpeg -ss 0 -t 3 -i input.mp4 -vf "fps=10,scale=320:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 output.gif
```
