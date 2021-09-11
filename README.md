# Vision Algorithm for Mobile Robotics (Fall 2020)

## Links

[The Course Website](http://rpg.ifi.uzh.ch/teaching2020.html).

## requirements

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
