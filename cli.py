import pexpect
import typer
from pathlib import Path
import os, pwd

app = typer.Typer()

DOCKER_IMAGE = "yosoufe/vamr"


def bash_cmd(command: str):
    """
    run a bash command
    """
    typer.echo(f"executing: {command}")
    child = pexpect.spawn(command)
    child.interact()
    child.close(True)
    return child.signalstatus


@app.command()
def build():
    """
    build the docker image
    """
    bash_cmd(f"docker build . -t {DOCKER_IMAGE}")


@app.command()
def run():
    """
    run the docker image
    """
    bash_cmd("xhost +")
    bash_cmd("docker run "
             "--name vamr "
             "-it "
             "--device /dev/video0 "
             "--network host "
             "-e DISPLAY "
             "--privileged "
             "--runtime=nvidia "
             "--gpus all "
             "--rm "
            #  "-v /etc/passwd:/etc/passwd "
            #  f"-u {pwd.getpwnam(os.getenv('USER')).pw_uid}:{pwd.getpwnam(os.getenv('USER')).pw_gid} "
             f"-v {Path(__file__).parent.resolve()}:/code "
             f"{DOCKER_IMAGE}")


@app.command()
def push():
    """
    push the docker image to docker hub as yosoufe/opencv_cuda
    """
    bash_cmd(f"docker push {DOCKER_IMAGE}")


@app.command()
def pull():
    """
    pull the docker image from the docker hub
    """
    bash_cmd(f'docker pull {DOCKER_IMAGE}')


if __name__ == "__main__":
    app()
