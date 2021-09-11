import pexpect
import typer
from pathlib import Path

app = typer.Typer()


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
    bash_cmd("docker build . -t vamr-base")


@app.command()
def run():
    """
    run the docker image
    """
    bash_cmd("xhost +")
    bash_cmd("docker run "
             "--rm "
             "-it "
             "--device /dev/video0 "
             "--network host "
             "-e DISPLAY "
             "--privileged "
             "--runtime=nvidia "
             "--gpus all "
             f"-v {Path(__file__).parent.resolve()}:/code "
             "vamr-base ")

@app.command()
def push():
    """
    push the docker image to docker hub as yosoufe/opencv_cuda
    """
    bash_cmd("docker tag vamr-base:latest yosoufe/opencv_cuda:4.5.0_11.4.1")
    bash_cmd("docker push yosoufe/opencv_cuda:4.5.0_11.4.1")


if __name__ == "__main__":
    app()
