import pexpect
import typer

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
    bash_cmd("docker run --rm -it --device /dev/video0 --network host -e DISPLAY --runtime=nvidia --gpus all vamr-base")


if __name__ == "__main__":
    app()
