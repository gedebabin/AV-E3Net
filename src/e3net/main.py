from lightning.pytorch.cli import LightningCLI
from datamodule_e3net import DataModule
from model_e3net import ModelAudioOnly
import sys


def cli_main():
    cli = LightningCLI(ModelAudioOnly, DataModule, save_config_callback=None)

    # if cli.subcommand == "fit":
    #     cli.trainer.test(cli.model, cli.datamodule, ckpt_path="best")


if __name__ == "__main__":
    sys.tracebacklimit = 0
    cli_main()
