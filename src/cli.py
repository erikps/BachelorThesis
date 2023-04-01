import click
import math
import time

import keyboard

from src.dataset import AttackInferenceDataset
from src.envs.environment import AttackInferenceEnvironment
from src.model import DeepQAgent, train_agent


def format_dictonary(dictionary: dict):
    return "; ".join([f"{k}: {v}" for k, v in dictionary.items()])


@click.group()
def cli():
    pass


@click.command()
@click.option(
    "--minimum-epsilon",
    default=0.1,
    type=click.FloatRange(0, 1.0),
    help="Minimum value for epsilon (exploration / exploitation tradeoff hyperparameter)",
    show_default=True,
)
@click.option(
    "--epsilon-decay",
    default=1e-3,
    type=click.FloatRange(0, 1.0),
    help="The amount by which epsilon decreases after each action.",
    show_default=True,
)
@click.option(
    "--learning-rate",
    default=1e-5,
    type=click.FloatRange(0, 1.0),
    help="Learning rate of the optimizer.",
    show_default=True,
)
@click.option(
    "--discount-rate",
    default=0.99,
    type=click.FloatRange(0, 1.0),
    help="Discount rate (gamma). Influences how much future rewards are discounted.",
    show_default=True,
)
@click.option(
    "--replay-buffer-length",
    default=1024,
    type=click.IntRange(0, math.inf),
    help="Length of the replay buffer in which past experiences are stored for further training.",
    show_default=True,
)
@click.option(
    "--batch-size",
    default=64,
    type=click.IntRange(0, math.inf),
    help="Number of past experiences to sample at every training step.",
    show_default=True,
)
@click.option(
    "--render-mode",
    default="none",
    type=click.Choice(["none", "human"]),
    help="If set to 'none', nothing is rendered, if set to 'human' a matplotlib window will render the current problem state",
    show_default=True,
)
def train(**kwargs):
    if kwargs["render_mode"] == "none":
        kwargs["render_mode"] = None

    settings_argument_names = [
        "minimum_epsilon",
        "epsilon_decay",
        "learning_rate",
        "discount_rate",
        "replay_buffer_length",
        "batch_size",
    ]

    settings_arguments = {
        k: v for k, v in kwargs.items() if k in settings_argument_names
    }

    # Set up the environment and agent
    settings = DeepQAgent.Settings(**settings_arguments)
    agent = DeepQAgent(settings=settings)
    environment = AttackInferenceEnvironment(
        AttackInferenceDataset.example_dataset(), render_mode=kwargs["render_mode"]
    )

    # The main training loop
    for step_info in train_agent(agent=agent, environment=environment):
        click.clear()
        click.echo(format_dictonary(step_info))
        click.echo("Press [q] to stop training.")

        if keyboard.is_pressed("q"):
            if click.confirm("Save model?"):
                click.echo("Saving model")
                agent.save_snapshot()
            exit()

    click.echo("Start training")


@click.command()
def run():
    # TODO: implement runnning a trained model
    pass


cli.add_command(train)


if __name__ == "__main__":
    cli()
