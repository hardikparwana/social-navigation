from argparse import ArgumentParser

import torch
import torch.multiprocessing
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import numpy as np

# from shapely.geometry import box

from neural_clbf.controllers import NeuralObsBFController
from neural_clbf.datamodules.episodic_datamodule import (
    EpisodicDataModule,
)
from neural_clbf.systems import TurtleBot2D
from neural_clbf.systems.planar_lidar_system import Scene
from neural_clbf.experiments import (
    ExperimentSuite,
    BFContourExperiment,
    # LFContourExperiment,
    RolloutStateSpaceExperiment,
)
from neural_clbf.training.utils import current_git_hash


torch.multiprocessing.set_sharing_strategy("file_system")

# batch_size = 1024
batch_size = 64
controller_period = 0.1

start_x = torch.tensor(
    [
        [4.5, 2.5, -np.pi / 2],
        [-4.5, 2.5, -np.pi / 2],
        [-4.5, -2.5, np.pi / 2],
        [4.5, -2.5, np.pi / 2],
    ]
)
simulation_dt = 0.01

# Scene parameters
room_size = 10.0
num_obstacles = 8
box_size_range = (0.75, 1.75)
position_range = (-4.0, 4.0)
rotation_range = (-np.pi, np.pi)

# Lidar parameters
num_rays = 32
field_of_view = (-np.pi, np.pi)
max_distance = 2 * room_size


def main(args):
    # Define the scenarios
    nominal_params = {"R": 0.0325, "L": 0.14}
    scenarios = [
        nominal_params,
    ]

    # Make the random scene
    scene = Scene([])
    scene.add_walls(room_size)
    scene.add_random_boxes(
        num_obstacles,
        box_size_range,
        position_range,
        position_range,
        rotation_range,
    )
    # scene.add_obstacle(box(-1, -1, 1, 1))

    # (spicy!) and make another random scene for validation
    validation_scene = Scene([])
    validation_scene.add_walls(room_size)
    validation_scene.add_random_boxes(
        num_obstacles,
        box_size_range,
        position_range,
        position_range,
        rotation_range,
    )

    # Define the dynamics model
    dynamics_model = TurtleBot2D(
        nominal_params,
        scene,
        dt=simulation_dt,
        controller_dt=controller_period,
        num_rays=num_rays,
        field_of_view=field_of_view,
        max_distance=max_distance,
    )

    # And define a second dynamics model for validation (with the different scene)
    validation_dynamics_model = TurtleBot2D(
        nominal_params,
        validation_scene,
        dt=simulation_dt,
        controller_dt=controller_period,
        num_rays=num_rays,
        field_of_view=field_of_view,
        max_distance=max_distance,
    )

    # Initialize the DataModule
    initial_conditions = [
        (-4.9, 4.9),  # x
        (-4.9, 4.9),  # y
        (-np.pi, np.pi),  # theta
    ]
    data_module = EpisodicDataModule(
        dynamics_model,
        initial_conditions,
        trajectories_per_episode=50,
        trajectory_length=100,
        fixed_samples=5000,
        max_points=20000,
        val_split=0.1,
        batch_size=batch_size,
    )

    # Define the experiment suite
    h_contour_experiment = BFContourExperiment(
        "h_Contour",
        domain=[(-5.0, 5.0), (-5.0, 5.0)],
        n_grid=60,
        x_axis_index=TurtleBot2D.X,
        y_axis_index=TurtleBot2D.Y,
        x_axis_label="$x$",
        y_axis_label="$y$",
    )
    # V_contour_experiment = LFContourExperiment(
    #     "V_Contour",
    #     domain=[(-5.0, 5.0), (-5.0, 5.0)],
    #     n_grid=60,
    #     x_axis_index=TurtleBot2D.X,
    #     y_axis_index=TurtleBot2D.Y,
    #     x_axis_label="$x$",
    #     y_axis_label="$y$",
    # )
    rollout_experiment = RolloutStateSpaceExperiment(
        "Rollout",
        start_x,
        plot_x_index=TurtleBot2D.X,
        plot_x_label="$x$",
        plot_y_index=TurtleBot2D.Y,
        plot_y_label="$y$",
        scenarios=scenarios,
        n_sims_per_start=1,
        t_sim=5.0,
    )
    experiment_suite = ExperimentSuite(
        [
            h_contour_experiment,
            # V_contour_experiment,
            rollout_experiment,
        ]
    )

    # Initialize the controller
    bf_controller = NeuralObsBFController(
        dynamics_model,
        data_module,
        experiment_suite=experiment_suite,
        encoder_hidden_layers=2,
        encoder_hidden_size=48,
        h_hidden_layers=2,
        h_hidden_size=48,
        h_alpha=0.3,
        lookahead_dual_penalty=1e3,
        lookahead_grid_n=8,
        controller_period=controller_period,
        validation_dynamics_model=validation_dynamics_model,
        epochs_per_episode=10,
    )

    # Initialize the logger and trainer
    tb_logger = pl_loggers.TensorBoardLogger(
        "logs/lidar_turtlebot",
        name=f"commit_{current_git_hash()}",
    )
    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=tb_logger,
        reload_dataloaders_every_epoch=True,
        check_val_every_n_epoch=1,
        stochastic_weight_avg=True,
        # track_grad_norm=2,
        max_epochs=400,
    )

    # Train
    torch.autograd.set_detect_anomaly(True)
    trainer.fit(bf_controller)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
