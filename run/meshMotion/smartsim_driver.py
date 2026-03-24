#!/usr/bin/env python

import argparse
import os
import sys
import time

from pathlib import Path

from smartsim import Experiment
from smartsim.status import TERMINAL_STATUSES


platform_configs = {
    "local": {
        "launcher": "local",
        "interface": "lo",
    },
}

def main(args):

    platform_config = platform_configs[args.platform]

    # ----------------------------------------------------------------
    # Create the SmartSim experiment
    # ----------------------------------------------------------------

    input_case_path = Path(args.case)
    case_name = input_case_path.stem
    experiment_name = f"{args.experiment}_{case_name}_{args.mesh_solver_type}"
    if args.mesh_solver_type == "PINN":
        experiment_name = f"{experiment_name}_{args.pinn_type}"

    exp = Experiment(experiment_name, launcher=platform_config["launcher"])

    # ----------------------------------------------------------------
    # Launch the database
    # ----------------------------------------------------------------

    db = exp.create_database(port=args.port, interface=platform_config["interface"])
    exp.generate(db, overwrite=True)

    # ----------------------------------------------------------------
    # Configure and create the OpenFOAM mesh-motion model
    # ----------------------------------------------------------------

    # Create OpenFOAM moveDynamicMesh run settings
    openfoam_rs = exp.create_run_settings(
        exe="moveDynamicMesh",
        run_command=platform_config["run_command"]
    )

    # Create the model from the OpenFOAM case argument
    openfoam_model = exp.create_model(
        name=args.case,
        run_settings=openfoam_rs
    )
    openfoam_model.attach_generator_files(to_copy=str(input_case_path.absolute()))

    # ----------------------------------------------------------------
    # Configure and create the ML training model
    # ----------------------------------------------------------------

    training_rs = exp.create_run_settings(
        exe="python",
        exe_args=f"ml_model_training.py 1 {args.pinn_type}"
    )

    ml_model_training = exp.create_model(
        name="ml_model_training",
        run_settings=training_rs
    )
    ml_model_training.attach_generator_files(
        to_copy=["ml_model_training.py", "networks/MLP.py", "networks/PINN.py"]
    )

    exp.generate(ml_model_training, overwrite=True)

    # ----------------------------------------------------------------
    # Run the experiment
    # ----------------------------------------------------------------

    try:
        exp.start(db)
        print(f"Database started at: {db.get_address()}")
        print("Running the OpenFOAM case")
        exp.generate(openfoam_model, overwrite=True)
        exp.start(openfoam_model, ml_model_training, block=False)

        while True:
            time.sleep(1)
            if exp.get_status(openfoam_model)[0] in TERMINAL_STATUSES:
                exp.stop(ml_model_training)
                break

    except Exception as e:
        print("Caught an exception:", e)

    finally:
        exp.stop(db)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a SmartSim Machine-Learning mesh deformation experiment"
    )
    parser.add_argument(
        "--port", "-e",
        required=True,
        help="Port used by the db to communicate over, e.g. 8000-9000"
    )
    parser.add_argument(
        "--experiment", "-e",
        default="meshMotion",
        help="Name of the SmartSim experiment (e.g., mesh_deformation)"
    )
    parser.add_argument(
        "--mesh-solver-type",
        default="PINN",
        choices=["Laplace", "PINN"],
        help="The solver type for mesh motion"
    )
    parser.add_argument(
        "--case", "-c",
        default="ellipsoid3d_MachineLearningMeshMotionBase",
        help="Name of the OpenFOAM case folder (e.g., ellipsoid3D)"
    )
    parser.add_argument(
        "--platform",
        default="local",
        help="The platform on which this is being run"
    )
    parser.add_argument(
        "--pinn-type",
        default="Laplace3d",
        help="The type of PINN to use"
    )
    args = parser.parse_args()
    main(args)
