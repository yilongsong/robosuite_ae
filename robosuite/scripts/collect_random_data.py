import numpy as np
import os
import time
import h5py
import json
from tqdm import tqdm
from glob import glob
import argparse  # Import argparse for command-line arguments

import robosuite as suite
from robosuite.controllers import load_controller_config
from robosuite.wrappers import DataCollectionWrapper

# Number of total steps to collect and demos to split them into
total_steps = 300000
num_steps_per_demo = 300
num_demos = total_steps // num_steps_per_demo

def gather_demonstrations_as_hdf5(directory, out_dir, env_info, file_name, env_args):
    """
    Gathers the demonstrations saved in @directory into a
    single hdf5 file.

    Args:
        directory (str): Path to the directory containing raw demonstrations.
        out_dir (str): Path to where to store the hdf5 file.
        env_info (str): JSON-encoded string containing environment information,
            including controller and robot info.
        file_name (str): The desired name of the HDF5 file.
        env_args (dict): The environment metadata to store in the HDF5 file.
    """
    hdf5_path = os.path.join(out_dir, file_name)
    f = h5py.File(hdf5_path, "w")

    # store some metadata in the attributes of one group
    grp = f.create_group("data")

    num_eps = -1
    env_name = None  # will get populated at some point

    for ep_directory in os.listdir(directory):
        state_paths = os.path.join(directory, ep_directory, "state_*.npz")
        states = []
        actions = []

        for state_file in sorted(glob(state_paths)):
            dic = np.load(state_file, allow_pickle=True)
            env_name = str(dic["env"])

            states.extend(dic["states"])
            for ai in dic["action_infos"]:
                actions.append(ai["actions"])

        if len(states) == 0:
            continue

        print("Saving demonstration")
        # Delete the last state. This is because when the DataCollector wrapper
        # recorded the states and actions, the states were recorded AFTER playing that action,
        # so we end up with an extra state at the end.
        del states[-1]
        assert len(states) == len(actions)

        num_eps += 1
        ep_data_grp = grp.create_group(f"demo_{num_eps}")

        # store model xml as an attribute
        xml_path = os.path.join(directory, ep_directory, "model.xml")
        with open(xml_path, "r") as xml_file:
            xml_str = xml_file.read()
        ep_data_grp.attrs["model_file"] = xml_str

        # write datasets for states and actions
        ep_data_grp.create_dataset("states", data=np.array(states))
        ep_data_grp.create_dataset("actions", data=np.array(actions))

    # write dataset attributes (metadata)
    now = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    grp.attrs["date"] = now.split()[0]
    grp.attrs["time"] = now.split()[1]
    grp.attrs["repository_version"] = suite.__version__
    grp.attrs["env"] = env_name
    grp.attrs["env_info"] = env_info

    # Add environment metadata (env_args) as an attribute
    grp.attrs["env_args"] = json.dumps(env_args)

    f.close()

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Collect demonstration data and save to HDF5 file.")
    parser.add_argument("--file_path", type=str, required=True, help="File path including the name to save the HDF5 file.")
    
    args = parser.parse_args()

    # Create options for the environment creation
    options = {}
    options["env_name"] = 'Lift'
    options["robots"] = 'Panda'
    options["controller_configs"] = load_controller_config(default_controller="OSC_POSE")

    # Get the directory and file name from the provided path
    file_path = args.file_path
    save_dir, hdf5_file_name = os.path.split(file_path)

    # Set directory to save the demos
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Initialize the environment
    env = suite.make(
        **options,
        has_renderer=False,            # No visual rendering
        has_offscreen_renderer=True,   # Use offscreen renderer to get images
        camera_names=["frontview", "agentview", "sideview"],    # Add both cameras
        camera_heights=128,
        camera_widths=128,
        use_camera_obs=True,           # Capture camera observations
        control_freq=20,
        ignore_done=True               # Ignore done flag to continue episodes
    )

    # Wrap the environment in the DataCollectionWrapper to enable data logging
    tmp_directory = "/tmp/{}".format(str(time.time()).replace(".", "_"))
    env = DataCollectionWrapper(env, directory=tmp_directory)

    # Get action limits
    low, high = env.action_spec

    # Collect data across demos
    for demo in range(num_demos):
        print(f"Collecting demo {demo}/{num_demos}...")

        # Reset the environment and start a new demo
        env.reset()

        # Execute random actions for each step in this demo
        for step in tqdm(range(num_steps_per_demo)):
            # Generate random action within the action limits
            action = np.random.uniform(low, high)

            # Step the environment with the random action
            env.step(action)

        # Flush any data still in memory to disk
        env._flush()

    # Define the env_args metadata format with "type" included
    env_args = {
        'env_name': options["env_name"],
        'type': 1,  # Example value, replace with appropriate type if needed
        'env_kwargs': {
            'robots': options["robots"],
            'controller_configs': options["controller_configs"],
            'camera_names': ["frontview", "agentview"],
            'camera_heights': 128,
            'camera_widths': 128,
            'control_freq': 20,
            'has_renderer': False,
            'has_offscreen_renderer': True,
            'use_camera_obs': True,
            'ignore_done': True
        }
    }

    # After all demos are collected, gather and save them in HDF5
    env_info = json.dumps(options)
    gather_demonstrations_as_hdf5(tmp_directory, save_dir, env_info, hdf5_file_name, env_args)

    print(f"Data collection completed. Data saved to: {file_path}")

    # Close the environment
    env.close()