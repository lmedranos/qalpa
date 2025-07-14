import os
import shutil
from pathlib import Path

class DataManager: 
    def __init__(self, data_source, data_directory):
        self.data_source = data_source
        self.data_directory = data_directory 
        self.current_data = None
        self.input_hdf5 = None
        self.output_hdf5 = None
        self.extra_prop_hdf5 = None
        self.hdf5_appending_flag = True
        self.hdf5_iter_flag = True
        self.extra_properties_flag = True
        

    def initialize_data_directory(self):
        """
        Create the initial data directory structure and copy the base dataset.
        """
        if not os.path.exists(self.data_directory):
            os.makedirs(self.data_directory)
        
        hdf5_dir = os.path.join(self.data_directory, "hdf5")
        if not os.path.exists(hdf5_dir):
            os.makedirs(hdf5_dir)
        
        try:
            data_base_name = os.path.splitext(os.path.basename(self.data_source))[0]
            self.input_hdf5 = os.path.join(hdf5_dir, f"{data_base_name}-input.hdf5")
            self.output_hdf5 = os.path.join(hdf5_dir, f"{data_base_name}-output.hdf5")
            self.extra_prop_hdf5 = os.path.join(hdf5_dir, f"{data_base_name}-extra_properties.hdf5") if self.extra_properties_flag else 'None'

            if not os.path.exists(self.input_hdf5):
                shutil.copy(self.data_source, self.input_hdf5)
                print(f"Copied base dataset to {self.input_hdf5}")
            else:
                print(f"Initial HDF5 file already exists at {self.input_hdf5}")
        except Exception as e:
            print(f"Error copying base dataset: {e}")
        
        if self.hdf5_appending_flag:
            shutil.copy(self.input_hdf5, self.output_hdf5)
            print(f"Copied input HDF5 to output HDF5 at {self.output_hdf5}")
        

    def create_iteration_directories(self, iteration):
        """
        Create a directories for a specific iteration.
        
        Parameters:
        - iteration: The iteration number.
        """
        self.current_data = os.path.join(self.data_directory, f"iter_{iteration}")
        if not os.path.exists(self.current_data):
            os.makedirs(self.current_data)
        # Create subdirectories for the iteration
        subfolders = ["splited", "generated", "relaxed", "filtered"]
        for subfolder in subfolders:
            subfolder_path = os.path.join(self.current_data, subfolder)
            if not os.path.exists(subfolder_path):
                os.makedirs(subfolder_path)
        if self.hdf5_iter_flag:
            hdf5_iter_path = os.path.join(self.current_data, "hdf5")
            if not os.path.exists(hdf5_iter_path):
                os.makedirs(hdf5_iter_path)
        print(f"Created iteration directory: {self.current_data}")

    def edit_config(self, config_path, model_dir, update_model_dir=False):
        """
        Edit the configuration file to include the model directory.
        
        Parameters:
        - config_path: Path to the configuration file.
        - model_dir: Directory of the model to be added to the config.
        """
        try:
            config_file = Path(config_path, 'diffusion_aqm.yaml')
            with open(config_file, 'r') as file:
                lines = file.readlines()

            # Remove existing 'load_ckpt_from_pretrained' lines
            lines = [line for line in lines if not line.strip().startswith("load_ckpt_from_pretrained:")]

            if update_model_dir:
                # Add new line at the beginning
                lines.insert(0, f"load_ckpt_from_pretrained: {model_dir}\n")

            # Write back to file
            with open(config_file, "w") as file:
                file.writelines(lines)

        except Exception as e:
            print(f"Error updating config file: {e}")
    


