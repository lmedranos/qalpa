import os
import shutil
from pathlib import Path

class DataManager: 
    def __init__(self, data_source, data_directory):
        self.data_source = Path(data_source)
        self.data_directory = Path(data_directory) 
        self.diffusion_data_dir = None
        self.diffusion_model_dir = None
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
        # Ensure the data directory exists
        self.data_directory.makedir(parents=True, exist_ok=True)
        # Create HDF5 directory
        hdf5_dir = self.data_directory / "hdf5"
        hdf5_dir.makedir(parents=True, exist_ok=True)
        # Copy the base dataset to HDF5 directory
        try:
            data_base_name = Path(self.data_source).stem
            self.input_hdf5 = hdf5_dir / f"{data_base_name}-input.hdf5"
            self.output_hdf5 = hdf5_dir / f"{data_base_name}-output.hdf5"
            self.extra_prop_hdf5 = hdf5_dir / f"{data_base_name}-extra_properties.hdf5" if self.extra_properties_flag else 'None'
            # Check if the input HDF5 file already exists
            if not self.input_hdf5.exists():
                shutil.copy(self.data_source, self.input_hdf5)
        except Exception as e:
            print(f"Error copying base dataset: {e}")
        
        # If appending to HDF5, copy the input HDF5 to output HDF5
        if self.hdf5_appending_flag:
            shutil.copy(self.input_hdf5, self.output_hdf5)
        
        # Create deffustion temp directory if it doesn't exist
        self.diffusion_data_dir = self.data_directory / "diffusion_wdir" / "data"
        self.diffusion_data_dir.mkdir(parents=True, exist_ok=True)
        self.diffusion_model_dir = self.data_directory / "diffusion_wdir" / "model_registry"
        self.diffusion_model_dir.mkdir(parents=True, exist_ok=True)
        Path(self.data_directory, "diffusion_wdir", "logs").mkdir(parents=True, exist_ok=True)


    def create_iteration_directories(self, iteration):
        """
        Create a directories for a specific iteration.
        
        Parameters:
        - iteration: The iteration number.
        """
        self.current_data = self.data_directory / f"iter_{iteration}"
        self.current_data.makedir(parents=True, exist_ok=True)
        # Create subdirectories for the iteration
        subfolders = ["generated", "relaxed", "filtered"]
        for subfolder in subfolders:
            (self.current_data / subfolder).mkdir(parents=True, exist_ok=True)

        if self.hdf5_iter_flag:
            (self.current_data / "hdf5").mkdir(parents=True, exist_ok=True)
            
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
    


