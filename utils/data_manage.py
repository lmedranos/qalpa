import os
import shutil
from pathlib import Path

class DataManager:
    def __init__(self):
        pass
    
    def __init__(self, data_source, data_directory):
        self.data_source = Path(data_source)
        self.data_directory = Path(data_directory) 
        self.diffusion_data_dir = None
        self.diffusion_log_dir = None
        self.diffusion_test_dir = None
        self.diffusion_model_dir = None
        self.current_data = None
        self.evaluation_data = None
        self.query_data = None
        self.input_hdf5 = None
        self.output_hdf5 = None
        self.extra_prop_hdf5 = None
        self.itr_hdf5_path = None
        self.itr_extra_prop_path = None
        # Remember to set these flags based on your requirements
        self.hdf5_appending_flag = True
        self.hdf5_iter_flag = False
        self.extra_properties_flag = False
        

    def initialize_data_directory(self):
        """
        Create the initial data directory structure and copy the base dataset.
        """
        # Ensure the data directory exists
        self.data_directory.mkdir(parents=True, exist_ok=True)
        # Create HDF5 directory
        hdf5_dir = self.data_directory / "hdf5"
        # if hdf5_dir.exists():
        #     shutil.rmtree(hdf5_dir)
        hdf5_dir.mkdir(parents=True, exist_ok=True)
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
            if not self.output_hdf5.exists():
                shutil.copy(self.input_hdf5, self.output_hdf5)
                print(f"Created output HDF5 file: {self.output_hdf5} from input HDF5 file: {self.input_hdf5}")
            else:
                print(f"Output HDF5 file already exists: {self.output_hdf5}, resuming from it.")
        else:
            if self.output_hdf5.exists():
                print(f"Removing existing output HDF5 file: {self.output_hdf5}")
                self.output_hdf5.unlink()
            else:
                print(f"Output HDF5 file does not exist, creating a new one: {self.output_hdf5}")

            
            
        
        # Create diffusion temp directory if it doesn't exist
        self.diffusion_data_dir = self.data_directory / "diffusion_wdir" / "data"
        # if self.diffusion_data_dir.exists():
        #     shutil.rmtree(self.diffusion_data_dir)
        self.diffusion_data_dir.mkdir(parents=True, exist_ok=True)
        self.diffusion_log_dir = Path(self.data_directory, "diffusion_wdir", "logs")
        # if self.diffusion_log_dir.exists():
        #     shutil.rmtree(self.diffusion_log_dir)
        self.diffusion_log_dir.mkdir(parents=True, exist_ok=True)
        self.diffusion_test_dir = Path(self.data_directory, "diffusion_wdir", "test")
        # if self.diffusion_test_dir.exists():
        #     shutil.rmtree(self.diffusion_test_dir)
        self.diffusion_test_dir.mkdir(parents=True, exist_ok=True)

        # Create model registry directory
        self.diffusion_model_dir = self.data_directory / "diffusion_wdir" / "model_registry"
        # if self.diffusion_model_dir.exists():
        #     shutil.rmtree(self.diffusion_model_dir)
        self.diffusion_model_dir.mkdir(parents=True, exist_ok=True)
        

    def create_iteration_directories(self, iteration):
        """
        Create a directories for a specific iteration.
        
        Parameters:
        - iteration: The iteration number.
        """
        self.current_data = self.data_directory / f"iter_{iteration}"
        # if self.current_data.exists():
        #     shutil.rmtree(self.current_data)
        self.current_data.mkdir(parents=True, exist_ok=True)
        subfolders = ["generated", "relaxed", "query", "score_filtered", "prop_calc", "accuracy_filtered"]
        for subfolder in subfolders:
            (self.current_data / subfolder).mkdir(parents=True, exist_ok=True)
        self.query_data = self.current_data / "query"
        # Create filtered directory which will be deprecated in the future
        # (self.current_data / "filtered").mkdir(parents=True, exist_ok=True)

        if self.hdf5_iter_flag:
            (self.current_data / "hdf5").mkdir(parents=True, exist_ok=True)
            self.itr_hdf5_path = self.current_data / "hdf5" / "output.hdf5"
            if self.extra_properties_flag:
                self.itr_extra_prop_path = self.current_data / "hdf5" / "extra_properties.hdf5"
            
        print(f"Created iteration directory: {self.current_data}")

    def create_evaluation_directories(self, target_indx, models_indx):
        """
        Create evaluation directory structure.
        """
        self.evaluation_data = self.data_directory / "evaluation"
        self.evaluation_data.mkdir(parents=True, exist_ok=True)
        self.current_data = self.evaluation_data / f"target{target_indx}" / f"model{models_indx}"
        self.current_data.mkdir(parents=True, exist_ok=True)
        subfolders = ["generated", "relaxed", "query", "prop_calc", "results"]
        for subfolder in subfolders:
            (self.current_data / subfolder).mkdir(parents=True, exist_ok=True)
        print(f"Created evaluation directories under: {self.evaluation_data} for target {target_indx} and model {models_indx}.")
        self.query_data = self.current_data / "query"
        self.results = self.current_data / "results"
        
        

    def edit_config(self, config_file, key, value, update=True):
        """
        Edit the configuration file to include the model directory.
        
        Parameters:
        - config_file: Path to the configuration file.
        - model_dir: Directory of the model to be added to the config.
        """
        try:
            with config_file.open('r') as file:
                lines = file.readlines()

            # replace the value of the specified key
            key_found = False
            for i, line in enumerate(lines):
                if line.strip().startswith(key + ":"):
                    key_found = True
                    if update:
                        lines[i] = f"{key}: {value}\n"
                    else:
                        lines[i] = ""  # delete line
                    break  # no need to continue after key is found
                

            if not key_found and update:
                lines.append(f"{key}: {value}\n")

            # Write back to file
            with config_file.open('w') as file:
                file.writelines(lines)

        except Exception as e:
            print(f"Error updating config file: {e}")
    
    # This meyhode is going to be deprecated in favor of clean_dir
    def clean_files(self, dir_path: Path, file_type: str = ''):
        if not dir_path.exists():
            return
        
        if not dir_path.is_dir():
            print(f"Not a directory: {dir_path}")
            return
   
        files = list(dir_path.rglob(f"*{file_type}"))
        if not files:
            return
        
        for file in files:
            try:
                file.unlink()
            except Exception as e:
                print(f"Error deleting {file}: {e}")

    def clean_dir(self, dir_path: Path, file_type: str = '', exclude_suffix: str = ''):
        if not dir_path.exists():
            return

        if not dir_path.is_dir():
            print(f"Not a directory: {dir_path}")
            return
        
        
        if file_type and not file_type.startswith('.'):
            file_type = '.' + file_type
        if exclude_suffix and not exclude_suffix.startswith('.'):
            exclude_suffix = '.' + exclude_suffix

        if exclude_suffix:
            files = [f for f in dir_path.rglob('*') if f.is_file() and f.suffix != exclude_suffix]
        elif file_type:
            files = list(dir_path.rglob(f'*{file_type}'))
        else:
            files = [f for f in dir_path.rglob('*') if f.is_file()]

        if not files:
            return
        
        for file in files:
            try:
                file.unlink()
            except Exception as e:
                print(f"Error deleting {file}: {e}")


    def unlink_mol(self, fnames, dir_path: Path):
        if not dir_path.exists():
            return
        
        if not dir_path.is_dir():
            print(f"Not a directory: {dir_path}")
            return
        for fname in fnames:
            file_path = dir_path / f"{fname}.xyz"
            if file_path.exists():
                try:
                    file_path.unlink()
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")