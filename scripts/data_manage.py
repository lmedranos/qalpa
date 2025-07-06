import os
import shutil

class DataManager: 
    def __init__(self, data_source, data_directory):
        self.data_source = data_source
        self.data_directory = data_directory 
        self.current_data = None
        self.input_hdf5 = None
        self.output_hdf5 = None
        

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
            data_base_name = os.path.basename(self.data_source)
            self.input_hdf5 = os.path.join(hdf5_dir, f"{data_base_name}-input.hdf5")
            self.output_hdf5 = os.path.join(hdf5_dir, f"{data_base_name}-output.hdf5")
            if not os.path.exists(self.input_hdf5):
                shutil.copy(self.data_source, self.input_hdf5)
                print(f"Copied base dataset to {self.input_hdf5}")
            else:
                print(f"Initial HDF5 file already exists at {self.input_hdf5}")
        except Exception as e:
            print(f"Error copying base dataset: {e}")


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
        subfolders = ["splited", "generated", "relaxed", "filtered", "hdf5"]
        for subfolder in subfolders:
            subfolder_path = os.path.join(self.current_data, subfolder)
            if not os.path.exists(subfolder_path):
                os.makedirs(subfolder_path)
        print(f"Created iteration directory: {self.current_data}")


    


