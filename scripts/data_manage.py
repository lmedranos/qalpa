import os
import shutil

class DataManager: 
    def __init__(self, data_source, data_directory):
        self.data_source = data_source
        self.data_directory = data_directory 
        self.current_data = None
        self.next_data = None
        self.hdf5 = None
        #self.next_hdf5_dir = None
        

    def initialize_data_directory(self):
        """
        Create the initial data directory structure and copy the base dataset.
        """
        if not os.path.exists(self.data_directory):
            os.makedirs(self.data_directory)
        
        intial_data_dir = os.path.join(self.data_directory, "iter_0")
        if not os.path.exists(intial_data_dir):
            os.makedirs(intial_data_dir)

        # initial_hdf5_dir = os.path.join(intial_data_dir, "hdf5")
        # if not os.path.exists(initial_hdf5_dir):
        #     os.makedirs(initial_hdf5_dir)
        initial_hdf5_dir = self.data_directory
        
        try:
            data_base_name = os.path.basename(self.data_source)
            initial_hdf5 = os.path.join(initial_hdf5_dir, data_base_name)
            if not os.path.exists(initial_hdf5):
                shutil.copy(self.data_source, initial_hdf5)
                print(f"Copied base dataset to {initial_hdf5}")
                self.hdf5 = initial_hdf5
            else:
                print(f"Initial HDF5 file already exists at {initial_hdf5}")
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
        subfolders = ["splited", "generated", "relaxed", "filtered"]
        for subfolder in subfolders:
            subfolder_path = os.path.join(self.current_data, subfolder)
            if not os.path.exists(subfolder_path):
                os.makedirs(subfolder_path)
        print(f"Created iteration directory: {self.current_data}")
        # Create next iteration directory
        self.next_data = os.path.join(self.data_directory, f"iter_{iteration + 1}")
        if not os.path.exists(self.next_data):
            os.makedirs(self.next_data)
        # self.next_hdf5_dir = os.path.join(self.next_data, "hdf5")
        # if not os.path.exists(self.next_hdf5_dir):
        #     os.makedirs(self.next_hdf5_dir)

    


