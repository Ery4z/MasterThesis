import os
from tqdm import tqdm
import numpy as np
import pickle
from PIL import Image


def check_timestamp(self):
        """Assert if some file cannot be putted in relation
        """
        
        available_pictures = [f for f in os.listdir(self.picture_dir) if f.split(".")[-1] == self.picture_extension_suffix]
        available_heatmaps = [f for f in os.listdir(self.heatmap_dir) if f.split(".")[-1] == self.heatmap_extension_suffix]
        
        iterable = tqdm(self.timestamps_to_load,desc="Checking file existence") if not self.silent else self.timestamps_to_load

        for timestamp in iterable:

            picture_name = self.picture_name_prefix + timestamp + "." + self.picture_extension_suffix
            
            heatmap_name = self.heatmap_name_prefix + timestamp + "." + self.heatmap_extension_suffix
            
            assert picture_name in available_pictures, "Picture {} does not exist".format(picture_name)
            assert heatmap_name in available_heatmaps, "Heatmap {} does not exist".format(heatmap_name)
    
def check_directory(self,directory):
        """Utility fucntion to check if the directory exist

        Args:
            directory (str): path
        """
        if not os.path.isdir(directory):
            assert os.path.isdir(directory), "Directory {} does not exist".format(directory)
    
def load_file(self,filename):
    """Utility function to load the data

    Args:
        filename (str): path

    Returns:
        np.array: data
    """
    with open(filename, "rb") as f:
        return pickle.load(f)

def load_heatmap_data(self):
    """Load the heatmap data to the good variable based on the path given on the class creation
    """
    tmp_data_list = []
    
    iterator = tqdm(self.timestamps_to_load,desc="Loading heatmap data") if not self.silent else self.timestamps_to_load

    
    for timestamp in iterator:
        
        heatmap_name = self.heatmap_name_prefix + timestamp + "." + self.heatmap_extension_suffix
        heatmap_path = os.path.join(self.heatmap_dir, heatmap_name)

        
        data = self.load_file(heatmap_path)
        data = np.abs(data)
        # data[data>10**7] = 0
        
        tmp_data_list.append(data)
    
        self.heatmap_data = np.array(tmp_data_list)
    
def load_picture_data(self):
    """Load the picture data to the good variable based on the path given on the class creation
    """
    tmp_data_list = []
    
    iterator = tqdm(self.timestamps_to_load,desc="Loading picture data") if not self.silent else self.timestamps_to_load
    
    for timestamp in iterator:
        
        picture_name = self.picture_name_prefix + timestamp + "." + self.picture_extension_suffix

        picture_path = os.path.join(self.picture_dir, picture_name)

        with Image.open(picture_path) as img:
            data = np.array(img)
        tmp_data_list.append(data)

    self.picture_data = np.array(tmp_data_list)
    self.picture_data_annotated = np.array(tmp_data_list)
    
    