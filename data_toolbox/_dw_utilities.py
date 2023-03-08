import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import pickle


def get_color_map(self,data):
    """Create a custom colormap forcing the 0 to be inside, useful to represent image if the data is sometime negative.

    Args:
        data (np.array): Data to plot

    Returns:
        cmap,norm: result to put into imshow
    """
    # define the colormap
    cmap = plt.get_cmap('PuOr')

    # extract all colors from the .jet map
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # create the new map
    cmap = cmap.from_list('Custom cmap', cmaplist, cmap.N)

    # define the bins and normalize and forcing 0 to be part of the colorbar!
    bounds = np.arange(np.min(data),np.max(data),.5)
    idx=np.searchsorted(bounds,0)
    bounds=np.insert(bounds,idx,0)
    norm = BoundaryNorm(bounds, cmap.N)
    return cmap, norm



def set_background_data(self, background_data_path):
    """Load the background data based on a saved file

    Args:
        background_data_path (str): Path for the file containing the background data 
    """
    self.background_data = np.abs(np.array(self.load_file(background_data_path)))
    
def set_mean_heatmap_data(self, mean_data_path):
    """Load the mean heatmap data based on the path of the file created using the function save_mean_heatmap_data

    Args:
        mean_data_path (str): path
    """
    self.heatmap_mean = np.abs(np.array(self.load_file(mean_data_path)))

def save_mean_heatmap_data(self, mean_data_path):
    """Save the mean heatmap data of the loaded data to the path

    Args:
        mean_data_path (str): path
    """
    assert self.heatmap_mean is not None, "Please calculate the heatmap mean data first."
    with open(mean_data_path,"wb") as f:
        pickle.dump(self.heatmap_mean, f)
