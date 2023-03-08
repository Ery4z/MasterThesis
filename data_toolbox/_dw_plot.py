import matplotlib as plt
import numpy as np
from .utilities import slog

def plot_background(self,logarithmic=True):
        """Plot the background data

        Args:
            logarithmic (bool, optional): Should the data being logarithmed. Defaults to True.
        """
        if self.background_data is None : return

        plt.figure()
        plt.subplot(121)
        self.plot_radar_wrapper(self.background_data)
        plt.subplot(122)
        self.plot_radar_wrapper(slog(self.background_data),negative_heatmap=True)
        
        plt.show()
    
def plot_mean(self,logarithmic=True):
    """Plot the mean data

    Args:
        logarithmic (bool, optional): Should the data being logarithmed. Defaults to True.
    """
    if self.heatmap_mean is None : self.calculate_heatmap_mean()

    plt.figure()
    if logarithmic:
        plt.imshow(slog(self.heatmap_mean),cmap="gray")
    else:
        plt.imshow(np.abs(self.heatmap_mean),cmap="gray")
    plt.show()


def plot(self,index,logarithmic=False,sign_color_map=False):
    """Plot the couple

    Args:
        index (int): index of the couple
        logarithmic (bool, optional): Should heatmap being log. Defaults to False.
        sign_color_map (bool, optional): Negative data ?. Defaults to False.
    """
    plt.figure()
    plt.subplot(1,2,1)
    data = self.heatmap_data[index]
    data = self.filter(data)
    if logarithmic:
        data = slog(data)
    

    if sign_color_map:
        
        cmap, norm = self.get_color_map(data)
        plt.imshow(data,cmap=cmap,norm=norm)
        plt.colorbar()
    else:
        self.plot_radar_wrapper(data)
        plt.colorbar()
        
    plt.subplot(1,2,2)
    plt.imshow(self.picture_data[index])
    
    plt.title(self.timestamps_to_load[index])
    
    plt.show()
def plot_radar_wrapper(self,data,negative_heatmap=False):
    """Utility function to wrap a heatmap plot.

    Args:
        data (np.array): heatmap data
    """
    if negative_heatmap:
        cmap, norm = self.get_color_map(data)
        plt.imshow(data,cmap=cmap,norm=norm,extent=self.radar_parameters["speed"]+self.radar_parameters["distance"],aspect='auto')
    else:
        plt.imshow(data,cmap=self.color_map,extent=self.radar_parameters["speed"]+self.radar_parameters["distance"],aspect='auto')
    plt.xlabel("Speed (km/h)")
    plt.ylabel("Distance (m)")
    plt.grid(color="green",linestyle="--",linewidth=0.5)
    
    
def plot_CFAR(self,index,annotated=False):
    """Utility function to plot the heatmap, CFAR result and picture, if annotated is true it will plot the potentialy annotated version by yolo.
    WARNING: THIS FUNCTION DO NOT CALL YOLO. SEE analyse_couple

    Args:
        index (int): index of the couple
        annotated (bool, optional): Should use the potentionnaly annotated data ?. Defaults to False.
    """
    plt.figure()
    plt.subplot(1,3,1)
    data = self.heatmap_data[index]
    data = self.filter(data)
    
    cfar_data,_,spotted = self.calculate_CFAR(data)

    
    self.plot_radar_wrapper(data)
    plt.colorbar()
    plt.title("Heatmap")
    
    plt.subplot(1,3,2)
    self.plot_radar_wrapper(spotted)
    plt.title("CFAR")

    plt.subplot(1,3,3)
    if annotated:
        plt.imshow(self.picture_data_annotated[index])
    else:
        plt.imshow(self.picture_data[index],cmap=self.color_map)
    plt.title(self.timestamps_to_load[index])
    
    plt.show()

def plot_comparison_filter(self,index):
    """Utility function to plot the raw heatmap, the filtered heatmap and the picture.

    Args:
        index (int): index of the couple
    """
    plt.figure()
    plt.subplot(2,3,1)
    
    
    
    data = self.heatmap_data[index]
    data = data - self.background_data
    data = data - self.heatmap_mean

    data = np.maximum(data, 0)
    
    
    
    filtred_data = self.filter(data.copy())
    

    
    # self.plot_radar_wrapper(data,negative_heatmap=True)
    plt.imshow(data)
    plt.colorbar()
    plt.title("Heatmap raw")
    
    plt.subplot(2,3,4)
    log_data = data.copy()
    log_data = slog(log_data)
    self.plot_radar_wrapper(log_data,negative_heatmap=True)
    # plt.imshow(log_data)
    
    plt.colorbar()
    plt.title("Heatmap raw log")
    
    plt.subplot(2,3,2)
    self.plot_radar_wrapper(filtred_data)
    plt.title("Heatmap filtered")

    plt.subplot(2,3,3)
    
    plt.imshow(self.picture_data[index],cmap=self.color_map)
    plt.title(self.timestamps_to_load[index])
    
    plt.show()