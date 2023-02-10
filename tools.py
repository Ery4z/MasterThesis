""" 
Thomas Bolteau - February 2023
"""

import numpy as np
import pickle
from scipy.ndimage import binary_dilation
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import random as rd
from matplotlib.colors import BoundaryNorm
import scipy


# from LELEC2885 - Image Proc. & Comp. Vision
def resize_and_fix_origin(kernel, size):
    pad0, pad1 = size[0]-kernel.shape[0], size[1]-kernel.shape[1]
    shift0, shift1 = (kernel.shape[0]-1)//2, (kernel.shape[1]-1)//2
    kernel = np.pad(kernel, ((0, pad0), (0, pad1)), mode='constant')
    kernel = np.roll(kernel, (-shift0, -shift1), axis=(0, 1))
    return kernel

k, l, m = 10, 30, 3
val = 1 / ((2*m+1)*(l-k))

kernelG = np.zeros((2*l+1, 2*l+1))
kernelG[l-m:l+m+1, :l-k] = val

kernelD = np.zeros((2*l+1, 2*l+1))
kernelD[l-m:l+m+1, l+k+1:] = val

kernelH = np.zeros((2*l+1, 2*l+1))
kernelH[:l-k, l-m:l+m+1] = val

kernelB = np.zeros((2*l+1, 2*l+1))
kernelB[l+k+1:, l-m:l+m+1] = val

# from LELEC2885 - Image Proc. & Comp. Vision
def fast_convolution(image, kernel):
    kernel_resized = resize_and_fix_origin(kernel, image.shape)
    image_fft = np.fft.fft2(image)
    kernel_fft = np.fft.fft2(kernel_resized)
    result = np.fft.ifft2(image_fft * kernel_fft)
    return np.real(result)

def load_file(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)
    
def slog(data):
    return np.nan_to_num(np.log(np.abs(data))*np.sign(data))


def triangle_kernel(kerlen):
    r = np.arange(kerlen)
    kernel1d = (kerlen + 1 - np.abs(r - r[::-1])) / 2
    kernel2d = np.outer(kernel1d, kernel1d)
    kernel2d /= kernel2d.sum()
    return kernel2d
    
    
class DataWrapper:
    def __init__(self, heatmap_dir, picture_dir, timestamps_to_load, picture_name_prefix="", picture_extension_suffix="jpeg", heatmap_extension_suffix="doppler", heatmap_name_prefix=""):

        self.radar_parameters = {
            "distance":[70,0],
            "speed":[-100,100],
        }

        self.heatmap_dir = heatmap_dir
        self.picture_dir = picture_dir
        self.timestamps_to_load = timestamps_to_load
        
        self.heatmap_name_prefix = heatmap_name_prefix

        self.picture_name_prefix = picture_name_prefix
        self.picture_extension_suffix = picture_extension_suffix
        self.heatmap_extension_suffix = heatmap_extension_suffix
        
        self.heatmap_data = np.array([])
        self.picture_data = np.array([])
        
        self.background_data = None

        
        self.heatmap_mean = None
        
        self.color_map = plt.get_cmap('gray')



        self.check_directory(self.heatmap_dir)
        self.check_directory(self.picture_dir)
        self.check_timestamp()
        
        self.load_heatmap_data()
        self.load_picture_data()
    
    def filter(self,data):
        self.calculate_heatmap_mean()
        data = data - self.background_data
        data = data - self.heatmap_mean

        data = np.maximum(data, 0)
        
        kernel = triangle_kernel(3)
        filtred_bg = scipy.signal.convolve2d(self.background_data, kernel, mode='same')
        
        
        data = scipy.signal.convolve2d(data, kernel, mode='same')
        
        # data = data - filtred_bg


        
        return data
        
    def get_color_map(self,data):
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
        self.background_data = np.abs(np.array(self.load_file(background_data_path)))
    
    def remove_background_data(self):
        self.heatmap_data = self.heatmap_data - self.background_data

    
    def load_file(self,filename):
        with open(filename, "rb") as f:
            return pickle.load(f)

    def load_heatmap_data(self):
        tmp_data_list = []

        
        for timestamp in tqdm(self.timestamps_to_load,desc="Loading heatmap data"):
            
            heatmap_name = self.heatmap_name_prefix + timestamp + "." + self.heatmap_extension_suffix
            heatmap_path = os.path.join(self.heatmap_dir, heatmap_name)

            
            data = self.load_file(heatmap_path)
            tmp_data_list.append(np.abs(data))
        
        self.heatmap_data = np.array(tmp_data_list)
    
    def load_picture_data(self):
        tmp_data_list = []
        
        for timestamp in tqdm(self.timestamps_to_load,desc="Loading picture data"):
            
            picture_name = self.picture_name_prefix + timestamp + "." + self.picture_extension_suffix

            picture_path = os.path.join(self.picture_dir, picture_name)

            
            data = np.array(Image.open(picture_path))
            tmp_data_list.append(data)
    
        self.picture_data = np.array(tmp_data_list)
        
    def check_directory(self,directory):
        if not os.path.isdir(directory):
            assert os.path.isdir(directory), "Directory {} does not exist".format(directory)
    
    def calculate_heatmap_mean(self):
        self.heatmap_mean = np.mean(self.heatmap_data,axis=0)

    
    def remove_temporal_mean(self):
        self.calculate_heatmap_mean()
        self.heatmap_data = self.heatmap_data - self.heatmap_mean
        
    def CFAR_loaded(self,data):
        """Calculate the CFAR

        Args:
            file_path (str): The path to the file to be processed. Has to be a .doppler file.
            output_dir (str): The path to the directory where the output files will be saved.
        """    
        signal = data

        moyG = fast_convolution(signal, kernelG)
        moyD = fast_convolution(signal, kernelD)
        moyH = fast_convolution(signal, kernelH)
        moyB = fast_convolution(signal, kernelB)
        maxmoy = np.maximum.reduce([moyG, moyD, moyH, moyB])

        magncfar = fast_convolution(signal, np.ones((4, 4))/16) - 1.1*maxmoy
        threshcfar = max(30, 1+0.5*(np.max(magncfar)-1))
        loccfar = np.where(magncfar >= threshcfar)

        spotted = np.zeros(signal.shape)
        spotted[loccfar] = 1

        
        return magncfar, loccfar, spotted
        


    def check_timestamp(self):
        
        available_pictures = [f for f in os.listdir(self.picture_dir) if f.split(".")[-1] == self.picture_extension_suffix]
        available_heatmaps = [f for f in os.listdir(self.heatmap_dir) if f.split(".")[-1] == self.heatmap_extension_suffix]

        for timestamp in tqdm(self.timestamps_to_load,desc="Checking file existence"):

            picture_name = self.picture_name_prefix + timestamp + "." + self.picture_extension_suffix
            
            heatmap_name = self.heatmap_name_prefix + timestamp + "." + self.heatmap_extension_suffix
            
            assert picture_name in available_pictures, "Picture {} does not exist".format(picture_name)
            assert heatmap_name in available_heatmaps, "Heatmap {} does not exist".format(heatmap_name)
    
    def plot_background(self,logarithmic=True):
        if self.background_data is None : return

        plt.figure()
        if logarithmic:
            plt.imshow(slog(self.background_data),cmap="gray")
        else:
            plt.imshow(np.abs(self.background_data),cmap="gray")
        plt.show()
    
    def plot_mean(self,logarithmic=True):
        if self.heatmap_mean is None : self.calculate_heatmap_mean()

        plt.figure()
        if logarithmic:
            plt.imshow(slog(self.heatmap_mean),cmap="gray")
        else:
            plt.imshow(np.abs(self.heatmap_mean),cmap="gray")
        plt.show()

    
    def plot(self,index,logarithmic=False,sign_color_map=False):
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
    def plot_radar_wrapper(self,data):
        plt.imshow(data,cmap=self.color_map,extent=self.radar_parameters["speed"]+self.radar_parameters["distance"],aspect='auto')
        plt.xlabel("Speed (km/h)")
        plt.ylabel("Distance (m)")
        plt.grid(color="orange",linestyle="--",linewidth=0.5)
        
        
    def plot_CFAR(self,index):
        plt.figure()
        plt.subplot(1,3,1)
        data = self.heatmap_data[index]
        data = self.filter(data)
        
        cfar_data,_,spotted = self.CFAR_loaded(data)

        
        self.plot_radar_wrapper(data)
        plt.colorbar()
        plt.title("Heatmap")
        
        plt.subplot(1,3,2)
        self.plot_radar_wrapper(spotted)
        plt.title("CFAR")

        plt.subplot(1,3,3)
        plt.imshow(self.picture_data[index],cmap=self.color_map)
        plt.title(self.timestamps_to_load[index])
        
        plt.show()

        



        


if __name__ == "__main__":
    FILE_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
    FILE_COUNT_TO_LOAD = 1000
    
    BACKGROUND_FILE = os.path.join(FILE_DIRECTORY,"data","graphes","1657880906.775549.doppler")
    
    

    heatmap_directory = os.path.join(FILE_DIRECTORY, "data","graphes")
    picture_directory = os.path.join(FILE_DIRECTORY, "data","images")
    
    

    timestamps_to_load = list([".".join(f.split(".")[:2]) for f in os.listdir(heatmap_directory) if "doppler" in f])
    timestamps_to_load = timestamps_to_load[0:min(FILE_COUNT_TO_LOAD,len(timestamps_to_load))]

    

    
    dataWrapper = DataWrapper(heatmap_directory, picture_directory, timestamps_to_load, picture_name_prefix="0", picture_extension_suffix="jpeg", heatmap_extension_suffix="doppler", heatmap_name_prefix="")
    
    dataWrapper.set_background_data(BACKGROUND_FILE)
    # dataWrapper.remove_background_data()

    
    
    random_sample = rd.sample(range(FILE_COUNT_TO_LOAD),10)
    
    for i in random_sample:
        # dataWrapper.plot(i,logarithmic=False,sign_color_map=False)
        dataWrapper.plot_CFAR(i)


    