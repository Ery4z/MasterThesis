""" 
Thomas Bolteau - February 2023
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import numpy as np
import pickle
from scipy.ndimage import binary_dilation
import os
import matplotlib
import matplotlib.backends
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import random as rd
from matplotlib.colors import BoundaryNorm
import scipy
import torch
import cv2
import math
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

CAMERA_PARAM = {'focal_length':5695.8, 'image_size':(1920,1080),'principal_point':(1920/2,1080/2), 'fov':30}
FILE_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
def calculate_energy(data):
    """Calculate the energy of an array. Useful to know if the data is meaningful

    Args:
        data (np.array): Array to calculate energy from

    Returns:
        float: Energy of the array
    """
    return np.sum(np.abs(data)**2)

def load_analysis_result(path:str):
    heatmap_data = pickle.load(open(path + "/detected_vehicle_heatmap.pkl", "rb"))    
    image_data = pickle.load(open(path + "/detected_vehicle_image.pkl", "rb"))
    energy = pickle.load(open(path + "/energy_heatmap.pkl", "rb"))
    pos_list = pickle.load(open(path + "/pos_list.pkl", "rb"))
    missmatch = pickle.load(open(path + "/count_missmatch.pkl", "rb"))
    
    return heatmap_data, image_data, energy, pos_list ,missmatch

def plot_analysis_result(heatmap_data, image_data, energy, pos_list,missmatch, camera_parameters):
    plot_3d_world_pos(pos_list,camera_parameters)
    
    plt.figure()
    plt.subplot(2,2,1)
    plt.hist(heatmap_data,bins=10)
    plt.title("Number of vehicle detected by heatmap")
    plt.subplot(2,2,2)
    plt.hist(image_data,bins=10)
    plt.title("Number of vehicle detected by image")
    plt.subplot(2,2,3)
    plt.hist(slog(energy),bins=10)
    plt.title("Energy of the heatmap")
    plt.subplot(2,2,4)
    plt.hist(missmatch,bins=(max(missmatch)-min(missmatch)))
    plt.title("Number of missmatch between heatmap and image")
    plt.show()

def get_yolo():
    '''At the time of writign this a bug happen after importing yolo (impossible to plot with matplotlib) this is the fix'''
    b = plt.get_backend()
    model = torch.hub.load("ultralytics/yolov5", "yolov5s")
    matplotlib.use(b)
    return model

class MultimodalAnalysisResult:
    def __init__(self,timestamp:str,heatmap_energy:float,heatmap_info:list,raw_heatmap_info:list,image_info:list):

        self.timestamp = timestamp
        self.heatmap_energy = heatmap_energy
        self.heatmap_info = heatmap_info
        self.raw_heatmap_info = raw_heatmap_info
        self.image_info = image_info


def plot_3d_world_pos(pos_list,camera_parameters):
    def plot_camera(ax,camera_parameters):
        
        
        v = np.array([[0, 0.3, 0.3], [0, 0.3, -0.3], [0, 0.3, -0.3],  [0, -0.3, -0.3], [-0.3, 0, 0]])
        ax.scatter3D(v[:, 0], v[:, 1], v[:, 2])
        
        verts = [ [v[0],v[1],v[4]], [v[0],v[2],v[4]],
        [v[2],v[1],v[4]], [v[2],v[3],v[4]], [v[0],v[1],v[2],v[3]]]
        
        ax.add_collection3d(Poly3DCollection(verts, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))

        camera_center = np.array([-0.3, 0, 0])
        fovY = camera_parameters['fov']
        image_size = camera_parameters['image_size']
        fovZ = fovY * image_size[1] / image_size[0]
        distanceX = 70
        
        point_FOV = [camera_center+np.array([70,distanceX*math.tan(fovY/2),-1]),
                    camera_center+np.array([70,distanceX*math.tan(fovY/2),10]),
                    camera_center+np.array([70,-distanceX*math.tan(fovY/2),10]),
                    camera_center+np.array([70,-distanceX*math.tan(fovY/2),-1])]
        lines = [ [camera_center,point_FOV[0]], [camera_center,point_FOV[1]], [camera_center,point_FOV[2]], [camera_center,point_FOV[3]]]
        for l in lines:
            ax.plot3D(*zip(*l), color='r')
    
    
    to_plot = np.array(pos_list)
    #Plot camera
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    
    
    ax.scatter3D(to_plot[:,0],to_plot[:,1],to_plot[:,2])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    
    plot_camera(ax,camera_parameters)
    ax.set_box_aspect(aspect=(1, 1, 1))
    plt.show()

    
    


def from_multimodal_analysis_result_to_3d(analysis:MultimodalAnalysisResult,camera_parameters:dict):
    """Convert the data extracted from the image and heatmap to 3D world position.

    Args:
        analysis (MultimodalAnalysisResult): Data returned from the analysis
        camera_parameters (dict): Intrinsict parameters from the camera

    Returns:
        [3]float: 3d pos of the item
    """
    fx, fy = camera_parameters['focal_length'], camera_parameters['focal_length']
    cx = camera_parameters['principal_point'][0]
    cy = camera_parameters['principal_point'][1]
    
    image_size = camera_parameters['image_size']
    fovX = camera_parameters['fov']
    fovY = fovX * image_size[1] / image_size[0]
    
    
    distance = analysis.heatmap_info[0]["distance"]
    x = analysis.image_info[0]["x"]
    y = analysis.image_info[0]["y"]
    
    yaw = math.atan2(x-cx, fx)
    pitch = math.atan2(y-cy, fy)
    
    absolute_position = [distance * math.cos(pitch) * math.cos(yaw), -distance * math.cos(pitch) * math.sin(yaw), distance * math.sin(pitch)]
    
    return absolute_position
    


class CounterVehicleCFAR:
    """This class is the implementation of the famous island problem. 
    As the Spotted array returned by CFAR is a 0 and 1 mask in the heatmap, counting the number of vehicle is exactly this problem.
    """
    def countVehicle(self, grid) -> int:
        m = len(grid)
        n = len(grid[0])
        result = 0
        index_list = []
        shape_list = []
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    new_shape = []
                    result += 1
                    index_list.append([i, j])
                    
                    self.waterLands(grid, i, j, new_shape)
                    shape_list.append(list(new_shape))
        return result, index_list, shape_list
        
    
    #given the grid and start point, submerge the adjacent land
    def waterLands(self, grid, i, j,new_shape):
        m = len(grid)
        n = len(grid[0])
        grid[i][j] = 0
        new_shape.append([i,j])
        queue = [[i, j]]
        x_dir = [0,0,1,-1]
        y_dir = [1,-1,0,0]
        
        while queue:
            curr = queue.pop(0)
            cx = curr[0]
            cy = curr[1]
            
            for i in range(4):
                nx = cx + x_dir[i]
                ny = cy + y_dir[i]
                
                if nx >= 0 and nx < m and ny >= 0 and ny < n and grid[nx][ny] == 1:
                    queue.append([nx, ny])
                    grid[nx][ny] = 0
                    new_shape.append([nx,ny])


def getPositionFromShapeAndData(shape, data):
    """TODO: Calculate the incertitude
    Given a list representing a shape of an object in the heatmap,
    return the position of the individual most energetic pixel in this object

    Args:
        shape_list ([[2]int]): list of index of the shape to analyse
        data ([256][256]int): heatmap_data

    Returns:
        [2]int: index of the most energetic pixel
    """
    most_energetic = shape[0]
    top_energy = data[most_energetic[0]][most_energetic[1]]
    
    for pixel_pos in shape:
        if data[pixel_pos[0]][pixel_pos[1]] > top_energy:
            most_energetic = pixel_pos
            top_energy = data[pixel_pos[0]][pixel_pos[1]]
    return most_energetic


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
    """Utility function to get the log value without loosing the sign

    Args:
        data (np.array): array

    Returns:
        np.array: array
    """
    out = np.where(data > 0) 
    return np.nan_to_num(np.log(np.abs(data[out]))*np.sign(data[out]))


def triangle_kernel(kerlen):
    """Generate a 2D triangle kernel given the length

    Args:
        kerlen (int): length of the kernel

    Returns:
        np.array([kerlen]float): Kernel
    """
    r = np.arange(kerlen)
    kernel1d = (kerlen + 1 - np.abs(r - r[::-1])) / 2
    kernel2d = np.outer(kernel1d, kernel1d)
    kernel2d /= kernel2d.sum()
    return kernel2d
    
    
class DataWrapper:
    """This class is used to wrap heatmap data and picture data.
    Motivation: When this work started, analysing the filtering of the heatmap was not easy. It was necessary to compare it with real picture.
    As teh data is multimodal, associating the two is not stupid and having utilities to compare the two modality for a certain time index is quite useful.
    """
    def __init__(self, heatmap_dir, picture_dir, timestamps_to_load, picture_name_prefix="", picture_extension_suffix="jpeg", heatmap_extension_suffix="doppler", heatmap_name_prefix=""):
        """Generate the class

        Args:
            heatmap_dir (str): Path of the heatmap directory
            picture_dir (str): Path of the picture directory
            timestamps_to_load ([]str]): List of file to load, it has to be the minimal common string between the two modalities, see the prefix variable.
            picture_name_prefix (str, optional): Prefix for the picture filename, in case of timestamp, this may be "0". Defaults to "".
            picture_extension_suffix (str, optional): Extension file for the picture. Defaults to "jpeg".
            heatmap_extension_suffix (str, optional): Extension file for the heatmap. Defaults to "doppler".
            heatmap_name_prefix (str, optional): Prefix for the heatmap filename, usually "". Defaults to "".
        """

        self.radar_parameters = {
            "distance":[70,0],
            "speed":[-27.78,27.78],
        }
        self.camera_parameters = CAMERA_PARAM

        self.heatmap_dir = heatmap_dir
        self.picture_dir = picture_dir
        self.timestamps_to_load = timestamps_to_load
        
        self.heatmap_name_prefix = heatmap_name_prefix

        self.picture_name_prefix = picture_name_prefix
        self.picture_extension_suffix = picture_extension_suffix
        self.heatmap_extension_suffix = heatmap_extension_suffix
        
        self.heatmap_data = np.array([])
        self.picture_data = np.array([])
        self.picture_data_annotated = np.array([])
        
        self.background_data = None

        
        self.heatmap_mean = None
        
        self.color_map = plt.get_cmap('gray')
        
        self.allowed_class = [1,2,3,5,7] # 1:bicycle, 2:car, 3:motorcycle, 5:bus, 7:truck

        self.yolo = get_yolo()  # or yolov5n - yolov5x6, custom
        self.yolo.classes = self.allowed_class
        self.conf = 0.4

        self.check_directory(self.heatmap_dir)
        self.check_directory(self.picture_dir)
        self.check_timestamp()
        
        self.load_heatmap_data()
        self.load_picture_data()
    
    def filter(self,data):
        """Main function to modify the filtering of the heatmap data. You may overwrite this function.

        Args:
            data (np.array): Data to filter

        Returns:
            np.array: Filtered Data
        """
        if self.heatmap_mean is None:
            self.calculate_heatmap_mean()
        data = data - self.background_data
        data = data - self.heatmap_mean

        data = np.maximum(data, 0)
        
        kernel = triangle_kernel(3)
        filtred_bg = scipy.signal.convolve2d(self.background_data, kernel, mode='same')
        
        
        data = scipy.signal.convolve2d(data, kernel, mode='same')
        
        # data = data - filtred_bg
        return data
    
    def set_filter(self,customfilter):
        """Set the filter to use for the heatmap data

        Args:
            filter (function): Filter function to use
        """
        self.filter = customfilter
    
    def calculate_energy(self,data):
        """Calculate the energy of an array. Useful to know if the data is meaningful

        Args:
            data (np.array): Array to calculate energy from

        Returns:
            float: Energy of the array
        """
        return np.sum(np.abs(data)**2)
    
    def analyse_image(self,index):
        """Use yolo to analyse the image at the given index.

        Args:
            index (int): index to analyse

        Returns:
            yolov5 result: Torch Hub result for yolov5 prediction
        """

        # Model
        

        # Images
        img = self.picture_data[index]  # PIL, OpenCV, numpy, Tensor, etc.

        # Inference
        results = self.yolo(img)

        # Results
        return results  # or .show(), .save(), .crop(), .pandas(), etc.
    
    def add_annotation(self,index,image_info):
        """Add the bounding box and annotation to the annotated image based on the index and data created by yolo

        Args:
            index (int): index of the image 
            image_info (pd.Dataframe): yolo prediction result converted to df
        """
        image = self.picture_data_annotated[index]
        thickness = 5
        
        for info in image_info:
            x1, y1, x2, y2 = info["bbox"]
            image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), thickness)
            image = cv2.putText(image, info["class"], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 8)
            image = cv2.putText(image, "{:.2f}".format(round(info["confidence"], 2)), (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 4, (255,0, 0), 8)
        
        self.picture_data_annotated[index] = image
        
        
    
    
    def analyse_couple(self,index,plot=False,cfar_threshold=30):
        """Utility function to analyse the couple of data.
        This function is not mean for data pipeline but for user end.

        Args:
            index (int): index of the couple
            plot (bool, optional): Plot or not the couple. Defaults to False.

        Returns:
            dict: The analyse
        """
        analyse = {
            "timestamp":self.timestamps_to_load[index],
            "heatmap_energy":0,
            "heatmap_info":[{"distance":0,"speed":0}]
        }
        
        solver = CounterVehicleCFAR()
        heatmap_data = self.heatmap_data[index]
        filtered_heatmap_data = self.filter(heatmap_data)
        
        cfar_data, _, spotted = self.CFAR_loaded(filtered_heatmap_data,threshold=cfar_threshold)
        
        result, index_list, shape_list = solver.countVehicle(spotted)
        
        vehicle_heatmap_info = []
        raw_vehicle_heatmap_info =[]
        
        heatmap_shape = self.heatmap_data.shape
        
        rdist = self.radar_parameters["distance"]
        rspeed = self.radar_parameters["speed"]
        if (len(shape_list)<=3):
            for shape in shape_list:
                pos = getPositionFromShapeAndData(shape, filtered_heatmap_data)
                
            
                
                
                raw_vehicle_heatmap_info.append(pos)
                # The weird index is because the [0] is the vertical axis and the [1] is the horizontal axis
                # the vertical axis is counted from top to bottom and the horizontal axis is counted from left to right
                vehicle_heatmap_info.append({
                    "distance": (pos[0]/heatmap_shape[1])*(rdist[0]-rdist[1])+rdist[1],
                    "speed": (pos[1]/heatmap_shape[2])*(rspeed[1]-rspeed[0])+rspeed[0]
                })
        
        analyse["heatmap_energy"] = self.calculate_energy(filtered_heatmap_data)
        analyse["heatmap_info"] = vehicle_heatmap_info
        analyse["raw_heatmap_info"] = raw_vehicle_heatmap_info
        
        yolo = self.analyse_image(index)
        yolo_result = yolo.pandas().xyxy[0]
        image_info = []
        for i in range(len(yolo_result.xmin)):
            image_info.append({
                "bbox": [int(yolo_result.xmin[i]),int(yolo_result.ymin[i]),int(yolo_result.xmax[i]),int(yolo_result.ymax[i])],
                "x": int((yolo_result.xmin[i]+yolo_result.xmax[i])/2),
                "y": int((yolo_result.ymin[i]+yolo_result.ymax[i])/2),
                "class": yolo_result.name[i],
                "confidence": yolo_result.confidence[i]
            })
        analyse["image_info"] = image_info
        
        # Get the image with bounding boxes
        if plot:
            # Extract the image to a numpy array
            print(analyse)
            self.add_annotation(index,image_info)
            # bb_array = np.array(Image.open("tmp.jpeg"))
            # self.picture_data_annotated[index] = bb_array
            self.plot_CFAR(index, annotated=True)

        struct_analyse = MultimodalAnalysisResult(analyse["timestamp"],analyse["heatmap_energy"],analyse["heatmap_info"],analyse["raw_heatmap_info"],analyse["image_info"])
        
        return struct_analyse
        
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
    
    def remove_background_data(self):
        """Subtract the background data from the loaded heatmap data
        """
        self.heatmap_data = self.heatmap_data - self.background_data

    
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

        
        for timestamp in tqdm(self.timestamps_to_load,desc="Loading heatmap data"):
            
            heatmap_name = self.heatmap_name_prefix + timestamp + "." + self.heatmap_extension_suffix
            heatmap_path = os.path.join(self.heatmap_dir, heatmap_name)

            
            data = self.load_file(heatmap_path)
            tmp_data_list.append(np.abs(data))
        
        self.heatmap_data = np.array(tmp_data_list)
    
    def load_picture_data(self):
        """Load the picture data to the good variable based on the path given on the class creation
        """
        tmp_data_list = []
        
        for timestamp in tqdm(self.timestamps_to_load,desc="Loading picture data"):
            
            picture_name = self.picture_name_prefix + timestamp + "." + self.picture_extension_suffix

            picture_path = os.path.join(self.picture_dir, picture_name)

            with Image.open(picture_path) as img:
                data = np.array(img)
            tmp_data_list.append(data)
    
        self.picture_data = np.array(tmp_data_list)
        self.picture_data_annotated = np.array(tmp_data_list)
        
        
    def check_directory(self,directory):
        """Utility fucntion to check if the directory exist

        Args:
            directory (str): path
        """
        if not os.path.isdir(directory):
            assert os.path.isdir(directory), "Directory {} does not exist".format(directory)
    
    def calculate_heatmap_mean(self):
        """Calculate the mean and assign it to the good variable
        """
        self.heatmap_mean = np.mean(self.heatmap_data,axis=0)

    
    def remove_temporal_mean(self):
        """Remove the mean of the loaded data (Please load the mean data before using set_mean_heatmap_data)
        """
        if self.heatmap_mean is None:
            self.calculate_heatmap_mean()
        self.heatmap_data = self.heatmap_data - self.heatmap_mean
        
    def CFAR_loaded(self,data,threshold=30):
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
        threshcfar = max(threshold, 1+0.5*(np.max(magncfar)-1))
        loccfar = np.where(magncfar >= threshcfar)

        spotted = np.zeros(signal.shape)
        spotted[loccfar] = 1

        
        return magncfar, loccfar, spotted
        


    def check_timestamp(self):
        """Assert if some file cannot be putted in relation
        """
        
        available_pictures = [f for f in os.listdir(self.picture_dir) if f.split(".")[-1] == self.picture_extension_suffix]
        available_heatmaps = [f for f in os.listdir(self.heatmap_dir) if f.split(".")[-1] == self.heatmap_extension_suffix]

        for timestamp in tqdm(self.timestamps_to_load,desc="Checking file existence"):

            picture_name = self.picture_name_prefix + timestamp + "." + self.picture_extension_suffix
            
            heatmap_name = self.heatmap_name_prefix + timestamp + "." + self.heatmap_extension_suffix
            
            assert picture_name in available_pictures, "Picture {} does not exist".format(picture_name)
            assert heatmap_name in available_heatmaps, "Heatmap {} does not exist".format(heatmap_name)
    
    def plot_background(self,logarithmic=True):
        """Plot the background data

        Args:
            logarithmic (bool, optional): Should the data being logarithmed. Defaults to True.
        """
        if self.background_data is None : return

        plt.figure()
        if logarithmic:
            plt.imshow(slog(self.background_data),cmap="gray")
        else:
            plt.imshow(np.abs(self.background_data),cmap="gray")
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
    def plot_radar_wrapper(self,data):
        """Utility function to wrap a heatmap plot.

        Args:
            data (np.array): heatmap data
        """
        plt.imshow(data,cmap=self.color_map,extent=self.radar_parameters["speed"]+self.radar_parameters["distance"],aspect='auto')
        plt.xlabel("Speed (km/h)")
        plt.ylabel("Distance (m)")
        plt.grid(color="orange",linestyle="--",linewidth=0.5)
        
        
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
        
        cfar_data,_,spotted = self.CFAR_loaded(data)

        
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

    def pipeline_process(self,index,debug=False,cfar_threshold=30):
        result_analysis = self.analyse_couple(index, plot=False, cfar_threshold=cfar_threshold)
        
        # Check if the number of detected object match the number of object in the picture
        detected_heatmap_count = len(result_analysis.heatmap_info)
        detected_image_count = len(result_analysis.image_info)
        if detected_heatmap_count != detected_image_count:
            if debug: 
                print("WARNING: {} heatmap objects detected but {} image objects detected".format(detected_heatmap_count,detected_image_count))
            return None,result_analysis
        
        # TODO: Continue the analysis part, Convert the cordinate to 3D base
        
        '''
        Content of the info_3D_object
        
        {
            "position": [x,y,z],
            "corners":[[x,y,z],[x,y,z],[x,y,z],[x,y,z]],
            "object_type": "car|truck|bike|motorcycle|bus",
        }
        '''
        
        
        # TODO: Support multiple object
        if detected_heatmap_count != 1:
            return None,result_analysis
        
        pos_3d = from_multimodal_analysis_result_to_3d(result_analysis,self.camera_parameters)
        
        
        
        return pos_3d, result_analysis
        
        
        # Check if the scale of the 3D object make sense
        


def test1 ():
    FILE_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
    FILE_COUNT_TO_LOAD = 100
    
    BACKGROUND_FILE = os.path.join(FILE_DIRECTORY,"background.doppler")
    MEAN_HEATMAP_FILE = os.path.join(FILE_DIRECTORY,"mean_heatmap.doppler")
    
    

    heatmap_directory = os.path.join(FILE_DIRECTORY, "data","graphes")
    picture_directory = os.path.join(FILE_DIRECTORY, "data","images")
    
    

    timestamps_to_load = list([".".join(f.split(".")[:2]) for f in os.listdir(heatmap_directory) if "doppler" in f])
    timestamps_to_load = timestamps_to_load[0:min(FILE_COUNT_TO_LOAD,len(timestamps_to_load))]

    

    
    dataWrapper = DataWrapper(heatmap_directory, picture_directory, timestamps_to_load, picture_name_prefix="0", picture_extension_suffix="jpeg", heatmap_extension_suffix="doppler", heatmap_name_prefix="")
    
    dataWrapper.set_background_data(BACKGROUND_FILE)
    dataWrapper.set_mean_heatmap_data(MEAN_HEATMAP_FILE)
    
    # dataWrapper.remove_background_data()

    
    
    random_sample = rd.sample(range(FILE_COUNT_TO_LOAD),10)
    
    for i in random_sample:
        # dataWrapper.plot(i,logarithmic=False,sign_color_map=False)
        dataWrapper.pipeline_process(i)
        ana = dataWrapper.analyse_couple(i,plot=True)
        res = from_multimodal_analysis_result_to_3d(ana,dataWrapper.camera_parameters)
        print(res)
        # dataWrapper.plot_CFAR(i)

def test2():
    analyse = {'timestamp': '1657880853.943706', 'heatmap_energy': 6732783934474.394, 'heatmap_info': [{'distance': 53.8671875, 'speed': 8.247187500000003}], 'raw_heatmap_info': [[197, 166]], 'image_info': [{'bbox': [538, 576, 804, 749], 'x': 671, 'y': 662, 'class': 'car', 'confidence': 0.8863056302070618}]}
    
    struct_analyse = MultimodalAnalysisResult(analyse["timestamp"],analyse["heatmap_energy"],analyse["heatmap_info"],analyse["raw_heatmap_info"],analyse["image_info"])

    
    
    
    camera_parameters = {'focal_length':5695.8, 'image_size':(1920,1080),'principal_point':(1920/2,1080/2), 'fov':30}
    
    res = from_multimodal_analysis_result_to_3d(struct_analyse,camera_parameters)
    print(res)
    
def analyse_dataset(BATCH_SIZE = 1000,FILE_COUNT_TO_LOAD = 10000,FILE_DIRECTORY = os.path.dirname(os.path.realpath(__file__)),customfilter=None,cfar_threshold=30,save=True):

    
    BACKGROUND_FILE = os.path.join(FILE_DIRECTORY,"background.doppler")
    MEAN_HEATMAP_FILE = os.path.join(FILE_DIRECTORY,"mean_heatmap.doppler")
    
    

    heatmap_directory = os.path.join(FILE_DIRECTORY, "data","graphes")
    picture_directory = os.path.join(FILE_DIRECTORY, "data","images")
    
    count_error = 0 
    detected_vehicle_heatmap = []
    detected_vehicle_image = []
    missmatch_count_heatmap_image = []
    
    energy_heatmap = []
    pos_list = []
    timestamps_to_load_total = list([".".join(f.split(".")[:2]) for f in os.listdir(heatmap_directory) if "doppler" in f])
    rd.shuffle(timestamps_to_load_total)
    timestamps_to_load_total = timestamps_to_load_total[0:min(FILE_COUNT_TO_LOAD,len(timestamps_to_load_total))]
    for batch_index in tqdm(range(0,math.ceil(len(timestamps_to_load_total)/BATCH_SIZE)),desc="Batch"):
    

        timestamps_to_load = timestamps_to_load_total[batch_index*BATCH_SIZE:min((batch_index+1)*BATCH_SIZE,len(timestamps_to_load_total))]

        dataWrapper = DataWrapper(heatmap_directory, picture_directory, timestamps_to_load, picture_name_prefix="0", picture_extension_suffix="jpeg", heatmap_extension_suffix="doppler", heatmap_name_prefix="")
        
        dataWrapper.set_background_data(BACKGROUND_FILE)
        dataWrapper.set_mean_heatmap_data(MEAN_HEATMAP_FILE)
        if customfilter is not None:
            dataWrapper.filter = customfilter
        # dataWrapper.filter = customfilter
        # dataWrapper.remove_background_data()

        
        
        # random_sample = rd.sample(range(FILE_COUNT_TO_LOAD),100)
        res_analyses = []
        
        for i in tqdm(range(len(timestamps_to_load)),desc="Analysing couple"): #(random_sample:
            # dataWrapper.plot(i,logarithmic=False,sign_color_map=False)
            res,res_ana = dataWrapper.pipeline_process(i,cfar_threshold=cfar_threshold)
            res_analyses.append(res_ana)
            if res is not None:
                pos_list.append(res)
                
        
        # Some metrics on the quality of the detection
        
        
        for ana in res_analyses:
            detected_vehicle_heatmap.append(len(ana.heatmap_info))
            detected_vehicle_image.append(len(ana.image_info))
            energy_heatmap.append(ana.heatmap_energy)
            missmatch_count_heatmap_image.append(len(ana.heatmap_info) - len(ana.image_info))
            if len(ana.heatmap_info) != len(ana.image_info):
                count_error += 1
    
    
    pos_list = np.array(pos_list)
    
    
    
    
    detected_vehicle_heatmap = np.array(detected_vehicle_heatmap)
    detected_vehicle_image = np.array(detected_vehicle_image)
    energy_heatmap = np.array(energy_heatmap)
    missmatch_count_heatmap_image = np.array(missmatch_count_heatmap_image)
    
    
    # Autoincrement of the save directory
    # Create a save directory if it does not exist
    if not os.path.exists(os.path.join(FILE_DIRECTORY,"dataset_analysis_save")):
        os.makedirs(os.path.join(FILE_DIRECTORY,"dataset_analysis_save"))
        
    if save:
        ANA_DIRECTORY = os.path.join(FILE_DIRECTORY,"dataset_analysis_save",f"analysis_{len(os.listdir(os.path.join(FILE_DIRECTORY,'dataset_analysis_save')))}")
        if not os.path.exists(ANA_DIRECTORY):
            os.makedirs(ANA_DIRECTORY)
        with open(os.path.join(ANA_DIRECTORY,"pos_list.pkl"),"wb") as f:
            pickle.dump(pos_list,f)
        
        with open(os.path.join(ANA_DIRECTORY,"detected_vehicle_heatmap.pkl"),"wb") as f:
            pickle.dump(detected_vehicle_heatmap,f)
        
        with open(os.path.join(ANA_DIRECTORY,"detected_vehicle_image.pkl"),"wb") as f:
            pickle.dump(detected_vehicle_image,f)
            
        with open(os.path.join(ANA_DIRECTORY,"energy_heatmap.pkl"),"wb") as f:
            pickle.dump(energy_heatmap,f)
            
        with open(os.path.join(ANA_DIRECTORY,"count_missmatch.pkl"),"wb") as f:
            pickle.dump(missmatch_count_heatmap_image,f)
        
        with open(os.path.join(ANA_DIRECTORY,"README.md"),"w") as f:
            f.write(f"Number of error : {count_error} / {len(timestamps_to_load_total)}")
    
    return pos_list,missmatch_count_heatmap_image,energy_heatmap,detected_vehicle_heatmap,detected_vehicle_image

def read_analysis():
    PATH_ANALYSIS = os.path.join(os.path.dirname(os.path.realpath(__file__)),"dataset_analysis_save","analysis_3")
    
    heatmap_data, image_data, energy, pos_list,missmatch = load_analysis_result(PATH_ANALYSIS)
    
    plot_analysis_result(heatmap_data, image_data, energy, pos_list,missmatch, CAMERA_PARAM)

def test5():
    FILE_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
    for i, cfar_threshold in enumerate(range(20,100,10)):
        dirname = os.path.join(os.path.join(FILE_DIRECTORY,"dataset_analysis_save"),f"analysis_{7+i}")
        heatmap_data, image_data, energy, pos_list,missmatch = load_analysis_result(path=dirname)
        loss = np.mean((missmatch)**2)
        
        print(f"CFAR_THRESHOLD: {cfar_threshold}")
        print(f"\tloss: {loss}")
        print(f"\tmean energy: {np.mean(energy)}")
        print(f"\tmean missmatch: {np.mean(missmatch)}")
        print(f"\tmean detected vehicle heatmap: {np.mean(heatmap_data)}")

def search_optimal_th():
    TH = 30000
    FILE_COUNT_TO_LOAD=3000
    FILE_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
    loss_list_all_sample = []
    error_count_list_all_sample = []
    
    loss_list_1_vehicle = []
    error_count_list_1_vehicle = []
    
    loss_list_0_vehicle = []
    error_count_list_0_vehicle = []
    
    search_space = np.logspace(3,6,num=100)
    for th in search_space:
        pos_list,missmatch_count_heatmap_image,energy_heatmap,detected_vehicle_heatmap,detected_vehicle_image = analyse_dataset(save=False,FILE_COUNT_TO_LOAD=FILE_COUNT_TO_LOAD,cfar_threshold=th)
        
        loss_all_sample = np.mean((missmatch_count_heatmap_image)**2)
        error_count_all_sample = np.sum(missmatch_count_heatmap_image != 0)
        
        loss_list_all_sample.append(loss_all_sample)
        error_count_list_all_sample.append(error_count_all_sample/len(detected_vehicle_image))
        
        loss_1_vehicle = np.mean((missmatch_count_heatmap_image[detected_vehicle_image==1])**2)
        error_count_1_vehicle = np.sum(missmatch_count_heatmap_image[detected_vehicle_image==1] != 0)
        
        loss_list_1_vehicle.append(loss_1_vehicle)
        error_count_list_1_vehicle.append(error_count_1_vehicle/len(detected_vehicle_image[detected_vehicle_image==1]))
        
        loss_0_vehicle = np.mean((missmatch_count_heatmap_image[detected_vehicle_image==0])**2)
        error_count_0_vehicle = np.sum(missmatch_count_heatmap_image[detected_vehicle_image==0] != 0)
        
        loss_list_0_vehicle.append(loss_0_vehicle)
        error_count_list_0_vehicle.append(error_count_0_vehicle/len(detected_vehicle_image==0))
        
        report_str = ""
    
        report_str += f"Analysis result for threshold {th} using {FILE_COUNT_TO_LOAD} files:\n"
        report_str += f"\tmean energy: {np.mean(energy_heatmap)}\n"
        report_str += f"\tmean missmatch: {np.mean(missmatch_count_heatmap_image)}\n"
        report_str += f"\terror count: {np.sum(missmatch_count_heatmap_image != 0)} / {len(missmatch_count_heatmap_image)} ({np.sum(missmatch_count_heatmap_image != 0)/len(missmatch_count_heatmap_image)*100}%)\n"
        report_str += f"\tloss: {loss_all_sample}\n"
        report_str += "\tfor datapoint having one vehicle (according to yolo):\n"
        report_str += f"\t\tmean missmatch: {loss_1_vehicle}\n"
        report_str += f"\t\tmean energy: {np.mean(energy_heatmap[detected_vehicle_image==1])}\n"
        report_str += f"\t\tloss: {np.mean((missmatch_count_heatmap_image[detected_vehicle_image==1])**2)}\n"
        report_str += f"\t\terror count: {error_count_1_vehicle} / {len(missmatch_count_heatmap_image[detected_vehicle_image==1])} ({np.sum(missmatch_count_heatmap_image[detected_vehicle_image==1] != 0)/len(missmatch_count_heatmap_image[detected_vehicle_image==1])*100}%)\n"
        report_str += "\tfor datapoint having no vehicle (according to yolo):\n"
        report_str += f"\t\tmean missmatch: {np.mean(missmatch_count_heatmap_image[detected_vehicle_image==0])}\n"
        report_str += f"\t\tmean energy: {np.mean(energy_heatmap[detected_vehicle_image==0])}\n"
        report_str += f"\t\tloss: {np.mean((missmatch_count_heatmap_image[detected_vehicle_image==0])**2)}\n"
        report_str += f"\t\terror count: {np.sum(missmatch_count_heatmap_image[detected_vehicle_image==0] != 0)} / {len(missmatch_count_heatmap_image[detected_vehicle_image==0])} ({np.sum(missmatch_count_heatmap_image[detected_vehicle_image==0] != 0)/len(missmatch_count_heatmap_image[detected_vehicle_image==0])*100}%)\n"
        
        print(report_str)
    
    with open(os.path.join(FILE_DIRECTORY,f"TH_CFAR_{FILE_COUNT_TO_LOAD}_threshold.pkl"),"wb") as f:
        pickle.dump(np.array(search_space),f)
    
    with open(os.path.join(FILE_DIRECTORY,f"TH_CFAR_{FILE_COUNT_TO_LOAD}_loss_all_sample.pkl"),"wb") as f:
        pickle.dump(np.array(loss_list_all_sample),f)
    
    with open(os.path.join(FILE_DIRECTORY,f"TH_CFAR_{FILE_COUNT_TO_LOAD}_errors_all_sample.pkl"),"wb") as f:
        pickle.dump(np.array(error_count_list_all_sample),f)
        
    with open(os.path.join(FILE_DIRECTORY,f"TH_CFAR_{FILE_COUNT_TO_LOAD}_loss_1_vehicle.pkl"),"wb") as f:
        pickle.dump(np.array(loss_list_1_vehicle),f)
    
    with open(os.path.join(FILE_DIRECTORY,f"TH_CFAR_{FILE_COUNT_TO_LOAD}_errors_1_vehicle.pkl"),"wb") as f:
        pickle.dump(np.array(error_count_list_1_vehicle),f)
        
    with open(os.path.join(FILE_DIRECTORY,f"TH_CFAR_{FILE_COUNT_TO_LOAD}_loss_0_vehicle.pkl"),"wb") as f:
        pickle.dump(np.array(loss_list_0_vehicle),f)
    
    with open(os.path.join(FILE_DIRECTORY,f"TH_CFAR_{FILE_COUNT_TO_LOAD}_errors_0_vehicle.pkl"),"wb") as f:
        pickle.dump(np.array(error_count_list_0_vehicle),f)
    
def load_plot_search_optimal_threshold():
    FILE_COUNT_TO_LOAD = 3000
    search_space = pickle.load(open(os.path.join(FILE_DIRECTORY,f"TH_CFAR_{FILE_COUNT_TO_LOAD}_threshold.pkl"),"rb"))
    loss_list_all_sample = pickle.load(open(os.path.join(FILE_DIRECTORY,f"TH_CFAR_{FILE_COUNT_TO_LOAD}_loss_all_sample.pkl"),"rb"))
    error_count_list_all_sample = pickle.load(open(os.path.join(FILE_DIRECTORY,f"TH_CFAR_{FILE_COUNT_TO_LOAD}_errors_all_sample.pkl"),"rb"))
    loss_list_1_vehicle = pickle.load(open(os.path.join(FILE_DIRECTORY,f"TH_CFAR_{FILE_COUNT_TO_LOAD}_loss_1_vehicle.pkl"),"rb"))
    error_count_list_1_vehicle = pickle.load(open(os.path.join(FILE_DIRECTORY,f"TH_CFAR_{FILE_COUNT_TO_LOAD}_errors_1_vehicle.pkl"),"rb"))
    loss_list_0_vehicle = pickle.load(open(os.path.join(FILE_DIRECTORY,f"TH_CFAR_{FILE_COUNT_TO_LOAD}_loss_0_vehicle.pkl"),"rb"))
    error_count_list_0_vehicle = pickle.load(open(os.path.join(FILE_DIRECTORY,f"TH_CFAR_{FILE_COUNT_TO_LOAD}_errors_0_vehicle.pkl"),"rb"))
    
    plt.figure()
    # plt.subplot(211)
    # plt.plot(search_space,loss_list_all_sample)
    # plt.plot(search_space,loss_list_1_vehicle)
    # plt.plot(search_space,loss_list_0_vehicle)
    # plt.grid()
    # plt.title("Loss")
    # plt.xlabel("Threshold value")
    # plt.ylabel("Loss")
    # plt.xscale("log")
    
    
    plt.subplot(212)
    plt.plot(search_space,error_count_list_all_sample,label="all samples")
    plt.plot(search_space,error_count_list_1_vehicle, label="1 vehicle")
    plt.plot(search_space,error_count_list_0_vehicle, label="0 vehicle")
    plt.grid()
    plt.legend()
    plt.title("Error count")
    plt.xlabel("Threshold value")
    plt.ylabel("Error count")
    plt.xscale("log")
    plt.show()
    
    
    
    
def rank_analysis(save=True,cfar_threshold=210000,FILE_COUNT_TO_LOAD=3000):
    pos_list,missmatch_count_heatmap_image,energy_heatmap,detected_vehicle_heatmap,detected_vehicle_image = analyse_dataset(save=True,FILE_COUNT_TO_LOAD=3000,cfar_threshold=210000)
    
    loss = np.mean((missmatch_count_heatmap_image)**2)
    error_count = np.sum(missmatch_count_heatmap_image != 0)
    
    report_str = ""
    
    report_str += f"Analysis result for threshold {cfar_threshold} using {FILE_COUNT_TO_LOAD} files:\n"
    report_str += f"\tmean energy: {np.mean(energy_heatmap)}\n"
    report_str += f"\tmean missmatch: {np.mean(missmatch_count_heatmap_image)}\n"
    report_str += f"\terror count: {np.sum(missmatch_count_heatmap_image != 0)} / {len(missmatch_count_heatmap_image)} ({np.sum(missmatch_count_heatmap_image != 0)/len(missmatch_count_heatmap_image)*100}%)\n"
    report_str += f"\tloss: {loss}\n"
    report_str += "\tfor datapoint having one vehicle (according to yolo):\n"
    report_str += f"\t\tmean missmatch: {np.mean(missmatch_count_heatmap_image[detected_vehicle_image==1])}\n"
    report_str += f"\t\tmean energy: {np.mean(energy_heatmap[detected_vehicle_image==1])}\n"
    report_str += f"\t\tloss: {np.mean((missmatch_count_heatmap_image[detected_vehicle_image==1])**2)}\n"
    report_str += f"\t\terror count: {np.sum(missmatch_count_heatmap_image[detected_vehicle_image==1] != 0)} / {len(missmatch_count_heatmap_image[detected_vehicle_image==1])} ({np.sum(missmatch_count_heatmap_image[detected_vehicle_image==1] != 0)/len(missmatch_count_heatmap_image[detected_vehicle_image==1])*100}%)\n"
    report_str += "\tfor datapoint having no vehicle (according to yolo):\n"
    report_str += f"\t\tmean missmatch: {np.mean(missmatch_count_heatmap_image[detected_vehicle_image==0])}\n"
    report_str += f"\t\tmean energy: {np.mean(energy_heatmap[detected_vehicle_image==0])}\n"
    report_str += f"\t\tloss: {np.mean((missmatch_count_heatmap_image[detected_vehicle_image==0])**2)}\n"
    report_str += f"\t\terror count: {np.sum(missmatch_count_heatmap_image[detected_vehicle_image==0] != 0)} / {len(missmatch_count_heatmap_image[detected_vehicle_image==0])} ({np.sum(missmatch_count_heatmap_image[detected_vehicle_image==0] != 0)/len(missmatch_count_heatmap_image[detected_vehicle_image==0])*100}%)\n"
    
    print(report_str)
    with open(os.path.join(FILE_DIRECTORY,f"rank_analysis_{FILE_COUNT_TO_LOAD}_threshold_{cfar_threshold}.txt"),"w") as f:
        f.write(report_str)
        
    plot_analysis_result(detected_vehicle_heatmap, detected_vehicle_image, energy_heatmap, pos_list,missmatch_count_heatmap_image,CAMERA_PARAM )

if __name__ == "__main__":
    # test1()
    # test2()
    # test3()
    # test4()
    # analyse()
    search_optimal_th()
    # load_plot_search_optimal_threshold()
    # analyse_dataset(save=True,FILE_COUNT_TO_LOAD=3000,cfar_threshold=210000)
    # rank_analysis(save=True,cfar_threshold=210000,FILE_COUNT_TO_LOAD=3000)
    