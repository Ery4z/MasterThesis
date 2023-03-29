import matplotlib.pyplot as plt
from .utilities import slog
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import math
import pickle
from ._math_utilities import area


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
    
def plot_3d_world_pos(pos_list,camera_parameters):
    def plot_camera(ax,camera_parameters):
        
        
        v = np.array([[0, 0.3, 0.3], [0, 0.3, -0.3], [0, 0.3, -0.3],  [0, -0.3, -0.3], [-0.3, 0, 0]])
        ax.scatter3D(v[:, 0], v[:, 1], v[:, 2])
        
        verts = [ [v[0],v[1],v[4]], [v[0],v[2],v[4]],
        [v[2],v[1],v[4]], [v[2],v[3],v[4]], [v[0],v[1],v[2],v[3]]]
        
        ax.add_collection3d(Poly3DCollection(verts, facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))

        camera_center = np.array([-0.3, 0, 0])
        fovY = camera_parameters['fov']*math.pi/180
        image_size = camera_parameters['image_size']
        fovZ = fovY * image_size[1] / image_size[0]
        distanceX = 70
        
        point_FOV = [camera_center+np.array([70,distanceX*math.tan(fovY/2),-1]),
                    camera_center+np.array([70,distanceX*math.tan(fovY/2),10]),
                    camera_center+np.array([70,-distanceX*math.tan(fovY/2),10]),
                    camera_center+np.array([70,-distanceX*math.tan(fovY/2),-1])]
        print(point_FOV)
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
    
    
def load_analysis_result(path:str):
    heatmap_data = pickle.load(open(path + "/detected_vehicle_heatmap.pkl", "rb"))    
    image_data = pickle.load(open(path + "/detected_vehicle_image.pkl", "rb"))
    energy = pickle.load(open(path + "/energy_heatmap.pkl", "rb"))
    pos_list = pickle.load(open(path + "/pos_list.pkl", "rb"))
    missmatch = pickle.load(open(path + "/count_missmatch.pkl", "rb"))
    
    return heatmap_data, image_data, energy, pos_list ,missmatch


class MultimodalAnalysisResult:
    def __init__(self,timestamp:str,heatmap_energy:float,heatmap_info:list,raw_heatmap_info:list,image_info:list):

        self.timestamp = float(timestamp)
        self.heatmap_energy = heatmap_energy
        self.heatmap_info = heatmap_info
        self.raw_heatmap_info = raw_heatmap_info
        self.image_info = image_info
        self.position = None
    
    def set_caracteristic_length(self,caracteristic_length):
        self.caracteristic_length = caracteristic_length
        
    def set_bbox_3d(self,points):
        self.bbox_3d = points
        
    def set_position(self,position):
        self.position = position
        
    def export_dict(self):
        to_export = {
            "couple_id":self.timestamp,
            "timestamp":self.timestamp,
            "bb":self.image_info[0]["bbox"],
            "x":self.position[0],
            "y":self.position[1],
            "z":self.position[2],
            "v": self.heatmap_info[0]["speed"],
            "d": self.heatmap_info[0]["distance"],
            
        }
        
        return to_export

def get_caracteristic_length(analysis_result: MultimodalAnalysisResult):
    """Get the caracteristic length of the detected vehicle

    Args:
        analysis_result (MultiModalAnalysisResult): Analysis result

    Returns:
        float: Caracteristic length of the heatmap
    """
    distance = analysis_result.heatmap_info[0]["distance"]
    # area = (analysis_result.image_info[0]["bbox"][2] - analysis_result.image_info[0]["bbox"][0]) * (analysis_result.image_info[0]["bbox"][3] - analysis_result.image_info[0]["bbox"][1])
    area_vehicle = area(analysis_result.bbox_3d)
    return np.sqrt(area_vehicle)


def from_multimodal_analysis_result_to_3d(analysis:MultimodalAnalysisResult,camera_parameters:dict):
    """Convert the data extracted from the image and heatmap to 3D world position.

    Args:
        analysis (MultimodalAnalysisResult): Data returned from the analysis
        camera_parameters (dict): Intrinsict parameters from the camera

    Returns:
        [3]float: 3d pos of the item
    """
    
    distance = analysis.heatmap_info[0]["distance"]
    
    def from_point_to_3d(x,y,distance,camera_parameters):
        fx, fy = camera_parameters['focal_length'], camera_parameters['focal_length']
        cx = camera_parameters['principal_point'][0]
        cy = camera_parameters['principal_point'][1]
        
            
        
        
        image_size = camera_parameters['image_size']
        fovX = camera_parameters['fov']
        fovY = fovX * image_size[1] / image_size[0]

        
        yaw = math.atan2(x-cx, fx)
        pitch = math.atan2(y-cy, fy)
        
        absolute_position = [distance * math.cos(pitch) * math.cos(yaw), -distance * math.sin(yaw), distance * math.sin(pitch)]
        return absolute_position
    
    absolute_position = from_point_to_3d(analysis.image_info[0]["x"],analysis.image_info[0]["y"],distance,camera_parameters)
    bb_box_3d_pos = []
    bb_box = analysis.image_info[0]["bbox"]
    bb_box_points = [[bb_box[0],bb_box[1]],[bb_box[0],bb_box[3]],[bb_box[2],bb_box[3]],[bb_box[2],bb_box[1]]]
    
    for points in bb_box_points:
        bb_box_3d_pos.append(from_point_to_3d(points[0],points[1],distance,camera_parameters))
    
    
    return absolute_position,bb_box_3d_pos


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

def calculate_energy(data):
    """Calculate the energy of an array. Useful to know if the data is meaningful

    Args:
        data (np.array): Array to calculate energy from

    Returns:
        float: Energy of the array
    """
    return np.sum(np.abs(data)**2)
