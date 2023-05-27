import math
import os
import json
from matplotlib import pyplot as plt



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


def main(DATASET_PATH):

    CAMERA_PARAM = {'skew':0,'distortion_coefficients':0,'focal_length':5695, 'image_size':(1920,1080),'principal_point':(1920/2,1080/2), 'fov':30}



    Points=[[[126,900,8.6],[420,920,8.4],[840,950,8.2],[1450,1000,8.0]],[[139,750,31.2],[339,747,33.7]],[[452,655,87],[854,704,64.4],[1200,700,55.3],[1360,720,46.5],[1450,725,45.3]]]


    d3_lines=[]

    for line in Points:
        lines_pos=[]
        for points in line :
            lines_pos.append(from_point_to_3d(points[0],points[1],points[2],CAMERA_PARAM))
        d3_lines.append(lines_pos)
        

    json_data = {"lines":d3_lines}
    with open('context_lines.json', 'w') as outfile:
        json.dump(json_data, outfile)

    

    dataset_directory = DATASET_PATH


    ax = plt.figure().add_subplot(projection='3d')
            
    for session_directory in os.listdir(dataset_directory):
        print(session_directory)
        if session_directory != '1' :
            continue
        
        if not os.path.isdir(os.path.join(dataset_directory,session_directory)):
            continue
        
        vehicle_file = os.path.join(dataset_directory,session_directory,"vehicles","vehicles.json")
        # Load the vehicle file

        vehicle_data = json.load(open(vehicle_file))



        for vehicle_identifier in vehicle_data:
            list_images = vehicle_data[vehicle_identifier]
            x = []
            y = []
            z = []
            for image in list_images:
                x.append(image['x'])
                y.append(image['y'])
                z.append(image['z'])
            
            
            ax.plot(x,y,z)
            
    for line in d3_lines:
        x = []
        y = []
        z = []
        for point in line:
            x.append(point[0])
            y.append(point[1])
            z.append(point[2])
        ax.plot(x,y,z,color='black')
    plt.show()



if __name__ == "__main__":
    DATASET_PATH = "F:\\RADAR_CAM_DATASET_THOMAS" # TODO: Change this to the path of the dataset
    main(DATASET_PATH)