""" 
Thomas Bolteau - February 2023
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import random as rd
import scipy

import cv2
import math
import json

from .analysis_utilities import plot_analysis_result, load_analysis_result
from .yolo_utilities import get_yolo
from ._math_utilities import area,corr_kernel,fast_convolution

from .analysis_utilities import MultimodalAnalysisResult, get_caracteristic_length,from_multimodal_analysis_result_to_3d

import sys, os


            
            

global yolo 
yolo = get_yolo()





CAMERA_PARAM = {'skew':0,'distortion_coefficients':0,'focal_length':5695, 'image_size':(1920,1080),'principal_point':(1920/2,1080/2), 'fov':30}
FILE_DIRECTORY = os.path.dirname(os.path.realpath(__file__))









    
class DataWrapper:
    """This class is used to wrap heatmap data and picture data.
    Motivation: When this work started, analysing the filtering of the heatmap was not easy. It was necessary to compare it with real picture.
    As teh data is multimodal, associating the two is not stupid and having utilities to compare the two modality for a certain time index is quite useful.
    """
    
    # Importing the method of the class
    from ._dw_plot import plot_background, plot_mean, plot, plot_radar_wrapper, plot_CFAR, plot_comparison_filter
    from ._dw_data_loading import check_timestamp, check_directory, load_file, load_heatmap_data,load_picture_data
    from ._dw_utilities import get_color_map,set_background_data,save_mean_heatmap_data,set_mean_heatmap_data,save_picture,save_heatmap,get_metadata
    from ._dw_processing import pipeline_process,filter, analyse_couple,remove_background_data,calculate_heatmap_mean,remove_temporal_mean,analyse_image,add_annotation,calculate_CFAR,add_identification
    
    def __init__(self, heatmap_dir, picture_dir, timestamps_to_load, picture_name_prefix="", picture_extension_suffix="jpeg", heatmap_extension_suffix="doppler", heatmap_name_prefix="",picture_output_dir=None,silent=False):
        """Generate the class

        Args:
            heatmap_dir (str): Path of the heatmap directory
            picture_dir (str): Path of the picture directory
            timestamps_to_load ([]str]): List of file to load, it has to be the minimal common string between the two modalities, see the prefix variable.
            picture_name_prefix (str, optional): Prefix for the picture filename, in case of timestamp, this may be "0". Defaults to "".
            picture_extension_suffix (str, optional): Extension file for the picture. Defaults to "jpeg".
            heatmap_extension_suffix (str, optional): Extension file for the heatmap. Defaults to "doppler".
            heatmap_name_prefix (str, optional): Prefix for the heatmap filename, usually "". Defaults to "".
            picture_output_dir (str, optional): Path of the output directory for the annotated picture. Defaults to None.
            silent (bool, optional): If True, no print will be done. Defaults to False.
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
        
        self.picture_output_dir = picture_output_dir
        
        self.background_data = None
        
        self.silent = silent
        if self.silent :
            os.environ["YOLOv5_VERBOSE"]="FALSE"

        
        self.heatmap_mean = None
        
        self.color_map = plt.get_cmap('gray')
        
        self.allowed_class = [1,2,3,5,7] # 1:bicycle, 2:car, 3:motorcycle, 5:bus, 7:truck

        global yolo
        if yolo is None:
            yolo = get_yolo()  # or yolov5n - yolov5x6, custom

        self.yolo = yolo  # or yolov5n - yolov5x6, custom
        self.yolo.classes = self.allowed_class
        self.conf = 0.4

        self.check_directory(self.heatmap_dir)
        self.check_directory(self.picture_dir)
        self.check_timestamp()
        
        self.load_heatmap_data()
        self.load_picture_data()

    

    

    
        


def test1 ():
    FILE_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
    FILE_COUNT_TO_LOAD = 1000
    cfar_threshold = 50000
    
    BACKGROUND_FILE = os.path.join(FILE_DIRECTORY,"background.doppler")
    MEAN_HEATMAP_FILE = os.path.join(FILE_DIRECTORY,"mean_heatmap.doppler")
    NEW_BACKGROUND = os.path.join(FILE_DIRECTORY,"new_new_background.doppler")
    
    

    heatmap_directory = os.path.join(FILE_DIRECTORY, "data","graphes")
    picture_directory = os.path.join(FILE_DIRECTORY, "data","images")
    
    

    timestamps_to_load = list([".".join(f.split(".")[:2]) for f in os.listdir(heatmap_directory) if "doppler" in f])
    timestamps_to_load = timestamps_to_load[0:min(FILE_COUNT_TO_LOAD,len(timestamps_to_load))]

    

    
    dataWrapper = DataWrapper(heatmap_directory, picture_directory, timestamps_to_load, picture_name_prefix="0", picture_extension_suffix="jpeg", heatmap_extension_suffix="doppler", heatmap_name_prefix="")
    
    dataWrapper.set_background_data(NEW_BACKGROUND)
    dataWrapper.set_mean_heatmap_data(MEAN_HEATMAP_FILE)
    
    # dataWrapper.remove_background_data()

    
    
    random_sample = range(10)
    dataWrapper.plot_background()
    
    bg = np.zeros((256,256))
    bg_list = []
    nb_bg = 0
    
    for i in random_sample:
        # dataWrapper.plot(i,logarithmic=False,sign_color_map=False)
        # dataWrapper.pipeline_process(i)
        # ana = dataWrapper.analyse_couple(i,plot=True)
        # res = from_multimodal_analysis_result_to_3d(ana,dataWrapper.camera_parameters)
        # print(res)
        # result_analysis = dataWrapper.analyse_couple(i, plot=False, cfar_threshold=cfar_threshold)
        
        # # Check if the number of detected object match the number of object in the picture
        # detected_heatmap_count = len(result_analysis.heatmap_info)
        # detected_image_count = len(result_analysis.image_info)
        
        # print("Detected image object: {} | ".format(detected_image_count)+"Detected heatmap object: {}".format(detected_heatmap_count))
        
        
        dataWrapper.plot_comparison_filter(i)
        # if detected_image_count == 0:
        #     bg_list.append(dataWrapper.heatmap_data[i])
            
        
        # dataWrapper.plot_CFAR(i)

    # bg = np.mean(np.array(bg_list),axis=0)
    # path = os.path.join(FILE_DIRECTORY,"new_new_background.doppler")
    # with open(path,"wb") as f:
    #     pickle.dump(bg,f)
    
    

def test1bis ():
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

    
    
    random_sample = [0,3,6]#
    heatmap_bg = np.zeros((256,256))
    
    for i in random_sample:
        # dataWrapper.plot(i,logarithmic=False,sign_color_map=False)
        # dataWrapper.pipeline_process(i)
        # ana = dataWrapper.analyse_couple(i,plot=True)
        # res = from_multimodal_analysis_result_to_3d(ana,dataWrapper.camera_parameters)
        # print(res)
        # dataWrapper.plot_comparison_filter(i)
        
        # dataWrapper.plot_CFAR(i)
        heatmap_bg += dataWrapper.heatmap_data[i]
    
    path = os.path.join(FILE_DIRECTORY,"new_background.doppler")
    dataWrapper.save_mean_heatmap_data(path)


def test2():
    analyse = {'timestamp': '1657880853.943706', 'heatmap_energy': 6732783934474.394, 'heatmap_info': [{'distance': 53.8671875, 'speed': 8.247187500000003}], 'raw_heatmap_info': [[197, 166]], 'image_info': [{'bbox': [538, 576, 804, 749], 'x': 671, 'y': 662, 'class': 'car', 'confidence': 0.8863056302070618}]}
    
    struct_analyse = MultimodalAnalysisResult(analyse["timestamp"],analyse["heatmap_energy"],analyse["heatmap_info"],analyse["raw_heatmap_info"],analyse["image_info"])

    
    
    
    camera_parameters = {'focal_length':5695.8, 'image_size':(1920,1080),'principal_point':(1920/2,1080/2), 'fov':30}
    
    res,bb_box_3d_pos = from_multimodal_analysis_result_to_3d(struct_analyse,camera_parameters)
    print(res)
    

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


    



    
    
    


def plot_carcteristic_lenght():
    FILE_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
    FILE_COUNT_TO_LOAD = 1000
    
    BACKGROUND_FILE = os.path.join(FILE_DIRECTORY,"background.doppler")
    MEAN_HEATMAP_FILE = os.path.join(FILE_DIRECTORY,"mean_heatmap.doppler")
    NEW_BACKGROUND = os.path.join(FILE_DIRECTORY,"new_new_background.doppler")
    
    

    heatmap_directory = os.path.join(FILE_DIRECTORY, "data","graphes")
    picture_directory = os.path.join(FILE_DIRECTORY, "data","images")
    
    

    timestamps_to_load = list([".".join(f.split(".")[:2]) for f in os.listdir(heatmap_directory) if "doppler" in f])
    timestamps_to_load = timestamps_to_load[0:min(FILE_COUNT_TO_LOAD,len(timestamps_to_load))]

    

    
    dataWrapper = DataWrapper(heatmap_directory, picture_directory, timestamps_to_load, picture_name_prefix="0", picture_extension_suffix="jpeg", heatmap_extension_suffix="doppler", heatmap_name_prefix="")
    
    dataWrapper.set_background_data(NEW_BACKGROUND)
    dataWrapper.set_mean_heatmap_data(MEAN_HEATMAP_FILE)
    
    # dataWrapper.remove_background_data()

    
    
    random_sample = range(1000)

    distance_list = []
    lenght_list = []

    
    for i in tqdm(random_sample):

        pos,analyse = dataWrapper.pipeline_process(i,cfar_threshold=5.1*(10**4))
        if pos is not None:
            distance_list.append(analyse.heatmap_info[0]["distance"])
            lenght_list.append(analyse.caracteristic_length)
            # if analyse.caracteristic_length < 1.1:
            #     dataWrapper.plot_CFAR(i,annotated=True)
    plt.figure()
    plt.scatter(distance_list,lenght_list)
    plt.show()
            
    
def analyse_dataset_image_count():
    FILE_COUNT_TO_LOAD=30000
    FILE_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
    loss_list_all_sample = []
    error_count_list_all_sample = []
    
    loss_list_1_vehicle = []
    error_count_list_1_vehicle = []
    
    loss_list_0_vehicle = []
    error_count_list_0_vehicle = []
    
    th = 5.2*(10**4)

    pos_list,missmatch_count_heatmap_image,energy_heatmap,detected_vehicle_heatmap,detected_vehicle_image,ok_1_vehicle = analyse_dataset(save=False,FILE_COUNT_TO_LOAD=FILE_COUNT_TO_LOAD,cfar_threshold=th)
    
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
    
    report_str += f"\ttotal valid 1 vehicle couple: {ok_1_vehicle}/{FILE_COUNT_TO_LOAD} ({round(ok_1_vehicle/FILE_COUNT_TO_LOAD*100,2)}%)"
    
    print(report_str)
    
    with open(os.path.join(FILE_DIRECTORY,f"TH_CFAR_{FILE_COUNT_TO_LOAD}_threshold.pkl"),"wb") as f:
        pickle.dump(np.array(th),f)
    
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
        
    with open(os.path.join(FILE_DIRECTORY,f"TH_CFAR_{FILE_COUNT_TO_LOAD}_result.txt"),"w") as f:
        f.write(report_str)


def labelize_dataset(cfar_threshold=5.2*(10**4),length_threshold=1.1):
    FILE_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
    FILE_COUNT_TO_LOAD = 1000
    cfar_threshold = 50000
    
    BACKGROUND_FILE = os.path.join(FILE_DIRECTORY,"background.doppler")
    MEAN_HEATMAP_FILE = os.path.join(FILE_DIRECTORY,"mean_heatmap.doppler")
    NEW_BACKGROUND = os.path.join(FILE_DIRECTORY,"new_new_background.doppler")
    
    

    heatmap_directory = os.path.join(FILE_DIRECTORY, "data","graphes")
    picture_directory = os.path.join(FILE_DIRECTORY, "data","images")
    
    

    timestamps_to_load = list([".".join(f.split(".")[:2]) for f in os.listdir(heatmap_directory) if "doppler" in f])
    timestamps_to_load = timestamps_to_load[0:min(FILE_COUNT_TO_LOAD,len(timestamps_to_load))]

    

    
    dataWrapper = DataWrapper(heatmap_directory, picture_directory, timestamps_to_load, picture_name_prefix="0", picture_extension_suffix="jpeg", heatmap_extension_suffix="doppler", heatmap_name_prefix="")
    
    dataWrapper.set_background_data(NEW_BACKGROUND)
    dataWrapper.set_mean_heatmap_data(MEAN_HEATMAP_FILE)
    
    # dataWrapper.remove_background_data()

    
    
    random_sample = range(10)
    

    
    for i in random_sample:
        
        pos,analysis = dataWrapper.pipeline_process(i,cfar_threshold=cfar_threshold,length_threshold=length_threshold)


if __name__ == "__main__":
    # test1()
    # test1bis()
    # test2()
    # test3()
    # test4()
    # analyse()
    # search_optimal_th()
    # load_plot_search_optimal_threshold()
    # analyse_dataset(save=True,FILE_COUNT_TO_LOAD=1000,cfar_threshold=210000)
    # rank_analysis(save=True,cfar_threshold=50000,FILE_COUNT_TO_LOAD=3000)
    # plot_carcteristic_lenght()
    # analyse_dataset_image_count()
    search_optimal_kernel_param()