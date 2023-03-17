import matplotlib.pyplot as plt
import pickle
from ..analysis_utilities import plot_analysis_result,load_analysis_result
import json
import numpy as np

import os

def load_plot_search_optimal_threshold(file_directory_path,analysis_file_count=10000):
    """Utility function to load and plot the results of the search for the optimal threshold

    Args:
        file_directory_path (path str): Absolute path of the directory containing the analysis. Usually <ROOT>/optimal_th_analysis/<analysis_name>
        analysis_file_count (int, optional): _description_. Defaults to 10000.
    """
    
    FILE_COUNT_TO_LOAD = analysis_file_count
    FILE_DIRECTORY_ANALYSE = file_directory_path
    search_space = pickle.load(open(os.path.join(FILE_DIRECTORY_ANALYSE,f"TH_CFAR_{FILE_COUNT_TO_LOAD}_threshold.pkl"),"rb"))
    loss_list_all_sample = pickle.load(open(os.path.join(FILE_DIRECTORY_ANALYSE,f"TH_CFAR_{FILE_COUNT_TO_LOAD}_loss_all_sample.pkl"),"rb"))
    error_count_list_all_sample = pickle.load(open(os.path.join(FILE_DIRECTORY_ANALYSE,f"TH_CFAR_{FILE_COUNT_TO_LOAD}_errors_all_sample.pkl"),"rb"))
    loss_list_1_vehicle = pickle.load(open(os.path.join(FILE_DIRECTORY_ANALYSE,f"TH_CFAR_{FILE_COUNT_TO_LOAD}_loss_1_vehicle.pkl"),"rb"))
    error_count_list_1_vehicle = pickle.load(open(os.path.join(FILE_DIRECTORY_ANALYSE,f"TH_CFAR_{FILE_COUNT_TO_LOAD}_errors_1_vehicle.pkl"),"rb"))
    loss_list_0_vehicle = pickle.load(open(os.path.join(FILE_DIRECTORY_ANALYSE,f"TH_CFAR_{FILE_COUNT_TO_LOAD}_loss_0_vehicle.pkl"),"rb"))
    error_count_list_0_vehicle = pickle.load(open(os.path.join(FILE_DIRECTORY_ANALYSE,f"TH_CFAR_{FILE_COUNT_TO_LOAD}_errors_0_vehicle.pkl"),"rb"))
    
    plt.figure()
    plt.subplot(211)
    plt.plot(search_space,loss_list_all_sample)
    plt.plot(search_space,loss_list_1_vehicle)
    plt.plot(search_space,loss_list_0_vehicle)
    plt.grid()
    plt.title("Loss")
    plt.xlabel("Threshold value")
    plt.ylabel("Loss")
    plt.xscale("log")
    
    
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
    
    
def plot_kernel_param_result(FilePath):
    """Utility function to read the results of the kernel param analysis

    Args:
        FilePath (path str): Absolute path of the file containing the analysis. Usually <ROOT>/optimal_th_analysis/<analysis_name>/kernel_param_results.pkl

    Returns:
        dict: Dictionary containing the results
    """
    with open(FilePath, "r") as f:
        data= json.load(f)
        
    # Creating the axis
    
    
    # The data is a list of dict
    # We need to create a list of list
    # We first create a dict of list to store the data with kernel size as key
    
    tmpX = {}
    
    for record in data:
        if record["kernel_size"] not in tmpX:
            tmpX[record["kernel_size"]] = []
        tmpX[record["kernel_size"]].append([record["sigma"],record["error_count"],record["error_count_0_vehicle"],record["error_count_1_vehicle"],record["error_count_2_vehicle"]])
    
    # We sort each list by sigma
    
    for key in tmpX:
        tmpX[key].sort(key=lambda x: x[0])
        tmpX[key] = np.array(tmpX[key])
    
    # We plot the data
    
    plt.figure()
    for error_index in range(4):
        for ksize,value in tmpX.items():
            plt.subplot(2,2,error_index+1)
            plt.plot(value[:,0],value[:,error_index+1],label=f"kernel size {ksize}")
        plt.grid()
        plt.legend()
        plt.ylabel("Error count (ratio)")
        plt.xlabel("Sigma")
        plt.title(["All samples","0 vehicle","1 vehicle","2 vehicles"][error_index])
        
    plt.show()
    
