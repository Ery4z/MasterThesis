import os
from tqdm import tqdm
from .._dw_ import DataWrapper
import math
import random as rd
import numpy as np
import pickle
import json
from ..analysis_utilities import plot_analysis_result

from .._identifier import VehicleIdentifier

from ..utilities import generate_video_annotated


import matplotlib.pyplot as plt

def analyse_dataset_batch(FILE_DIRECTORY,BATCH_SIZE = 200,FILE_COUNT_TO_LOAD = 10000,customfilter=None,cfar_threshold=52000,save=True,gauss_kernel_length=11,gauss_sigma=0.5,silent=False):
    """Function used to analyse the dataset. This is the low level wrapper allowing to analyse the dataset in batches. This is useful when the dataset is too large to be loaded in memory.
    Use this function as an utility for more complex analysis.

    Args:
        FILE_DIRECTORY (path str): Root directory of the project. Expect to have a background.doppler file and data folder as child.
        BATCH_SIZE (int, optional): Number of couple per batch. Defaults to 200.
        FILE_COUNT_TO_LOAD (int, optional): Total count of file to load. Defaults to 10000.
        cfar_threshold (int, optional): Threshold of the CFAR. Defaults to 52000.
        save (bool, optional): Should it be saved or used only in ram. Defaults to True.
        gauss_kernel_length (int, optional): Length of the gaussian kernel used for the convolution during filtering. Defaults to 11.
        gauss_sigma (float, optional): Sigma parameter of the gaussian kernel used during filtering. Defaults to 0.5.
        silent (bool, optional): Should it not print to the console ?. Defaults to False.

    Returns:
        _type_: _description_
    """
    
    
    BACKGROUND_FILE = os.path.join(FILE_DIRECTORY,"new_new_background.doppler")
    MEAN_HEATMAP_FILE = os.path.join(FILE_DIRECTORY,"mean_heatmap.doppler")
    

    heatmap_directory = os.path.join(FILE_DIRECTORY, "data","graphes")
    picture_directory = os.path.join(FILE_DIRECTORY, "data","images")
    
    count_error = 0 
    ok_1_vehicle = 0
    detected_vehicle_heatmap = []
    detected_vehicle_image = []
    missmatch_count_heatmap_image = []
    
    energy_heatmap = []
    pos_list = []
    timestamps_to_load_total = list([".".join(f.split(".")[:2]) for f in os.listdir(heatmap_directory) if "doppler" in f])
    rd.shuffle(timestamps_to_load_total)
    timestamps_to_load_total = timestamps_to_load_total[0:min(FILE_COUNT_TO_LOAD,len(timestamps_to_load_total))]
    for batch_index in tqdm(range(0,math.ceil(len(timestamps_to_load_total)/BATCH_SIZE)),desc="Batch",leave=False):
    

        timestamps_to_load = timestamps_to_load_total[batch_index*BATCH_SIZE:min((batch_index+1)*BATCH_SIZE,len(timestamps_to_load_total))]

        dataWrapper = DataWrapper(heatmap_directory, picture_directory, timestamps_to_load, picture_name_prefix="0", picture_extension_suffix="jpeg", heatmap_extension_suffix="doppler", heatmap_name_prefix="",silent=silent)
        
        dataWrapper.set_background_data(BACKGROUND_FILE)
        dataWrapper.set_mean_heatmap_data(MEAN_HEATMAP_FILE)
        if customfilter is not None:
            dataWrapper.filter = customfilter
        # dataWrapper.filter = customfilter
        # dataWrapper.remove_background_data()

        
        
        # random_sample = rd.sample(range(FILE_COUNT_TO_LOAD),100)
        res_analyses = []
        
        
        for i in tqdm(range(len(timestamps_to_load)),desc="Analysing couple",leave=False): #(random_sample:
            # dataWrapper.plot(i,logarithmic=False,sign_color_map=False)
            res,res_ana = dataWrapper.pipeline_process(i,cfar_threshold=cfar_threshold,gauss_kernel_length=gauss_kernel_length,gauss_sigma=gauss_sigma)
            res_analyses.append(res_ana)
            if res is not None:
                pos_list.append(res)
                ok_1_vehicle += 1
                
        
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
    
    return pos_list,missmatch_count_heatmap_image,energy_heatmap,detected_vehicle_heatmap,detected_vehicle_image,ok_1_vehicle


def search_optimal_th():
    TH = 30000
    FILE_COUNT_TO_LOAD=10000
    FILE_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
    loss_list_all_sample = []
    error_count_list_all_sample = []
    
    loss_list_1_vehicle = []
    error_count_list_1_vehicle = []
    
    loss_list_0_vehicle = []
    error_count_list_0_vehicle = []
    
    search_space = np.logspace(3,6,num=100)
    for th in search_space:
        pos_list,missmatch_count_heatmap_image,energy_heatmap,detected_vehicle_heatmap,detected_vehicle_image,ok_1_vehicle = analyse_dataset_batch(save=False,FILE_COUNT_TO_LOAD=FILE_COUNT_TO_LOAD,cfar_threshold=th)
        
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

def search_optimal_kernel_param():
    TH = 30000
    FILE_COUNT_TO_LOAD=10000
    FILE_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
    SAVE_DIRECTORY = os.path.join(FILE_DIRECTORY,"results_analysis_kernel_param")
    
    # create directory if not exist
    if not os.path.exists(SAVE_DIRECTORY):
        os.makedirs(SAVE_DIRECTORY)
    
    
    min_kernel_size,max_kernel_size = 5,15
    min_sigma,max_sigma,sigma_step = 0.1,1,0.1
    
    
    results_dict_list = []
    
    search_space = np.logspace(3,6,num=100)
    
    k_size_space = range(min_kernel_size,max_kernel_size+1,2)
    sigma_space = np.arange(min_sigma,max_sigma,sigma_step)
    
    for k_size in tqdm(k_size_space,desc="k_size"):
        
        for sigma in tqdm(sigma_space,desc="sigma",leave=False):
            sub_dict = {}
            
            
            pos_list,missmatch_count_heatmap_image,energy_heatmap,detected_vehicle_heatmap,detected_vehicle_image,ok_1_vehicle = analyse_dataset_batch(save=False,FILE_COUNT_TO_LOAD=FILE_COUNT_TO_LOAD, gauss_kernel_length=k_size,gauss_sigma=sigma,silent=True)
        
            loss_all_sample = np.mean((missmatch_count_heatmap_image)**2)
            error_count_all_sample = np.sum(missmatch_count_heatmap_image != 0)
            

            
            loss_1_vehicle = np.mean((missmatch_count_heatmap_image[detected_vehicle_image==1])**2)
            error_count_1_vehicle = np.sum(missmatch_count_heatmap_image[detected_vehicle_image==1] != 0)
            

            
            loss_0_vehicle = np.mean((missmatch_count_heatmap_image[detected_vehicle_image==0])**2)
            error_count_0_vehicle = np.sum(missmatch_count_heatmap_image[detected_vehicle_image==0] != 0)
            

            
            loss_2_vehicle = np.mean((missmatch_count_heatmap_image[detected_vehicle_image==2])**2)
            error_count_2_vehicle = np.sum(missmatch_count_heatmap_image[detected_vehicle_image==2] != 0)
            

            
            report_str = ""
        
            report_str += f"Analysis result for kernel size {k_size} kernel sigma {sigma} using {FILE_COUNT_TO_LOAD} files:\n"
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
            
            report_str += "\tfor datapoint having two vehicle (according to yolo):\n"
            report_str += f"\t\tmean missmatch: {loss_2_vehicle}\n"
            report_str += f"\t\tmean energy: {np.mean(energy_heatmap[detected_vehicle_image==2])}\n"
            report_str += f"\t\tloss: {np.mean((missmatch_count_heatmap_image[detected_vehicle_image==2])**2)}\n"
            report_str += f"\t\terror count: {error_count_2_vehicle} / {len(missmatch_count_heatmap_image[detected_vehicle_image==2])} ({np.sum(missmatch_count_heatmap_image[detected_vehicle_image==2] != 0)/len(missmatch_count_heatmap_image[detected_vehicle_image==2])*100}%)\n"

            
            # print(report_str)
            
            sub_dict["sigma"] = sigma
            sub_dict["kernel_size"] = k_size
            
            sub_dict["error_count"] = error_count_all_sample/len(detected_vehicle_image)
            sub_dict["loss"] = loss_all_sample
            
            sub_dict["error_count_1_vehicle"] = error_count_1_vehicle/len(detected_vehicle_image[detected_vehicle_image==1])
            sub_dict["loss_1_vehicle"] = loss_1_vehicle
            
            sub_dict["error_count_0_vehicle"] = error_count_0_vehicle/len(detected_vehicle_image[detected_vehicle_image==0])
            sub_dict["loss_0_vehicle"] = loss_0_vehicle
            
            sub_dict["error_count_2_vehicle"] = error_count_2_vehicle/len(detected_vehicle_image[detected_vehicle_image==2])
            sub_dict["loss_2_vehicle"] = loss_2_vehicle
            
            sub_dict["mean_energy"] = np.mean(energy_heatmap)
            sub_dict["mean_missmatch"] = np.mean(missmatch_count_heatmap_image)
            
            results_dict_list.append(sub_dict)
            
            


    
    with open(os.path.join(SAVE_DIRECTORY,f"analysis_ks<{min_kernel_size},{max_kernel_size}>_sigma<{min_sigma},{max_sigma},{sigma_step}>.json"),"w") as f:
        json.dump(results_dict_list,f)
    
def rank_analysis(FILE_DIRECTORY,save=True,cfar_threshold=210000,FILE_COUNT_TO_LOAD=3000):
    pos_list,missmatch_count_heatmap_image,energy_heatmap,detected_vehicle_heatmap,detected_vehicle_image,ok_1_vehicle = analyse_dataset_batch(save=True,FILE_COUNT_TO_LOAD=FILE_COUNT_TO_LOAD,cfar_threshold=210000)
    
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


def dataset_visual_identification_analysis(FILE_DIRECTORY,export_video=True,show_plot=False,BATCH_SIZE=200,cfar_threshold=210000,FILE_COUNT_TO_LOAD=3000):
    BACKGROUND_FILE = os.path.join(FILE_DIRECTORY,"new_new_background.doppler")
    MEAN_HEATMAP_FILE = os.path.join(FILE_DIRECTORY,"mean_heatmap.doppler")
    

    heatmap_directory = os.path.join(FILE_DIRECTORY, "data","graphes")
    picture_directory = os.path.join(FILE_DIRECTORY, "data","images")
    output_directory = os.path.join(FILE_DIRECTORY, "data","annotated")


    timestamps_to_load_total = list([".".join(f.split(".")[:2]) for f in os.listdir(heatmap_directory) if "doppler" in f])
    timestamps_to_load_total = timestamps_to_load_total[0:min(FILE_COUNT_TO_LOAD,len(timestamps_to_load_total))]
    
    identifier = VehicleIdentifier().set_metric_example().set_metric_euclidian_distance().set_metric_bb_common_area().set_metric_time_delay()
    
    identified_lenght = {}
    
    
    # Delete content of output directory
    
    for f in os.listdir(output_directory):
        os.remove(os.path.join(output_directory,f))
    
    
    for batch_index in tqdm(range(0,math.ceil(len(timestamps_to_load_total)/BATCH_SIZE)),desc="Batch",leave=False):
    

        timestamps_to_load = timestamps_to_load_total[batch_index*BATCH_SIZE:min((batch_index+1)*BATCH_SIZE,len(timestamps_to_load_total))]

        dataWrapper = DataWrapper(heatmap_directory, picture_directory, timestamps_to_load, picture_name_prefix="0", picture_extension_suffix="jpeg", heatmap_extension_suffix="doppler", heatmap_name_prefix="",picture_output_dir=output_directory)
        
        dataWrapper.set_background_data(BACKGROUND_FILE)
        dataWrapper.set_mean_heatmap_data(MEAN_HEATMAP_FILE)


        
        
        # random_sample = rd.sample(range(FILE_COUNT_TO_LOAD),100)
        res_analyses = []
        
        
        for i in tqdm(range(len(timestamps_to_load)),desc="Analysing couple",leave=False): #(random_sample:
            # dataWrapper.plot(i,logarithmic=False,sign_color_map=False)
            res,res_ana = dataWrapper.pipeline_process(i,cfar_threshold=cfar_threshold)
            res_analyses.append(res_ana)
            if res is not None:
                vehicle_identifier,score_detail = identifier.identify(res_ana,get_score_detail=True)
                
                if vehicle_identifier not in identified_lenght:
                    identified_lenght[vehicle_identifier] = []
                    
                identified_lenght[vehicle_identifier].append([res_ana.timestamp,res_ana.caracteristic_length])
                
                dataWrapper.add_identification(i,res_ana.image_info[0]["bbox"],vehicle_identifier,score_details=score_detail)
                if show_plot:
                    dataWrapper.plot_CFAR(i,annotated=True)
            dataWrapper.save_picture(i,annotated=True)
    
    if export_video:
        
        generate_video_annotated(output_directory)
    
    
    plt.figure()
    
    for vehicle in identified_lenght:
        X = [x[0] for x in identified_lenght[vehicle]]
        Y = [x[1] for x in identified_lenght[vehicle]]
        
        plt.plot(X,Y,label=vehicle)
        
    plt.grid()
    # plt.legend()
    plt.show()
    
    # Remove the pictures from the output directory
    
    """for f in os.listdir(output_directory):
        if "jpg" in f:
            os.remove(os.path.join(output_directory,f))"""
            
                
    
    
