import os
import random as rd
from tqdm import tqdm
from ._dw_ import DataWrapper
import math
import numpy as np
import pickle

def labelize_dataset(FILE_DIRECTORY,output_dir,FILE_COUNT_TO_LABELIZE=None,silent=False,BATCH_SIZE=200,cfar_threshold=52000,length_threshold=1.1,gauss_kernel_length=11,gauss_sigma=0.5):
    """Main function used to labellize a dataset according to the specification.

    Args:
        input_dir (str): absolute path of the input directory.
        output_dir (str): absolute path of the output directory
        cfar_threshold (int, optional): Threshold for object detection in the heatmap. Defaults to 52000.
        length_threshold (float, optional): Threshold for detecting abnormally small object. Defaults to 1.1.
        gauss_kernel_length (int, optional): Length of the gaussian kernel. Defaults to 11.
        gauss_sigma (float, optional): Variance parameter of the gaussian kernel. Defaults to 0.5.
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
    if FILE_COUNT_TO_LABELIZE is not None:
        
        timestamps_to_load_total = timestamps_to_load_total[0:min(FILE_COUNT_TO_LABELIZE,len(timestamps_to_load_total))]
    
    for batch_index in tqdm(range(0,math.ceil(len(timestamps_to_load_total)/BATCH_SIZE)),desc="Batch",leave=False):
    

        timestamps_to_load = timestamps_to_load_total[batch_index*BATCH_SIZE:min((batch_index+1)*BATCH_SIZE,len(timestamps_to_load_total))]

        dataWrapper = DataWrapper(heatmap_directory, picture_directory, timestamps_to_load, picture_name_prefix="0", picture_extension_suffix="jpeg", heatmap_extension_suffix="doppler", heatmap_name_prefix="",silent=silent)
        
        dataWrapper.set_background_data(BACKGROUND_FILE)
        dataWrapper.set_mean_heatmap_data(MEAN_HEATMAP_FILE)

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
