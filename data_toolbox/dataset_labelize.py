import os
import random as rd
from tqdm import tqdm
from ._dw_ import DataWrapper
import math
import numpy as np
import pickle
from ._identifier import VehicleIdentifier
import json
import copy

def labelize_dataset(base_path,dataset_path,FILE_COUNT_TO_LABELIZE=None,silent=False,BATCH_SIZE=200,cfar_threshold=52000,length_threshold=1.1,gauss_kernel_length=11,gauss_sigma=0.5):
    """Main function used to labellize a dataset according to the specification.

    Args:
        input_dir (str): absolute path of the input directory.
        output_dir (str): absolute path of the output directory
        cfar_threshold (int, optional): Threshold for object detection in the heatmap. Defaults to 52000.
        length_threshold (float, optional): Threshold for detecting abnormally small object. Defaults to 1.1.
        gauss_kernel_length (int, optional): Length of the gaussian kernel. Defaults to 11.
        gauss_sigma (float, optional): Variance parameter of the gaussian kernel. Defaults to 0.5.
    """
    
    # Generate the file structure for the dataset
    session_path = generate_file_structure(dataset_path)
    DATASET_CAMERA_PATH = os.path.join(session_path,"camera")
    DATASET_RADAR_PATH = os.path.join(session_path,"radar")
    DATASET_FILTRED_RADAR_PATH = os.path.join(session_path,"filtred_radar")
    DATASET_VEHICLE_PATH = os.path.join(session_path,"vehicles")
    DATASET_VEHICLE_FILE = os.path.join(session_path,"vehicles","vehicles.json")


    
    
    
    #TODO: Need to copy file, do generate heatmap from raw data
    
    # Analysis of the dataset and creation of it
    
    
    BACKGROUND_FILE = os.path.join(base_path,"new_new_background.doppler")
    MEAN_HEATMAP_FILE = os.path.join(base_path,"mean_heatmap.doppler")
    

    heatmap_directory = os.path.join(base_path, "data","graphes")
    picture_directory = os.path.join(base_path, "data","images")
    
    count_error = 0 
    ok_1_vehicle = 0
    detected_vehicle_heatmap = []
    detected_vehicle_image = []
    missmatch_count_heatmap_image = []
    
    energy_heatmap = []
    pos_list = []
    timestamps_to_load_total = list([".".join(f.split(".")[:2]) for f in os.listdir(heatmap_directory) if "doppler" in f])
    
    identifier = VehicleIdentifier().set_metric_example().set_metric_euclidian_distance().set_metric_bb_common_area().set_metric_time_delay(weight=2)
    vehicle_id_dict = {}
    
    is_metadata_saved = False
    
    
    
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
        
        if not is_metadata_saved:
            with open(os.path.join(session_path,"metadata.json"),"w") as f:
                json.dump(dataWrapper.get_metadata(),f)
        
        
        
        for i in tqdm(range(len(timestamps_to_load)),desc="Analysing couple",leave=False): #(random_sample:
            # dataWrapper.plot(i,logarithmic=False,sign_color_map=False)
            
            # The filtred heatmap has to be saved during pipeline process as it is not saved in the datawrapper
            res,res_ana = dataWrapper.pipeline_process(i,cfar_threshold=cfar_threshold,gauss_kernel_length=gauss_kernel_length,gauss_sigma=gauss_sigma,save_path_filtred_heatmap=DATASET_FILTRED_RADAR_PATH)
            
            res_analyses.append(res_ana)
            if res is not None:
                vehicle_identifier,score_detail = identifier.identify(res_ana,get_score_detail=True)
                dataWrapper.add_identification(i,res_ana.image_info[0]["bbox"],vehicle_identifier,score_details=score_detail)

                if vehicle_identifier not in vehicle_id_dict:
                    vehicle_id_dict[vehicle_identifier] = []
                    
                vehicle_id_dict[vehicle_identifier].append(res_ana.export_dict())
            dataWrapper.save_picture(i,output_dir=DATASET_CAMERA_PATH,annotated=True)
            dataWrapper.save_heatmap(i,output_dir=DATASET_RADAR_PATH)
        
        # Flushing and saving the archived vehicle id to avoid memory overflow
        relevant_vehicles = identifier.get_identifier_pool().keys()
        archived_vehicle_id_dict = {}
        
        for vehicle_identifier in vehicle_id_dict:
            if vehicle_identifier not in relevant_vehicles:
                archived_vehicle_id_dict[vehicle_identifier] = copy.deepcopy(vehicle_id_dict[vehicle_identifier])
        for vehicle_identifier in archived_vehicle_id_dict:
            del vehicle_id_dict[vehicle_identifier]

        if len(archived_vehicle_id_dict) > 0:
            append_to_json_on_disk(DATASET_VEHICLE_FILE,archived_vehicle_id_dict)
    
    # Append the last vehicle id
    append_to_json_on_disk(DATASET_VEHICLE_FILE,vehicle_id_dict)
    
        
                
                


def generate_file_structure(FILE_PATH):
    # Create a save directory if it does not exist
    if not os.path.exists(FILE_PATH):
        os.makedirs(FILE_PATH)
        
    # Get the capture_session_id
    session_id = len(os.listdir(FILE_PATH))
    session_folder = os.path.join(FILE_PATH,str(session_id))
    
    os.makedirs(session_folder)
    
    # Create the different subdirectories
    
    os.makedirs(os.path.join(session_folder,"camera"))
    os.makedirs(os.path.join(session_folder,"radar"))
    os.makedirs(os.path.join(session_folder,"filtred_radar"))
    os.makedirs(os.path.join(session_folder,"vehicles"))
    
    return session_folder
    
    
    
def labelize(dataset_path):
    generate_file_structure(dataset_path)
    
    
def append_to_json_on_disk(json_path, data):
    if not os.path.exists(json_path):
        with open(json_path,"w") as f:
            f.write(json.dumps(data))
        return 
    
    # This is used to remove the last } and add a comma to the json file
    # This is done to avoid having to load the whole json file in memory
    with open(json_path,"rb+") as f:
        f.seek(-1, os.SEEK_END)
        f.truncate()
        f.write(b",")
        
    with open(json_path,"a") as f:
        f.write(json.dumps(data)[1:])
    