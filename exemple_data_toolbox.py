from data_toolbox import DataWrapper
import os
import numpy as np
import pickle
import random as rd
from tqdm import tqdm
import math

def analyse_dataset(BATCH_SIZE = 200,FILE_COUNT_TO_LOAD = 10000,FILE_DIRECTORY = os.path.dirname(os.path.realpath(__file__)),customfilter=None,cfar_threshold=52000,save=True,gauss_kernel_length=11,gauss_sigma=0.5,silent=False):
    
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



def analyse_dataset_image_count():
    FILE_COUNT_TO_LOAD=1000
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
        
        
if __name__ == "__main__":
    analyse_dataset_image_count()