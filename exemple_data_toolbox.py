from data_toolbox.dataset_analysis.read import plot_kernel_param_result
from data_toolbox.dataset_analysis.generate import dataset_visual_identification_analysis
from data_toolbox.dataset_labelize import labelize_dataset

from data_toolbox.dataset_analysis.generate import analyse_labelized_dataset, analyse_labelized_dataset_session
import os

def plot_kernel_parameter_study():
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    RELATIVE_PATH = os.path.join("results_analysis_kernel_param","analysis_ks_5,15__sigma_0.1,1,0.1_.json")
    PATH = os.path.join(BASE_PATH, RELATIVE_PATH)
    plot_kernel_param_result(PATH)
    
    
def visual_identification_analysis():
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    dataset_visual_identification_analysis(BASE_PATH,FILE_COUNT_TO_LOAD=3000)

def labelize_the_dataset():
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    DATASET_PATH = os.path.join(BASE_PATH,"TEST_DATASET")
    labelize_dataset(BASE_PATH,DATASET_PATH,FILE_COUNT_TO_LABELIZE=3000)
    
def analyse_the_dataset():
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    DATASET_PATH = "F:\\RADARCAM_DATASET"
    analyse_labelized_dataset(DATASET_PATH)
    
def analyse_the_dataset_night_session():
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    DATASET_PATH = "F:\\RADARCAM_DATASET\\7"
    analyse_labelized_dataset_session(DATASET_PATH,extract=True)
    
if __name__ == "__main__":
    pass