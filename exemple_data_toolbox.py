from data_toolbox.dataset_analysis.read import plot_kernel_param_result
from data_toolbox.dataset_analysis.generate import dataset_visual_identification_analysis
import os

def main():
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    RELATIVE_PATH = os.path.join("results_analysis_kernel_param","analysis_ks_5,15__sigma_0.1,1,0.1_.json")
    PATH = os.path.join(BASE_PATH, RELATIVE_PATH)
    plot_kernel_param_result(PATH)
    
    
def main2():
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    dataset_visual_identification_analysis(BASE_PATH,FILE_COUNT_TO_LOAD=500)
    
    
if __name__ == "__main__":
    main2()