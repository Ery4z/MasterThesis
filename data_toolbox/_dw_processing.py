import numpy as np
from .analysis_utilities import MultimodalAnalysisResult,CounterVehicleCFAR, getPositionFromShapeAndData, calculate_energy,from_multimodal_analysis_result_to_3d,get_caracteristic_length
from ._math_utilities import corr_kernel,fast_convolution
from scipy.signal import convolve2d
import cv2


def filter(self,data,gauss_kernel_length=11,gauss_sigma=0.5):
    """Main function to modify the filtering of the heatmap data. You may overwrite this function.

    Args:
        data (np.array): Data to filter

    Returns:
        np.array: Filtered Data
    """
    if self.heatmap_mean is None:
        self.calculate_heatmap_mean()
    data = data - self.background_data
    data = data - self.heatmap_mean

    data = np.maximum(data, 0)
    
    # kernel = triangle_kernel(3,3)
    kernel = corr_kernel(gauss_kernel_length,gauss_kernel_length,gauss_sigma)
    # filtred_bg = scipy.signal.convolve2d(self.background_data, kernel, mode='same')
    
    
    data = convolve2d(data, kernel, mode='same')
    
    # data = data - filtred_bg
    return data

def pipeline_process(self,index,debug=False,cfar_threshold=52000,length_threshold=1.1,gauss_kernel_length=11,gauss_sigma=0.5):
        """Main function to process data couple.

        Args:
            index (int): Index of the couple to process.
            debug (bool, optional): Print warning message on error. Defaults to False.
            cfar_threshold (int, optional): Threshold for object detection in the heatmap. Defaults to 52000.
            length_threshold (float, optional): Threshold for detecting abnormally small object. Defaults to 1.1.
            gauss_kernel_length (int, optional): Length of the gaussian kernel. Defaults to 11.
            gauss_sigma (float, optional): Variance parameter of the gaussian kernel. Defaults to 0.5.

        Returns:
            [[3]int,MultimodalAnalysisResult]: Couple position of the detected object, result of the analysis.
        """
        result_analysis = self.analyse_couple(index, plot=False, cfar_threshold=cfar_threshold,gauss_kernel_length=gauss_kernel_length,gauss_sigma=gauss_sigma)
        
        # Check if the number of detected object match the number of object in the picture
        detected_heatmap_count = len(result_analysis.heatmap_info)
        detected_image_count = len(result_analysis.image_info)
        if detected_heatmap_count != detected_image_count:
            if debug: 
                print("WARNING: {} heatmap objects detected but {} image objects detected".format(detected_heatmap_count,detected_image_count))
            return None,result_analysis
        
        # TODO: Continue the analysis part, Convert the cordinate to 3D base
        
        '''
        Content of the info_3D_object
        
        {
            "position": [x,y,z],
            "corners":[[x,y,z],[x,y,z],[x,y,z],[x,y,z]],
            "object_type": "car|truck|bike|motorcycle|bus",
        }
        '''
        
        
        # TODO: Support multiple object
        if detected_heatmap_count != 1:
            return None,result_analysis
        
        pos_3d,bb_box_3d_pos = from_multimodal_analysis_result_to_3d(result_analysis,self.camera_parameters)
        result_analysis.set_bbox_3d(bb_box_3d_pos)
        # Check if the detected heatmap object is coherent with the picture object
        caracteristic_lenght = get_caracteristic_length(result_analysis)
        result_analysis.set_caracteristic_length(caracteristic_lenght)
        
        if caracteristic_lenght < length_threshold:
            return None,result_analysis
        
        result_analysis.set_position(pos_3d)
        
        
        return pos_3d, result_analysis
        
        
        # Check if the scale of the 3D object make sense



def analyse_couple(self,index,plot=False,cfar_threshold=30, gauss_kernel_length=11,gauss_sigma=0.5):
    """Utility function to analyse the couple of data.
    This function is not mean for data pipeline but for user end.

    Args:
        index (int): index of the couple
        plot (bool, optional): Plot or not the couple. Defaults to False.

    Returns:
        dict: The analyse
    """
    analyse = {
        "timestamp":self.timestamps_to_load[index],
        "heatmap_energy":0,
        "heatmap_info":[{"distance":0,"speed":0}]
    }
    
    solver = CounterVehicleCFAR()
    heatmap_data = self.heatmap_data[index]
    filtered_heatmap_data = self.filter(heatmap_data,gauss_kernel_length=gauss_kernel_length,gauss_sigma=gauss_sigma)
    
    cfar_data, _, spotted = self.calculate_CFAR(filtered_heatmap_data,threshold=cfar_threshold)
    
    result, index_list, shape_list = solver.countVehicle(spotted)
    
    vehicle_heatmap_info = []
    raw_vehicle_heatmap_info =[]
    
    heatmap_shape = self.heatmap_data.shape
    
    rdist = self.radar_parameters["distance"]
    rspeed = self.radar_parameters["speed"]
    if (len(shape_list)<=3):
        for shape in shape_list:
            pos = getPositionFromShapeAndData(shape, filtered_heatmap_data)
            
        
            
            
            raw_vehicle_heatmap_info.append(pos)
            # The weird index is because the [0] is the vertical axis and the [1] is the horizontal axis
            # the vertical axis is counted from top to bottom and the horizontal axis is counted from left to right
            vehicle_heatmap_info.append({
                "distance": (pos[0]/heatmap_shape[1])*(rdist[0]-rdist[1])+rdist[1],
                "speed": (pos[1]/heatmap_shape[2])*(rspeed[1]-rspeed[0])+rspeed[0]
            })
    
    analyse["heatmap_energy"] = calculate_energy(filtered_heatmap_data)
    analyse["heatmap_info"] = vehicle_heatmap_info
    analyse["raw_heatmap_info"] = raw_vehicle_heatmap_info
    
    yolo = self.analyse_image(index)
    yolo_result = yolo.pandas().xyxy[0]
    image_info = []
    for i in range(len(yolo_result.xmin)):
        image_info.append({
            "bbox": [int(yolo_result.xmin[i]),int(yolo_result.ymin[i]),int(yolo_result.xmax[i]),int(yolo_result.ymax[i])],
            "x": int((yolo_result.xmin[i]+yolo_result.xmax[i])/2),
            "y": int((yolo_result.ymin[i]+yolo_result.ymax[i])/2),
            "class": yolo_result.name[i],
            "confidence": yolo_result.confidence[i]
        })
    analyse["image_info"] = image_info
    
    # Get the image with bounding boxes
    #self.add_annotation(index,image_info)
    if plot:
        # Extract the image to a numpy array
        print(analyse)
        # bb_array = np.array(Image.open("tmp.jpeg"))
        # self.picture_data_annotated[index] = bb_array
        self.plot_CFAR(index, annotated=True)

    struct_analyse = MultimodalAnalysisResult(analyse["timestamp"],analyse["heatmap_energy"],analyse["heatmap_info"],analyse["raw_heatmap_info"],analyse["image_info"])
    
    return struct_analyse
    

def remove_background_data(self):
    """Subtract the background data from the loaded heatmap data
    """
    self.heatmap_data = self.heatmap_data - self.background_data




def calculate_heatmap_mean(self):
    """Calculate the mean and assign it to the good variable
    """
    self.heatmap_mean = np.mean(self.heatmap_data,axis=0)


def remove_temporal_mean(self):
    """Remove the mean of the loaded data (Please load the mean data before using set_mean_heatmap_data)
    """
    if self.heatmap_mean is None:
        self.calculate_heatmap_mean()
    self.heatmap_data = self.heatmap_data - self.heatmap_mean
    
    
def analyse_image(self,index):
    """Use yolo to analyse the image at the given index.

    Args:
        index (int): index to analyse

    Returns:
        yolov5 result: Torch Hub result for yolov5 prediction
    """

    # Model
    

    # Images
    img = self.picture_data[index]  # PIL, OpenCV, numpy, Tensor, etc.

    # Inference
    results = self.yolo(img)

    # Results
    return results  # or .show(), .save(), .crop(), .pandas(), etc.

def add_annotation(self,index,image_info):
    """Add the bounding box and annotation to the annotated image based on the index and data created by yolo

    Args:
        index (int): index of the image 
        image_info (pd.Dataframe): yolo prediction result converted to df
    """
    image = self.picture_data_annotated[index]
    thickness = 5
    
    for info in image_info:
        x1, y1, x2, y2 = info["bbox"]
        image = cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), thickness)
        image = cv2.putText(image, info["class"], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 0, 0), 8)
        image = cv2.putText(image, "{:.2f}".format(round(info["confidence"], 2)), (x2, y2), cv2.FONT_HERSHEY_SIMPLEX, 4, (255,0, 0), 8)
    
    self.picture_data_annotated[index] = image
    
def add_identification(self,index,bbox,label,score_details=None):
    image = self.picture_data_annotated[index]
    thickness = 5
    
    x1, y1, x2, y2 = bbox
    
    color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    hash_id = hash(label) % len(color)
    
    color_for_box = color[hash_id]
    
    image = cv2.rectangle(image, (x1, y1), (x2, y2), color_for_box, thickness)
    image = cv2.putText(image, str(label), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 16)
    image = cv2.putText(image, str(label), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 8)

    if score_details is not None:
        header = "Metric Name: Score_top_id | New"
        image = cv2.putText(image, header, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 10)
        image = cv2.putText(image, header, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
        
        
        for i,metric_detail in enumerate(score_details):
            text = str(metric_detail[0]) +":  "+ str(metric_detail[1]) + " | " + str(metric_detail[2])
            
            image = cv2.putText(image, text, (0, (i+2)*50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 10)
            image = cv2.putText(image,  text, (0, (i+2)*50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)
            

    self.picture_data_annotated[index] = image
    
def calculate_CFAR(self,data,threshold=30):
    """Calculate the CFAR

    Args:
        file_path (str): The path to the file to be processed. Has to be a .doppler file.
        output_dir (str): The path to the directory where the output files will be saved.
    """
    k, l, m = 10, 30, 3
    val = 1 / ((2*m+1)*(l-k))

    kernelG = np.zeros((2*l+1, 2*l+1))
    kernelG[l-m:l+m+1, :l-k] = val

    kernelD = np.zeros((2*l+1, 2*l+1))
    kernelD[l-m:l+m+1, l+k+1:] = val

    kernelH = np.zeros((2*l+1, 2*l+1))
    kernelH[:l-k, l-m:l+m+1] = val

    kernelB = np.zeros((2*l+1, 2*l+1))
    kernelB[l+k+1:, l-m:l+m+1] = val
    
    
    
    
    signal = data

    moyG = fast_convolution(signal, kernelG)
    moyD = fast_convolution(signal, kernelD)
    moyH = fast_convolution(signal, kernelH)
    moyB = fast_convolution(signal, kernelB)
    maxmoy = np.maximum.reduce([moyG, moyD, moyH, moyB])

    magncfar = fast_convolution(signal, np.ones((4, 4))/16) - 1.1*maxmoy
    threshcfar = max(threshold, 1+0.5*(np.max(magncfar)-1))
    loccfar = np.where(magncfar >= threshcfar)

    spotted = np.zeros(signal.shape)
    spotted[loccfar] = 1

    
    return magncfar, loccfar, spotted
    
