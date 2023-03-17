import numpy as np
import pickle
import os

def slog(data):
    """Utility function to get the log value without loosing the sign

    Args:
        data (np.array): array

    Returns:
        np.array: array
    """
    return np.nan_to_num(np.log(np.abs(data))*np.sign(data))

def load_file(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def generate_video_annotated(directory,video_name="annotated_video.mp4",extention="jpg",fps=12):
    #Renaming the files as ffmpeg does not like the timestamp format but want a sequence of numbers
    
    extention = "jpg"
    count = 0
    
    for i in os.listdir(directory):
        if i.endswith(extention):
            old_name = os.path.join(directory, i)
            new_name = os.path.join(directory, str(count) + "." + extention)
            
            os.rename(old_name, new_name)
            count += 1
    
    # Executing the ffmpeg command
    
    os.system(f"ffmpeg -f image2 -r {fps} -i {directory}/%d.{extention} -vcodec libx264 -crf 30 -pix_fmt yuv420p {directory}/{video_name}")