"""
Code réalisant l'alignement des séquences et l'extraction du contenu des trames radar
pour les prises de vue réalisées dans le cadre du mémoire :

Synchronization of Multimodal Data Flows for Real-Time AI Analysis
Juin 2022, LEDENT François, 2023 BOLTEAU Thomas
"""


import numpy as np
import os
import shutil
import pickle
import time

from multiprocessing import Pool, cpu_count
from multiprocessing.managers import BaseManager
from PIL import Image
import PIL
from tqdm import tqdm


NCPU = cpu_count()-1
SAVE_CONTENT = True
SAVE_DOPPLER = True
SAVE_MAGN = True

def verify_and_delete_images_in_dir(directory):
    for file in tqdm(os.listdir(directory),desc="Verifying images",unit="images",leave=False):
        data = None
        try:
            with Image.open(os.path.join(directory, file)) as img:
                data = np.array(img)
        except:
            os.remove(os.path.join(directory, file))
        


def check_subdirs(*paths):
    for path in paths:
        if not os.path.isdir(path):
            return False
    return True


def create_subdirs(*paths):
    for path in paths:
        if not os.path.isdir(path):
            os.makedirs(path)


def get_files_times(folder, extension):
    files = np.array(
        [file.replace(extension, "") for file in os.listdir(folder) if extension in file])
    files.sort()
    times = np.array([file for file in files], dtype=float)
    return files, times


def compare(value1, value2, refvalue):
    return value1 if abs(value1-refvalue) <= abs(value2-refvalue) else value2


def manage(source_file, dest_file):
    shutil.copy(source_file, dest_file)


def get_filename(directory, timestamp, extension):
    return "{}{:018.6f}{}".format(directory, timestamp, extension)


def align_seq(path_source_radar, path_source_camera, path_dest_radar, path_dest_camera):
    assert check_subdirs(path_source_camera, path_source_radar), "assertion"
    create_subdirs(path_dest_radar, path_dest_camera)

    _, times_radar = get_files_times(path_source_radar, ".raw")
    _, times_camera = get_files_times(path_source_camera, ".jpeg")

    progress_bar = ProgressBar(times_radar.shape[0], "Align Sequences", max_upd_rate=50)
    i_camera = 1
    for i_radar in range(times_radar.shape[0]):
        try:
            while times_camera[i_camera] <= times_radar[i_radar]:
                i_camera += 1
        except IndexError:
            os.remove(os.path.join(path_source_radar , "{:017.6f}".format(times_radar[i_radar]) + ".raw"))
            continue
        choice = compare(times_camera[i_camera-1], times_camera[i_camera], times_radar[i_radar])
        source_file = os.path.join(path_source_camera, "{:018.6f}".format(choice)+ ".jpeg")
        dest_file = os.path.join(path_dest_camera, "{:018.6f}".format(times_radar[i_radar])+ ".jpeg")
        manage(source_file, dest_file)
        progress_bar.increment()


def save_file(filename, content):
    with open(filename, "wb+") as f:
        pickle.dump(content, f)


def load_file(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def content_doppler_magn(file, inst, path_source_radar, path_dest_radar,save_one_file):

    content = np.empty((256, 256), dtype="complex")

    bFile = open(os.path.join(path_source_radar,file+".raw"), "rb").readlines()[0]
    bFile = np.frombuffer(bFile, dtype=np.uint16)

    size = 2 * 256 * 256
    if not save_one_file:
        for i in range(3):
            subarray = bFile[i * size: (i+1) * size]
            content = (subarray[0::2] + 1j * subarray[1::2]).reshape((256, 256))
            if SAVE_CONTENT:
                save_file(os.path.join(path_dest_radar,file+f".content{i}"), content)

            if SAVE_DOPPLER or SAVE_MAGN:
                doppler = np.fft.fftshift(np.fft.fft2(content), 0).T
                if SAVE_DOPPLER:
                    save_file(os.path.join(path_dest_radar,file+f".doppler{i}"), doppler)

            if SAVE_MAGN:
                doppler[doppler == 0] = 1e-12
                magn = 20*np.log(np.abs(doppler))
                save_file(os.path.join(path_dest_radar,file+f".magn{i}"), magn)
    else:
        contentToProcess = np.zeros((256, 256), dtype="complex")

        for i in range(3):
            subarray = bFile[i * size: (i+1) * size]
            content = (subarray[0::2] + 1j * subarray[1::2]).reshape((256, 256))
            contentToProcess += content
            
        if SAVE_CONTENT:
            save_file(os.path.join(path_dest_radar,file+f".content"), contentToProcess)

        if SAVE_DOPPLER or SAVE_MAGN:
            doppler = np.fft.fftshift(np.fft.fft2(contentToProcess), 0).T
            if SAVE_DOPPLER:
                save_file(os.path.join(path_dest_radar,file+f".doppler"), doppler)

        if SAVE_MAGN:
            doppler[doppler == 0] = 1e-12
            magn = 20*np.log(np.abs(doppler))
            save_file(os.path.join(path_dest_radar,file+f".magn"), magn)

    inst.increment()


class ProgressBar(object):
    def __init__(self, total, title="undefined", length=50, max_upd_rate=5):
        self.title = title
        self.total = total
        self.length = length
        self.min_delay = 1 / max_upd_rate
        self.init_time = time.time()
        self.value = 0
        self.last_update = 0

    @property
    def working_time(self):
        return time.time()-self.init_time

    def update_bar(self):
        if not (self.value >= self.total or time.time()-self.last_update > self.min_delay):
            return

        work_time = self.working_time
        pourc = self.value/self.total
        estim = work_time * (self.total/self.value - 1)
        complete = int(pourc*self.length)  # █
        not_complete = self.length-complete  # ▫▪■□

        if self.value == self.total:
            print("\r{:18} |{}{}| {:6d}/{:6d} in {:7.3f} sec, {:7.3f} sec/it. (terminated){}".format(self.title,
                                                                                                     "█"*complete, "▫"*not_complete, self.value, self.total, work_time, work_time/self.value, " "*2))
        else:
            print("\r{:18} |{}{}| {:6d}/{:6d} in {:7.3f} sec, {:7.3f} sec/it. (>{:7.3f} sec)\r".format(self.title,
                                                                                                     "█"*complete, "▫"*not_complete, self.value, self.total, work_time, work_time/self.value, estim), end="")
            self.last_update = time.time()

    def warning(*text):
        print(*text)

    def check_value(self, _value):
        if _value < 0:
            raise RuntimeError(
                "Can not set the value of the ProgressBar to {} because it can not be negative.".format(_value))
        if _value > self.total:
            self.warning("\nCan not set the value of the ProgressBar to {} because total is {}.\n\
                            Execution is not stopped and continues.".format(_value, self.total))
            return False
        return True

    def increment(self):
        if self.check_value(self.value+1):
            self.value += 1
        self.update_bar()


def get_graphs(path_source_radar, path_dest_radar,save_one_file=False):
    assert check_subdirs(path_source_radar), "assertion"
    create_subdirs(path_dest_radar)

    files_radar, _ = get_files_times(path_source_radar, ".raw")
    pool = Pool(processes=NCPU)

    BaseManager.register('ProgressBar', ProgressBar)
    manager = BaseManager()
    manager.start()
    inst = manager.ProgressBar(files_radar.shape[0], "Doppler Graphs")
    #inst = ProgressBar(files_radar.shape[0], "Doppler Graphs")

    for file in files_radar:
        pool.apply_async(content_doppler_magn, args=(file, inst, path_source_radar, path_dest_radar,save_one_file))
        #content_doppler_magn(file, inst)

    pool.close()
    pool.join()


# not used
if __name__ == "__mp_main__":
    pass

if __name__ == "__main__":
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"data")
    path_source_radar = os.path.join(path,"raw")
    path_source_camera = os.path.join(path,"jpeg")
    path_dest_radar = os.path.join(path,"graphes")
    path_dest_camera = os.path.join(path,"images")
    
    # Some images are corrupted, we delete them
    verify_and_delete_images_in_dir(path_source_camera)
    
    # We align the images and the radar data
    align_seq(path_source_radar, path_source_camera, path_dest_radar, path_dest_camera)
    
    # We generate the radar heatmaps
    get_graphs(path_source_radar, path_dest_radar,True)
