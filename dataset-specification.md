# Dataset Specification

This document describes the specification, justification and structure of the dataset created using this project.

## 1. Motivation and Justification

The original data gathered by François Ledent during the summer 2022 were pairs of images captured by a camera and radar data of the scene.
The provided scripts allowed to quickly gets hands on data and provide a base for more elaborated, polished and general purpose scripts to be developed.

The goal of this dataset is to provide labelized data of bimodal camera-radar system. An use of this dataset would be to train a model to predict vehicle position using camera and radar data.

Using the provided data it is possible to extract the 3D position of the vehicle using yolo information, distance of the object from the radar and teh intrinsic parameters of the camera and radar.

For the start, only one vehicle will be present per pair of data as the radar used for data collection does not allow to discriminate between multiple objects.

---

## 2. Dataset Structure

The idea between this dataset is to provide two possible way to access the data:

-   Going through each pair of data and extract the information from the images and the radar data in a raw way (as it was provided by François Ledent) or using filtred heatmap data.
-   Going through each vehicles and being able to retrieve more abstractly the 3D information of the vehicle.

### 2.1 Folder Structure

The data set is structured as follow:

```bash
dataset
├───<capture_session_id>
│   ├───camera
│   │   └───<image_id>.jpg
│   ├───radar
│   │   └───<radar_id>.radar
│   ├───filtered_radar
│   │   └───<radar_id>.fradar
│   ├───metadata
│   │   └───<capture_session_id>_meta.json
│   └───vehicles
│       └───<capture_session_id>_vehicles.json
└───...

```

---

## 2.2 Data Format

### 2.2.1 Camera

The camera data is provided as a set of images in the `jpg` format. The image id is constructed as follow: `<capture_session_id>_<timestamp>`. The timestamp is the time at which the image was captured in milliseconds (TO REVIEW) since the Unix epoch. The camera id is the id of the camera used to capture the image.

> Note: On the first version of the dataset the image data is a 1920x1080 jpg.

### 2.2.2 Radar

The radar data is provided as a set or numpy array saved in the pickle format. The use of `.radar` extension is to distinguish the filtred radar data from the raw radar data. The radar id is constructed as follow: `<capture_session_id>_<timestamp>`. The timestamp is the time at which the image was captured in milliseconds (TO REVIEW) since the Unix epoch. The radar id is the id of the radar used to capture the image.

> Note: On the first version of the dataset the radar data is a 256x256 numpy array. The radar data is not normalized.

### 2.2.3 Filtred Radar

The filtred radar data is in the same format as the radar one. The filtering has been the following:

-   The background has been removed using mean value of the radar heatmap when no vehicle is present.
-   The data has been thresholded to remove negative values.
-   The data has been filtred using a gaussian filter with a sigma of 0.5 and a kernel size of 11x11.

### 2.2.4 Metadata

The metadata file is a json file with the following structure:

```json
{
    "camera": {
        "resolution": "<resolution>",
        "intrinsic_parameters": {
            "focal_length": "<focal_length>",
            "principal_point": "<principal_point>",
            "skew": "<skew>", // In the first version of the dataset this value is 0
            "distortion_coefficients": "<distortion_coefficients>" // In the first version of the dataset this value is 0
        }
    },
    "radar": {
        "d_max": "<d_max>",
        "d_min": "<d_min>",
        "s_max": "<s_max>",
        "s_min": "<s_min>"
    }
}
```

### 2.2.5 Vehicles

The vehicles file is a json file with the following structure:

```json
{
    "vehicle_<id>": [
        {
            "couple_id": "<couple_id>",
            "timestamp": "<timestamp>",
            "bb": ["<x_min>", "<y_min>", "<x_max>", "<y_max>"],
            "x": "<x>",
            "y": "<y>",
            "z": "<z>",
            "v": "<v>",
            "d": "<d>"
        },
        ...
    ],
    ...
}
```

This structure allow to cycle through each vehicle and get the information of each couple of data in which the vehicle is present in a linear way.
