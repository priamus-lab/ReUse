# ReUse
## Project Description
Recently, the United Nations Framework Convention on Climate Change(UNFCCC) established the Reducing Emissions from Deforestation and forest Degradation (REDD+) program 
that requires countries to report their carbon emissions and sink estimates through national greenhouse gas inventories (NGHGI). Developing systems
that can estimate the carbon absorbed by forests becomes essential. This work introduces a pixel-wise Regressive UNet to estimate carbon absorbed by forests and nature reserves without in situ 
observations, using open data from ESA's CCI Biomass project https://climate.esa.int/en/projects/biomass/ for Above Ground Biomass (AGB) together with Sentinel-2 imagery
https://scihub.copernicus.eu/. 
In addition to UNet, two classical machine learning approaches with related feature extraction are implemented.
## Getting started
git clone https://github.com/priamus-lab/ReUse.git
## How to Install and Run the Project
### install dependencies with conda
The following line allows you to create a conda environment with all dependencies installed (if you do not have miniconda, you can download it from this 
page https://docs.conda.io/en/latest/miniconda.html; to install instead, tensorflow compatible with the use of a GPU see the link https://www.tensorflow.org/install/pip)

```
conda env create -n ENVNAME --file environment.yml
```

### set variables in the main.py
from line 8 to 16 in main.py

```
area = "area-name"                                                  #i.e., "europa"                                                  
path_agb = "Data/{}/agb.tif".format(area)                           #leave like this
path_sentinel = "sentinel2-path"                                    #i.e.,"Data/{}/sentinel/2018/4/3/".format(area)
strategy = "DL"                                                     #choose one of the following: "ML", "DL",
model = "UNet-Paper2"                                               #choose one of the following: "Paper1", "Paper2", "UNet", "UNet-Paper2" 
goal = "inference"                                                  #choose one of the following: "training","error-with-cv","inference" 
raster_src_uri = path_agb                                           #band path to be used for reprojecting
raster_dst_uri = "Results/area-inference/name-prediction.tif"       #path to save predictions
```

### add folders to the project
#### "Data" folder
From the root, create a folder 'Data'. In the folder "Data" create a folder with the name of the study area (the same name you assigned to the variable 'area' above). 
In this new folder of the study area, load the AGB raster and, call it "agb" and create the following folders: "haralick", "haralick-pca", "sentinel".
In the "sentinel" folder, upload the sentinel-2 data using the following path: "sentinel/year/month/day/".
Note that in "haralick" folder, you will find the texture features extracted for "Paper1" and in "haralick-pca" folder, you will find the texture features extracted for "Paper2" after that you launch the code.
#### "models" folder
From the root, create a folder "models". Inside this folder, create another folder with the name of the study area (the same name you assigned to the variable 'area' above).
In this new folder of the study area, create a folder named "models_dl". Here you will find the saved UNet models for different models trained using eight-fold cross-validation.
#### "Results" folder
From the root create a folder "Results". Inside this folder, create another folder with the name of the study area (the same name you assigned to the variable 'area' above).
Here you will find the results in different metrics after training using eight-fold cross-validation.
### Run the project
run  main.py

```
python main.py
```

## How to cite

Pascarella, A.E.; Giacco, G.; Rigiroli, M.; Marrone, S.; Sansone, C. ReUse: REgressive Unet for Carbon Storage and above Ground Biomass Estimation. Preprints 2022, 2022120158 (doi: 10.20944/preprints202212.0158.v1).



