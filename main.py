from utils_preprocess import Dataset, get_all_haralick
from utils_models import get_dl_model
import rioxarray
import numpy as np
import joblib
from utils_inference import predict, save_prediction_to_raster

area = "area-name"                                                  #i.e., "europa"                                                  
path_agb = "Data/{}/agb.tif".format(area)
path_sentinel = "sentinel2-path"                                    #i.e.,"Data/{}/sentinel/2018/4/3/".format(area)
strategy = "DL"                                                     #"ML", "DL",
model = "UNet-Paper2"                                               #"Paper1", "Paper2", "UNet", "UNet-Paper2" 
goal = "inference"                                                  #"training","error-with-cv","inference" 
raster_src_uri = path_agb                                           #path of band to use in order to reproject
raster_dst_uri = "Results/area-inference/name-prediction.tif"       #path to save predictions
crop_size = 16

data = Dataset(path_agb, path_sentinel)
data.load_data(is_sentinel=False)
data.load_data(is_sentinel=True)
data.reprojection(on_agb = True)                                    
data.to_numpy(attribute="agb", is_list=False)
data.to_numpy(attribute="sentinel", is_list=True)
data.list_to_tensor(attribute="sentinel")
data.normalize(attribute="sentinel")

if strategy == "ML" and model=="Paper1":
    print("Paper1")
    if compute_haralick:
        get_all_haralick(data.sentinel, padding = 2, kernel = 5,  area = area, ispca=False) 
    data.load_haralick(path = "Data/{}/haralick".format(area))
    data.get_spectral_index(model = model)
    data.stack_tensors(["normalized_sentinel", "haralick", "spectral_indexes"])

elif strategy == "ML" and model=="Paper2":
    print("Paper2")
    data.compute_pca(n_components=1)
    if compute_haralick:
        get_all_haralick(data.pca, padding = 2, kernel = 5, area = area, ispca=True) 
    data.compute_wavelets(level=3)
    data.load_haralick(path = "Data/{}/haralick-pca".format(area))
    data.get_spectral_index(model = model)
    data.stack_tensors(["normalized_sentinel", "haralick", "wavelet", "spectral_indexes"])

elif strategy == "DL" and model=="UNet":
    print("UNet")
    data.stack_tensors(["normalized_sentinel"])

elif strategy == "DL" and model=="UNet-Paper2":
    print("UNet-Paper2")
    data.compute_pca(n_components=1)
    if compute_haralick:
        get_all_haralick(data.pca, padding = 2, kernel = 5, area = area, ispca=True)
    data.compute_wavelets(level=3)
    data.load_haralick(path = "Data/{}/haralick-pca".format(area))
    data.get_spectral_index(model = "Paper1")                  
    data.normalize(attribute="wavelet", norm_coeff=np.amax(np.abs(data.wavelet)))
    data.normalize(attribute="haralick", norm_coeff=np.amax(np.abs(data.haralick)))
    data.stack_tensors(["normalized_sentinel", "normalized_haralick", "normalized_wavelet", "spectral_indexes"])

if goal=="error-with-cv":
    print("Cross-Validation")
    data.split_to_patch(crop = crop_size)
    data.cross_validation(area, strategy=strategy, model_str=model)    #should be the method of a model class
elif goal=="training":
    print("training")
    data.split_to_patch(crop = crop_size)
    data.training(area, strategy=strategy, model_str=model)            #should be the method of a model class
elif goal=="inference":
    print("inference")
    sentinel_tensor = np.nan_to_num(data.long_tensor, copy=True, nan=-1.0)
    if strategy == "DL":
        model_instance, callback = get_dl_model(crop_size, crop_size, sentinel_tensor.shape[-1], "", "")
        model_instance.load_weights('models/area-train/models_dl/fold0.h5')
    elif strategy == "ML" and model == "Paper1":
        model_instance = joblib.load('models/area-train/models_ml/Paper1-fold0.joblib')
    elif strategy == "ML" and model == "Paper2":
        model_instance = joblib.load('models/area-train/models_ml/Paper2-fold0.joblib')
    labels = predict(sentinel_tensor, model_instance, strategy, crop_size)
    save_prediction_to_raster(labels, raster_src_uri, raster_dst_uri)



