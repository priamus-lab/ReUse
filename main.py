from utils_preprocess import Dataset, get_haralick

area = "europa"                                                   #"vietnam", "myanmar","europa"
path_agb = "Data/{}/agb.tif".format(area)
path_sentinel = "Data/{}/sentinel/2018/7/27/".format(area)          #"Data/{}/sentinel/2018/4/3/".format(area), "Data/{}/sentinel/2018/3/7/".format(area), "Data/{}/sentinel/2018/7/27/".format(area)
compute_haralick = False                                            #True, False
strategy = "DL"                                                     #"ML", "DL"
model = "UNet"                                                      #"Paper1", "Paper2", "UNet"

data = Dataset(path_agb, path_sentinel)
data.load_data(is_sentinel=False)
data.load_data(is_sentinel=True)
data.reprojection()
data.to_numpy(attribute="agb", is_list=False)
data.to_numpy(attribute="sentinel", is_list=True)
data.list_to_tensor(attribute="sentinel")
data.normalize(attribute="sentinel")

if strategy == "ML" and model=="Paper1":
    if compute_haralick:
        get_haralick(data.sentinel, band = 9 , padding = 2, kernel = 5, namefile = "banda12", area = area, ispca=False) 
    data.load_haralick(path = "Data/{}/haralick".format(area))
    data.get_spectral_index(model = model)
    data.stack_tensors(["normalized_sentinel", "haralick", "spectral_indexes"])

elif strategy == "ML" and model=="Paper2":
    data.compute_pca(n_components=1)
    if compute_haralick:
        get_haralick(data.pca, band = 0, padding = 2, kernel = 5, namefile = "first_component", area = area, ispca=True) 
    data.compute_wavelets(level=3)
    data.load_haralick(path = "Data/{}/haralick-pca".format(area))
    data.get_spectral_index(model = model)
    data.stack_tensors(["normalized_sentinel", "haralick", "wavelet", "spectral_indexes"])

elif strategy == "DL":
    data.stack_tensors(["normalized_sentinel"])

data.split_to_patch()
data.cross_validation(area, strategy=strategy, model_str=model)




