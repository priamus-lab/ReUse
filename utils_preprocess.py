import os
import mahotas
import numpy as np
import rioxarray
from sklearn.decomposition import PCA
import pywt
import cv2
from utils_models import get_dl_model, get_ml_model


class Dataset():
    def __init__(self, path_agb, path_sentinel):
        self.path_agb = path_agb
        self.path_sentinel = path_sentinel
        self.sentinel = []
        self.normalized_sentinel = []
        self.agb = []
        self.haralick = []
        self.spectral_indexes = []
        self.long_tensor = []
        self.patches = []
        self.agb_patches = []
        self.pca = []
        self.wavelet = []

    def load_data(self, is_sentinel):
        if is_sentinel:
            sentinel_list = []
            anno = self.path_sentinel.split("/")[-4]   #self.path_sentinel[-9:-5]
            mese = self.path_sentinel.split("/")[-3]   #self.path_sentinel[-4:-3]
            giorno = self.path_sentinel.split("/")[-2] #self.path_sentinel[-2:-1]
            if len(giorno)<2:
                giorno = "0" + giorno
            for band in range(1,12):
                if (band+1)<9:
                    sentinel_list.append(rioxarray.open_rasterio(self.path_sentinel + "B0{}_{}0{}{}.tif".format(band+1,anno,mese,giorno)))
                if (band+1)==9:
                    sentinel_list.append(rioxarray.open_rasterio(self.path_sentinel + "B8A_{}0{}{}.tif".format(anno,mese,giorno)))
                elif (band+1)>10:
                    sentinel_list.append(rioxarray.open_rasterio(self.path_sentinel + "B{}_{}0{}{}.tif".format(band+1,anno,mese,giorno)))  
            self.sentinel =  sentinel_list  
        else:
            agb = rioxarray.open_rasterio(self.path_agb)
            self.agb = agb

    def reprojection(self):
        sentinel_reprojected_list = []
        for idx in range(len(self.sentinel)):
            sentinel_reprojected = self.sentinel[idx].rio.reproject_match(self.agb)
            sentinel_reprojected_list.append(sentinel_reprojected)
        self.sentinel = sentinel_reprojected_list

    def to_numpy(self, attribute, is_list):
        if is_list:
            for i in range(len(self.__dict__[attribute])):
                self.__dict__[attribute][i] = self.__dict__[attribute][i].to_numpy()
        else:
            self.__dict__[attribute] = self.__dict__[attribute].to_numpy()
            
    def list_to_tensor(self, attribute):
        #self.sentinel = np.stack(self.sentinel, axis=-1)
        self.__dict__[attribute] = np.stack(self.__dict__[attribute], axis=-1) 

    def normalize(self, attribute, norm_coeff = 10000):
        self.__dict__["normalized_{}".format(attribute)] = self.__dict__[attribute]/norm_coeff
    
    def compute_pca(self, n_components = 1):
        tabular = self.sentinel.reshape((-1,self.sentinel.shape[3]))
        pca = PCA(n_components=n_components)
        principal_component = pca.fit_transform(tabular).astype(int)
        self.pca = principal_component.reshape((self.sentinel.shape[0],self.sentinel.shape[1],
                                                self.sentinel.shape[2],n_components))

    def compute_wavelets(self, level):
        LL = self.pca
        details_list = []
        for i in range(level):
            LL, (LH, HL, HH) = pywt.dwt2(np.average(LL, axis = 3).reshape((1,LL.shape[1],LL.shape[2],1)), 'coif1')
            details_list.append(np.average(LH, axis =3)) 
            details_list.append(np.average(HL, axis =3)) 
            details_list.append(np.average(HH, axis =3)) 
        wavelet_list_resized = []
        for image in details_list:
            wavelet_list_resized.append(cv2.resize(image[0,:,:], (self.pca.shape[2], self.pca.shape[1]),
            interpolation=cv2.INTER_NEAREST).reshape((self.pca.shape[0],self.pca.shape[1],self.pca.shape[2],self.pca.shape[3])))
        self.wavelet = np.concatenate((wavelet_list_resized), axis = 3)
            
    def load_haralick(self, path):
        file_list = os.listdir(path)
        haralick_list = []
        for file in file_list:
            haralick_list.append(np.load(os.path.join(path, file))[:,:,[5,3,4,1,9,8,0,2]])
        self.haralick = np.expand_dims(np.dstack(haralick_list), axis = 0)
    
    def get_spectral_index(self, model):
        if model == "Paper1":
            ng = self.normalized_sentinel[0,:,:,1]/(self.normalized_sentinel[0,:,:,6] + self.normalized_sentinel[0,:,:,1] + self.normalized_sentinel[0,:,:,2])
            ndii = (self.normalized_sentinel[0,:,:,6] - self.normalized_sentinel[0,:,:,8])/(self.normalized_sentinel[0,:,:,6] + self.normalized_sentinel[0,:,:,8])
            gndvi = (self.normalized_sentinel[0,:,:,6] - self.normalized_sentinel[0,:,:,1])/(self.normalized_sentinel[0,:,:,6] + self.normalized_sentinel[0,:,:,1])
            ndwi = (self.normalized_sentinel[0,:,:,1] - self.normalized_sentinel[0,:,:,6]) / (self.normalized_sentinel[0,:,:,1] + self.normalized_sentinel[0,:,:,6])
            cig = (self.normalized_sentinel[0,:,:,6] / self.normalized_sentinel[0,:,:,1]) - 1.0
            msi = self.normalized_sentinel[0,:,:,8]/self.normalized_sentinel[0,:,:,6]
            vari_g = (self.normalized_sentinel[0,:,:,1] - self.normalized_sentinel[0,:,:,2]) / (self.normalized_sentinel[0,:,:,1] + self.normalized_sentinel[0,:,:,2])
            cire = (self.normalized_sentinel[0,:,:,6] / self.normalized_sentinel[0,:,:,3]) - 1.0
            dvi = self.normalized_sentinel[0,:,:,6] - self.normalized_sentinel[0,:,:,2]
            evi = 2.5 * (self.normalized_sentinel[0,:,:,6] - self.normalized_sentinel[0,:,:,2]) / (self.normalized_sentinel[0,:,:,6] + 6 * self.normalized_sentinel[0,:,:,2] - 7.5 * self.normalized_sentinel[0,:,:,0] + 1.0)   
            evire1 = 2.5*(self.normalized_sentinel[0,:,:,3]-self.normalized_sentinel[0,:,:,2])/(1+self.normalized_sentinel[0,:,:,3]+6*self.normalized_sentinel[0,:,:,2]-7.5*self.normalized_sentinel[0,:,:,0])
            evire2 = 2.5*(self.normalized_sentinel[0,:,:,4]-self.normalized_sentinel[0,:,:,2])/(1+self.normalized_sentinel[0,:,:,4]+6*self.normalized_sentinel[0,:,:,2]-7.5*self.normalized_sentinel[0,:,:,0])
            evire3 = 2.5*(self.normalized_sentinel[0,:,:,5]-self.normalized_sentinel[0,:,:,2])/(1+self.normalized_sentinel[0,:,:,5]+6*self.normalized_sentinel[0,:,:,2]-7.5*self.normalized_sentinel[0,:,:,0])
            evinir2 = 2.5*(self.normalized_sentinel[0,:,:,7]-self.normalized_sentinel[0,:,:,2])/(1+self.normalized_sentinel[0,:,:,7]+6*self.normalized_sentinel[0,:,:,2]-7.5*self.normalized_sentinel[0,:,:,0])
            gari = (self.normalized_sentinel[0,:,:,6] - (self.normalized_sentinel[0,:,:,1] - (self.normalized_sentinel[0,:,:,0] - self.normalized_sentinel[0,:,:,2]))) / (self.normalized_sentinel[0,:,:,6] - (self.normalized_sentinel[0,:,:,1] + (self.normalized_sentinel[0,:,:,0] - self.normalized_sentinel[0,:,:,2])))
            gdvi = self.normalized_sentinel[0,:,:,6] - self.normalized_sentinel[0,:,:,1]
            ireci = (self.normalized_sentinel[0,:,:,5] - self.normalized_sentinel[0,:,:,2]) / (self.normalized_sentinel[0,:,:,3] / self.normalized_sentinel[0,:,:,4])
            mcari = ((self.normalized_sentinel[0,:,:,3] - self.normalized_sentinel[0,:,:,2]) - 0.2 * (self.normalized_sentinel[0,:,:,3] - self.normalized_sentinel[0,:,:,1])) * (self.normalized_sentinel[0,:,:,3] / self.normalized_sentinel[0,:,:,2])
            msavi = 0.5 * (2.0 * self.normalized_sentinel[0,:,:,6] + 1 - (((2 * self.normalized_sentinel[0,:,:,6] + 1) ** 2) - 8 * (self.normalized_sentinel[0,:,:,6] - self.normalized_sentinel[0,:,:,2])) ** 0.5)
            msr = (self.normalized_sentinel[0,:,:,6] / self.normalized_sentinel[0,:,:,2] - 1) / ((self.normalized_sentinel[0,:,:,6] / self.normalized_sentinel[0,:,:,2] + 1) ** 0.5)
            msrre1 = (self.normalized_sentinel[0,:,:,3] / self.normalized_sentinel[0,:,:,2] - 1) / ((self.normalized_sentinel[0,:,:,3] / self.normalized_sentinel[0,:,:,2] + 1) ** 0.5)
            msrre2 = (self.normalized_sentinel[0,:,:,4] / self.normalized_sentinel[0,:,:,2] - 1) / ((self.normalized_sentinel[0,:,:,4] / self.normalized_sentinel[0,:,:,2] + 1) ** 0.5)
            msrre3 = (self.normalized_sentinel[0,:,:,5] / self.normalized_sentinel[0,:,:,2] - 1) / ((self.normalized_sentinel[0,:,:,5] / self.normalized_sentinel[0,:,:,2] + 1) ** 0.5)
            msrnir2 = (self.normalized_sentinel[0,:,:,7] / self.normalized_sentinel[0,:,:,2] - 1) / ((self.normalized_sentinel[0,:,:,7] / self.normalized_sentinel[0,:,:,2] + 1) ** 0.5)
            ndvi = (self.normalized_sentinel[0,:,:,6] - self.normalized_sentinel[0,:,:,2])/(self.normalized_sentinel[0,:,:,6] + self.normalized_sentinel[0,:,:,2])
            ndvi705 = (self.normalized_sentinel[0,:,:,4] - self.normalized_sentinel[0,:,:,3]) / (self.normalized_sentinel[0,:,:,4] + self.normalized_sentinel[0,:,:,3])
            nli = ((self.normalized_sentinel[0,:,:,6] ** 2) - self.normalized_sentinel[0,:,:,2])/((self.normalized_sentinel[0,:,:,6] ** 2) + self.normalized_sentinel[0,:,:,2])
            nlire1 = ((self.normalized_sentinel[0,:,:,3] ** 2) - self.normalized_sentinel[0,:,:,2])/((self.normalized_sentinel[0,:,:,3] ** 2) + self.normalized_sentinel[0,:,:,2])    
            nlire2 = ((self.normalized_sentinel[0,:,:,4] ** 2) - self.normalized_sentinel[0,:,:,2])/((self.normalized_sentinel[0,:,:,4] ** 2) + self.normalized_sentinel[0,:,:,2]) 
            nlire3 = ((self.normalized_sentinel[0,:,:,5] ** 2) - self.normalized_sentinel[0,:,:,2])/((self.normalized_sentinel[0,:,:,5] ** 2) + self.normalized_sentinel[0,:,:,2])  
            nlinir2 = ((self.normalized_sentinel[0,:,:,7] ** 2) - self.normalized_sentinel[0,:,:,2])/((self.normalized_sentinel[0,:,:,7] ** 2) + self.normalized_sentinel[0,:,:,2])
            nnir = self.normalized_sentinel[0,:,:,5]/(self.normalized_sentinel[0,:,:,5] + self.normalized_sentinel[0,:,:,2] +self.normalized_sentinel[0,:,:,1])
            normR = self.normalized_sentinel[0,:,:,2]/(self.normalized_sentinel[0,:,:,6] + self.normalized_sentinel[0,:,:,1] + self.normalized_sentinel[0,:,:,2])
            psri = (self.normalized_sentinel[0,:,:,2] - self.normalized_sentinel[0,:,:,0])/self.normalized_sentinel[0,:,:,4]
            psrinir = (self.normalized_sentinel[0,:,:,6] - self.normalized_sentinel[0,:,:,0])/self.normalized_sentinel[0,:,:,4]
            pssr = self.normalized_sentinel[0,:,:,5] * self.normalized_sentinel[0,:,:,2]
            rdvi = (self.normalized_sentinel[0,:,:,6] - self.normalized_sentinel[0,:,:,2]) / ((self.normalized_sentinel[0,:,:,6] + self.normalized_sentinel[0,:,:,2]) ** 0.5)
            savi = (1.5) * (self.normalized_sentinel[0,:,:,6] - self.normalized_sentinel[0,:,:,2]) / (self.normalized_sentinel[0,:,:,6] + self.normalized_sentinel[0,:,:,2] + 0.5)
            tsavi =  (self.normalized_sentinel[0,:,:,6] -   self.normalized_sentinel[0,:,:,2] ) / (  self.normalized_sentinel[0,:,:,6] + self.normalized_sentinel[0,:,:,2] )
            wdrvi = (0.01 * self.normalized_sentinel[0,:,:,6] - self.normalized_sentinel[0,:,:,2]) / (0.01 * self.normalized_sentinel[0,:,:,6] + self.normalized_sentinel[0,:,:,2])
            wdrvirededge = (0.01 * self.normalized_sentinel[0,:,:,6] - self.normalized_sentinel[0,:,:,3]) / (0.01 * self.normalized_sentinel[0,:,:,6] + self.normalized_sentinel[0,:,:,3])
            self.spectral_indexes = np.expand_dims(np.stack([ng, ndii, gndvi, ndwi, cig, msi, vari_g, cire, dvi, evi, evire1, evire2, evire3, evinir2, 
                                    gari, gdvi, ireci, mcari, msavi, msr, msrre1, msrre2, msrre3, msrnir2, ndvi,
                                    ndvi705, nli, nlire1, nlire2, nlire3, nlinir2, nnir, normR, psri, psrinir, pssr,
                                    rdvi, savi, tsavi, wdrvi, wdrvirededge], axis = -1), axis = 0)
        elif model == "Paper2":
            ndvi = (self.normalized_sentinel[0,:,:,6] - self.normalized_sentinel[0,:,:,2])/(self.normalized_sentinel[0,:,:,6] + self.normalized_sentinel[0,:,:,2])
            savi = (1.5) * (self.normalized_sentinel[0,:,:,6] - self.normalized_sentinel[0,:,:,2]) / (self.normalized_sentinel[0,:,:,6] + self.normalized_sentinel[0,:,:,2] + 0.5)
            evi = 2.5 * (self.normalized_sentinel[0,:,:,6] - self.normalized_sentinel[0,:,:,2]) / (self.normalized_sentinel[0,:,:,6] + 6 * self.normalized_sentinel[0,:,:,2] - 7.5 * self.normalized_sentinel[0,:,:,0] + 1.0)   
            gndvi = (self.normalized_sentinel[0,:,:,6] - self.normalized_sentinel[0,:,:,1])/(self.normalized_sentinel[0,:,:,6] + self.normalized_sentinel[0,:,:,1])
            ndwi = (self.normalized_sentinel[0,:,:,1] - self.normalized_sentinel[0,:,:,6]) / (self.normalized_sentinel[0,:,:,1] + self.normalized_sentinel[0,:,:,6])
            wdvi = self.normalized_sentinel[0,:,:,6] - 0.5*self.normalized_sentinel[0,:,:,2]
            sr = self.normalized_sentinel[0,:,:,6]/self.normalized_sentinel[0,:,:,2]
            ndi45 = (self.normalized_sentinel[0,:,:,3] - self.normalized_sentinel[0,:,:,2])/(self.normalized_sentinel[0,:,:,3] + self.normalized_sentinel[0,:,:,2])
            mtci = (self.normalized_sentinel[0,:,:,4] - self.normalized_sentinel[0,:,:,3])/(self.normalized_sentinel[0,:,:,3] - self.normalized_sentinel[0,:,:,2] + 0.1)
            rendvi = self.normalized_sentinel[0,:,:,6] - self.normalized_sentinel[0,:,:,3]/self.normalized_sentinel[0,:,:,6] + self.normalized_sentinel[0,:,:,3]
            reevi = 2.5*(self.normalized_sentinel[0,:,:,6] - self.normalized_sentinel[0,:,:,3]/self.normalized_sentinel[0,:,:,6] + 2.4*self.normalized_sentinel[0,:,:,3]+1)
            self.spectral_indexes = np.expand_dims(np.stack([ndvi, savi, evi, gndvi, ndwi, wdvi, sr, ndi45, mtci, rendvi, reevi], axis = -1), axis = 0)

    def stack_tensors(self, attribute_list):
        tensor_list = []
        for attribute, value in self.__dict__.items():
            if attribute in attribute_list:
                tensor_list.append(value)
        self.long_tensor = np.concatenate(tensor_list, axis = 3)

    def split_to_patch(self, crop = 16):
        patches_list = extract_patches_non_overlap(self.long_tensor,  (crop,crop))
        self.patches = np.concatenate(patches_list,axis=0)
        patches_agb_list = extract_patches_non_overlap(self.agb, (crop,crop))
        self.agb_patches = np.concatenate(patches_agb_list,axis=0)
    
    def cross_validation(self, area, strategy, model_str):
        k=8
        num_samples = len(self.patches) // k
        test_mae_dl = []
        test_rmse_dl = []
        test_mae_ml = []
        test_rmse_ml = []
        stopped_epochs = []
        num_trees = []
        for fold in range(k):
            print("fold: ", fold)
            if strategy == "DL":
                test_data = self.patches[num_samples * fold:num_samples * (fold + 1)]
                test_agb = self.agb_patches[num_samples * fold:num_samples * (fold + 1)]
                training_data = np.concatenate([self.patches[:num_samples * fold], self.patches[num_samples * (fold + 1):]], axis = 0)
                training_agb = np.concatenate([self.agb_patches[:num_samples * fold], self.agb_patches[num_samples * (fold + 1):]], axis = 0)
                if fold != (k-1):
                    validation_data = training_data[num_samples * fold:num_samples * (fold + 1)].copy()
                    validation_agb = training_agb[num_samples * fold:num_samples * (fold + 1)].copy()
                    training_data = np.concatenate([training_data[:num_samples * fold], training_data[num_samples * (fold + 1):]], axis = 0).copy()
                    training_agb = np.concatenate([training_agb[:num_samples * fold], training_agb[num_samples * (fold + 1):]], axis = 0).copy()
                else:
                    validation_data = training_data[:num_samples].copy()
                    validation_agb = training_agb[:num_samples].copy()
                    training_data = training_data[num_samples:].copy()
                    training_agb = training_agb[num_samples:].copy()
                training = np.nan_to_num(training_data, copy=True, nan=-1.0)
                validation = np.nan_to_num(validation_data, copy=True, nan=-1.0)
                test = np.nan_to_num(test_data, copy=True, nan=-1.0)
                model, callbacks = get_dl_model(training.shape[1], training.shape[2], training.shape[3], fold, area)
                model.fit(training, training_agb, batch_size=8, epochs=500, callbacks=callbacks,         #epoche = 500
                        validation_data=(validation, validation_agb))
                model.load_weights('models/{}/models_dl/fold{}.h5'.format(area, fold))
                mae, mse = model.evaluate(test, test_agb, verbose=1)
                test_mae_dl.append(mae)
                test_rmse_dl.append(np.sqrt(mse))
                stopped_epochs.append(callbacks[0].stopped_epoch)
            elif strategy == "ML":
                test_data = self.patches[num_samples * fold:num_samples * (fold + 1)]
                test_agb = self.agb_patches[num_samples * fold:num_samples * (fold + 1)]
                training_data = np.concatenate([self.patches[:num_samples * fold], self.patches[num_samples * (fold + 1):]], axis = 0)
                training_agb = np.concatenate([self.agb_patches[:num_samples * fold], self.agb_patches[num_samples * (fold + 1):]], axis = 0)
                if fold != (k-1):
                    validation_data = training_data[num_samples * fold:num_samples * (fold + 1)].copy()
                    validation_agb = training_agb[num_samples * fold:num_samples * (fold + 1)].copy()
                    training_data = np.concatenate([training_data[:num_samples * fold], training_data[num_samples * (fold + 1):]], axis = 0).copy()
                    training_agb = np.concatenate([training_agb[:num_samples * fold], training_agb[num_samples * (fold + 1):]], axis = 0).copy()
                else:
                    validation_data = training_data[:num_samples].copy()
                    validation_agb = training_agb[:num_samples].copy()
                    training_data = training_data[num_samples:].copy()
                    training_agb = training_agb[num_samples:].copy()
                training = np.nan_to_num(training_data, copy=True, nan=-1.0).reshape((-1,self.patches.shape[3]))
                validation = np.nan_to_num(validation_data, copy=True, nan=-1.0).reshape((-1,self.patches.shape[3]))
                test = np.nan_to_num(test_data, copy=True, nan=-1.0).reshape((-1,self.patches.shape[3]))
                model_1 = get_ml_model(n_estimators = 250)                                                        #n_estimators = 250
                model_1.fit(training, training_agb.reshape((-1))) 
                val_score_1 = np.mean(abs(model_1.predict(validation) - validation_agb.reshape((-1))),axis=0)
                model_2 = get_ml_model(n_estimators = 500)                                                        #n_estimators = 500
                model_2.fit(training, training_agb.reshape((-1))) 
                val_score_2 = np.mean(abs(model_2.predict(validation) - validation_agb.reshape((-1))),axis=0)
                if val_score_1<=val_score_2:
                    model = model_1
                    num_trees.append(250)
                    print("vince ml1 con 250 alberi")
                else:
                    model = model_2
                    num_trees.append(500)
                    print("vince ml2 con 500 alberi")
                mae = np.mean(abs(model.predict(test) - test_agb.reshape((-1))),axis=0)
                rmse = np.sqrt(np.mean((model.predict(test) - test_agb.reshape((-1)))**2,axis=0))
                test_mae_ml.append(mae)
                test_rmse_ml.append(rmse)
        if strategy == "DL":
            overall_mae_dl = np.average(test_mae_dl)
            overall_rmse_dl = np.average(test_rmse_dl)
            with open("Results/{}/dl-{}.txt".format(area, model_str), "w") as f:
                f.write("Results for CV with U-Net:\n")
                f.write("           MAE                 RMSE\n")
                f.write('\n'.join("fold: " + str(mae) +"    "+str(rmse) for mae, rmse in zip(test_mae_dl,test_rmse_dl)))
                f.write("\n")
                f.write("Avg: {}      {}\n".format(str(overall_mae_dl),str(overall_rmse_dl)))
                f.write("number of epochs: {}\n".format(stopped_epochs))
        elif strategy == "ML":
            overall_mae_ml = np.average(test_mae_ml)
            overall_rmse_ml = np.average(test_rmse_ml)
            with open("Results/{}/ml-{}.txt".format(area, model_str), "w") as f:
                f.write("Results for CV with RandomForest:\n")
                f.write("           MAE                 RMSE\n")
                f.write('\n'.join("fold: " + str(mae) +"    "+str(rmse) for mae, rmse in zip(test_mae_ml,test_rmse_ml)))
                f.write("\n")
                f.write("Avg: {}      {}\n".format(str(overall_mae_ml),str(overall_rmse_ml)))
                f.write("num. of trees: {}\n".format(num_trees))
            
         
def get_haralick(sentinel, band, padding, kernel, namefile, area, ispca):
    #padding 2 con kernel 5
    padded = np.expand_dims(np.expand_dims(np.pad(sentinel[0,:,:,band], ((padding,padding),(padding,padding)), mode='constant'), axis = 0), axis = -1)
    padded_patch = extract_patches(padded, kernel, 1)
    h_feature_list = []
    for i in range(sentinel.shape[1]*sentinel.shape[2]):
        h_feature_list.append(np.mean(mahotas.features.haralick(padded_patch[i,:,:,0].astype(np.uint8)), axis = 0)) 
        print(i)
    haralick_array = np.asarray(h_feature_list).reshape((sentinel.shape[1], sentinel.shape[2], 13))
    if ispca:
        np.save('Data/{}/haralick-pca/{}'.format(area, namefile), haralick_array)
    elif not ispca:
        np.save('Data/{}/haralick/{}'.format(area, namefile), haralick_array)

def get_all_haralick(sentinel, padding, kernel, area, ispca):
    if not ispca:
        mapping = {0:"banda2",1:"banda3",2:"banda4",3:"banda5",4:"banda6",5:"banda7",6:"banda8",7:"banda8a",8:"banda11",9:"banda12"}
    else:
        mapping = {0:"first_component"}
    for k,v in mapping.items():
        get_haralick(sentinel, k, padding, kernel, v, area, ispca)
   
def extract_patches(full_imgs, crop_size, stride_size):
    #OVERLAPPING
    patch_height = crop_size
    patch_width = crop_size
    stride_height = stride_size
    stride_width = stride_size

    assert (len(full_imgs.shape) == 4)  # 4D arrays
    img_h = full_imgs.shape[1]          # height of the full image
    img_w = full_imgs.shape[2]          # width of the full image

    assert ((img_h - patch_height) % stride_height == 0 and (img_w - patch_width) % stride_width == 0)
    N_patches_img = ((img_h - patch_height) // stride_height + 1) * (
            (img_w - patch_width) // stride_width + 1)  # // --> division between integers
    N_patches_tot = N_patches_img * full_imgs.shape[0]

    patches = np.empty((N_patches_tot, patch_height, patch_width, full_imgs.shape[3]))
    iter_tot = 0  # iter over the total number of patches (N_patches)
    for i in range(full_imgs.shape[0]):  # loop over the full images
        for h in range((img_h - patch_height) // stride_height + 1):
            for w in range((img_w - patch_width) // stride_width + 1):
                patch = full_imgs[i, h * stride_height:(h * stride_height) + patch_height,
                        w * stride_width:(w * stride_width) + patch_width, :]
                patches[iter_tot] = patch
                iter_tot += 1  # total
    assert (iter_tot == N_patches_tot)
    return patches 

def extract_patches_non_overlap(X, patch_size):
    #NON OVERLAPPING
    list_X = []
    list_row_idx = [(i * patch_size[0], (i + 1) * patch_size[0]) for i in range(X.shape[1] // patch_size[0])]
    list_col_idx = [(i * patch_size[1], (i + 1) * patch_size[1]) for i in range(X.shape[2] // patch_size[1])]

    if len(X.shape) == 4:
      for row_idx in list_row_idx:
          for col_idx in list_col_idx:
              list_X.append(X[:, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1], :])
    elif len(X.shape) ==3:
      for row_idx in list_row_idx:
          for col_idx in list_col_idx:
              list_X.append(X[:, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1]])

    return list_X



















