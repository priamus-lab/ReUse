import numpy as np
import rasterio

def predict(x, model, strategy, crop_size):
    #chip_size = 16

    batch, height, width, bands = x.shape  
    print(x.shape)
    if strategy == "ML":
        labels = model.predict(x.reshape((-1,bands))).reshape((1, height, width))
        return labels
    boxes, list_row_idx, list_col_idx = extract_patches(x, (crop_size,crop_size))
    print("len boxes: {}".format(len(boxes)), boxes[0].shape, len(list_row_idx), len(list_col_idx))
    labels = np.zeros((1, height, width), dtype=np.float16)
    prediction_list = []
    
    for i in range(len(boxes)):
        # Box to compute inference on

        prediction_list.append(np.squeeze(model.predict(boxes[i]),0)) 

    for i in range(len(list_row_idx)):
      for j in range(len(list_col_idx)):
        labels[0, list_row_idx[i][0]:list_row_idx[i][1], list_col_idx[j][0]:list_col_idx[j][1]] = np.squeeze(prediction_list[i*len(list_col_idx) + j], axis=-1)
          
    
    return labels



def save_prediction_to_raster(labels, raster_src_uri, raster_dst_uri):
    """
    Save prediction to GeoTiff raster
    :param raster_src_uri: reference GeoTiff used for georeferencing !!
    :param raster_dst_uri: raster where save prediction
    :return:
    """
    # leggo i metadata da copiare
    with rasterio.open(raster_src_uri) as raster_src:
        profile = raster_src.profile

    batch, height, width = labels.shape # le bande sulla terza dim., in questo caso e` 1 sola
    #num_bands = 1
    # And then change the band count to 1, set the
    # dtype to uint8, and specify LZW compression.
    profile.update(
        height=height,
        width=width,
        dtype=rasterio.float32,
        #count=num_bands,
        compress='lzw')

    with rasterio.open(raster_dst_uri, 'w', **profile) as raster_dst:
        #for i in range(0, num_bands):
            #raster_dst.write(labels[i, :, :], i + 1)
          raster_dst.write(labels)


def extract_patches(X, patch_size):
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

    return list_X, list_row_idx, list_col_idx

