import numpy as np
import pymannkendall as mk
import os
import rasterio as ras

def sen_mk_test(image_path, outputPath):

    global path1
    filepaths = []
    for file in os.listdir(path1):
        filepath1 = os.path.join(path1, file)
        filepaths.append(filepath1)


    num_images = len(filepaths)

    img1 = ras.open(filepaths[0])

    transform1 = img1.transform
    height1 = img1.height
    width1 = img1.width
    array1 = img1.read()
    img1.close()


    for path1 in filepaths[1:]:
        if path1[-3:] == 'tif':
            print(path1)
            img2 = ras.open(path1)
            array2 = img2.read()
            array1 = np.vstack((array1, array2))
            img2.close()

    nums, width, height = array1.shape


    def writeImage(image_save_path, height1, width1, para_array, bandDes, transform1):
        with ras.open(
                image_save_path,
                'w',
                driver='GTiff',
                height=height1,
                width=width1,
                count=1,
                dtype=para_array.dtype,
                crs='+proj=latlong',
                transform=transform1,
        ) as dst:
            dst.write_band(1, para_array)
            dst.set_band_description(1, bandDes)
        del dst


    slope_array = np.full([width, height], -9999.0000)
    z_array = np.full([width, height], -9999.0000)
    Trend_array = np.full([width, height], -9999.0000)
    Tau_array = np.full([width, height], -9999.0000)
    s_array = np.full([width, height], -9999.0000)
    p_array = np.full([width, height], -9999.0000)

    c1 = np.isnan(array1)
    sum_array1 = np.sum(c1, axis=0)
    nan_positions = np.where(sum_array1 == num_images)

    positions = np.where(sum_array1 != num_images)


    print("all the pixel counts are {0}".format(len(positions[0])))
    # mk test
    for i in range(len(positions[0])):
        print(i)
        x = positions[0][i]
        y = positions[1][i]
        mk_list1 = array1[:, x, y]
        trend, h, p, z, Tau, s, var_s, slope, intercept = mk.original_test(mk_list1)


        if trend == "decreasing":
            trend_value = -1
        elif trend == "increasing":
            trend_value = 1
        else:
            trend_value = 0
        slope_array[x, y] = slope  # senslope
        s_array[x, y] = s
        z_array[x, y] = z
        Trend_array[x, y] = trend_value
        p_array[x, y] = p
        Tau_array[x, y] = Tau

    all_array = [slope_array, Trend_array, p_array, s_array, Tau_array, z_array]

    slope_save_path = os.path.join(result_path, "slope.tif")
    Trend_save_path = os.path.join(result_path, "Trend.tif")
    p_save_path = os.path.join(result_path, "p.tif")
    s_save_path = os.path.join(result_path, "s.tif")
    tau_save_path = os.path.join(result_path, "tau.tif")
    z_save_path = os.path.join(result_path, "z.tif")
    image_save_paths = [slope_save_path, Trend_save_path, p_save_path, s_save_path, tau_save_path, z_save_path]
    band_Des = ['slope', 'trend', 'p_value', 'score', 'tau', 'z_value']
    for i in range(len(all_array)):
        writeImage(image_save_paths[i], height1, width1, all_array[i], band_Des[i], transform1)

# 调用
path1 = 'inputpath'
result_path ='outpath'
sen_mk_test(path1, result_path)