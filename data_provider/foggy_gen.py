import os, sys
import numpy as np
from scipy.io import loadmat
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.model_selection import train_test_split
import random
import tifffile as tiff
import math
import matplotlib.pyplot as plt
import tifffile



def max_min_norm(temp_data):
    maxv, minv = temp_data.max(), temp_data.min()
    temp_data = (temp_data - minv) / (maxv - minv)
    return temp_data


def cal_rou(dn_data, params):
    norm_dn = max_min_norm(dn_data)
    # 注意这里使用线性映射替换了辐射定标，假定DN值与反射率之间为线性关系 使得[0,1] -> [t, 1]
    # TODO: 如果需要更精确的话，尽量进行辐射定标
    alpha = params.get('alpha', 0.01)
    beta = params.get('beta', 1.0)
    res = norm_dn * (beta-alpha) + alpha
    return res

def cal_tx(rou, params):
    omega = params.get('omega', 0.9)
    t1 = 1 - omega * rou
    t1 = np.where(t1 > 1, 1, t1)
    t1 = np.where(t1 < 0, 0, t1)
    return t1

def cal_t1(fog_data_patten, params):
    rou = cal_rou(fog_data_patten, params)
    t1 = cal_tx(rou, params)
    t1 = np.where(t1>0.999999, 0.99999, t1)

    specific_rou_list = np.asarray([0.108, 0.255, 0.334, 0.412, 0.490, 0.765]).astype(np.float64)
    specific_t_list = np.asarray([cal_tx(v, params) for v in specific_rou_list]).astype(np.float64)
    specific_gamma_list = np.asarray([4,2,1,0.7,0.5,0]).astype(np.float64)
    specitic_log_t_list = np.log(np.log(1/specific_t_list))

    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(np.expand_dims(specitic_log_t_list, 1) , specific_gamma_list)

    ln_t1 = np.log(np.log(1/t1))

    gamma_temp = model.predict(np.expand_dims(ln_t1.reshape(-1), 1))
    gamma = np.where(gamma_temp<0,0,gamma_temp)
    gamma = np.where(gamma>4,4,gamma)
    gamma = gamma.reshape(rou.shape)

    return rou, t1, gamma


def simu_fog(J, lambda_j, lambda_base, t1, gamma):
    pows = np.power(lambda_base / lambda_j, gamma)
    t = np.exp(pows * np.log(t1))
    top_dn_num = int(0.0001 * J.size)
    A = np.mean(np.sort(J.reshape(-1))[-top_dn_num:])
    I = J * t + A *( 1 - t)
    return I, t, A



def run_sim(data, fog_data_patten, params):
    # data need to be simulated fog, data.shape [h, w, c]
    # fog_data_patten.shape [h, w]
    spe_start = params['spe_start']
    spe_end = params['spe_end']
    band_width = params['band_width']
    h, w, c = data.shape 
    rou, t1, gamma = cal_t1(fog_data_patten, params)

    simu_data = np.zeros_like(data)
    t_data = np.zeros_like(data).astype(np.float64)
    lambda_base = spe_start
    for index_j in range(0,c):
        lambda_j = spe_start + index_j*band_width + band_width//2
        J = data[:,:,index_j]
        simu_data[:,:,index_j], t_data[:,:, index_j], _ = simu_fog(J, lambda_j, lambda_base, t1, gamma)
    return simu_data


def load_fog_data(path):
    fog_data = tifffile.imread(path)
    fog_channel = fog_data[:, :, 0]
    return fog_channel


def load_data(data_sign, data_path_prefix):
    if data_sign == "Indian":
        data = sio.loadmat('%s/Indian_pines_corrected.mat' % data_path_prefix)['indian_pines_corrected']
        labels = sio.loadmat('%s/Indian_pines_gt.mat' % data_path_prefix)['indian_pines_gt']
        spe_start, spe_end, band_width = 400, 2500, 10
    elif data_sign == "PaviaU":
        data = sio.loadmat('%s/PaviaU.mat' % data_path_prefix)['paviaU']
        labels = sio.loadmat('%s/PaviaU_gt.mat' % data_path_prefix)['paviaU_gt'] 
        spe_start, spe_end, band_width = 430, 860, 4
    elif data_sign == "Houston":
        data = sio.loadmat('%s/Houston.mat' % data_path_prefix)['img']
        labels = sio.loadmat('%s/Houston_gt.mat' % data_path_prefix)['Houston_gt']
        # spe_start, spe_end, band_width = 400, 2500, 10
    elif data_sign == 'Salinas':
        data = sio.loadmat('%s/Salinas_corrected.mat' % data_path_prefix)['salinas_corrected']
        labels = sio.loadmat('%s/Salinas_gt.mat' % data_path_prefix)['salinas_gt']
        spe_start, spe_end, band_width = 400, 2500, 10
    elif data_sign == 'WH' or data_sign=='Honghu':
        data = sio.loadmat('%s/WHU_Hi_HongHu.mat' % data_path_prefix)['WHU_Hi_HongHu']
        labels = sio.loadmat('%s/WHU_Hi_HongHu_gt.mat' % data_path_prefix)['WHU_Hi_HongHu_gt']
        spe_start, spe_end, band_width = 400, 1000, 2
    elif data_sign == 'WHLK':
        data = sio.loadmat('%s/WHU_Hi_LongKou.mat' % data_path_prefix)['WHU_Hi_LongKou']
        labels = sio.loadmat('%s/WHU_Hi_LongKou_gt.mat' % data_path_prefix)['WHU_Hi_LongKou_gt']
        spe_start, spe_end, band_width = 400, 1000, 2
    return data, labels, spe_start, spe_end, band_width


def gen_fog_data_patten(data, fog_channel, data_sign):
    if data_sign == "Indian":
        h, w, c = data.shape
        fog_data_patten = np.zeros([h,w])
        temp_h, temp_w= fog_channel.shape
        use_h, use_w = min(h,temp_h), min(temp_w, w)
        fog_data_patten[:,:] = fog_channel[-use_h:, -use_w:]
        return fog_data_patten
    elif data_sign == "Pavia":
        h, w, c = data.shape
        fog_data_patten = np.zeros([h,w])
        fog_h, fog_w = fog_channel.shape
        temp = fog_channel[:h//2,:w]
        temp_flip = np.flipud(temp)
        hh, ww = temp.shape
        fog_data_patten[:hh, :] = temp
        fog_data_patten[hh:,:] = temp_flip
        return fog_data_patten
        # h, w, c = data.shape
        # fog_data_patten = np.zeros([h,w])
        # fog_h, fog_w = fog_channel.shape
        # fog_data_patten[:fog_h,:w] = fog_channel[:,:w]
        # fog_data_patten[fog_h:,:w] = fog_channel[fog_h-h:,:w]
        # return fog_data_patten
    elif data_sign == "Salinas":
        h, w, c = data.shape
        fog_data_patten = np.zeros([h,w])
        fog_h, fog_w = fog_channel.shape
        fog_data_patten[:,:] = fog_channel[:,:w]
        return fog_data_patten
    elif data_sign == "WH":
        h, w, c = data.shape
        fog_data_patten = np.zeros([h,w])
        fog_h, fog_w = fog_channel.shape
        temp = fog_channel[:h//2,:w]
        temp_flip = np.flipud(temp)
        hh, ww = temp.shape
        fog_data_patten[:hh, :] = temp
        fog_data_patten[hh:,:] = temp_flip
        return fog_data_patten
    elif data_sign == "WHLK":
        h, w, c = data.shape
        fog_data_patten = np.zeros([h, w])
        fog_h, fog_w = fog_channel.shape
        temp = fog_channel[:h // 2, :w]
        temp_flip = np.flipud(temp)
        hh, ww = temp.shape
        fog_data_patten[:hh, :] = temp
        fog_data_patten[hh:, :] = temp_flip
        return fog_data_patten


def get_rgb_index(params):
    red_spe = (620 + 750) // 2
    green_spe = (495 + 570) // 2
    blue_spe = (450 + 495) // 2
    spe_start = params['spe_start']

    spe_end = params['spe_end']
    band_width = params['band_width']
    r = (red_spe - spe_start) // band_width
    g = (green_spe - spe_start) // band_width
    b = (blue_spe - spe_start) // band_width

    return r, g, b


def norm_0_255(data):
    h, w, c = data.shape
    res = np.zeros_like(data)
    params = np.zeros([c, 2])
    for ci in range(c):
        ss = data[:, :, ci]
        res[:, :, ci] = (ss - ss.min()) / (ss.max() - ss.min()) * 255
        params[ci] = np.asarray([ss.min(), ss.max()])
    res = res.astype(np.uint8)
    
    return res


def plot_image(path_save, data, params):
    red, green, blue = get_rgb_index(params)    
    img = norm_0_255(data[:, :, [red, green, blue]])
    plt.figure(figsize=(15, 10))
    plt.imsave(path_save, img,)


def api_run(data_sign, data, params):
    path_fog_data = "A3.tif"
    fog_channel = load_fog_data(path_fog_data)
    fog_data_patten = gen_fog_data_patten(data, fog_channel, data_sign)
    print(fog_data_patten.shape, data.shape)

    
    simu_data = run_sim(data, fog_data_patten, params)

    input_data = simu_data  # input 是模拟后的数据
    return input_data

def run(data_sign, omega): 
    res_path_prefix = "../data/fog"
    if not os.path.exists(res_path_prefix):
        os.makedirs(res_path_prefix)

    path_data_prefix = "../data"
    path_fog_data = "../data/AVIRIS/A3.tif"
    fog_channel = load_fog_data(path_fog_data)
    data, labels, spe_start, spe_end, band_width = load_data(data_sign, path_data_prefix)
    fog_data_patten = gen_fog_data_patten(data, fog_channel, data_sign)
    print(fog_data_patten.shape, data.shape)

    params = {
        'spe_start': spe_start,
        'spe_end': spe_end,
        'band_width': band_width,
        'omega': omega,
        'alpha': 0.1,
        'beta': 0.9
    }
    simu_data = run_sim(data, fog_data_patten, params)

    # 提取 input 和 y 数据
    input_data = simu_data  # input 是模拟后的数据
    y_data = labels  # y 是标签数据

    # 保存到 .mat 文件
    res = {
        "input": input_data,  # 模拟的图像数据
        "y": y_data,  # 标签数据
        "params": params  # 额外的参数
    }

    path = "%s/%s_%s.mat" % (res_path_prefix, data_sign, "fog_%s" % omega)
    sio.savemat(path, res)

    path_img = "%s/fog_%s_%s.jpg" % (res_path_prefix, data_sign, "fog_%s" % omega)
    plot_image(path_img, simu_data, params)


if __name__ == "__main__":
    # data_signs = ['Indian', 'Pavia', 'Salinas', 'WH']
    # data_signs = ['Indian', 'Pavia']
    data_signs = ['PaviaU']
    # omegas = [0.1, 0.3, 0.5, 0.5, 0.6, 0.7, 0.8, 0.9]
    omegas = [0.125]

    for sign in data_signs:
        for omega in omegas:
            print("start to simulate %s_%s" % (sign, omega))
            run(sign, omega)