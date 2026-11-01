import os, sys
from scipy.io import loadmat
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.model_selection import train_test_split
import random
import tifffile as tiff
import math
import matplotlib.pyplot as plt
import info
from scipy.stats import poisson
from skimage.util import random_noise
import scipy.io as sio
import matplotlib.pyplot as plt
import glymur
import scipy.ndimage as ndimage
import foggy_gen

DATA_PATH_PREFIX = "../data"  # ../data
def load_data(data_sign, data_path_prefix):
    if data_sign == "Indian":
        data = sio.loadmat('%s/Indian_pines_corrected.mat' % data_path_prefix)['indian_pines_corrected']
        labels = sio.loadmat('%s/Indian_pines_gt.mat' % data_path_prefix)['indian_pines_gt']
    elif data_sign == "Pavia":
        data = sio.loadmat('%s/PaviaU.mat' % data_path_prefix)['paviaU']
        labels = sio.loadmat('%s/PaviaU_gt.mat' % data_path_prefix)['paviaU_gt'] 
    elif data_sign == "Houston":
        data = sio.loadmat('%s/Houston.mat' % data_path_prefix)['img']
        labels = sio.loadmat('%s/Houston_gt.mat' % data_path_prefix)['Houston_gt']
    elif data_sign == 'Salinas':
        data = sio.loadmat('%s/Salinas_corrected.mat' % data_path_prefix)['salinas_corrected']
        labels = sio.loadmat('%s/Salinas_gt.mat' % data_path_prefix)['salinas_gt']
    elif data_sign == 'WH' or data_sign=='Honghu':
        data = sio.loadmat('%s/WHU_Hi_HongHu.mat' % data_path_prefix)['WHU_Hi_HongHu']
        labels = sio.loadmat('%s/WHU_Hi_HongHu_gt.mat' % data_path_prefix)['WHU_Hi_HongHu_gt']
    elif data_sign == 'WHLK':
        data = sio.loadmat('%s/WHU_Hi_LongKou.mat' % data_path_prefix)['WHU_Hi_LongKou']
        labels = sio.loadmat('%s/WHU_Hi_LongKou_gt.mat' % data_path_prefix)['WHU_Hi_LongKou_gt']
    return data, labels


def norm_0_255(data):
    h, w, c = data.shape
    res = np.zeros_like(data)
    params = np.zeros([c,2])
    for ci in range(c):
        ss = data[:,:,ci]
        res[:,:,ci] = (ss - ss.min()) / (ss.max() - ss.min()) * 255
        params[ci] = np.asarray([ss.min(), ss.max()])
    res = res.astype(np.uint8)
    
    return res


class DataNoiseGeneratorBase(object):
    def __init__(self, data_sign) -> None:
        self.noise_type = "Base"
        self.data_sign = data_sign

        self.params = {}

    def set_params(self, noise_type, params):
        self.noise_type = noise_type
        self.params = params
    
    def gen(self, data):
        return data


    def save_data(self, data, path_prefix):
        if not os.path.exists(path_prefix):
            os.makedirs(path_prefix)

        res = {
            "noise_type": self.noise_type,
            "data_sign": self.data_sign,
            "data" : data,
            "params": self.params
        }

        path = "%s/%s.mat" % (path_prefix, self.noise_type)
        sio.savemat(path, res)

    def plot_img(self, data, path_prefix):
        assert len(data.shape) == 3
        h, w, c = data.shape
        assert c >= 3
        r, g, b = info.get_rgb_index_by_sign(self.data_sign)
        img = norm_0_255(data[:,:,[r, g, b]])

        path = "%s/%s_%s.jpg" % (path_prefix, self.noise_type, self.data_sign)
        if not os.path.exists(path_prefix):
            os.makedirs(path_prefix)
        plt.figure(figsize=(15,10))
        plt.imsave(path, img)

        

class JPEGGenerator(DataNoiseGeneratorBase):
    def __init__(self, data_sign) -> None:
        super().__init__(data_sign)

        self.noise_type = "jpeg"
        self.params = {
            'compression_ratios' : [10]
        }


    def gen(self, data):
        import glymur
        compression_ratios = self.params['compression_ratios']
        temp_file = 'output_lossy.jp2'
        # print(data.dtype, data)
        data = data.astype(np.uint16)
        jp2k = glymur.Jp2k(temp_file, data, cratios=compression_ratios)
        res_data = jp2k[:]
        return res_data

class ZMGauss(DataNoiseGeneratorBase):
    def __init__(self, data_sign):
        super().__init__(data_sign)

        self.noise_type = "zero_mean_guass"

        self.params = {
            "sigma": 0.1 # 0.16
        }
        #self.sigma = self.params['sigma']

    def normalize_band(self, band):
        min_val, max_val = band.min(), band.max()
        return (band - min_val) / (max_val - min_val)

    def denormalize_band(self, band, min_val, max_val):
        return band * (max_val - min_val) + min_val

    def gen(self, data):
        img_clean = data
        row, column, bands = img_clean.shape
        np.random.seed(0)

        # Step 1: Normalize each band to the [0, 1] range
        img_clean_normalized = np.zeros_like(img_clean, dtype=np.float32)
        for i in range(bands):
            img_clean_normalized[:, :, i] = self.normalize_band(img_clean[:, :, i].astype(np.float32))

        # Step 2: Generate noisy image (Zero-mean Gaussian noise)
        sigma = self.params['sigma']
        noise = np.random.normal(0, sigma, img_clean.shape)
        img_noisy = img_clean_normalized + noise
        img_noisy = np.clip(img_noisy, 0, 1)

        # Step 3 (Optional): Denormalize the noisy image back to the original scale
        img_noisy_denorm = np.zeros_like(img_clean)
        for i in range(bands):
            min_val, max_val = img_clean[:, :, i].min(), img_clean[:, :, i].max()
            img_noisy_denorm[:, :, i] = self.denormalize_band(img_noisy[:, :, i], min_val, max_val).astype(img_clean.dtype)

        return img_noisy_denorm
        

class Additive(DataNoiseGeneratorBase):
    def __init__(self, data_sign):
        super().__init__(data_sign)
        self.noise_type = "additive"

        self.params = {
            "noise_level": 0.3 
        }
        # self.noise_level = self.params['noise_level']

    def gen(self, data):
        # Different variance zero-mean Gaussian noise is added to each band
        # The std values are randomly selected from 0 to 0.1.
        img_clean = data
        row, column, bands = img_clean.shape
        N = row * column  # img_clean has dimensions [row, column, band]
        img_noisy = None
        np.random.seed(0)
        # Step 1: Normalize each band to the [0, 1] range
        img_clean_normalized = np.zeros_like(img_clean, dtype=np.float32)  # initialize for normalized image
        for i in range(bands):
            band = img_clean[:, :, i].astype(np.float32)
            min_val, max_val = band.min(), band.max()
            img_clean_normalized[:, :, i] = (band - min_val) / (max_val - min_val)  # normalize to [0, 1]

        # Step 2: Generate noisy image (Zero-mean Gaussian noise)
        noise_level = self.params['noise_level']
        sigma = np.random.rand(bands) * noise_level
        noise = np.random.randn(*img_clean.shape)

        for cb in range(bands):
            noise[:, :, cb] = sigma[cb] * noise[:, :, cb]

        img_noisy = img_clean_normalized + noise

        # Clip noisy image to ensure values stay within [0, 1] range
        img_noisy = np.clip(img_noisy, 0, 1)

        # Step 3 (Optional): Denormalize the noisy image back to the original scale
        img_noisy_denorm = np.zeros_like(img_clean)
        for i in range(bands):
            min_val, max_val = img_clean[:, :, i].min(), img_clean[:, :, i].max()
            img_noisy_denorm[:, :, i] = (img_noisy[:, :, i] * (max_val - min_val) + min_val).astype(img_clean.dtype)

        img_noisy = img_noisy_denorm
        return img_noisy



class Poisson(DataNoiseGeneratorBase):
    def __init__(self, data_sign):
        super().__init__(data_sign)
        self.noise_type = "poisson"
        self.params = {
            "snr_db" : 50  # 13
        } 
        # self.snr_db = self.params['snr_db']

    def gen(self, data):
        img_clean = data
        row, column, bands = img_clean.shape

        # 参数设置
        snr_db = self.params['snr_db']
        snr_linear = np.exp(snr_db * np.log(10) / 10)  # dB转线性值
        np.random.seed(0)  # 固定随机种子

        # 初始化结果矩阵
        img_noisy = np.zeros_like(img_clean, dtype=np.float32)

        # 遍历每个波段
        for band_idx in range(bands):
            # 获取当前波段数据并展平
            band_data = img_clean[:, :, band_idx].reshape(-1)
            band_data = np.maximum(band_data, 0)  # 确保非负

            # 计算统计量
            sum_x = np.sum(band_data)
            sum_x2 = np.sum(band_data ** 2)

            # 处理全零波段
            if sum_x <= 1e-6 or sum_x2 <= 1e-6:
                img_noisy[:, :, band_idx] = 0
                continue

            # 修正后的缩放因子公式
            scaling_factor = snr_linear * sum_x / sum_x2

            # 生成泊松噪声
            noisy_band = np.random.poisson(scaling_factor * band_data).astype(np.float32)

            # 恢复形状并存储
            img_noisy[:, :, band_idx] = noisy_band.reshape(row, column)
        # img_clean = data
        # row, column, bands = img_clean.shape
        # N = row * column  # img_clean has dimensions [row, column, band]
        # img_noisy = None
        # np.random.seed(0)
        #
        # snr_db = self.params['snr_db']
        #
        # # 计算信噪比对应的线性值
        # snr_set = np.exp(snr_db * np.log(10) / 10)
        #
        # # 图像的维度
        # row, column, band = img_clean.shape
        # N = row * column
        #
        # # 初始化结果矩阵
        # img_wN_scale = np.zeros((band, N))
        # img_wN_noisy = np.zeros((band, N))
        #
        # # 遍历每个波段
        # for i in range(band):
        #     # 获取当前波段的数据并展平为一维
        #     img_wNtmp = img_clean[:, :, i].reshape(-1)
        #     img_wNtmp = np.maximum(img_wNtmp, 0)  # 确保非负
        #
        #     # 计算缩放因子
        #     factor = snr_set / (np.sum(img_wNtmp ** 2) / np.sum(img_wNtmp))
        #     img_wN_scale[i, :] = factor * img_wNtmp
        #
        #     # 生成泊松噪声
        #     img_wN_noisy[i, :] = np.random.poisson(factor * img_wNtmp)
        #
        # # 重塑为原始图像形状
        # img_noisy = img_wN_noisy.T.reshape((row, column, band))
        # img_clean_scaled = img_wN_scale.T.reshape((row, column, band))
        #
        # # 计算差分图像
        # diff_image = img_noisy - img_clean
        #
        # mse = np.mean((img_noisy - img_clean) ** 2)
        # print("MSE between noisy and clean image:", mse)

        # 可视化差分图像
        # plt.imshow(np.mean(diff_image, axis=2), cmap='gray')
        # plt.colorbar()
        # plt.title("Difference Image")
        # plt.show()
        return img_noisy

class SaltPepper(DataNoiseGeneratorBase):
    def __init__(self, data_sign):
        super().__init__(data_sign)

        self.noise_type = "salt_pepper"

        self.params = {
            "amount": 0.1
        }

    def gen(self, data):
        img_clean = data
        row, column, bands = img_clean.shape
        N = row * column  # img_clean has dimensions [row, column, band]
        img_noisy = None
        np.random.seed(0)
        # --------------------- Case 4: Salt & Pepper Noise ---------------------
        amount = self.params['amount'] 
        bands = img_clean.shape[2]  # number of bands
        # Step 1: Normalize each band to the [0, 1] range
        img_clean_normalized = np.zeros_like(img_clean, dtype=np.float32)  # initialize for normalized image
        for i in range(bands):
            band = img_clean[:, :, i].astype(np.float32)
            min_val, max_val = band.min(), band.max()
            img_clean_normalized[:, :, i] = (band - min_val) / (max_val - min_val)  # normalize to [0, 1]

        # Step 2: Add salt-and-pepper noise
        img_noisy = np.zeros_like(img_clean_normalized)
        for i in range(bands):
            img_noisy[:, :, i] = random_noise(img_clean_normalized[:, :, i], mode='s&p', amount=amount)

        # Step 3 (Optional): If you need the result back in the original scale, denormalize
        img_noisy_denorm = np.zeros_like(img_clean)
        for i in range(bands):
            min_val, max_val = img_clean[:, :, i].min(), img_clean[:, :, i].max()
            img_noisy_denorm[:, :, i] = (img_noisy[:, :, i] * (max_val - min_val) + min_val).astype(img_clean.dtype)
        return  img_noisy_denorm

class Stripes(DataNoiseGeneratorBase):
    def __init__(self, data_sign):
        super().__init__(data_sign)

        self.noise_type = "stripes"

        self.params = {
            "start": 0,
            "end" : 6
        }

    def gen(self, data):
        img_clean = data
        row, column, bands = img_clean.shape
        N = row * column  # img_clean has dimensions [row, column, band]
        img_noisy = None
        np.random.seed(0)
        # --------------------- Case 5: Stripes Noise ---------------------
        img_noisy = np.copy(img_clean)

        # Define stripe noise parameters
        stripenum = np.random.randint(self.params['start'], self.params['end'])  # Randomly generate 6-15 stripes
        for cb in range(bands):
            locolumn = np.random.choice(column, size=stripenum, replace=False)  # Random stripe positions
            img_noisy[:, locolumn, cb] = 0.2 * np.random.rand(1) + 0.6  # Add stripes noise
        return img_noisy

class Deadlines(DataNoiseGeneratorBase):
    def __init__(self, data_sign):
        super().__init__(data_sign)

        self.noise_type = "deadlines"

        self.params = {
            "start": 30,
            "end" : 33
        }


    def gen(self, data):
        img_clean = data
        row, column, bands = img_clean.shape
        N = row * column  # img_clean has dimensions [row, column, band]
        img_noisy = None
        np.random.seed(0)
        # --------------------- Case 6: Deadlines Noise ---------------------
        img_noisy = np.copy(img_clean)

        # Define deadline noise parameters
        deadlinenum = np.random.randint(self.params['start'], self.params['end'])  # Randomly generate 6-10 deadlines
        for cb in range(bands):
            locolumn = np.random.choice(column - 3, size=deadlinenum, replace=False)  # Deadline positions
            # an = np.random.randint(1, 4, size=deadlinenum)  # Randomly select width (1-3)
            an = np.random.randint(1, 4, size=deadlinenum)  # Randomly select width (1-3)
            for idx in range(deadlinenum):
                if an[idx] == 1:
                    img_noisy[:, locolumn[idx], cb] = 0
                elif an[idx] == 2:
                    img_noisy[:, locolumn[idx]:locolumn[idx] + 2, cb] = 0
                else:
                    img_noisy[:, locolumn[idx]:locolumn[idx] + 3, cb] = 0

        return img_noisy


class Kernel(DataNoiseGeneratorBase):
    def __init__(self, data_sign):
        super().__init__(data_sign)

        self.noise_type = "kernel"

        self.params = {
            "kernel_size": 3,
            'noise_level': 0.1
        }

    def gen(self, data):
        img_clean = data
        row, column, bands = img_clean.shape
        img_noisy = np.copy(img_clean)
        np.random.seed(0)  # 固定随机种子，保证可重复性

        # 定义卷积核
        kernel_size = self.params['kernel_size']
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size * kernel_size)  # 归一化

        # 获取噪声强度参数（需要提前在 self.params 中定义 'noise_level'）
        noise_level = self.params.get('noise_level', 0.1)  # 默认噪声强度为0.1

        # 对每个波段添加噪声并卷积
        for cb in range(bands):
            # 生成高斯噪声并缩放
            noise = np.random.randn(row, column) * noise_level  # 关键修改：用 noise_level 控制噪声幅度
            img_noisy[:, :, cb] = img_clean[:, :, cb] + noise  # 添加噪声

            # 对噪声图像进行卷积
            img_noisy[:, :, cb] = ndimage.convolve(
                img_noisy[:, :, cb], kernel, mode='constant', cval=0
            )

        return img_noisy


class Fog(DataNoiseGeneratorBase):
    def __init__(self, data_sign):
        super().__init__(data_sign)
        self.noise_type = "fog"
        
        self.params = {
            'spe_start': 0,
            'spe_end': 0,
            'band_width': 0,
            'omega': 0.5,
            'alpha': 0.01,
            'beta': 1.0
    }

    def gen(self, data):
        spe_start, spe_end, band_width = info.get_spetral_info(self.data_sign)
        self.params['spe_start'] = spe_start
        self.params['spe_end'] = spe_end
        self.params['band_width'] = band_width
        sim_data = foggy_gen.api_run(self.data_sign, data, self.params)
        return sim_data




# kvs_params = info.Indian_kvs_params
# kvs_params = info.Pavia_kvs_params
kvs_params = info.WHLK_kvs_params

kvs = {
    'jpeg': JPEGGenerator,
    'zmguass': ZMGauss,
    'additive': Additive,
    'poisson': Poisson,
    'salt_pepper': SaltPepper,
    'stripes': Stripes,
    'deadlines': Deadlines,
    'kernel': Kernel,
    'thin_fog': Fog,
    'thick_fog': Fog
}

def run_gen(data_sign):
    data, labels = load_data(data_sign, data_path_prefix=DATA_PATH_PREFIX)
    NOISE_DATA_SAVE_PREFIX = "%s/noise_%s" % (DATA_PATH_PREFIX, data_sign)
    NOISE_DATA_SAVE_PREFIX_DATA = "%s" % NOISE_DATA_SAVE_PREFIX
    NOISE_DATA_SAVE_PREFIX_IMG= "%s/img" % NOISE_DATA_SAVE_PREFIX
    # start to gen
    for k, cls in kvs.items():
        obj = cls(data_sign)
        print("start to gen noise data_sign=%s, noise_type=%s .." % (data_sign, obj.noise_type))
        params = kvs_params[k]
        obj.set_params(noise_type=k, params=params)
        res_data = obj.gen(data) 
        obj.save_data(res_data, NOISE_DATA_SAVE_PREFIX_DATA)
        obj.plot_img(res_data, NOISE_DATA_SAVE_PREFIX_IMG)
        print("start to gen noise data_sign=%s, noise_type=%s .." % (data_sign, obj.noise_type))



def run_all():
    # data_signs = ['Pavia', 'Indian', 'Salinas', 'WH', 'Honghu', 'WHLK']
    data_signs = ['WHLK']
    # data_signs = ['Pavia']
    # data_signs = ['WH', 'Honghu']
    for data_sign in data_signs:
        run_gen(data_sign)



if __name__ == "__main__":
    run_all()

