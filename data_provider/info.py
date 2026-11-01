import numpy as np
import pandas as pd
import os, sys

def get_spetral_info(data_sign):
    if data_sign == "Indian":
        spe_start, spe_end, band_width = 400, 2500, 10
        return spe_start, spe_end, band_width
    elif data_sign == "Pavia":
        spe_start, spe_end, band_width = 430, 860, 4
        return spe_start, spe_end, band_width
    elif data_sign == "Salinas":
        spe_start, spe_end, band_width = 400, 2500, 10
        return spe_start, spe_end, band_width
    elif data_sign == "Honghu" or data_sign == "WH":
        spe_start, spe_end, band_width = 400, 1000, 2
        return spe_start, spe_end, band_width
    elif data_sign == 'WHLK':
        spe_start, spe_end, band_width = 400, 1000, 2
        return spe_start, spe_end, band_width
    else:
        assert Exception("not implement")

def get_rgb_index(spe_start, spe_end, band_width):
    red_spe = (620 + 750) // 2
    green_spe = (495 + 570) // 2
    blue_spe = (450 + 495) // 2
    r = (red_spe - spe_start) // band_width
    g = (green_spe - spe_start) // band_width
    b = (blue_spe - spe_start) // band_width

    return r, g, b


def get_rgb_index_by_sign(data_sign):
    spe_start, spe_end, band_width = get_spetral_info(data_sign)
    r, g, b = get_rgb_index(spe_start, spe_end, band_width)
    return r, g, b

noise_type_list = ['jpeg', 'zmguass', 'additive', 'poisson', 'salt_pepper', 'stripes', 'deadlines', 'kernel', 'thin_fog', 'thick_fog']


Indian_kvs_params = {
    "jpeg" : {
            'compression_ratios' : [110]  # 10 100
        },

    "zmguass": {
            "sigma": 0.30
    },

    "additive": {
        #     "noise_level": 0.3 
            "noise_level": 0.3
    },

    "poisson": {
            "snr_db" : 7 # 30
    },

    "salt_pepper": {
        #     "amount": 0.15 
            "amount": 0.15
    },

    "stripes": {
            "start": 3,
            "end" : 5
    },

    "deadlines":{
        "start": 3,
        "end" : 5
    },

    "kernel": {
            "kernel_size": 2 # PU 10
    },

    "thin_fog": {
            'spe_start': 0,
            'spe_end': 0,
            'band_width': 0,
            'omega': 0.5,
            'alpha': 0.01,
            'beta': 1.0
    },

    "thick_fog": {
            'spe_start': 0,
            'spe_end': 0,
            'band_width': 0,
            'omega': 0.35,
            'alpha': 0.01,
            'beta': 1.0
    }
}

Pavia_kvs_params = {
    "jpeg" : {
            'compression_ratios' : [110]  # 10 100
        },

    "zmguass": {
            "sigma": 0.25
    },

    "additive": {
        #     "noise_level": 0.3 
            "noise_level": 0.2
    },

    "poisson": {
            "snr_db" : -10  # 30
    },

    "salt_pepper": {
        #     "amount": 0.15 
            "amount": 0.10 
    },

    "stripes": {
            "start": 30,
            "end" : 35
    },

    "deadlines":{
        "start": 30,
        "end" : 35
    },

    "kernel": {
            "kernel_size": 3 # PU 10
    },

    "thin_fog": {
            'spe_start': 0,
            'spe_end': 0,
            'band_width': 0,
            'omega': 0.3,
            'alpha': 0.01,
            'beta': 1.0
    },

    "thick_fog": {
            'spe_start': 0,
            'spe_end': 0,
            'band_width': 0,
            'omega': 0.35,
            'alpha': 0.01,
            'beta': 1.0
    }
}

WH_kvs_params = {
    "jpeg" : {
            'compression_ratios' : [10]
        },

    "zmguass": {
            "sigma": 0.3
    },

    "additive": {
            "noise_level": 0.3 
    },

    "poisson": {
            "snr_db" : 9 
    },

    "salt_pepper": {
            "amount": 0.15 
    },

    "stripes": {
            "start": 30,
            "end" : 35
    },

    "deadlines":{
        "start": 30,
        "end" : 35
    },

    "kernel": {
            "kernel_size": 5 # PU
    },

    "thin_fog": {
            'spe_start': 0,
            'spe_end': 0,
            'band_width': 0,
            'omega': 0.3,
            'alpha': 0.01,
            'beta': 1.0
    },

    "thick_fog": {
            'spe_start': 0,
            'spe_end': 0,
            'band_width': 0,
            'omega': 0.5,
            'alpha': 0.01,
            'beta': 1.0
    }
}


# WHLK_kvs_params = {
#     "jpeg" : {
#             'compression_ratios': [300]
#         },
#
#     "zmguass": {
#             "sigma": 0.5  # 0.3 0.1
#     },
#
#     "additive": {
#             "noise_level": 0.6
#     },
#
#     "poisson": {
#             "snr_db" : 1  # 22
#     },
#
#     "salt_pepper": {
#             "amount": 0.09  # 0.15
#     },
#
#     "stripes": {
#             "start": 35,
#             "end" : 40
#     },
#
#     "deadlines":{
#         "start": 20, # 30
#         "end" : 25  # 35
#     },
#
#     "kernel": {
#             "kernel_size": 6, # PU 3
#             'noise_level': 0.001
#     },
#
#     "thin_fog": {
#             'spe_start': 0,
#             'spe_end': 0,
#             'band_width': 0,
#             'omega': 0.9,  # 0.25
#             'alpha': 0.01,
#             'beta': 1.0
#     },
#
#     "thick_fog": {
#             'spe_start': 0,
#             'spe_end': 0,
#             'band_width': 0,
#             'omega': 0.35,  # 0.29
#             'alpha': 0.01,
#             'beta': 1.0
#     }
# }

WHLK_kvs_params = {
    "jpeg" : {
            'compression_ratios': [15]
        },

    "zmguass": {
            "sigma": 0.15  # 0.3 0.1
    },

    "additive": {
            "noise_level": 0.25
    },

    "poisson": {
            "snr_db" : 15  # 22
    },

    "salt_pepper": {
            "amount": 0.09  # 0.15
    },

    "stripes": {
            "start": 35,
            "end" : 40
    },

    "deadlines":{
        "start": 25, # 30
        "end" : 30  # 35
    },

    "kernel": {
            "kernel_size": 3, # PU 3
            'noise_level': 0.001
    },

    "thin_fog": {
            'spe_start': 0,
            'spe_end': 0,
            'band_width': 0,
            'omega': 0.2,  # 0.25
            'alpha': 0.01,
            'beta': 1.0
    },

    "thick_fog": {
            'spe_start': 0,
            'spe_end': 0,
            'band_width': 0,
            'omega': 0.25,  # 0.29
            'alpha': 0.01,
            'beta': 1.0
    }
}