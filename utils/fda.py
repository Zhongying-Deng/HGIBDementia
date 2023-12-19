'''
   Code from https://github.com/YanchaoYang/FDA

   Yanchao Yang and Stefano Soatto. 'FDA: Fourier Domain Adaptation for Semantic Segmentation'. CVPR 2020
'''
import torch
import numpy as np
from random import uniform


def extract_ampl_phase(fft_im):
    # fft_im: size should be bx3xhxwx2
    fft_amp = fft_im[...,0]**2 + fft_im[...,1]**2
    fft_amp = torch.sqrt(fft_amp)
    fft_pha = torch.atan2( fft_im[...,1], fft_im[...,0] )
    return fft_amp, fft_pha

def mix_freq_mutate(amp_src, amp_trg):
    bs_src = amp_src.size(0)
    bs_trg = amp_trg.size(0)
    if bs_trg < bs_src:
        times = bs_src // bs_trg
        amp_tmp = torch.cat([amp_trg for _ in range(times + 1)], 0)
        amp_trg = amp_tmp[:bs_src, :, :, :]
    else:
        amp_trg = amp_trg[:bs_src, :, :, :]

    lmda = uniform(0., 1.0)
    amp_src = lmda * amp_src + (1 - lmda) * amp_trg
    return amp_src

def mix_amplitude(src_img, trg_img):
    # exchange magnitude
    # input: src_img, trg_img

    # get fft of both source and target
    # fft_src = torch.fft.rfft( src_img.clone(), signal_ndim=2, onesided=False )
    # fft_trg = torch.fft.rfft( trg_img.clone(), signal_ndim=2, onesided=False )
    fft_src = torch.fft.rfftn(src_img.clone(), dim=(-3, -2, -1), norm="ortho")
    fft_src = torch.stack((fft_src.real, fft_src.imag), -1)
    fft_trg = torch.fft.rfftn(trg_img.clone(), dim=(-3, -2, -1), norm="ortho")
    fft_trg = torch.stack((fft_trg.real, fft_trg.imag), -1)
    
    # extract amplitude and phase of both ffts
    amp_src, pha_src = extract_ampl_phase( fft_src.clone())
    amp_trg, pha_trg = extract_ampl_phase( fft_trg.clone())

    # replace the low frequency amplitude part of source with that from target
    amp_src_ = mix_freq_mutate(amp_src.clone(), amp_trg.clone())

    # recompose fft of source
    fft_src_ = torch.zeros(fft_src.size(), dtype=torch.float )
    fft_src_[...,0] = torch.cos(pha_src.clone()) * amp_src_.clone()
    fft_src_[...,1] = torch.sin(pha_src.clone()) * amp_src_.clone()

    # get the recomposed image: source content, target style
    #src_in_trg = torch.fft.irfft( fft_src_, signal_ndim=2, onesided=False, signal_sizes=[imgH,imgW] )
    real_part = fft_src_[..., 0]
    imaginary_part = fft_src_[..., 1]
    complex_tensor = real_part + 1j * imaginary_part
    src_in_trg = torch.fft.irfftn(complex_tensor, dim=(-3, -2, -1), s=src_img.size()[-3:], norm="ortho")

    return src_in_trg

