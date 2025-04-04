import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def ssim(clean, noisy, normalized=True, raw=False):
    """Use skimage.meamsure.compare_ssim to calculate SSIM
    Args:
        clean (Tensor): (B, 1, H, W)
        noisy (Tensor): (B, 1, H, W)
        normalized (bool): If True, the range of tensors are [0., 1.] else [0, 255]
    Returns:
        SSIM per image: (B, )
    """
    if normalized:
        clean = clean.mul(255).clamp(0, 255)
        noisy = noisy.mul(255).clamp(0, 255)

    clean = clean.cpu().detach().numpy().astype(np.float32).transpose(0,2,3,1)
    noisy = noisy.cpu().detach().numpy().astype(np.float32).transpose(0,2,3,1)

    # squeeze the last dimension
    clean = np.squeeze(clean, axis=-1)
    noisy = np.squeeze(noisy, axis=-1)
    
    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    clean = clean + epsilon
    noisy = noisy + epsilon

    if raw:
        noisy = (np.uint16(noisy*(2**12-1-240)+240).astype(np.float32)-240)/(2**12-1-240)
    
    if normalized:
        return np.array([structural_similarity(c, n, data_range=255, multichannel=False) for c, n in zip(clean, noisy)]).mean()
    else:
        return np.array([structural_similarity(c, n, data_range=1.0, multichannel=False) for c, n in zip(clean, noisy)]).mean()


def psnr(clean, noisy, normalized=True, raw=False):
    """Use skimage.meamsure.compare_ssim to calculate SSIM
    Args:
        clean (Tensor): (B, 1, H, W)
        noisy (Tensor): (B, 1, H, W)
        normalized (bool): If True, the range of tensors are [0., 1.]
            else [0, 255]
    Returns:
        SSIM per image: (B, )
    """
    if normalized:
        clean = clean.mul(255).clamp(0, 255)
        noisy = noisy.mul(255).clamp(0, 255)

    clean = clean.cpu().detach().numpy().astype(np.float32).transpose(0,2,3,1)
    noisy = noisy.cpu().detach().numpy().astype(np.float32).transpose(0,2,3,1)
    
    # add small epsilon to avoid division by zero
    epsilon = 1e-10
    clean = clean + epsilon
    noisy = noisy + epsilon

    if raw:
        noisy = (np.uint16(noisy*(2**12-1-240)+240).astype(np.float32)-240)/(2**12-1-240)
    
    if normalized:
        return np.array([peak_signal_noise_ratio(c, n, data_range=255) for c, n in zip(clean, noisy)]).mean()
    else:
        return np.array([peak_signal_noise_ratio(c, n, data_range=1.0) for c, n in zip(clean, noisy)]).mean()
