import cv2
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# ---------- Helper Functions ----------

def motion_blur_psf(size, angle):
    psf = np.zeros((size, size))
    psf[size // 2, :] = 1
    M = cv2.getRotationMatrix2D((size // 2, size // 2), angle, 1)
    psf = cv2.warpAffine(psf, M, (size, size))
    psf /= psf.sum()
    return psf

def fft_blur(image, psf):
    psf_pad = np.zeros_like(image)
    h, w = psf.shape
    psf_pad[:h, :w] = psf
    psf_pad = np.roll(psf_pad, -h // 2, axis=0)
    psf_pad = np.roll(psf_pad, -w // 2, axis=1)
    blurred = np.real(ifft2(fft2(image) * fft2(psf_pad)))
    return blurred, psf_pad

def tikhonov_deblur(blurred, psf, lambd):
    H = fft2(psf)
    G = fft2(blurred)
    H_conj = np.conj(H)
    denominator = H_conj * H + lambd
    F_hat = (H_conj * G) / denominator
    f_rec = np.real(ifft2(F_hat))
    return np.clip(f_rec, 0, 1)

def evaluate_metrics(original, restored):
    psnr = peak_signal_noise_ratio(original, restored, data_range=1.0)
    ssim = structural_similarity(original, restored, data_range=1.0)
    return psnr, ssim

# ---------- Main ----------

# Load image
image = cv2.imread("image\\lion.png", cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError("Image not found.")
image = image.astype(np.float32) / 255.0

# Create motion blur PSF
psf = motion_blur_psf(size=21, angle=15)

# Apply FFT-based blur
blurred, psf_centered = fft_blur(image, psf)

# Add realistic Gaussian noise
noise = np.random.normal(0, 0.005, image.shape)
blurred_noisy = np.clip(blurred + noise, 0, 1)

# Lambda range to test
lambda_values = [0.0001, 0.0005, 0.001, 0.005]
results = []
psnr_scores = []
ssim_scores = []

# Run Tikhonov for each lambda
for lambd in lambda_values:
    restored = tikhonov_deblur(blurred_noisy, psf_centered, lambd)
    psnr, ssim = evaluate_metrics(image, restored)
    results.append((lambd, restored))
    psnr_scores.append(psnr)
    ssim_scores.append(ssim)
    print(f"λ={lambd} → PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f}")

# ---------- Display Side-by-Side Images ----------

# Set number of rows and columns
total_images = len(results) + 2  # original + blurred + restored
cols = 3
rows = int(np.ceil(total_images / cols))

fig = plt.figure(figsize=(5 * cols, 4 * rows))
gs = gridspec.GridSpec(rows, cols, figure=fig, wspace=0.3, hspace=0.3)

# Add Original image
ax = fig.add_subplot(gs[0, 0])
ax.imshow(image, cmap='gray')
ax.set_title("Original")
ax.axis('off')

# Add Blurred + Noise
ax = fig.add_subplot(gs[0, 1])
ax.imshow(blurred_noisy, cmap='gray')
ax.set_title("Blurred + Noise")
ax.axis('off')

# Add Restored images
for i, (lambd, restored) in enumerate(results):
    row = (i + 2) // cols
    col = (i + 2) % cols
    ax = fig.add_subplot(gs[row, col])
    ax.imshow(restored, cmap='gray')
    ax.set_title(f"λ={lambd}")
    ax.axis('off')

plt.tight_layout()
plt.show()

# ---------- Plot PSNR and SSIM vs Lambda ----------

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(lambda_values, psnr_scores, marker='o')
plt.title("PSNR vs λ")
plt.xlabel("λ")
plt.ylabel("PSNR (dB)")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(lambda_values, ssim_scores, marker='o', color='orange')
plt.title("SSIM vs λ")
plt.xlabel("λ")
plt.ylabel("SSIM")
plt.grid(True)

plt.tight_layout()
plt.show()
