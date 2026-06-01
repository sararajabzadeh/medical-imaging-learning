import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn.datasets import load_mni152_template

img = load_mni152_template(resolution=2)
data = img.get_fdata()
print("shape   :", data.shape)
print("dtype   :", data.dtype)
print("min/max :", data.min(), data.max())

affine = img.affine
print("\naffine:\n", affine)
print("voxel sizes (mm):", img.header.get_zooms())

voxel = np.array([50, 60, 40, 1])
world = affine @ voxel
print("voxel (50,60,40) -> world mm:", world[:3])
print("\norientation codes:", nib.aff2axcodes(affine))


ni, nj, nk = data.shape
ci, cj, ck = ni // 2, nj // 2, nk // 2

sagittal = data[ci, :, :].T
coronal = data[:, cj, :].T
axial = data[:, :, ck].T

fig, axes = plt.subplots(1, 3, figsize=(13, 5))
for ax, sl, title in zip(axes, [sagittal, coronal, axial], ["Sagittal", "Coronal", "Axial"]):
    ax.imshow(sl, cmap="gray", origin="lower")
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
plt.savefig("brain_slices.png", dpi=110, bbox_inches="tight")
print("\nsaved brain_slices.png")
    