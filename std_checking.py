import h5py, numpy as np
p="/lambda/nfs/Kiana/PhysioWave-main9/DB6_processed_8ch/train.h5"
with h5py.File(p,"r") as f:
    x=f["data"][:1000]   # sample
print("shape:", x.shape)
print("min/max:", x.min(), x.max())

print("mean/std:", x.mean(), x.std())
