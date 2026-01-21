python - << 'EOF'
import h5py

f = h5py.File("/lambda/nfs/Kiana/PhysioWave-main9/DB6_processed_8ch/train.h5", "r")
print("H5 dataset shape:", f["data"].shape)
print("Single sample shape:", f["data"][0].shape)
f.close()
EOF
