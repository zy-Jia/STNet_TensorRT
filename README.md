# STNet

The Pytorch implementation is [xuexingyu24/License_Plate_Detection_Pytorch](https://github.com/xuexingyu24/License_Plate_Detection_Pytorch).

## How to Run

1. generate STNet.wts from pytorch

```
git clone https://github.com/xuexingyu24/License_Plate_Detection_Pytorch.git

// gen_wts.py to License_Plate_Detection_Pytorch/
// go to License_Plate_Detection_Pytorch/
python genwts.py
// a file 'STNet.wts' will be generated.
```

2. build STNet and run

```
// put STNet.wts into STNet_TensorRT
// go to STNet_TensorRT
mkdir build
cd build
cmake ..
make
sudo ./STNet -s  // serialize model to file i.e. 'STNet.engine'
sudo ./STNet -d  // deserialize model and run inference
```

