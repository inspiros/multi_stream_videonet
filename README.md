# Code đề tài AFORS2022

### Train:

#### 1. Dataset IPN
##### 1.1. RGB
Lệnh chạy:
```terminal
bash scripts/run_train_ipn_rgb.sh [model_arch]
```
##### 1.2. OF
Lệnh chạy:
```terminal
bash scripts/run_train_ipn_of.sh [model_arch]
```
##### 1.3. RGB + OF
_TODO_

#### 2. Dataset AFORS
##### 2.1. RGB
Lệnh chạy:
```terminal
bash scripts/run_train_afors_rgb.sh [model_arch]
```
##### 2.2. OF
_TODO_

### Note:

Argument `model_arch` là tên của architecture. Các model cần chạy:
- `c3d_bn`
- `r2plus1d_18`
- `r3d_18`
- `densenet3d_21`
- `mobilenet3d`
- `efficientnet3d_b0`
