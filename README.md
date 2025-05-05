# WGLIN
Code for paper: Wavelet-based Global-Local Interaction Network with Cross-Attention for Multi-View Diabetic Retinopathy Detection(ICME2025)

You can train the model using `python train.py` and test it with `python test.py`. Pretrained weights can be downloaded from [this link](https://pan.baidu.com/s/1KlB6FfsHDfAePHqQG-0f6g?pwd=k6m5).

For **four-view inference**, each of the four images must be pre-processed using the provided script `process_image.py`, and lesion information should be obtained via [HACDR-Net](https://github.com/xqh180110910537/HACDR-Net).

The dataset is available at [MVCINN](https://github.com/XiaolingLuo/MVCINN).
