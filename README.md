# Edge Device Realtime SuperResolution based on eSR

## Introduction
This project is based on [eSR](https://github.com/pnavarre/eSR). Adding interface for RGB image and video input and output, support more image enhancement methods and estimation of the model's performance on both cpu and gpu.

## Run project
1. Environment setup
```bash
pip install -r requirements.txt
```

2. Run the project
```bash
python run.py
```

## demo
- image
<div style="float:left;border:solid 1px 000;margin:2px;"><img src="./images/SR1.jpg"  width = "300" height = "300" alt="SR1" align=center /></div>
<div style="float:left;border:solid 1px 000;margin:2px;"><img src="./images/SR2.jpg"  width = "300" height = "300" alt="SR2" align=center /></div>
- mp4

![video](./images/compared.mp4)