# UA-BCD

The official implementation of **"Overcoming the Uncertainty Challenges in Detecting Building Changes from Remote Sensing Images"**.

We are delighted to share that our paper has been successfully accepted by the **ISPRS Journal of Photogrammetry and Remote Sensing (ISPRS 2024)**.

This repository contains the full implementation of our model, including training, testing, and a large-scale inference framework.

---

## 📦 Pretrained Backbones

We provide the pretrained backbone **PVT-v2-b2** for your convenience.  
You can download it via Baidu Disk:

- [Download Link](https://pan.baidu.com/s/16sA3ZejzcItAWa2JE1G6vg?pwd=abmg)  
  Code: `abmg`

---

## 🏋️‍♀️ Training Instructions

To train the UA-BCD model, follow these steps:

1. Set the hyperparameters for training.
2. Run the following command:

   ```bash
   python train.py --batchsize 32 --data_name LEVIR
