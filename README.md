# HB_CLS

The corresponded code of the paper "Accurate identification of hemangioblastomas from other cerebellar-and-brainstem tumors using a contrast-enhanced MR dataset: A convolutional neural network model."


## Folder architecture

├─model (The configurement file forvarious run)  
│  ├─cls_2d   
│  ├─cls_2d_angiocavernoma  
│  ├─cls_2d_glioma  
│  └─cls_2d_maoxing  
├─nets ( Network )  
├─scripts ( Code for model training)  
└─utils (Some useful utils)  

## How to run
```python
python scripts/train_cls_2d.py -i /cinfig_path
```
