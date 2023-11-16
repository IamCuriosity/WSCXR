# WSCXR

**💡 This is the official implementation of the paper "Weakly Supervised Anomaly Detection for Chest X-Ray Image"**.  


WSCXR is a weakly supervised anomaly detection framework for Chest X-Ray (CXR) image. WSCXR can effectively leverage medical cues from few-shot real anomalous images for anomaly detection, thereby improving the model’s anomaly detection performance. Additionally, WSCXR employs a linear mixing strategy to augment the anomaly features, facilitating the training of anomaly detector with few-shot anomaly images. 

<div align=center><img width="850" src="assets/pipeline.PNG"/></div>  

## 🔧 Installation

To run experiments, first clone the repository and install `requirements.txt`.

```
$ git clone https://github.com/IamCuriosity/WSCXR.git
$ cd WSCXR
$ pip install -r requirements.txt
```  
### Data preparation 
Download the following datasets:
- **ZhangLab  [[Baidu Cloud]]() or [[Google Drive]]()**  
- **CheXpert [[Baidu Cloud]]() or [[Google Drive]]()**  

Unzip them to the `data`. Please refer to [data/README](data/README.md).  
  
## 🚀 Experiments

To train the WSCXR on the ZhangLab dataset:  
```
$ python  train.py --dataset_name zhanglab  
```  
   
To test the WSCXR on the ZhangLab dataset:  
```
$ python  test.py --dataset_name zhanglab  
```  

<div align=center><img width="500" src="assets/results.png"/></div>  

## 🔗 Citation  

If this work is helpful to you, please cite it as:
```
coming soon.
```
