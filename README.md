# Towards High-Resolution 3D Anomaly Detection: A Scalable Dataset and Real-Time Framework for Subtle Industrial Defects
🌐 [Hugging Face Dataset](https://huggingface.co/datasets/ChengYuQi99/MiniShift)  


> 📚 [**Paper**](https://arxiv.org/abs/2507.07435) • 🏠 [**Homepage**](https://hustcyq.github.io/MiniShift-Simple3D)  
> by , [Yuqi Cheng*](https://hustcyq.github.io/), [Yihan Sun*](), [Hui Zhang]() [Weiming Shen](https://scholar.google.com/citations?user=FuSHsx4AAAAJ&hl=en), [Yunkang Cao](https://caoyunkang.github.io/)


## 🚀 Updates  
We're committed to open science! Here's our progress:  
* **2025/07/10**: 📄 Paper released on [ArXiv](https://arxiv.org/abs/2507.07435).  
* **2025/07/08**: 🌐 Dataset homepage launched.  

Code will be available soon!

## 📊 Introduction  
In industrial point cloud analysis, detecting subtle anomalies demands high-resolution spatial data, yet prevailing benchmarks emphasize low-resolution inputs. To address this disparity, we propose a scalable pipeline for generating realistic and subtle 3D anomalies. Employing this pipeline, we developed **MiniShift**, the inaugural high-resolution 3D anomaly detection dataset, encompassing 2,577 point clouds, each with 500,000 points and anomalies occupying less than 1% of the total. We further introduce **Simple3D**, an efficient framework integrating Multi-scale Neighborhood Descriptors (MSND) and Local Feature Spatial Aggregation (LFSA) to capture intricate geometric details with minimal computational overhead, achieving real-time inference exceeding 20 fps. Extensive evaluations on **MiniShift** and established benchmarks demonstrate that **Simple3D** surpasses state-of-the-art methods in both accuracy and speed, highlighting the pivotal role of high-resolution data and effective feature aggregation in advancing practical 3D anomaly detection.


## 🔍 Overview of MiniShift

### 12 Categories and 4 defect types  
<img src="./static/images/dataset.png" width="800px">  


### Anchor-Guided Geometric Anomaly Synthesis
<img src="./static/images/dataset_pipe.png" width="400px">  

### Download
You are welcome to try our dataset: [Hugging Face Dataset](https://huggingface.co/datasets/ChengYuQi99/MiniShift)  


## Simple3D
<img src="./static/images/methods.png" width="800px">  



## 📊 Main Results  
### 1. Performance on MiniShift
<img src="./static/images/minishift.png" width="800px">  

### 2. Performance on Real3D-AD, Anomaly-ShapeNet, and MulSenAD 
<img src="./static/images/public_dataset.png" width="800px">  


## 🙏 Acknowledgements  
Grateful to these projects for inspiration:  
- 🌟 [BTF](https://github.com/eliahuhorwitz/3D-ADS)
- 🎨 [GLFM](https://github.com/hustCYQ/GLFM-Multi-class-3DAD)


## 📖 Citation  
If our work aids your research, please cite:  
```bibtex  
@article{MiniShift_Simple3D,  
  title={Towards High-Resolution 3D Anomaly Detection: A Scalable Dataset and Real-Time Framework for Subtle Industrial Defects},
  author={Cheng, Yuqi and Sun, Yihan and Zhang, Hui and Shen, Weiming and Cao, Yunkang},
  journal={arXiv preprint arXiv:2507.07435},
  year={2025}
}  
```  

## Contact
If you have any questions about our work, please do not hesitate to contact [yuqicheng@hust.edu.cn](mailto:yuqicheng@hust.edu.cn).
