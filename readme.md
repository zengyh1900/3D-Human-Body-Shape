# 3D Human Body Reshaping with Anthropometric Modeling  

![creating by deform-based global mapping](https://raw.githubusercontent.com/1900zyh/3D-Human-Body-Shape/master/docs/dg-h.png)

### [Conference Paper](https://link.springer.com/chapter/10.1007/978-981-10-8530-7_10) | [Arxiv](https://arxiv.org/abs/2104.01762) | [Demo](https://sites.google.com/view/1900zyh/3dhumanbody)
3D Human Body Reshaping with Anthropometric Modeling<br>
In ICIMCS 2017 (Oral). <br>
[Yanhong Zeng](https://sites.google.com/view/1900zyh),  [Jianlong Fu](https://jianlong-fu.github.io/), [Hongyang Chao](https://scholar.google.com/citations?user=qnbpG6gAAAAJ&hl).<br>


## Introduction  
In this paper, we design a user-friendly and accurate system for 3D human body reshaping with limited anthropometric parameters (e.g., height and weight). Specifically, we leverage MICE technique for missing data imputation and we propose a feature-selection-based local mapping method for accurate shape modeling. The proposed feature-selection-based local mapping method can select the most relevant parameters for each facet automatically for linear regression learning, which eliminates heavy human efforts for utilizing topology information of body shape, and thus a more approximate body mesh can be obtained.

## Approach 
The overview of the proposed 3D human body reshaping system is shown as below. The system consists of three parts, i.e., the Imputer, the Selector and the Mapper in both online stage and offline stage. In offline stage, the Selector takes the dataset of 3D body meshes (a) and corresponding anthropometric parameters (b) as inputs to learn the relevance masks (c) by the proposed feature-selection-based local mapping technique. The mapping matrices (d) are further learned by linear regression from the parameters selected by (c) to mesh-based body representation. In online stage, MICE is leveraged in the Imputer for the imputation of the parameters from user input (e), which is introduced in Sect. 2.1. ‘?’ in (e) indicates the missing parameters from user inputs, yet could be complemented in (f) by the proposed approach. After imputation, the vector of parameters (f) will be passed to the Mapper. By adopting (c) and (d), 3D body mesh (g) will be generated from (f) in the Mapper.

![framework](https://raw.githubusercontent.com/1900zyh/3D-Human-Body-Shape/master/docs/framework.PNG)


## Experiments
<div align="center">
	<img src="https://raw.githubusercontent.com/1900zyh/3D-Human-Body-Shape/master/docs/table2.PNG" alt="Editor" width="500">
</div>


## Citation  
If any part of our paper and code is helpful to your work, 
please generously cite and star us :kissing_heart: :kissing_heart: :kissing_heart: !

```
@inproceedings{zeng20173d,
  title={3D Human Body Reshaping with Anthropometric Modeling},
  author={Zeng, Yanhong and Fu, Jianlong and Chao, Hongyang},
  booktitle={International Conference on Internet Multimedia Computing and Service},
  pages={96--107},
  year={2017},
  organization={Springer}
}
```


## Quick start

### Environment  
1. Windows/OSX/Linux, python3.5
2. install all packages you need from [.whl files](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyqt4) or by running:
```
pip install -r requirements.txt
```

### Preparation
1. download training data from [SPRING](https://graphics.soe.ucsc.edu/data/BodyModels/index.html)
2. put the datasets under 3D-human-Body-Shape/data folder 
3. download codes by running
```
git clone https://github.com/1900zyh/3D-Human-Body-Shape.git
cd 3D-Human-Body-Shape/
```


### Training
You need to download your own datasets and run the scripts as below:
```
cd src/ 
python train.py
```

### Testing
You can test the demo using [released model](https://github.com/1900zyh/3D-Human-Body-Shape/tree/master/release_model) by running:
```
cd src/
python demo.py
```

### Demo
1. adjust size (the numbers represents times of std, e.g. 30 means +3 std)
<img src="https://raw.githubusercontent.com/1900zyh/3D-Human-Body-Shape/master/docs/fig1.png" width="500" hegiht="313" align=center />

2. 'Ctrl + s' to save obj file

3. choose different mapping method
<img src="https://raw.githubusercontent.com/1900zyh/3D-Human-Body-Shape/master/docs/fig2.png" width="500" hegiht="313" align=center />

4. press 'PREDICT' button to input the numbers(You don't need to fill out the form, the defualt can be estimated)
<img src="https://raw.githubusercontent.com/1900zyh/3D-Human-Body-Shape/master/docs/fig3.png" width="500" hegiht="313" align=center />

### Different Mapping Methods
1. global mapping
2. [local_with_mask](https://dl.acm.org/citation.cfm?id=2758217)
3. [local_with_rfemat](https://link.springer.com/chapter/10.1007/978-981-10-8530-7_10)

