# 3D Human Body Reshaping with Anthropometric Modeling  
source code of paper: [3D Human Body Reshaping with Anthropometric Modeling](https://link.springer.com/chapter/10.1007/978-981-10-8530-7_10) 

* please contact with me by email for the full-text paper. 

## Examples  
- Input: your anthropometric measurements such as height, weight etc. 
- Output: your 3D body shape (i.e., obj file)

![creating by deform-based global mapping](https://raw.githubusercontent.com/1900zyh/3D-Human-Body-Shape/master/pics/dg-h.png)


## Instructions  

### Training data
1. download training data from [SPRING](https://graphics.soe.ucsc.edu/data/BodyModels/index.html)
2. put the datasets under TrainingData folder 
3. refer to src/body_utils.py for building models

(Note: you can directly run the demo without training data. )


### Environment  
Windows/OSX/Linux

### Package  
```
pip install -r requirements.txt
```

(Note: if you want to run on Windows, strongly recommend to use 'pip install' from .whl files download [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyqt4))


### Training
You need to download your own datasets and run the scripts as below:
```
git clone https://github.com/1900zyh/3D-Human-Body-Shape.git
cd 3D-Human-Body-Shape/
cd src/ 
python body_utils.py
```

### Testing
I have put the data needed for running demo in data folder, you can run the demo directly by running the scripts as below:
```
cd src/
python demo.py
```

### Running API
1. adjust size (the numbers represents times of std, e.g. 30 means +3 std)
<!-- ![creating by deform-based global mapping](https://raw.githubusercontent.com/1900zyh/3D-Human-Body-Shape/master/pics/fig1.png) -->
<img src="https://raw.githubusercontent.com/1900zyh/3D-Human-Body-Shape/master/pics/fig1.png" width="500" hegiht="313" align=center />

2. 'Ctrl + s' to save obj file

3. choose different mapping method
<!-- ![creating by deform-based global mapping](https://raw.githubusercontent.com/1900zyh/3D-Human-Body-Shape/master/pics/fig2.png) -->
<img src="https://raw.githubusercontent.com/1900zyh/3D-Human-Body-Shape/master/pics/fig2.png" width="500" hegiht="313" align=center />

4. press 'PREDICT' button to input the numbers(You don't need to fill out the form, the defualt can be estimated)
<!-- ![creating by deform-based global mapping](https://raw.githubusercontent.com/1900zyh/3D-Human-Body-Shape/master/pics/fig3.png){:height="50%" width="50%"} -->
<img src="https://raw.githubusercontent.com/1900zyh/3D-Human-Body-Shape/master/pics/fig3.png" width="500" hegiht="313" align=center />

### Different Mapping Methods
1. global mapping
2. [local_with_mask](https://dl.acm.org/citation.cfm?id=2758217)
3. [local_with_rfemat](https://link.springer.com/chapter/10.1007/978-981-10-8530-7_10)


## Citation  
If you find this paper useful, please cite:

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
