# 3D Human Body Reshaping with Anthropometric Modeling  
source code of paper: [3D Human Body Reshaping with Anthropometric Modeling](https://link.springer.com/chapter/10.1007/978-981-10-8530-7_10) 

## Examples  
- Input: your anthropometric measurements such as height, weight etc. 
- Output: your 3D body shape (i.e., obj file)

![creating by deform-based global mapping](https://raw.githubusercontent.com/1900zyh/3D-Human-Body-Shape/master/pics/dg-h.png)


## Instructions  

### Environment  
Windows/OSX/Linux

### Package  
1. python 3.5
2. cython
3. cvxpy  
4. numpy
5. scipy
6. openpyxl
7. vtk
8. ecos
9. fancyimpute=0.3.2
10. traits
11. opengl
12. ctypes
13. pyqt4
14. mayavi  

(Note: if you want to run on Windows, strongly recommend to use 'pip install' from .whl files download [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyqt4))


### Training
```
git clone https://github.com/1900zyh/3D-Human-Body-Shape.git
cd 3D-Human-Body-Shape/
cd src/ 
python body_utils.py
```

### Testing
```
cd src/
python demo.py
```

### Running API
1. adjust size (the numbers represents times of std, e.g. 30 means +3 std)
![creating by deform-based global mapping](https://raw.githubusercontent.com/1900zyh/3D-Human-Body-Shape/master/pics/fig1.png)

2. 'Ctrl + s' to save obj file

3. choose different mapping method
![creating by deform-based global mapping](https://raw.githubusercontent.com/1900zyh/3D-Human-Body-Shape/master/pics/fig2.png)

4. press 'PREDICT' button to input the numbers(You don't need to fill out the form, the defualt can be estimated)
![creating by deform-based global mapping](https://raw.githubusercontent.com/1900zyh/3D-Human-Body-Shape/master/pics/fig3.png)


# Different Mapping Methods
1. global mapping
2. [local_with_mask](https://dl.acm.org/citation.cfm?id=2758217)
3. [local_with_rfemat](https://link.springer.com/chapter/10.1007/978-981-10-8530-7_10)
