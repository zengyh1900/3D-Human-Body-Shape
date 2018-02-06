![creating by deform-based global mapping](https://raw.githubusercontent.com/1900zyh/3D-Human-Body-Shape/master/pics/dg-h.png)

# What's this?
1. It's a personal project for synthesizing 3D Human Body Shape.
2. Input: your anthropometric measurements such as height, weight etc.
3. Output: your 3D body shape

# Installation
1. python 3.5
2. Numpy
3. Scipy
4. openpyxl
5. fancyimpute
6. pyqt4
7. mayavi
8. ctypes
9. opengl


# Instructions

It works well on Windows/OSX/Linux.


## Training
1. cd 3D-Human-Body-Shape
2. download related data
2. cd src 
3. python body_utils

## Testing
1. cd src
2. python demo


# During Testing
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
3. [local_with_rfemat](https://raw.githubusercontent.com/1900zyh/3D-Human-Body-Shape/master/pics/3D_Human_Body_Reshaping_with_Anthropometric_Modeling.pdf)