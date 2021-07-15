# PracticaLaboratorio04-ComputoParalelo-CUDA-GaussianBlur
Implementar una función en el GPU a través de PyCuda en el lenguaje Python que realice la convolución de una imagen con un filtro gaussiano.
### Instalacion de PyCuda en Ubuntu 16.04 LTS
#### Versión de python==3.5, si utiliza anaconda, ver documentación: 
- sudo pip3 install --global-option=build_ext --global-option="-I/usr/local/cuda/include" --global-option="-L/usr/local/cuda/lib64" pycuda 
#### Para comprobar ejecutar en la consola de ubuntu, lo siguiente: 
- python3
- import pycuda.driver as drv 
### Referencias
- https://github.com/harrytang/cuda-gaussian-blur
- https://github.com/dino8890/img-filters/tree/14e1e30b1bac330358cd0288755846934bb60194
