"""
Implementation of gaussian filter algorithm
"""
from itertools import product
from PIL import Image 
from numpy import dot, exp, mgrid, pi, ravel, square, uint8, zeros
import numpy as np
import math
import sys
import timeit
import pycuda.autoinit
import pycuda.driver as drv
import pycuda.compiler as compiler

def gaussFilter(x,y, sigma):
    #formula: w(x,y) =  e^(-(x^2+ y^2)/(sigma^2))/(2*pi*sigma^2)
    return ( np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2))

#Crear una matriz de numpy tipo float32
def gen_gaussian_kernel(kernel , sigma):
    # Creacion de una matriz vacia de NxN de tipo float#
    kernel_matrix = np.empty((kernel, kernel), np.float32)
    # Obtenemos el valor central de la matriz 5//2 = 2.5 = 2
    centro_del_kernel = kernel // 2

    # Iteramos desde -2 hasta +3 teniendo 5 posiciones si el kernel es de 5
    for i in range(-centro_del_kernel, centro_del_kernel + 1):
        # Iteramos desde -2 hasta +3 teniendo 5 posiciones si el kernel es de 5
        for j in range(-centro_del_kernel, centro_del_kernel + 1):
            # creamos la matriz en la posicion i,j, va desde 0 a 4 cuando el kernel es 5
            kernel_matrix[i + centro_del_kernel][j + centro_del_kernel] = gaussFilter(i,j, sigma)
            #print(" i + centro_del_kernel ", i + centro_del_kernel)
            #print(" j + centro_del_kernel ", j + centro_del_kernel)
            #print("valor = ", gaussFilter(i,j))
    print("***************Matriz Resultado***************")
    print("Divisor comun para la matriz: ", kernel_matrix.sum())
    # dividimos la matriz para el resultado. 
    kernel_matrix = kernel_matrix / kernel_matrix.sum()
    print(kernel_matrix)
    return kernel_matrix

if __name__ == "__main__":
    
    # Carga de imagen en RGB en la matriz y extraer sus canales de color #
    try:
        # read original image #
        img = Image.open("imagenNormal.jpg")
        # Pasamos la imagen a un array de numpy para obtener los canales #
        img_input_array = np.array(img)
        
        # Creacion de arrays de numpy para cada canal. #
        red = img_input_array[:, :, 0].copy()
        green = img_input_array[:, :, 1].copy()
        blue = img_input_array[:, :, 2].copy()

    except FileNotFoundError:
        sys.exit("No se pudo cargar la imagen")
    

    # Generando gaussian kernel (size of N * N) #1234 35
    kernel = 5
    sigma = 4
    
    # LLamada a la funcion para la generacion de matriz de Gauss de NxN #
    gaussian_kernel = gen_gaussian_kernel(kernel, sigma)
    #gaussian_kernel = gaussian_kernel/gaussian_kernel.sum()
    # Calculo de threats/blocks/gird basado en el ancho y altura de una imagen #
    
    # Obtencion del alto y ancho de una imagen hacemos uso de un canal blue. 
    alto, ancho = img_input_array.shape[:2]
    print("Dimension de la imagen: ", alto," | ",ancho)
    # height, width = img_input_array.shape[:2]
    
    # Dimension maxima por bloque
    dimension_por_bloque = 32
    
    # Dimension de cuadrilla para "x" y "y" 
    # ceil nos devuelve un valor entero de la division obtenida.
    dim_grid_x = int(math.ceil(ancho / dimension_por_bloque))
    dim_grid_y = int(math.ceil(alto / dimension_por_bloque))
    
    #Obtencion del maximo numero de bloques 
    max_num_blocks = (pycuda.autoinit.device.get_attribute(drv.device_attribute.MAX_GRID_DIM_X)
                    * pycuda.autoinit.device.get_attribute(drv.device_attribute.MAX_GRID_DIM_Y))

    if(dim_grid_x*dim_grid_y) > max_num_blocks:
        raise ValueError(
            "La imagen supera el maximo numero de bloques"
        )


    # Llamada a funcion de pycuda para obtener respuesta.
    # Leemos la funcion almacenada en el archivo gaussFilter.cu 
    mod = compiler.SourceModule(open('gaussFilter.cu').read())
    # Obtencion de la funcion de CUDA
    filtroGauss = mod.get_function('aplicarFiltroGauss')


    # Aplicacion de Filtro de Gauss #
    # paso de parametros para la funcion filtroGauss 
    # Toma de tiempo para el programa
    time_started = timeit.default_timer()
    for espacioColor in (red, green, blue):
        # Parametros:
        # 1. Input: canal que pasamos
        # 2. Output: canal que se recupera y se almacena en la misma variable
        # 3. ancho imagen
        # 4. alto imagen
        # 5. Matriz de Gauss
        # 6. Size de kernel
        # 7. block
        # 8. grid
        filtroGauss(
            drv.In(espacioColor),
            drv.Out(espacioColor),
            np.uint32(ancho),
            np.uint32(alto),
            drv.In(gaussian_kernel),
            np.uint32(kernel),
            block=(dimension_por_bloque, dimension_por_bloque, 1),
            grid=(dim_grid_x, dim_grid_y)
        )
    time_ended = timeit.default_timer()
    # display total time
    print('Tiempo de ejecucion del progrma: ', time_ended - time_started, 's')
    # Crear la matriz de salida con la misma forma y tipo que la matriz de entrada #
    # Creacion de una matriz de ceros del mismo size de la imagen original
    img_output_array = np.empty_like(img_input_array)
    # Union de cada canal rojo, azul y verde
    #print(red)
    img_output_array[:, :, 0] = red
    img_output_array[:, :, 1] = green
    img_output_array[:, :, 2] = blue
    
    # Guardar imagen Resultados
    Image.fromarray(img_output_array).save("Resultados/imagenFiltroGauss5x4.png")
    #imgSuavizado = Image.open("imagenFiltroGauss.png")
    # Mostrar las dos imagenes para comparar resultados.
    #img.show()
    #imgSuavizado.show()