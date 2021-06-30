"""
Implementation of gaussian filter algorithm
"""
from itertools import product
from PIL import Image 
from numpy import dot, exp, mgrid, pi, ravel, square, uint8, zeros
import numpy as np
import math
import sys
import pycuda.autoinit
import pycuda.driver as drv
import pycuda.compiler as compiler

def gen_gaussian_kernel(k_size, sigma):
    # Cociente cuando a se divide por b, redondeado al siguiente número entero más pequeño
    center = k_size // 2
    # Creacion de la matriz [1,0,1; 1,0,1; 1,0,1]
    x, y = mgrid[0 - center : k_size - center, 0 - center : k_size - center]
    print(x," \n \n " ,y)
    #Multiplicacion de la matriz por la varianza solicitada
    matrizGauss = 1 / (2 * pi * sigma) * exp(-(square(x) + square(y)) / (2 * square(sigma)))
    return matrizGauss


if __name__ == "__main__":
    
    # Carga de imagen en RGB en la matriz y extraer sus canales de color #
    try:
        # read original image #
        img = Image.open("lubuntu.png")
        # Pasamos la imagen a un array de numpy para obtener los canales #
        img_input_array = np.array(img)
        
        # Creacion de arrays de numpy para cada canal. #
        # El método copy () devuelve una nueva lista. No modifica la lista original. #
        red = img_input_array[:, :, 0].copy()
        green = img_input_array[:, :, 1].copy()
        blue = img_input_array[:, :, 2].copy()

    except FileNotFoundError:
        sys.exit("No se pudo cargar la imagen")
    

    # Generando gaussian kernel (size of N * N) #
    # Tamaño de sigma y del kernel tres, cinco. #
    kernel = 5
    sigma = 1
    # LLamada a la funcion para la generacion de matriz de Gauss de NxN #
    gaussian_kernel = gen_gaussian_kernel(kernel, sigma)


    # Calculo de threats/blocks/gird basado en el ancho y altura de una imagen #
    
    # Obtencion del alto y ancho de una imagen hacemos uso de un canal blue. 
    alto, ancho = img_input_array.shape[:2]
    # height, width = img_input_array.shape[:2]
    
    # Dimension maxima por bloque
    dimension_por_bloque = 32
    
    # Dimension de cuadrilla para "x" y "y"
    dim_grid_x = math.ceil(ancho / dimension_por_bloque)
    dim_grid_y = math.ceil(alto / dimension_por_bloque)

    # Llamada a funcion de pycuda para obtener respuesta.
    # Leemos la funcion almacenada en el archivo gaussFilter.cu 
    mod = compiler.SourceModule(open('gaussFilter.cu').read())
    # Obtencion de la funcion de CUDA
    filtroGauss = mod.get_function('aplicarFiltroGauss')


    # Aplicacion de Filtro de Gauss #
    # paso de parametros para la funcion filtroGauss 
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
            grid=(dim_grid_x, dim_grid_y,1)
        )
    
    # Crear la matriz de salida con la misma forma y tipo que la matriz de entrada #
    # Creacion de una matriz de ceros del mismo size de la imagen original
    img_output_array = np.empty_like(img_input_array)
    # Union de cada canal rojo, azul y verde
    img_output_array[:, :, 0] = red
    img_output_array[:, :, 1] = green
    img_output_array[:, :, 2] = blue
    
    # Guardar imagen Resultados
    Image.fromarray(img_output_array).save("imagenFiltroGauss.png")
    imgSuavizado = Image.open("imagenFiltroGauss.png")
    # Mostrar las dos imagenes para comparar resultados.
    img.show()
    imgSuavizado.show()