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
import multiprocessing
import signal

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
    #print("***************Matriz Resultado***************")
    #print("Divisor comun para la matriz: ", kernel_matrix.sum())
    # dividimos la matriz para el resultado. 
    kernel_matrix = kernel_matrix / kernel_matrix.sum()
    #print(kernel_matrix)
    return kernel_matrix


def filtroGauss(img_input_array, sigma, kernel, gauss_matriz):
    # Crear la matriz de salida con la misma forma y tipo que la matriz de entrada #
    # Creacion de una matriz de ceros del mismo size de la imagen original #
    result_array = np.empty_like(img_input_array)
    # Obtencion del alto y ancho de una imagen hacemos uso de un canal blue. #
    alto, ancho = img_input_array.shape[:2]
    # Obtenemos el valor central de la matriz 5//2 = 2.5 = 2 #
    centro_del_kernel = kernel // 2
    
    # Recorrido de cada pixel de la imagen de manera secuencial
    # i va a obtener los valores del alto de la imagen y j los valores del ancho 
    for i in range(0, alto):
        for j in range(0, ancho):
            # Variables que van a almacenar la suma de producto de los canales rgb por cada valor de la matriz de gauss
            red = 0.0
            green = 0.0
            blue = 0.0
            
            # Bucles para recorrer la matriz de gauss#
            for k in range(-centro_del_kernel, centro_del_kernel + 1):
                for l in range(-centro_del_kernel, centro_del_kernel + 1):

                    # Obtenemos la posicion "x" y "y" de un pixel en especifico para manipularlo. #
                    # Con el min aseguramos no pasarnos del ancho o alto de la imagen #
                    # Con el max aseguramos obtener solo valores positivos y no se problemas con la dimension de la imagen                    
                    x = max(0, min(img_input_array.shape[1] - 1, j + l))
                    y = max(0, min(img_input_array.shape[0] - 1, i + k))
                    #print("-----------------------------------")
                    #print(i," | ",j," | ",x ," | ", y, " | ", img_input_array.shape[1], " | ", img_input_array.shape[1], " | ", img_input_array[y][x])
                    
                    # Obtenemos la posicion del pixel: img_input_array[y][x] = [0,72,166]
                    # Obtenemos los valores de la matriz de Gauss: gauss_matriz[0][0] = 0.0032....
                    # Multiplicamos cada canal rgb por el valores de la matriz de gauss y los almacenamos 
                    r, g, b = (img_input_array[y][x] * gauss_matriz[k + centro_del_kernel][l + centro_del_kernel])
                    # Se suma y almacera los resultados para llegar a obtener los valores reales para nuestro pixel #
                    red += r
                    green += g
                    blue += b
            # Colocamos nuestro nuevo valor para el pixel con gauss aplicado, en los tres espacios de color.#
            result_array[i][j] = (red, green, blue)
    return result_array


if __name__ == "__main__":
    
    # Carga de imagen en RGB en la matriz y extraer sus canales de color #
    try:
        # read original image #
        img = Image.open("imagenNormal.jpg")
        # Pasamos la imagen a un array de numpy para obtener los canales #
        img_input_array = np.array(img)

    except FileNotFoundError:
        sys.exit("No se pudo cargar la imagen")
    
    # Generando gaussian kernel (size of N * N) #1234 35
    kernel = 3
    sigma = 4
    
    # Crear la matriz de salida con la misma forma y tipo que la matriz de entrada #
    # Creacion de una matriz de ceros del mismo size de la imagen original #
    img_output_array = np.empty_like(img_input_array)
    
    #LLamada a la funcion para la generacion de matriz de Gauss de NxN #
    gauss_matriz = gen_gaussian_kernel(kernel, sigma)
    
    # Aplicacion de Filtro de Gauss #
    # Toma de tiempo para el programa
    time_started = timeit.default_timer()
    # LLamda a la funcion de Gauss para su respectivo calculo#
    img_output_array = filtroGauss(img_input_array, sigma, kernel, gauss_matriz)
    time_ended = timeit.default_timer()
    # mostrar tiempo total
    print('Tiempo de ejecucion del progrma: ', time_ended - time_started, 's')
    

    # Union de cada canal rojo, azul y verde
    #print(red)
    
    # Guardar imagen Resultados
    Image.fromarray(img_output_array).save("Resultados-CPU/imagenSecuencialFiltroGauss5x4.png")
    imgSuavizado = Image.open("Resultados-CPU/imagenSecuencialFiltroGauss5x4.png")
    # Mostrar las dos imagenes para comparar resultados.
    img.show()
    imgSuavizado.show()