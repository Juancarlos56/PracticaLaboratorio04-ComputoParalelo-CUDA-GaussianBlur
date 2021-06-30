__global__ void aplicarFiltroGauss(const unsigned char *inputEspacioColor, 
                                    unsigned char *outputEspacioColor, 
                                    const unsigned int ancho, 
                                    const unsigned int alto, 
                                    const float *gausskernel, 
                                    const unsigned int kernel) {

    const unsigned int columnas = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int filas = threadIdx.y + blockIdx.y * blockDim.y;

    if(filas < alto && columnas < ancho) {
        const int mitadSizeKernel = kernel / 2;
        float pixel = 0.0;
        for(int i = -mitadSizeKernel; i <= mitadSizeKernel; i++) {
            for(int j = -mitadSizeKernel; j <= mitadSizeKernel; j++) {

                const unsigned int y = max(0, min(alto - 1, filas + i));
                const unsigned int x = max(0, min(ancho - 1, columnas + j));

                const float w = gausskernel[(j + mitadSizeKernel) + (i + mitadSizeKernel) * kernel];
                pixel += w * inputEspacioColor[x + y * ancho];
            }
        }
        outputEspacioColor[columnas + filas * ancho] = static_cast<unsigned char>(pixel);
    }
}
