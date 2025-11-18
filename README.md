# GPU-Flip-Image
Project: GPU - Generate Fractal Images
Author: George Barrinuevo
Date: 11/17/2025

## Purpose
The purpose of this repository is to demonstrate how GPU programming using CUDA NPP libraries can be used to generate fractal images. 

## Description
The code will read a data.txt input file where each line contains the "C" parameter to the fractal, first column is the real number, second column is the imaginary number. The mathematics used to generate these fractals can be found on the internet. The program can generate Julia and Mandelbot Set fractals, but the input data.txt file contains only "C" parameters for the Julia Set. The user can create another file to contain the "C" parameters for the Mandelbot Set. Each fractal is saved to a JPG file for later viewing.
- Input File
The input file data.txt contains the "C" parameters for the Julia Set in this form: '1.23 4.56', where the first number is the real number, second number is the imaginary number. This would be expressed mathematically like this: C = 1.23 + 4.56i.
- Output Directory
Once each fractal is generated will output the image to a JPG file in the out_data/ directory. You can click on a factal_xxxx.jpg file to view it. I have included the fractal images this program will generate in this directory.
- Development Environment
I used Coursera's VS Code GUI to develop this code using NVidia's CUDA C++ libraries. The GPU was included in the VS Code GUI environment is a GPU is required to run this code. 
- Artifacts
The output.txt contains the runtime output to the fractal program.
The generated fractal JPG images are found in the out_data/ directory.
- Lessons Learned
While debugging the code, I found out some older version of the CUDA functions are deprecated which requires using the latest versions and methods assuming you have the latest libraries/packages installed. I also learned that the best starting point on a CUDA project is to obtain a code template that demonstrates how to use the CUDA in say an image manipulation project. This template will have a lot of the allocate, de-allocate, copy, kernel, and etc that is required for CUDA programming. I also learned when calling a CUDA function, to check for any errors.

## Pseudocode
Here is a pseudocode of what this code does. Developing pseudocode is a good preparation before diving in to the CUDA code.
- Allocate device memory.
- Read the fractal "C" parameter values.
- Iterate through all of the "C" parameter values.
- Call the kernel to calculate the fractal math.
- Wait for all kernel processes are completed.
- Copy memory from device to host.
- Generate the fractal JPG image file.
- Free the device and host memory.

## CUDA Library Functions
- cudaFree(d_img)
This de-allocates GPU memory.
d_img - This points to the memory to de-allocate.
- cudaMemcpy(d_mem, s_mem, imgSize, cudaMemcpyDeviceToHost)
This copies data between CPU and GPU. 
cudaMemcpyDeviceToHost - Copies data from GPU device to host.
cudaMemcpyHostToDevice - Copies data from host to GPU device.
cudaMemcpyDeviceToDevice - Copies data from GPU device to GPU device.
d_mem - Destination memory.
s_mem - Source memory
imgSize - The number of bytes to copy.
- cudaDeviceSynchronize()
This blocks the CPU until all previously launched GPU work in finished.
- fractalKernel<<grid, block>>()
This will run the fractal algorithm in parallel on the GPU.
- cudaMalloc(&d_img, imgSize)
It allocates memory on the GPU device.
d_img - This is a pointer to the allocated memory.
imgSize - This is the size of the memory in bytes to allocate.
- CUDA_CHECK()
This is a user-defined error check wrapper. After calling a CUDA function, call this to check for any errors.

## Fractal Math
Here is a list of links I used to study the mathematics of fractals. It include both Julia and Mandelbot Set fractals.
- https://en.wikipedia.org/wiki/Julia_set
- https://mathworld.wolfram.com/JuliaSet.html
- https://e.math.cornell.edu/people/belk/dynamicalsystems/NotesJuliaMandelbrot.pdf
- https://www.cantorsparadise.com/the-julia-set-e03c29bed3d0
- https://e.math.cornell.edu/people/belk/dynamicalsystems/NotesJuliaMandelbrot.pdf
- https://en.wikipedia.org/wiki/Mandelbrot_set
- https://alonso-delarte.medium.com/a-quick-explanation-of-the-mandelbrot-set-41102d7182b

## Setup & Run Code

```bash
git clone https://github.com/geo1590/GPU-fractals.git
make all
make run
cat output.txt

# View fractal images in out_data/ directory.
```


