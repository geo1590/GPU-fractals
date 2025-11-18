#define STB_IMAGE_WRITE_IMPLEMENTATION

#include "fractal.h"

// A macro to check for errors when calling cudaMalloc(), cudaDeviceSynchronize(), cudaMemcpy().
#define CUDA_CHECK(err) \
    do { \
        cudaError_t err_ = (err); \
        if (err_ != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err_) \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Math for the Mandelbrot Set.
__device__ int escapeTimeMandelbrot(float cx, float cy, int maxIter) {
    float zx = 0.0f, zy = 0.0f;
    int iter = 0;

    while (zx*zx + zy*zy <= 4.0f && iter < maxIter) {
        float zx2 = zx*zx - zy*zy + cx;
        float zy2 = 2.0f * zx * zy + cy;
        zx = zx2;
        zy = zy2;
        iter++;
    }
    return iter;
}

// Math for the Julia Set.
__device__ int escapeTimeJulia(float zx, float zy, float cx, float cy, int maxIter) {
    int iter = 0;

    while (zx*zx + zy*zy <= 4.0f && iter < maxIter) {
        float zx2 = zx*zx - zy*zy + cx;
        float zy2 = 2.0f * zx * zy + cy;
        zx = zx2;
        zy = zy2;
        iter++;
    }
    return iter;
}

// Perform fractal calculations.
__global__ void fractalKernel(
    unsigned char* img,
    int width,
    int height,
    float xMin,
    float xMax,
    float yMin,
    float yMax,
    int maxIter,
    int mode,
    float cRe,
    float cIm
) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= width || py >= height) return;

    float x = xMin + (xMax - xMin) * px / (width - 1);
    float y = yMin + (yMax - yMin) * py / (height - 1);

    int iter = (mode == 0)
        ? escapeTimeMandelbrot(x, y, maxIter)
        : escapeTimeJulia(x, y, cRe, cIm, maxIter);

    float t = (float)iter / maxIter;

    unsigned char r, g, b;
    if (iter >= maxIter) {
        r = g = b = 0;
    } else {
        r = (unsigned char)(9*(1-t)*t*t*t*255);
        g = (unsigned char)(15*(1-t)*(1-t)*t*t*255);
        b = (unsigned char)(8.5*(1-t)*(1-t)*(1-t)*t*255);
    }

    int idx = (py * width + px) * 3;
    img[idx + 0] = r;
    img[idx + 1] = g;
    img[idx + 2] = b;
}

// Read the file that has the "C" parameters for the fractal equation.
int read_file(std::vector<double> &real_num, std::vector<double> &img_num) {
    double a, b;

    // Read the input file to get the "C" parameters.
    std::ifstream infile("data.txt");
    if (!infile.is_open()) {
        std::cerr << "Error: could not open data.txt\n";
        return 1;
    }

    while (infile >> a >> b) {
        real_num.push_back(a);
        img_num.push_back(b);
    }

    infile.close();

    return 0;
}

int main() {
    int width = 1920;
    int height = 1080;
    int maxIter = 500;
    float xMin = -2.5f, xMax = 1.0f;
    float yMin = -1.25f, yMax = 1.25f;
    int mode = 1; // 0=Mandelbrot, 1=Julia
    int quality;
    size_t imgSize = width * height * 3;
    unsigned char* h_img = new unsigned char[imgSize];
    unsigned char* d_img = nullptr;
    std::vector<double> real_num;
    std::vector<double> img_num;

    // Allocate the device memory to store the image file.
    CUDA_CHECK(cudaMalloc(&d_img, imgSize));

    dim3 block(16,16);
    dim3 grid(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y
    );
  
    // Read the fractal "C" parameter values.
    read_file(real_num, img_num);

    // Iterate through all of the "C" parameter values.
    for (size_t i = 0; i < real_num.size(); i++) {
        // Call the kernel.
        fractalKernel<<<grid, block>>>(
            d_img, width, height,
            xMin, xMax, yMin, yMax,
            maxIter, mode, real_num[i], img_num[i]
        );

        // Wait for all kernel processes to complete.
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy memory from device to host.
        CUDA_CHECK(cudaMemcpy(h_img, d_img, imgSize, cudaMemcpyDeviceToHost));

        // Generate the fractal JPG image file.
        quality = 95; // 1–100
        std::ostringstream ss;
        ss << "out_data/fractal_" << std::setw(4) << std::setfill('0') << i << ".jpg";
        std::string name = ss.str();
        stbi_write_jpg(name.c_str(), width, height, 3, h_img, quality);
        std::cout << "Using 'C = " << real_num[i] << " + " << img_num[i] << "' -- created " << name.c_str() << "\n";
    }

    // Free the device and host memory.
    cudaFree(d_img);
    delete[] h_img;

    return 0;
}
