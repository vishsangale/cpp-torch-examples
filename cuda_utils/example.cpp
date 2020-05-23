#include <iostream>

#include <torch/torch.h>

int main ()
{
    torch::manual_seed(13);

    torch::Device device = torch::kCPU;
    if (torch::cuda::is_available())
    {
        std::cout << "CUDA is available, using  GPU." << std::endl;
        device = torch::kCUDA;
    }
    else
    {
        std::cout << "CUDA not available, using CPU." << std::endl;
    }
    return 0;
}
