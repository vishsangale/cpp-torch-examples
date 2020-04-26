
#include <iostream>
#include <torch/torch.h>


int main()
{
    torch::manual_seed(13);
    torch::Tensor tensor = torch::rand({5, 4});
    std::cout << tensor << std::endl;

    auto acc = tensor.accessor<float, 2>();

    auto data_ptr = tensor.data_ptr<float>();

    std::cout << acc[0][0] << "\t" << *data_ptr << std::endl;

    std::vector<float> vec(data_ptr, data_ptr + 20);

    for (const auto item: vec)
    {
        std::cout << item << "\t";
    }

    std::cout << std::endl;

    auto acc_data = tensor.accessor<float, 2>().data();

    std::vector<float> acc_vector(acc_data, acc_data + 20);

    for (const auto item: acc_vector)
    {
        std::cout << item << "\t";
    }
    std::cout << std::endl;

    std::cout << "Tensor size: " << tensor.sizes() << std::endl;
}

