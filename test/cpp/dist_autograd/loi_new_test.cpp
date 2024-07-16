#include <torch/torch.h>

int main(){
  auto x = torch::randn(3, torch::requires_grad());

  auto y = x * 2;
  // while (y.norm().item<double>() < 1000) {
  //   y = y * 2;
  // }
  std::cout << y << std::endl;
  std::cout << y.grad_fn()->name() << std::endl;
  auto v = torch::tensor({0.1, 1.0, 0.0001}, torch::kFloat);
  y.backward(v);

  std::cout << x.grad() << std::endl;
}
