import torch

if __name__ == '__main__':
    # print(torch.rand((3, 2)) * torch.tensor([N_CTPS - 2, 2.]) + torch.tensor([1., -1.]))
    print(torch.tensor([5., 2.]))
    a = torch.rand((3, 2))
    print(a * torch.tensor([5., 2.]))
    print(a)
