import torch
import torch.nn as nn

from utils import get_losses, get_weights


def test_mlp() -> None:
    in_ = 2
    outs = [3, 5, 7, 4, 1]

    model = nn.Sequential()
    model.add_module("layer_0", nn.Linear(2, 3))
    model.add_module("tanh_0", nn.Tanh())
    model.add_module("layer_1", nn.Linear(3, 5))
    model.add_module("relu_0", nn.ReLU())
    model.add_module("layer_2", nn.Linear(5, 7))
    model.add_module("tanh_1", nn.Tanh())
    model.add_module("layer_3", nn.Linear(7, 4))
    model.add_module("relu_1", nn.ReLU())
    model.add_module("layer_4", nn.Linear(4, 1))

    init_weights = get_weights("cases/mlp_init_weights.bin", in_, outs)
    model.load_state_dict(init_weights)

    loss_func = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), 0.0001)

    losses = get_losses("cases/mlp_loss.txt")

    pred = model(torch.tensor([1, 2], dtype=torch.float32))
    exp_loss = loss_func(pred, torch.tensor([0.1], dtype=torch.float32))
    torch.testing.assert_close(losses[0], exp_loss)

    exp_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    act_updated_weights = get_weights("cases/mlp_updated_weights.bin", in_, outs)
    exp_updated_weights = model.state_dict()
    torch.testing.assert_close(act_updated_weights, exp_updated_weights)
    model.load_state_dict(act_updated_weights)

    pred_prime = model(torch.tensor([3, 2], dtype=torch.float32))
    exp_loss_prime = loss_func(pred_prime, torch.tensor([0.2], dtype=torch.float32))
    torch.testing.assert_close(losses[1], exp_loss_prime)

    exp_loss_prime.backward()
    optimizer.step()

    act_final_weights = get_weights("cases/mlp_final_weights.bin", in_, outs)
    exp_final_weights = model.state_dict()
    torch.testing.assert_close(act_final_weights, exp_final_weights)
