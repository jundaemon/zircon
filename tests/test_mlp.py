import torch
import torch.nn as nn

from utils import get_losses, get_weights


def test_mlp_1() -> None:
    model = nn.Sequential()
    model.add_module("layer_0", nn.Linear(3, 10))
    model.add_module("tanh_0", nn.Tanh())
    model.add_module("layer_1", nn.Linear(10, 15))
    model.add_module("tanh_1", nn.Tanh())
    model.add_module("layer_2", nn.Linear(15, 3))

    init_weights = get_weights("cases/mlp_1_init.bin", 3, [10, 15, 3])
    model.load_state_dict(init_weights)

    loss_func = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), 0.0001)

    pred = model(torch.tensor([1, 2, 3], dtype=torch.float32))
    actual_loss = get_losses("cases/mlp_1_loss.txt")[0]
    exp_loss = loss_func(pred, torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32))
    torch.testing.assert_close(actual_loss, exp_loss, rtol=0.01, atol=0.01)

    exp_loss.backward()
    optimizer.step()

    actual_final_weights = get_weights("cases/mlp_1_final.bin", 3, [10, 15, 3])
    exp_final_weights = model.state_dict()
    torch.testing.assert_close(
        actual_final_weights, exp_final_weights, rtol=0.01, atol=0.01
    )


def test_mlp_2() -> None:
    model = nn.Sequential()
    model.add_module("layer_0", nn.Linear(4, 7))
    model.add_module("tanh_0", nn.Tanh())
    model.add_module("layer_1", nn.Linear(7, 10))
    model.add_module("tanh_1", nn.Tanh())
    model.add_module("layer_2", nn.Linear(10, 11))
    model.add_module("tanh_2", nn.Tanh())
    model.add_module("layer_3", nn.Linear(11, 5))

    init_weights = get_weights("cases/mlp_2_init.bin", 4, [7, 10, 11, 5])
    model.load_state_dict(init_weights)

    loss_func = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), 0.0001)

    losses = get_losses("cases/mlp_2_loss.txt")
    pred_1 = model(torch.tensor([1, 2, 3, 4], dtype=torch.float32))
    exp_loss_1 = loss_func(
        pred_1, torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float32)
    )
    torch.testing.assert_close(losses[0], exp_loss_1, rtol=0.01, atol=0.01)

    exp_loss_1.backward()
    optimizer.step()

    actual_updated_weights = get_weights("cases/mlp_2_updated.bin", 4, [7, 10, 11, 5])
    exp_updated_weights = model.state_dict()
    torch.testing.assert_close(
        actual_updated_weights, exp_updated_weights, rtol=0.01, atol=0.01
    )
    optimizer.zero_grad()

    pred_2 = model(torch.tensor([1, 2, 3, 4], dtype=torch.float32))
    exp_loss_2 = loss_func(
        pred_2, torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float32)
    )
    torch.testing.assert_close(losses[1], exp_loss_2, rtol=0.01, atol=0.01)

    exp_loss_2.backward()
    optimizer.step()

    actual_final_weights = get_weights("cases/mlp_2_final.bin", 4, [7, 10, 11, 5])
    exp_final_weights = model.state_dict()
    torch.testing.assert_close(
        actual_final_weights, exp_final_weights, rtol=0.01, atol=0.01
    )
