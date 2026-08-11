import torch
import torch.nn as nn

from utils import get_losses, get_weights


def test_mlp_1() -> None:
    in_channels = 2
    out_channels = [3, 5, 7, 4, 1]

    model = nn.Sequential()
    model.add_module("layer_0", nn.Linear(2, 3))
    model.add_module("tanh_0", nn.Tanh())
    model.add_module("layer_1", nn.Linear(3, 5))
    model.add_module("tanh_1", nn.Tanh())
    model.add_module("layer_2", nn.Linear(5, 7))
    model.add_module("tanh_2", nn.Tanh())
    model.add_module("layer_3", nn.Linear(7, 4))
    model.add_module("tanh_3", nn.Tanh())
    model.add_module("layer_4", nn.Linear(4, 1))

    initial_weights = get_weights(
        "cases/mlp_1_initial_weights", in_channels, out_channels
    )
    model.load_state_dict(initial_weights)

    loss_func = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), 0.0001)

    losses = get_losses("cases/mlp_1_losses")

    pred = model(torch.tensor([1, 2], dtype=torch.float32))
    expected_loss = loss_func(pred, torch.tensor([0.1], dtype=torch.float32))
    torch.testing.assert_close(losses[0], expected_loss)

    expected_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    actual_updated_weights = get_weights(
        "cases/mlp_1_updated_weights", in_channels, out_channels
    )
    expected_updated_weights = model.state_dict()
    torch.testing.assert_close(actual_updated_weights, expected_updated_weights)

    pred_prime = model(torch.tensor([3, 2], dtype=torch.float32))
    expected_loss_prime = loss_func(
        pred_prime, torch.tensor([0.2], dtype=torch.float32)
    )
    torch.testing.assert_close(losses[1], expected_loss_prime)

    expected_loss_prime.backward()
    optimizer.step()

    actual_final_weights = get_weights(
        "cases/mlp_1_final_weights", in_channels, out_channels
    )
    expected_final_weights = model.state_dict()
    torch.testing.assert_close(actual_final_weights, expected_final_weights)


def test_mlp_2() -> None:
    in_channels = 1
    out_channels = [2, 4, 6, 5, 3]

    model = nn.Sequential()
    model.add_module("layer_0", nn.Linear(1, 2))
    model.add_module("relu_0", nn.ReLU())
    model.add_module("layer_1", nn.Linear(2, 4))
    model.add_module("relu_1", nn.ReLU())
    model.add_module("layer_2", nn.Linear(4, 6))
    model.add_module("relu_2", nn.ReLU())
    model.add_module("layer_3", nn.Linear(6, 5))
    model.add_module("relu_3", nn.ReLU())
    model.add_module("layer_4", nn.Linear(5, 3))

    initial_weights = get_weights(
        "cases/mlp_2_initial_weights", in_channels, out_channels
    )
    model.load_state_dict(initial_weights)

    loss_func = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), 0.0001, 0.9)

    losses = get_losses("cases/mlp_2_losses")

    pred = model(torch.tensor([2], dtype=torch.float32))
    expected_loss = loss_func(pred, torch.tensor([0.5, 0.2, 0.3], dtype=torch.float32))
    torch.testing.assert_close(losses[0], expected_loss)

    expected_loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    actual_updated_weights = get_weights(
        "cases/mlp_2_updated_weights", in_channels, out_channels
    )
    expected_updated_weights = model.state_dict()
    torch.testing.assert_close(actual_updated_weights, expected_updated_weights)

    pred_prime = model(torch.tensor([3], dtype=torch.float32))
    expected_loss_prime = loss_func(
        pred_prime, torch.tensor([0.7, 0.1, 0.2], dtype=torch.float32)
    )
    torch.testing.assert_close(losses[1], expected_loss_prime)

    expected_loss_prime.backward()
    optimizer.step()

    actual_final_weights = get_weights(
        "cases/mlp_2_final_weights", in_channels, out_channels
    )
    expected_final_weights = model.state_dict()
    torch.testing.assert_close(actual_final_weights, expected_final_weights)
