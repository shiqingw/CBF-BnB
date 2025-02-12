import torch
import numpy as np
import time
from datetime import datetime
from dreal import Variable, tanh, Expression, Config, logical_and, logical_not, logical_imply, Min, Max

from ..lip_nn.layers import SandwichFc, SandwichLin
from .utils import format_time

def get_dreal_lyapunov_exp(vars, lyapunov_nn, dtype, device):
    # Check the input
    assert len(vars) == lyapunov_nn.in_features

    # Construct the Lyapunov function
    num_layers = len(lyapunov_nn.layers)
    activations = lyapunov_nn.activations

    # For input transform
    input_bias = lyapunov_nn.input_bias.detach().cpu().numpy()
    input_transform = lyapunov_nn.input_transform.detach().cpu().numpy()
    out = input_transform * (vars - input_bias)

    print("> Start constructing the dReal expression for lyapunov function ...")
    start_time = time.time()
    print("> Start time:", datetime.fromtimestamp(start_time))

    # For the SandwichFc layers
    for i in range(num_layers-1):
        layer = lyapunov_nn.layers[i]
        if not isinstance(layer, SandwichFc):
            raise ValueError("The layer is not SandwichFc!")
        
        weight = layer.weight.detach().cpu().numpy()
        fout, fin = weight.shape # shape: (fout=n_out, fin=n_out+n_in)

        Q = layer.Q.detach().cpu().numpy() # shape: (fout, fin)
        scale = layer.scale
        psi = layer.psi.detach().cpu().numpy()
        bias = layer.bias.detach().cpu().numpy() if layer.bias is not None else None

        B = Q[:, fout:] # shape: (n_out, n_in)
        A = Q[:, :fout] # shape: (n_in, n_out)
        weight1 = (2 ** 0.5) * scale * np.diag(np.exp(-psi)) @ B
        out = np.dot(weight1, out) # shape: (n_out, )
        if bias is not None:
            out += bias
        if activations[i] == "tanh":
            tmp = []
            for j in range(fout):
                tmp.append(tanh(out[j]))
            out = np.array(tmp)
        else:
            raise ValueError("Activation function not implemented!")
        weight2 = (2 ** 0.5) * np.dot(A.T, np.diag(np.exp(psi)))
        out = np.dot(weight2, out)
    
    # For the last SandwichLin layer
    layer = lyapunov_nn.layers[-1]
    if not isinstance(layer, SandwichLin):
        raise ValueError("The layer is not SandwichLin!")
    weight = layer.weight.detach().cpu().numpy()
    fout, fin = weight.shape # shape: (fout=n_out, fin=n_out+n_in)

    Q = layer.Q.detach().cpu().numpy() # shape: (fout, fin)
    scale = layer.scale
    AB = layer.AB
    bias = layer.bias.detach().cpu().numpy() if layer.bias is not None else None

    B = Q[:, fout:] # shape: (n_out, n_in)
    A = Q[:, :fout] # shape: (n_in, n_out)
    out = np.dot(B, scale*out) # shape: (n_out, )
    if AB:
        out = np.dot(2*A.T, out)
    if bias is not None:
        out += bias
    V = out[0] # dReal scalar expression
    
    # Substract the value at zero
    if lyapunov_nn.zero_at_zero:
        env = {var:0.0 for var in vars}
        zero_value = V.Evaluate(env)
        V = V - Expression(zero_value)
        assert abs(V.Evaluate(env)) < 1e-7
    
    stop_time = time.time()
    print("> Stop time:", datetime.fromtimestamp(stop_time))
    print(f"> Time used: {format_time(stop_time-start_time)} = {stop_time-start_time} s")

    # Test the dReal expression
    print("> Checking consistency for lyapunov function ...")
    N = 20
    test_input = torch.rand((N, lyapunov_nn.in_features), dtype=dtype, device=device)
    for i in range(N):
        env = {var:test_input[i, j].item() for j, var in enumerate(vars)}
        t1 = time.time()
        dreal_value = V.Evaluate(env)
        t2 = time.time()
        pytorch_value = lyapunov_nn(test_input[i].unsqueeze(0)).item()
        if abs(dreal_value-pytorch_value) > 1e-5:
            print(f"> Test input {i+1}: {test_input[i]}")
            print(f"> dReal value: {dreal_value} | Time used: {format_time(t2-t1)}")
            print(f"> pytorch value: {pytorch_value}")
            print(f"> Difference: {np.linalg.norm(np.array(dreal_value)-pytorch_value)}")
            raise ValueError("The dReal expression for lyapunov function is not correct!")

    return V

def get_dreal_controller_exp(vars, controller_nn, dtype, device):
    # Check the input
    assert len(vars) == controller_nn.in_features

    # Construct the controller
    in_features = controller_nn.in_features
    out_features = controller_nn.out_features
    num_layers = len(controller_nn.layers)
    activations = controller_nn.activations

    # For input transform
    input_bias = controller_nn.input_bias.detach().cpu().numpy()
    input_transform = controller_nn.input_transform.detach().cpu().numpy()
    out = input_transform * (vars - input_bias)

    print("> Start constructing the dReal expression for controller ...")
    start_time = time.time()
    print("> Start time:", datetime.fromtimestamp(start_time))

    # For the SandwichFc layers
    for i in range(num_layers-1):
        layer = controller_nn.layers[i]
        if not isinstance(layer, SandwichFc):
            raise ValueError("The layer is not SandwichFc!")
        
        weight = layer.weight.detach().cpu().numpy()
        fout, fin = weight.shape # shape: (fout=n_out, fin=n_out+n_in)

        Q = layer.Q.detach().cpu().numpy() # shape: (fout, fin)
        scale = layer.scale
        psi = layer.psi.detach().cpu().numpy()
        bias = layer.bias.detach().cpu().numpy() if layer.bias is not None else None

        B = Q[:, fout:] # shape: (n_out, n_in)
        A = Q[:, :fout] # shape: (n_in, n_out)
        weight1 = (2 ** 0.5) * scale * np.diag(np.exp(-psi)) @ B
        out = np.dot(weight1, out) # shape: (n_out, )
        if bias is not None:
            out += bias
        if activations[i] == "tanh":
            tmp = []
            for j in range(fout):
                tmp.append(tanh(out[j]))
            out = np.array(tmp)
        elif activations[i] == "relu":
            tmp = []
            for j in range(fout):
                tmp.append(Max(out[j], Expression(0)))
            out = np.array(tmp)
        else:
            raise ValueError("Activation function not implemented!")
        weight2 = (2 ** 0.5) * np.dot(A.T, np.diag(np.exp(psi)))
        out = np.dot(weight2, out)
    
    # For the last SandwichLin layer
    layer = controller_nn.layers[-1]
    if not isinstance(layer, SandwichLin):
        raise ValueError("The layer is not SandwichLin!")
    weight = layer.weight.detach().cpu().numpy()
    fout, fin = weight.shape # shape: (fout=n_out, fin=n_out+n_in)

    Q = layer.Q.detach().cpu().numpy() # shape: (fout, fin)
    scale = layer.scale
    AB = layer.AB
    bias = layer.bias.detach().cpu().numpy() if layer.bias is not None else None

    B = Q[:, fout:] # shape: (n_out, n_in)
    A = Q[:, :fout] # shape: (n_in, n_out)
    out = np.dot(B, scale*out) # shape: (n_out, )
    if AB:
        out = np.dot(2*A.T, out)
    if bias is not None:
        out += bias

    if controller_nn.zero_at_zero:
        env = {var:0.0 for var in vars}

        if out_features == 1:
            zero_value = out[0].Evaluate(env)
            out = out[0] - Expression(zero_value)
            assert abs(out.Evaluate(env)) < 1e-7
        else:
            for i in range(out_features):
                zero_value = out[i].Evaluate(env)
                out[i] = out[i] - Expression(zero_value)
                assert abs(out[i].Evaluate(env)) < 1e-7

    if controller_nn.lower_bound is not None:
        lower_bound = controller_nn.lower_bound.detach().cpu().numpy()
        if out_features == 1:
            out = Max(out, Expression(lower_bound))
        else:
            for i in range(out_features):
                out[i] = Max(out[i], Expression(lower_bound[i]))
    
    if controller_nn.upper_bound is not None:
        upper_bound = controller_nn.upper_bound.detach().cpu().numpy()
        if out_features == 1:
            out = Min(out, Expression(upper_bound))
        else:
            for i in range(out_features):
                out[i] = Min(out[i], Expression(upper_bound[i]))
    
    stop_time = time.time()
    print("> Stop time:", datetime.fromtimestamp(stop_time))
    print(f"> Time used: {format_time(stop_time-start_time)} = {stop_time-start_time} s")

    # Test the dReal expression
    print("> Checking consistency for controller ...")
    N = 20
    test_input = 1*(torch.rand((N, controller_nn.in_features), dtype=dtype, device=device) - 0.5)
    for i in range(N):
        env = {var:test_input[i, j].item() for j, var in enumerate(vars)}
        t1 = time.time()
        if out_features == 1:
            dreal_value = out.Evaluate(env)
        else:
            dreal_value = [out[j].Evaluate(env) for j in range(out_features)]
        t2 = time.time()
        pytorch_value = controller_nn(test_input[i].unsqueeze(0)).detach().cpu().numpy().squeeze()
        if np.linalg.norm(np.array(dreal_value)-pytorch_value) > 1e-5:
            print(f"> Test input {i+1}: {test_input[i]}")
            print(f"> dReal value: {dreal_value} | Time used: {format_time(t2-t1)}")
            print(f"> pytorch value: {pytorch_value}")
            print(f"> Difference: {np.linalg.norm(np.array(dreal_value)-pytorch_value)}")
            raise ValueError("The dReal expression for controller is not correct!")
    
    return out
