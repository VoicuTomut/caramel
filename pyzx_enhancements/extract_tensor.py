import numpy as np
import math
import pyzx.tensor as pyzx_tensor


def input_to_tensor() -> np.ndarray:
    return np.identity(2)  # np.identity(2) #np.array([1, 0])


def output_to_tensor() -> np.ndarray:
    return np.identity(2)  # np.array([1, 0])


def get_tensor_from_g(pyzx_graph, v):
    phase = math.pi * pyzx_graph.phases()[v]
    v_type = pyzx_graph.types()[v]
    arity = len(pyzx_graph.neighbors(v))

    # input
    # output
    if v in pyzx_graph.inputs():
        return input_to_tensor()
    if v in pyzx_graph.outputs():
        return output_to_tensor()  # np.identity(2)

    if v_type == 1:
        t = pyzx_tensor.Z_to_tensor(arity, phase)
    elif v_type == 2:
        t = pyzx_tensor.X_to_tensor(arity, phase)
    elif v_type == 3:
        t = pyzx_tensor.H_to_tensor(arity, phase)
    else:
        raise ValueError(
            "Vertex %s has non-ZXH type but is not an input or output" % str(v)
        )

    return t
