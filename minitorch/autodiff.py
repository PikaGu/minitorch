from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol

# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    f_x = f(*vals)
    vals = list(vals)
    vals[arg] += epsilon
    vals = tuple(vals)
    f_delta_x = f(*vals)
    return (f_delta_x - f_x) / epsilon

variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    sort_res = []
    vis = {}
    done = {}
    
    def dfs_toposort(v: Variable):
        if v.unique_id in vis:
            return
        vis[v.unique_id] = True
        for p in v.parents:
            if not p.unique_id in done:
                dfs_toposort(p)
        sort_res.append(v)
        done[v.unique_id] = True
        vis.pop(v.unique_id)
    dfs_toposort(variable)
    sort_res.reverse()
    return sort_res


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    d_chain = topological_sort(variable)
    d_map = {variable.unique_id: deriv}
    
    for node in d_chain:
        d_out = d_map.get(node.unique_id)
        print(node.unique_id, d_out)
        if node.is_leaf():
            node.accumulate_derivative(d_out)
        else:
            new_grads = node.chain_rule(d_out)
            for (input, diff) in new_grads:
                print(input.unique_id, diff, end=".\n")
                pre = d_map.get(input.unique_id, 0.0)
                d_map.update({input.unique_id: pre+diff})
    print(d_map)


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
