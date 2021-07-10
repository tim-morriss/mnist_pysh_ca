import numpy as np
from pyshgp.push.atoms import CodeBlock
from pyshgp.push.interpreter import PushInterpreter
from pyshgp.push.program import Program


class ErrorFunction:
    """
    The given error function must take a push program in the form of a
    CodeBlock and then return an np.ndarray of numeric errors. These errors
    will be used as the program's error vector.

    The error functions will typically instantiate its own PushInterpreter
    an run the given program as needed.

    Parameters
    ----------
    error_function : Callable
        A function which takes a program to evaluate and returns a
        np.ndarray of errors.
    """

    @staticmethod
    def mnist_error_function(program: Program, interpreter: PushInterpreter, inputs: np.ndarray) -> np.ndarray:
        print(program.pretty_str())
        output = interpreter.run(program, inputs)
        print(output)
        return np.ndarray(output)
