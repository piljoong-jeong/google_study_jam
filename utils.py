import functools
import itertools
from contextlib import ContextDecorator
from time import perf_counter

import torch

class ContextDecorator(object):
    
    def __init__(self, context_manager):
        self._cm = context_manager

    def __enter__(self):
        return self._cm.__enter__()
    
    def __exit__(self, *args, **kwargs):
        return self._cm.__exit__(*args, **kwargs)
    
    def __call__(self, func):
        # NOTE: wrap `func` to be decorated with this  
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
            
        return wrapper
    

class measure_time(ContextDecorator):
    def __init__(self, str_desc: str = "", precision: int = 6, *, print=True) -> None:
        super().__init__(self)

        assert -1 <= precision
        self.precision = precision

        self.t = None
        if "" != str_desc:
            self.str_desc = f"for codeblock '{str_desc}'"

        self.print = print

    def __enter__(self):
        self.t = perf_counter()

    def __exit__(self, type, value, traceback): # NOTE: `type`, `value`, `traceback` is unused but raises an error when deleted
        
        self.t = perf_counter() - self.t
        self.t_with_precision = f"{self.t:.{self.precision}f}" if -1 != self.precision else f"{self.t}"
        if self.print:
            print(f"[PERF ] execution time {self.str_desc}: {self.t_with_precision}s")

    def get_time_taken(self, is_with_precision: bool = True) -> float:
        """
        ### spellbook.perf.measure_time.get_time_taken()

        Returns time taken within a code block in float type

        Args:
            - `is_with_precision` (bool): If true, returns time with specified precision defined in the construction stage

        Returns:
            - `time_taken` (float): time taken, either plain, or formatted with desired precision
        """

        # fmt: off
        time_taken = {
            True: float(self.t_with_precision),
            False: float(self.t)
        }.get(is_with_precision, KeyError)
        # fmt: on

        return time_taken


def humanbytes(B):
    """Return the given bytes as a human friendly KB, MB, GB, or TB string."""
    B = float(B)
    KB = float(1024)
    MB = float(KB ** 2) # 1,048,576
    GB = float(KB ** 3) # 1,073,741,824
    TB = float(KB ** 4) # 1,099,511,627,776

    if B < KB:
        return '{0} {1}'.format(B,'Bytes' if 0 == B > 1 else 'Byte')
    elif KB <= B < MB:
        return '{0:.2f} KB'.format(B / KB)
    elif MB <= B < GB:
        return '{0:.2f} MB'.format(B / MB)
    elif GB <= B < TB:
        return '{0:.2f} GB'.format(B / GB)
    elif TB <= B:
        return '{0:.2f} TB'.format(B / TB)
    
def get_gpu_memory_usage(device = 0):
    tup_memory_category = ("Free", "Total")
    return tuple(
        f"{category}: {humanbytes(mb)}" for category, mb 
        in zip(tup_memory_category, torch.cuda.mem_get_info()))