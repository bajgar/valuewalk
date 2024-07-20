import pydantic
import logging
import functools
from typing import Callable, Any, Dict, Union


def resolve_configurable_factories(config: pydantic.BaseModel) -> None:
    """
    Resolves configurable factories in the given config object.

    Args:
        config (pydantic.BaseModel): The config object to resolve configurable factories in.
    """

    def traverse(a_dict: Dict[str, Any]) -> None:
        """
        Traverses the given dictionary and resolves configurable factories.

        Args:
            a_dict (Dict[str, Any]): The dictionary to traverse.
        """
        for key, value in a_dict.items():
            if callable(value):
                if is_configurable_factory(value):
                    factory = value
                    a_dict[key] = factory(config)
                else:
                    logging.debug(f"Skipping resolution of {key}={value} "
                                  f"as it is not marked as a @configurable_factory.")
            if isinstance(value, pydantic.BaseModel):
                traverse(value.__dict__)
            if isinstance(value, dict):
                traverse(value)

    traverse(config.__dict__)


def configurable_factory(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator to mark a function as a configurable factory.

    Args:
        func (Callable[..., Any]): The function to mark as a configurable factory.

    Returns:
        Callable[..., Any]: The decorated function.
    """
    func.configurable_factory = True
    return func


def is_configurable_factory(func: Union[Callable[..., Any], functools.partial]) -> bool:
    """
    Checks if the given function is a configurable factory.

    Args:
        func (Union[Callable[..., Any], functools.partial]): The function to check.

    Returns:
        bool: True if the function is a configurable factory, False otherwise.
    """
    if isinstance(func, functools.partial):
        return is_configurable_factory(func.func)
    return getattr(func, 'configurable_factory', False)
