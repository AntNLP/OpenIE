from typing import Dict, List, TypeVar
from ...utils.case_sensitive_configurator import CaseSensConfigParser
import argparse
import os
import ast
from configparser import ExtendedInterpolation


BaseObj = TypeVar("BaseObj", int, float, str, list, set, dict)
BASEOBJ = {int, float, str, list, set, dict}


class safe_var_sub(dict):
    """ Safe Variable Substitution """

    def __miss__(self, key):
        raise RuntimeError('Attribute (%s) does not exist.' % (key))


def str_to_baseobj(s: str) -> BaseObj:
    """
    Converts a string to the corresponding base python type value.

    Parameters
    ----------
    s : ``str``
        string like "123", "12.3", "[1, 2, 3]" ...

    Returns
    -------
    res : ``BaseObj``
        "123" -> int(123)
        "12.3" -> float(12.3)
        ...
    """
    try:
        res = eval(s)
        # res = ast.literal_eval(s.format_map(vars()))
    except BaseException:
        return s
    if (s in globals() or s in locals()) and type(res) not in BASEOBJ:
        return s
    else:
        return res


class IniConfigurator:
    """
    Reads and stores the configuration in the ini Format file.

    Parameters
    ----------
    config_file : ``str``
        Path to the configuration file.
    extra_args : ``Dict[str, str]``, optional (default=``dict()``)
        The configuration of the command line input.
    """

    def __init__(self,
                 config_file: str,
                 extra_args: Dict[str, str] = dict()) -> None:

        config = CaseSensConfigParser(interpolation=ExtendedInterpolation())
        config.read(config_file)
        if extra_args:
            extra_args = (
                dict([(k[2:], v)
                      for k, v in zip(extra_args[0::2], extra_args[1::2])]))
        attr_name = set()
        for section in config.sections():
            for k, v in config.items(section):
                if k in extra_args:
                    v = type(v)(extra_args[k])
                    config.set(section, k, v)
                    if k in attr_name:
                        raise RuntimeError(
                            'Attribute (%s) has already appeared.' % (k))
                    else:
                        attr_name.update(k)
                    super(IniConfigurator, self).__setattr__(k, str_to_baseobj(v))

        for section in config.sections():
            for k, v in config.items(section):
                if k not in extra_args:
                    if k in attr_name:
                        raise RuntimeError(
                            'Attribute (%s) has already appeared.' % (k))
                    else:
                        attr_name.update(k)
                    super(IniConfigurator, self).__setattr__(k, str_to_baseobj(v))

        with open(config_file, 'w') as fout:
            config.write(fout)

        print('Loaded config file sucessfully.')
        for section in config.sections():
            for k, v in config.items(section):
                print(k, v)

    def __setattr__(self, name, value):
        raise RuntimeError('Try to set the attribute (%s) of the constant '
                           'class (%s).' % (name, self.__class__.__name__))
