# -*- coding: utf-8 -*-

"""
Created on July 05, 2019

@author: Stefan Lattner

Sony CSL Paris, France

"""

from configobj import ConfigObj, flatten_errors, ConfigObjError
import os
import logging
import validate
from validate import ValidateError
from pprint import pprint

LOGGER = logging.getLogger(__name__)

PATH = os.path.dirname(os.path.abspath(__file__))
config_spec_path = os.path.join(PATH, "config_spec.ini")


def get_config(files, spec = None):
    """
    Loads one or more configuration files and validates it against a
    specification.
    
    Parameters
    ----------
    
    files : string or array-like
        One or more strings defining the path to configuration file(s)
        
    spec : string, optional
        Specification of configuration file. Used to validate the config file
        and to cast its parameters to desired variable types.
        Default: util.config_spec.ini
        
    Returns
    -------
        Dictionary reflecting the structure of the config file
    """
    if spec == None:
        spec = config_spec_path

    if isinstance(files, str):
        files = (files,)

    configspec = ConfigObj(spec, interpolation=True,
                           list_values=False, _inspec=True)

    if os.access(files[0], os.R_OK):
        try:
            c_mrg = ConfigObj(files[0], unrepr=False, configspec=configspec)
        except ConfigObjError as e:
            LOGGER.error("Configuration file validation failed "
                         "(see 'util/config_spec.ini' for reference).")
            pprint(e.__dict__["errors"])
            raise e
    else:
        if len(files) == 1:
            msg = 'cannot open config file: {0}'.format(files[0])
            LOGGER.error(msg)
            raise Exception(msg)

    for fn in files[1:]:
        if os.access(fn, os.R_OK):
            c_mrg.merge(ConfigObj(fn, unrepr=False, configspec=configspec))
        else:
            LOGGER.warning('cannot open config file: {0}'.format(fn))

    valid(c_mrg)
    return c_mrg


def write_config(config, fn):
    """ Write configuration (dict) into .ini file """
    c = ConfigObj()
    c.merge(config)
    f = open(fn, 'w')
    c.write(f)
    f.close()


def config_to_args(config, args):
    for (key, value) in config.items():
        vars(args)[key] = value

    return args


def eval_mixed_list(list_in, *types):
    result = []
    for element, type in zip(list_in, types):
        command = "result.append({0}(element))".format(type)
        exec(command)
    return result


def valid(config):
    vtor = validate.Validator()
    vtor.functions['mixed_list'] = eval_mixed_list
    res = config.validate(vtor, preserve_errors=True)
    any_error = False
    report = ""
    for entry in flatten_errors(config, res):
        any_error = True
        # each entry is a tuple
        section_list, key, error = entry
        if key is not None:
            section_list.append(key)
        else:
            section_list.append('[missing section]')
        section_string = ', '.join(section_list)
        if error == False:
            error = 'missing value or section.'
        report += "{0}: {1} \n".format(section_string, error)

    if any_error:
        raise ValidateError("Configuration file validation failed, "
                            "(see 'util/config_spec.ini' for reference):\n{0}"
                            .format(report))
