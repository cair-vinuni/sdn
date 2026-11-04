"""Command-line helpers shared by the evaluation entry points of all backbones."""

import re


def parse_value(value):
    # parse value from string to list/int/float/bool
    if isinstance(value, str) and ',' in value:
        return [parse_value(v) for v in value.split(',')]
    if re.match(r'^-?\d+$', value):
        return int(value)
    if re.match(r'^-?\d+\.\d+$', value):
        return float(value)
    if value == 'True' or value == 'False':
        return value == 'True'
    if value == 'None':
        return None
    return value


def parse_opts(opts):
    if len(opts) == 0:
        return dict()
    # [a, b, c, d] -> {a: b, c: d}
    if len(opts) % 2 != 0:
        raise ValueError(f"options must be key value pairs, got {opts}")
    return {opts[i]: parse_value(opts[i+1]) for i in range(0, len(opts), 2)}
