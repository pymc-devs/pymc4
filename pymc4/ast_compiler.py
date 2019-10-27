# Taken from http://code.activestate.com/recipes/578353-code-to-source-and-back/

import __future__
import ast
import inspect
import re
from types import CodeType as code

from . import distributions

PyCF_MASK = sum(v for k, v in vars(__future__).items() if k.startswith("CO_FUTURE"))
ALL_RVs = [rv for rv in dir(distributions) if rv[0].isupper()]


class SourceCodeNotFoundError(Exception):
    pass


def uncompile(c):
    """uncompile(codeobj) -> [source, filename, mode, flags, firstlineno, privateprefix]."""
    if c.co_flags & inspect.CO_NESTED or c.co_freevars:
        raise NotImplementedError("nested functions not supported")
    if c.co_name == "<lambda>":
        raise NotImplementedError("lambda functions not supported")
    if c.co_filename == "<string>":
        raise NotImplementedError("code without source file not supported")

    filename = inspect.getfile(c)
    try:
        lines, firstlineno = inspect.getsourcelines(c)
    except IOError:
        raise SourceCodeNotFoundError("source code not available")
    source = "".join(lines)

    # __X is mangled to _ClassName__X in methods. Find this prefix:
    privateprefix = None
    for name in c.co_names:
        m = re.match("^(_[A-Za-z][A-Za-z0-9_]*)__.*$", name)
        if m:
            privateprefix = m.group(1)
            break

    return [source, filename, "exec", c.co_flags & PyCF_MASK, firstlineno, privateprefix]


def recompile(source, filename, mode, flags=0, firstlineno=1, privateprefix=None):
    """Recompile output of uncompile back to a code object. source may also be preparsed AST."""
    if isinstance(source, ast.AST):
        a = source
    else:
        a = parse_snippet(source, filename, mode, flags, firstlineno)
    node = a.body[0]
    if not isinstance(node, ast.FunctionDef):
        raise Error("Expecting function AST node")

    c0 = compile(a, filename, mode, flags, True)

    # This code object defines the function. Find the function's actual body code:
    for c in c0.co_consts:
        if not isinstance(c, code):
            continue
        if c.co_name == node.name and c.co_firstlineno == node.lineno:
            break
    else:
        raise Error("Function body code not found")

    # Re-mangle private names:
    if privateprefix is not None:

        def fixnames(names):
            isprivate = re.compile("^__.*(?<!__)$").match
            return tuple(privateprefix + name if isprivate(name) else name for name in names)

        c = code(
            c.co_argcount,  # pylint: disable=undefined-loop-variable
            c.co_nlocals,  # pylint: disable=undefined-loop-variable
            c.co_stacksize,  # pylint: disable=undefined-loop-variable
            c.co_flags,  # pylint: disable=undefined-loop-variable
            c.co_code,  # pylint: disable=undefined-loop-variable
            c.co_consts,  # pylint: disable=undefined-loop-variable
            fixnames(c.co_names),  # pylint: disable=undefined-loop-variable
            fixnames(c.co_varnames),  # pylint: disable=undefined-loop-variable
            c.co_filename,  # pylint: disable=undefined-loop-variable
            c.co_name,  # pylint: disable=undefined-loop-variable
            c.co_firstlineno,  # pylint: disable=undefined-loop-variable
            c.co_lnotab,  # pylint: disable=undefined-loop-variable
            c.co_freevars,  # pylint: disable=undefined-loop-variable
            c.co_cellvars,  # pylint: disable=undefined-loop-variable
        )
    return c


def parse_snippet(source, filename, mode, flags, firstlineno, privateprefix_ignored=None):
    """Like ast.parse, but accepts indented code snippet with a line number offset."""
    args = filename, mode, flags | ast.PyCF_ONLY_AST, True
    prefix = "\n"
    try:
        a = compile(prefix + source, *args)
    except IndentationError:
        # Already indented? Wrap with dummy compound statement
        prefix = "with 0:\n"
        a = compile(prefix + source, *args)
        # peel wrapper
        a.body = a.body[0].body
    ast.increment_lineno(a, firstlineno - 2)
    return a


class AutoNameVisitor(ast.NodeVisitor):
    def __init__(self):
        self.random_variable_names = []

    def visit_Assign(self, node):
        names = node.targets  # LHS of assignment expression
        assigned = node.value  # RHS of assignment expression

        # If the assigned value is not a yield expression, skip it.
        # I.e. do nothing to assignments like `N = 10` or `mu = x.mean()`
        if not any([isinstance(assigned, expr) for expr in [ast.Yield, ast.YieldFrom]]):
            return

        yielded = assigned.value  # Yielded expression

        # We expect the yielded expression to be a function call. If it is not,
        # raise an exception.
        if not isinstance(yielded, ast.Call):
            msg = "Unable to auto-name: a yielded expression is not a function call."
            raise RuntimeError(msg)

        # We expect there to be only one target. If there are more, raise an
        # exception.
        if len(names) > 1:
            msg = "Unable to auto-name: expected one target."
            raise RuntimeError(msg)

        name = names[0].id
        self.random_variable_names.append(name)

        # Recursively visit child nodes.
        self.generic_visit(node)


def parse_random_variable_names(model):
    """
    Parameters
    ----------
    model : function
        Model function.

    Returns
    -------
    random_variable_names : list
        List of random variable names.
    """

    visitor = AutoNameVisitor()
    uncompiled = uncompile(model.__code__)
    tree = parse_snippet(*uncompiled)
    visitor.visit(tree)
    return visitor.random_variable_names
