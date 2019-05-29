# Taken from http://code.activestate.com/recipes/578353-code-to-source-and-back/

import ast, inspect, re
from types import CodeType as code

import __future__

PyCF_MASK = sum(v for k, v in vars(__future__).items() if k.startswith("CO_FUTURE"))

from . import random_variables

ALL_RVs = [rv for rv in dir(random_variables) if rv[0].isupper()]


class Error(Exception):
    pass


class Unsupported(Error):
    pass


class NoSource(Error):
    pass


def uncompile(c):
    """uncompile(codeobj) -> [source, filename, mode, flags, firstlineno, privateprefix]."""
    if c.co_flags & inspect.CO_NESTED or c.co_freevars:
        raise Unsupported("nested functions not supported")
    if c.co_name == "<lambda>":
        raise Unsupported("lambda functions not supported")
    if c.co_filename == "<string>":
        raise Unsupported("code without source file not supported")

    filename = inspect.getfile(c)
    try:
        lines, firstlineno = inspect.getsourcelines(c)
    except IOError:
        raise NoSource("source code not available")
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


class AutoNameTransformer(ast.NodeTransformer):
    def visit_Assign(self, tree_node):
        try:
            rv_name = tree_node.targets[0].id
            # Test if creation of known RV
            func = tree_node.value.func
            if hasattr(func, "attr"):
                call = func.attr
            else:
                call = func.id

            if call not in ALL_RVs:
                return tree_node

            # Test if name keyword is already set
            if any(kwarg.arg == "name" for kwarg in tree_node.value.keywords):
                return tree_node
            else:
                tree_node.value.keywords.append(ast.keyword("name", ast.Str(rv_name)))
        except AttributeError:
            pass

        return tree_node
