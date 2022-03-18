# -*- coding: utf-8 -*-
"""
This module enables runtime type collection.
Collected information can be used to automatically generate
mypy annotation for the executed code paths.

It uses python profiler callback to examine frames and record
type info about arguments and return type.

For the module consumer, the workflow looks like that:
1) call init_types_collection() from the main thread once
2) call start() to start the type collection
3) call stop() to stop the type collection
4) call dump_stats(file_name) to dump all collected info to the file as json

You can repeat start() / stop() as many times as you want.

The module is based on Tony's 2016 prototype D219371.
"""

from __future__ import (
    absolute_import,
    division,
    print_function,
)

import collections
import inspect
import json
import opcode
import dis
import os
import sys, traceback
import threading
from inspect import ArgInfo, trace
from threading import Thread
from types import MethodType
import gc

from mypy_extensions import TypedDict
from six import iteritems
from six.moves import range
from six.moves.queue import Queue  # type: ignore  # No library stub yet
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Set,
    Sized,
    Tuple,
    TypeVar,
    Union,
)
from contextlib import contextmanager
from .var_analysis import VarAnalysis
from .extract_info import ExtractConstant, ExtractRaise
import ast
import copy

if sys.version_info[0] == 3 :
    import asyncio

if sys.version_info[0] < 3:
    try :
        sys.setdefaultencoding('utf-8')
    except :
        pass

MYPY=False
if MYPY:
    # MYPY is True when mypy is running
    # 'Type' is only required for running mypy, not for running pyannotate
    from typing import Type


# pylint: disable=invalid-name

CO_GENERATOR = inspect.CO_GENERATOR  # type: ignore


def _my_hash(arg_list):
    # type: (List[Any]) -> int
    """Simple helper hash function"""
    res = 0
    for arg in arg_list:
        res = res * 31 + hash(arg)
    return res


# JSON object representing the collected data for a single function/method
FunctionData = TypedDict('FunctionData', {'path': str,
                                          'line': int,
                                          'func_name': str,
                                          'type_comments': List[str],
                                          'samples': int})


class UnknownType(object):
    pass


class NoReturnType(object):
    pass


class TypeWasIncomparable(object):
    pass


class FakeIterator(Iterable[Any], Sized):
    """
    Container for iterator values.

    Note that FakeIterator([a, b, c]) is akin to list([a, b, c]); this
    is turned into IteratorType by resolve_type().
    """

    def __init__(self, values):
        # type: (List[Any]) -> None
        self.values = values

    def __iter__(self):
        # type: () -> Iterator[Any]
        for v in self.values:
            yield v

    def __len__(self):
        # type: () -> int
        return len(self.values)


_NONE_TYPE = type(None)  # type: Type[None]
InternalType = Union['DictType', 'ListType', 'TupleType', 'SetType', 'IteratorType', 'type', 'NdarrayType', "TestType"]


class DictType(object):
    """
    Internal representation of Dict type.
    """

    def __init__(self, key_type, val_type):
        # type: (TentativeType, TentativeType) -> None
        self.key_type = key_type
        self.val_type = val_type

    def __repr__(self):
        # type: () -> str
        if repr(self.key_type) == 'None':
            # We didn't see any values, so we don't know what's inside
            return 'Dict'
        else:
            return 'Dict[%s => %s]' % (repr(self.key_type), repr(self.val_type))

    def __hash__(self):
        # type: () -> int
        return hash(self.key_type) if self.key_type else 0

    def __eq__(self, other):
        # type: (object) -> bool
        if not isinstance(other, DictType):
            return False

        return self.val_type == other.val_type and self.key_type == other.key_type

    def __ne__(self, other):
        # type: (object) -> bool
        return not self.__eq__(other)


class ListType(object):
    """
    Internal representation of List type.
    """

    def __init__(self, val_type):
        # type: (TentativeType) -> None
        self.val_type = val_type

    def __repr__(self):
        # type: () -> str
        if repr(self.val_type) == 'None':
            # We didn't see any values, so we don't know what's inside
            return 'List'
        else:
            return 'List[%s]' % (repr(self.val_type))

    def __hash__(self):
        # type: () -> int
        return hash(self.val_type) if self.val_type else 0

    def __eq__(self, other):
        # type: (object) -> bool
        if not isinstance(other, ListType):
            return False

        return self.val_type == other.val_type

    def __ne__(self, other):
        # type: (object) -> bool
        return not self.__eq__(other)


class SetType(object):
    """
    Internal representation of Set type.
    """

    def __init__(self, val_type):
        # type: (TentativeType) -> None
        self.val_type = val_type

    def __repr__(self):
        # type: () -> str
        if repr(self.val_type) == 'None':
            # We didn't see any values, so we don't know what's inside
            return 'Set'
        else:
            return 'Set[%s]' % (repr(self.val_type))

    def __hash__(self):
        # type: () -> int
        return hash(self.val_type) if self.val_type else 0

    def __eq__(self, other):
        # type: (object) -> bool
        if not isinstance(other, SetType):
            return False

        return self.val_type == other.val_type

    def __ne__(self, other):
        # type: (object) -> bool
        return not self.__eq__(other)


class IteratorType(object):
    """
    Internal representation of Iterator type.
    """

    def __init__(self, val_type):
        # type: (TentativeType) -> None
        self.val_type = val_type

    def __repr__(self):
        # type: () -> str
        if repr(self.val_type) == 'None':
            # We didn't see any values, so we don't know what's inside
            return 'Iterator'
        else:
            return 'Iterator[%s]' % (repr(self.val_type))

    def __hash__(self):
        # type: () -> int
        return hash(self.val_type) if self.val_type else 0

    def __eq__(self, other):
        # type: (object) -> bool
        if not isinstance(other, IteratorType):
            return False

        return self.val_type == other.val_type

    def __ne__(self, other):
        # type: (object) -> bool
        return not self.__eq__(other)


class TupleType(object):
    """
    Internal representation of Tuple type.
    """

    def __init__(self, val_types):
        #  type: (List[InternalType]) -> None
        self.val_types = val_types

    def __repr__(self):
        # type: () -> str
        return 'Tuple[%s]' % '/'.join([name_from_type(vt) for vt in self.val_types])

    def __hash__(self):
        # type: () -> int
        return _my_hash(self.val_types)

    def __eq__(self, other):
        # type: (object) -> bool
        if not isinstance(other, TupleType):
            return False

        if len(self.val_types) != len(other.val_types):
            return False

        for i, v in enumerate(self.val_types):
            if v != other.val_types[i]:
                return False

        return True

    def __ne__(self, other):
        # type: (object) -> bool
        return not self.__eq__(other)


class NdarrayType(object):
    """
    Internal representation of List type.
    """

    def __init__(self, dtype):
        # type: (str) -> None
        self.dtype = dtype

    def __repr__(self):
        # type: () -> str
        return 'numpy.ndarray<<%s>>' % (repr(self.dtype)[1:-1])

    def __hash__(self):
        # type: () -> int
        return hash(self.dtype) if self.dtype else 0

    def __eq__(self, other):
        # type: (object) -> bool
        if not isinstance(other, NdarrayType):
            return False

        return self.dtype == other.dtype

    def __ne__(self, other):
        # type: (object) -> bool
        return not self.__eq__(other)

class TestType(object):
    """
    Internal representation of List type.
    """

    def __init__(self, dtype, parent):
        # type: (str) -> None
        self.dtype = dtype
        self.parent = parent

    def __repr__(self):
        # type: () -> str
        return '%s::%s' % (name_from_type(self.dtype), name_from_type(self.parent))

    def __hash__(self):
        # type: () -> int
        return hash(self.dtype) if self.dtype else 0

    def __eq__(self, other):
        # type: (object) -> bool
        if not isinstance(other, TestType):
            return False

        return (self.dtype == other.dtype and self.parent == other.parent)

    def __ne__(self, other):
        # type: (object) -> bool
        return not self.__eq__(other)


class TentativeType(object):
    """
    This class serves as internal representation of type for a type collection process.
    It can be merged with another instance of TentativeType to build up a broader sample.
    """

    def __init__(self):
        # type: () -> None
        self.types_hashable = set()  # type: Set[InternalType]
        self.types = []  # type: List[InternalType]

    def __hash__(self):
        # type: () -> int

        # These objects not immutable because there was a _large_ perf impact to being immutable.
        # Having a hash on a mutable object is dangerous, but is was much faster.
        # If you do change it, you need to
        # (a) pull it out of the set/table
        # (b) change it,
        # (c) stuff it back in
        return _my_hash([self.types, len(self.types_hashable)]) if self.types else 0

    def __eq__(self, other):
        # type: (object) -> bool
        if not isinstance(other, TentativeType):
            return False

        if self.types_hashable != other.types_hashable:
            return False

        if len(self.types) != len(other.types):
            return False

        for i in self.types:
            if i not in other.types:
                return False

        return True

    def __ne__(self, other):
        # type: (object) -> bool
        return not self.__eq__(other)

    def add(self, type):
        # type: (InternalType) -> None
        """
        Add type to the runtime type samples.
        """
        try:
            if isinstance(type, SetType):
                if EMPTY_SET_TYPE in self.types_hashable:
                    self.types_hashable.remove(EMPTY_SET_TYPE)
            elif isinstance(type, ListType):
                if EMPTY_LIST_TYPE in self.types_hashable:
                    self.types_hashable.remove(EMPTY_LIST_TYPE)
            elif isinstance(type, IteratorType):
                if EMPTY_ITERATOR_TYPE in self.types_hashable:
                    self.types_hashable.remove(EMPTY_ITERATOR_TYPE)
            elif isinstance(type, DictType):
                if EMPTY_DICT_TYPE in self.types_hashable:
                    self.types_hashable.remove(EMPTY_DICT_TYPE)
                for item in self.types_hashable:
                    if isinstance(item, DictType):
                        if item.key_type == type.key_type:
                            item.val_type.merge(type.val_type)
                            return
            self.types_hashable.add(type)

        except (TypeError, AttributeError):
            try:
                if type not in self.types:
                    self.types.append(type)
            except AttributeError:
                if TypeWasIncomparable not in self.types:
                    self.types.append(TypeWasIncomparable)

    def merge(self, other):
        # type: (TentativeType) -> None
        """
        Merge two TentativeType instances
        """
        for hashables in other.types_hashable:
            self.add(hashables)
        for non_hashbles in other.types:
            self.add(non_hashbles)

    def __repr__(self):
        # type: () -> str
        if (len(self.types) + len(self.types_hashable) == 0) or (
                len(self.types_hashable) == 1 and _NONE_TYPE in self.types_hashable):
            return 'None'
        else:
            type_format = '%s'
            filtered_types = self.types + [i for i in self.types_hashable if i != _NONE_TYPE]
            if _NONE_TYPE in self.types_hashable:
                type_format = 'Optional[%s]'
            if len(filtered_types) == 1:
                return type_format % name_from_type(filtered_types[0])
            else:
                # use sorted() for predictable type order in the Union
                return type_format % (
                    'Union[' + '/'.join(sorted([name_from_type(s) for s in filtered_types])) + ']')


FunctionKey = NamedTuple('FunctionKey', [('path', str), ('line', int), ('func_name', str)])

# Inferred types for a function call
ResolvedTypes = NamedTuple('ResolvedTypes',
                           [('pos_args', List[InternalType]),
                            ('varargs', Optional[List[InternalType]])])

# Task queue entry for calling a function with specific argument types
KeyAndTypes = NamedTuple('KeyAndTypes', [('key', FunctionKey), ('types', ResolvedTypes)])

# Task queue entry for returning from a function with a value
KeyAndReturn = NamedTuple('KeyAndReturn', [('key', FunctionKey), ('return_type', InternalType)])

# Combined argument and return types for a single function call
Signature = NamedTuple('Signature', [('args', 'ArgTypes'), ('return_type', InternalType)])


BUILTIN_MODULES = {'__builtin__', 'builtins', 'exceptions'}


def name_from_type(type_):
    # type: (InternalType) -> str
    """
    Helper function to get PEP-484 compatible string representation of our internal types.
    """
    if isinstance(type_, (DictType, ListType, TupleType, SetType, IteratorType, NdarrayType, TestType)):
        return repr(type_)
    else:
        if type_.__name__ != 'NoneType':
            module = type_.__module__
            if module in BUILTIN_MODULES or module == '<unknown>':
                # Omit module prefix for known built-ins, for convenience. This
                # makes unit tests for this module simpler.
                # Also ignore '<unknown>' modules so pyannotate can parse these types
                return type_.__name__
            else:
                name = getattr(type_, '__qualname__', None) or type_.__name__
                delim = '.' if '.' not in name else ':'
                return '%s%s%s' % (module, delim, name)
        else:
            return 'None'


EMPTY_DICT_TYPE = DictType(TentativeType(), TentativeType())
EMPTY_LIST_TYPE = ListType(TentativeType())
EMPTY_SET_TYPE = SetType(TentativeType())
EMPTY_ITERATOR_TYPE = IteratorType(TentativeType())


# TODO: Make this faster
def get_function_name_from_frame(frame):
    # type: (Any) -> str
    """
    Heuristic to find the class-specified name by @guido

    For instance methods we return "ClassName.method_name"
    For functions we return "function_name"
    """

    def bases_to_mro(cls, bases):
        # type: (type, List[type]) -> List[type]
        """
        Convert __bases__ to __mro__
        """
        mro = [cls]
        for base in bases:
            if base not in mro:
                mro.append(base)
            sub_bases = getattr(base, '__bases__', None)
            if sub_bases:
                sub_bases = [sb for sb in sub_bases if sb not in mro and sb not in bases]
                if sub_bases:
                    mro.extend(bases_to_mro(base, sub_bases))
        return mro

    code = frame.f_code
    # This ought to be aggressively cached with the code object as key.
    funcname = code.co_name
    if code.co_varnames:
        varname = code.co_varnames[0]
        if varname == 'self':
            inst = frame.f_locals.get(varname)
            if inst is not None:
                try:
                    mro = inst.__class__.__mro__
                except AttributeError:
                    mro = None
                    try:
                        bases = inst.__class__.__bases__
                    except AttributeError:
                        bases = None
                    else:
                        mro = bases_to_mro(inst.__class__, bases)
                except ImportError :
                    mro = None
                
                if mro:
                    for cls in mro:
                        bare_method = cls.__dict__.get(funcname)
                        if bare_method and getattr(bare_method, '__code__', None) is code:
                            return '%s.%s' % (cls.__name__, funcname)
    return funcname


def resolve_type(arg):
    # type: (object) -> InternalType
    """
    Resolve object to one of our internal collection types or generic built-in type.

    Args:
        arg: object to resolve
    """
    arg_type = type(arg)
    if arg_type == list:
        assert isinstance(arg, list)  # this line helps mypy figure out types
        sample = arg[:min(4, len(arg))]
        tentative_type = TentativeType()
        for sample_item in sample:
            tentative_type.add(resolve_type(sample_item))
        return ListType(tentative_type)
    elif arg_type == set:
        assert isinstance(arg, set)  # this line helps mypy figure out types
        sample = []
        iterator = iter(arg)
        for i in range(0, min(4, len(arg))):
            sample.append(next(iterator))
        tentative_type = TentativeType()
        for sample_item in sample:
            tentative_type.add(resolve_type(sample_item))
        return SetType(tentative_type)
    elif arg_type == FakeIterator:
        assert isinstance(arg, FakeIterator)  # this line helps mypy figure out types
        sample = []
        iterator = iter(arg)
        for i in range(0, min(4, len(arg))):
            sample.append(next(iterator))
        tentative_type = TentativeType()
        for sample_item in sample:
            tentative_type.add(resolve_type(sample_item))
        return IteratorType(tentative_type)
    elif arg_type == tuple:
        assert isinstance(arg, tuple)  # this line helps mypy figure out types
        sample = list(arg[:min(10, len(arg))])
        return TupleType([resolve_type(sample_item) for sample_item in sample])
    elif arg_type == dict:
        assert isinstance(arg, dict)  # this line helps mypy figure out types
        key_tt = TentativeType()
        val_tt = TentativeType()
        for i, (k, v) in enumerate(iteritems(arg)):
            if i > 4:
                break
            key_tt.add(resolve_type(k))
            val_tt.add(resolve_type(v))
        return DictType(key_tt, val_tt)
    elif str(arg_type) == "<class 'numpy.ndarray'>" :
        return NdarrayType(arg.dtype.type.__name__)

    elif "test" in str(arg_type) :
        # test에서 클래스를 만드는 경우
        # 상속을 뭘로 받았는지 알필요가 있음
        # test에서 만든 클래스가 아닌 클래스 중 가장 가까운 클래스 리턴
        classes = arg_type.__mro__

        for c in classes :
            if "test" in str(c) :
                continue
            
            return TestType(arg_type, c)

    else:
        return type(arg)


def prep_args(arg_info, select_args=None):
    # type: (ArgInfo) -> ResolvedTypes
    """
    Resolve types from ArgInfo
    """

    # pull out any varargs declarations
    if select_args is None :
        filtered_args = [a for a in arg_info.args if getattr(arg_info, 'varargs', None) != a] 
    else : # select_arg는 얘네 정보만 이끌어내기
        filtered_args = select_args

    # we don't care about self/cls first params (perhaps we can test if it's an instance/class method another way?)
    if filtered_args and (filtered_args[0] in ('self', 'cls')):
        filtered_args = filtered_args[1:]

    pos_args = []  # type: List[InternalType]
    if filtered_args:
        for arg in filtered_args:
            if isinstance(arg, str) and arg in arg_info.locals:
                # here we know that return type will be of type "type"
                resolved_type = resolve_type(arg_info.locals[arg])
                pos_args.append(resolved_type)
            else:
                pos_args.append(type(UnknownType()))

    varargs = None  # type: Optional[List[InternalType]]
    if arg_info.varargs:
        varargs_tuple = arg_info.locals[arg_info.varargs]
        # It's unclear what all the possible values for 'varargs_tuple' are,
        # so perform a defensive type check since we don't want to crash here.
        if isinstance(varargs_tuple, tuple):
            varargs = [resolve_type(arg) for arg in varargs_tuple[:4]]

    return ResolvedTypes(pos_args=pos_args, varargs=varargs)

def prep_select_args(arg_info, skip_keys=None, subs_list=None): # 변수 : 타입으로 내보내기
    # type: (ArgInfo) -> ResolvedTypes
    """
    Resolve types from ArgInfo
    """

    filtered_args = arg_info.locals

    pos_args = dict()  # type: List[InternalType]
    if filtered_args:
        for arg in filtered_args :
            if isinstance(arg, str):
                if skip_keys is not None and arg in skip_keys : # skip_key는 생략
                    continue

                if subs_list is not None :
                    subs = subs_list.get(arg, None)
                    if subs is not None :
                        try :
                            subs_obj = filtered_args.get(subs, subs)

                            if isinstance(subs_obj, str) :
                                if subs_obj[0] == '\'' and subs_obj[-1] == '\'' :
                                    subs_obj = subs_obj[1:-1] 

                            target_sub = filtered_args[arg][subs_obj]
                            resolved_type = resolve_type(target_sub)

                            try :
                                if resolved_type.__name__ == 'method' :
                                    func_resolved_type = resolve_type(filtered_args[arg]())
                                    pos_args[arg+'()'] = func_resolved_type
                            except :
                                pass

                            if subs == subs_obj :
                                subs_name = '\'' + subs + '\''
                            else :
                                subs_name = subs
                            subs_name = arg+'['+subs_name+']'

                            pos_args[subs_name] = resolved_type
                        except Exception as e:
                            pass
                # here we know that return type will be of type "type"
                resolved_type = resolve_type(filtered_args[arg])

                try :
                    if resolved_type.__name__ == 'method' :
                        func_resolved_type = resolve_type(filtered_args[arg]())
                        pos_args[arg+'()'] = func_resolved_type

                        #print(pos_args)
                except :
                    pass

                pos_args[arg] = resolved_type
            else:
                pos_args[arg] = type(UnknownType())

    varargs = None  # type: Optional[List[InternalType]]
    if arg_info.varargs:
        varargs_tuple = arg_info.locals[arg_info.varargs]
        # It's unclear what all the possible values for 'varargs_tuple' are,
        # so perform a defensive type check since we don't want to crash here.
        if isinstance(varargs_tuple, tuple):
            varargs = [resolve_type(arg) for arg in varargs_tuple[:4]]

    if varargs is not None :
        pass
        #print(varargs)
        #print("!!! varargs is not None !!!")

    return (pos_args, varargs)


class ArgTypes(object):
    """
    Internal representation of argument types in a single call
    """

    def __init__(self, resolved_types):
        # type: (ResolvedTypes) -> None
        self.pos_args = [TentativeType() for _ in range(len(resolved_types.pos_args))]
        if resolved_types.pos_args:
            for i, arg in enumerate(resolved_types.pos_args):
                self.pos_args[i].add(arg)

        self.varargs = None  # type: Optional[TentativeType]
        if resolved_types.varargs:
            self.varargs = TentativeType()
            for arg in resolved_types.varargs:
                self.varargs.add(arg)

    def __repr__(self):
        # type: () -> str
        return str({'pos_args': self.pos_args, 'varargs': self.varargs})

    def __hash__(self):
        # type: () -> int
        return _my_hash(self.pos_args) + hash(self.varargs)

    def __eq__(self, other):
        # type: (object) -> bool
        return (isinstance(other, ArgTypes)
                and other.pos_args == self.pos_args and other.varargs == self.varargs)

    def __ne__(self, other):
        # type: (object) -> bool
        return not self.__eq__(other)


# Collect at most this many type comments for each function.
MAX_ITEMS_PER_FUNCTION = 20

# The most recent argument types collected for each function. Once we encounter
# a corresponding return event, an item will be flushed and moved to
# 'collected_comments'.
collected_args = {}  # type: Dict[FunctionKey, ArgTypes]

# Collected unique type comments for each function, of form '(arg, ...) -> ret'.
# There at most MAX_ITEMS_PER_FUNCTION items.
collected_signatures = {}  # type: Dict[FunctionKey, Set[Tuple[ArgTypes, InternalType]]]

# Number of samples collected per function (we also count ones ignored after reaching
# the maximum comment count per function).
num_samples = {}  # type: Dict[FunctionKey, int]
type_samples = {} 

def _make_type_comment(args_info, return_type):
    #print("args_info : ", args_info)
    #print("return_type : ", return_type)
    # type: (ArgTypes, InternalType) -> str
    """Generate a type comment of form '(arg, ...) -> ret'."""
    if not args_info.pos_args:
        args_string = ''
    else:
        args_string = ', '.join([repr(t) for t in args_info.pos_args])
    if args_info.varargs:
        varargs = '*%s' % repr(args_info.varargs)
        if args_string:
            args_string += ', %s' % varargs
        else:
            args_string = varargs
    return_name = name_from_type(return_type)
    return '(%s) -> %s' % (args_string, return_name)


def _flush_signature(key, return_type):
    # type: (FunctionKey, InternalType) -> None
    """Store signature for a function.

    Assume that argument types have been stored previously to
    'collected_args'. As the 'return_type' argument provides the return
    type, we now have a complete signature.

    As a side effect, removes the argument types for the function from
    'collected_args'.
    """
    #print("key : ", key)
    signatures = collected_signatures.setdefault(key, set())
    args_info = collected_args.pop(key)
    #print(args_info)
    #if len(signatures) < MAX_ITEMS_PER_FUNCTION:
    signatures.add((args_info, return_type))
    num_samples[key] = num_samples.get(key, 0) + 1
    type_samples[(key, args_info)] = type_samples.get((key, args_info), 0) + 1


def type_consumer():
    # type: () -> None
    """
    Infinite loop of the type consumer thread.
    It gets types to process from the task query.
    """

    # we are not interested in profiling type_consumer itself
    # but we start it before any other thread
    while True:
        try :
            item = _task_queue.get()
            if isinstance(item, KeyAndTypes):
                if item.key in collected_args: # item.key -> function_key
                    # Previous call didn't get a corresponding return, perhaps because we
                    # stopped collecting types in the middle of a call or because of
                    # a recursive function.
                    _flush_signature(item.key, UnknownType)
                #print("item : ", item)
                #print("item.types : ", item.types)
                collected_args[item.key] = ArgTypes(item.types)
            else:
                assert isinstance(item, KeyAndReturn)
                if item.key in collected_args:
                    _flush_signature(item.key, item.return_type)
            _task_queue.task_done()
        except Exception as e:
            print(e)

_task_queue = Queue()  # type: Queue[Union[KeyAndTypes, KeyAndReturn]]
_consumer_thread = Thread(target=type_consumer)
_consumer_thread.daemon = True
_consumer_thread.start()


running = False

CURRENT_FILENAME = None
TOP_DIR = os.path.join(os.getcwd(), '')     # current dir with trailing slash
TOP_DIR_DOT = os.path.join(TOP_DIR, '.')
TOP_DIR_LEN = len(TOP_DIR)


def _make_sampling_sequence(n):
    # type: (int) -> List[int]
    """
    Return a list containing the proposed call event sampling sequence.

    Return events are paired with call events and not counted separately.

    This is 0, 1, 2, ..., 4 plus 50, 100, 150, 200, etc.

    The total list size is n.
    """
    seq = list(range(50))
    i = 55
    while len(seq) < n:
        seq.append(i)
        if i < 100 :
            i += 5
        if i < 300 :
            i += 10
        elif i < 1000 :
            i += 20
        elif i < 10000 :
            i += 50
        else :
            i += 100
    return seq


# We pre-compute the sampling sequence since 'x in <set>' is faster.
MAX_SAMPLES_PER_FUNC = 50000
sampling_sequence = frozenset(_make_sampling_sequence(MAX_SAMPLES_PER_FUNC))
LAST_SAMPLE = max(sampling_sequence)

# Array of counters indexed by ID of code object.
sampling_counters = {}  # type: Dict[int, Optional[int]]
# IDs of code objects for which the previous event was a call (awaiting return).
call_pending = set()  # type: Set[int]

_pos_args = dict()
_fail_args = None
ERROR_INFO = dict()
ERROR_INFO_LIST = list()
ERROR_LIST = list()
ERROR_MSG = list()
CLEAR_FRAME = False

ADDITIONAL_INFO = dict()

TEST_OPTION = None

def traceback_manager(reverse=False) :
    global ERROR_INFO, ERROR_INFO_LIST, ERROR_LIST

    info_list = list()
    constant_dict = dict()
    raise_info = tuple()

    global TEST_OPTION
    if TEST_OPTION :
        for test_option in TEST_OPTION :
            if not "::" in test_option :
                continue

            test_split = test_option.split("::")

            test_filename = test_split[0]
            test_funcname = test_split[-1]

            if test_funcname in TEST_FUNC :
                #frame_summary = traceback.extract_tb(last_tb)[0]
                #lineno = frame_summary.lineno

                if sys.version_info[0] < 3:
                    file_code = open(test_filename, 'r').read()
                else :
                    file_code = open(test_filename, 'r', -1, "utf-8").read()
                file_ast = ast.parse(file_code)

                extract_constant = ExtractConstant(test_funcname)
                constant_dict = extract_constant.get_constant(file_ast)

                extract_raise = ExtractRaise(test_funcname)
                raise_info = extract_raise.get_raise_info(file_ast)
            

    #print("SPARTA", ERROR_INFO_LIST)

    ERROR_INFO_LIST.append(ERROR_INFO)

    empty_count = 0
    
    for info_idx, error_info in enumerate(ERROR_INFO_LIST) :
        if not error_info :
            empty_count += 1
            continue
        # Error가 안났으면 skip
        if not error_info.get('tb') :
            return

        last_tb = error_info['tb']
        #traceback.print_tb(last_tb)

        '''
        if last_tb.tb_frame.f_code.co_name == 'assertIsInstance' :
            # Type Assert 실패한 것
            cur_frame = last_tb.tb_frame
            cur_code = cur_frame.f_code
            cur_line = last_tb.tb_lineno

            var_analyzer = VarAnalysis()

            file_code = open(cur_code.co_filename, 'r').read()
            file_ast = ast.parse(file_code)
            usage_list = var_analyzer.get_var_info(file_ast, cur_line)

            arg_info = inspect.getargvalues(cur_frame)

            fail_types = prep_select_args(arg_info)
            neg = fail_types[0]['obj']
            pos = arg_info.locals['cls']

            error_info = dict()
            error_info['info'] = "AssertionError"
            error_info['neg'] = name_from_type(neg)
            error_info['pos'] = name_from_type(pos)

            ERROR_LIST.append(error_info)

            return
        '''



        # cpython, unittest 파일은 제외하기
        if last_tb.tb_next : # TypeError 하나면, Test 쪽에서 난 거임
            while True :
                test_filename = last_tb.tb_frame.f_code.co_filename
                test_funcname = last_tb.tb_frame.f_code.co_name

                if test_filename.find("/tests/") != -1 or test_filename.find("/test/") != -1 :
                    if last_tb.tb_next :
                        last_tb = last_tb.tb_next
                    else :
                        return
                else :
                    break

        #error_info['error'] = "No Call"
        error_info['funcname'] = None

        ERROR_MSG.append({'msg' : str(error_info['msg'])})

        prev_tb = list()

        tb_list = [last_tb]

        while last_tb.tb_next :
            last_tb = last_tb.tb_next
            tb_list.insert(0, last_tb)

        if reverse :
            tb_list.reverse()


        candidate_tb = []
        new_extern_module = False

        first_info = dict()

        for tb in tb_list :
            filename = _filter_filename(tb.tb_frame.f_code.co_filename)

            if not filename and not new_extern_module :
                #first_info['error'] = "Call"
                new_extern_module = True
            elif filename :
                if new_extern_module is True :
                    prev_tb = list()
                new_extern_module = False

                prev_tb.append(tb)
                last_tb = tb

        '''
        while last_tb.tb_next :
            # 대상 함수이면...
            filename = _filter_filename(last_tb.tb_next.tb_frame.f_code.co_filename)
            if not filename :
                ERROR_INFO['error'] = "Call"
                ERROR_INFO['funcname'] = last_tb.tb_next.tb_frame.f_code.co_name
                break

            prev_tb = last_tb
            last_tb = last_tb.tb_next
        '''

        '''
        # user가 exception 낸거면 그 이전 함수 이용
        error_line = traceback.format_tb(last_tb)[0].splitlines()[1]

        if error_line.strip().startswith("raise TypeError") :
            #ERROR_INFO['error'] = "Call"
            #ERROR_INFO['funcname'] = last_tb.tb_frame.f_code.co_name
            prev_tb = prev_tb[:-1]
        '''

        prev_tb.reverse()

        prev_arguments = []

        
        for i, cur_tb in enumerate(prev_tb) :
            neg_info = dict()

            if i == 0 : # 제일 처음 불러진거
                #neg_info['error'] = first_info.get('error', 'No Call')
                neg_info['funcname'] = cur_tb.tb_frame.f_code.co_name
            else :
                #neg_info['error'] = "Call"
                neg_info['funcname'] = cur_tb.tb_frame.f_code.co_name
            #    ERROR_INFO['funcname'] = last_tb.tb_frame.f_code.co_name

            cur_frame = cur_tb.tb_frame
            cur_code = cur_frame.f_code
            cur_line = cur_tb.tb_lineno

            neg_info['filename'] = cur_code.co_filename
            test_filename = cur_code.co_filename
            neg_info['classname'] = get_function_name_from_frame(cur_frame)
            neg_info['line'] = cur_line

            
            var_analyzer = VarAnalysis()

            if sys.version_info[0] < 3:
                file_code = open(test_filename, 'r').read()
            else :
                file_code = open(test_filename, 'r', -1, "utf-8").read()
            file_ast = ast.parse(file_code)
            usage_list, subs_list = var_analyzer.get_var_info(file_ast, cur_line)

            #print("Usage_list : ", usage_list)
            #print("Subs list : ", subs_list)

            arg_info = inspect.getargvalues(cur_frame)

            #arg_info = copy.deepcopy(arg_info)

            skip_keys = []
            for arg in arg_info.locals.keys() :
                if usage_list :
                    if arg not in usage_list :
                        skip_keys.append(arg)

            # prev argument => 즉, 다음 호출된 함수의 인자들, 이게 존재하면 이거만 확인해보자 일단
            #prev_arguments = []
            #prev_arguments.extend(arg_info.args)
            #prev_arguments.extend([k for k in arg_info.locals.get('kwargs', dict()).keys()])

            #attribute_info = copy.deepcopy(cur_code.co_names) # attribute들
            #attribute_info = list(filter(lambda attribute : attribute in usage_list, attribute_info))

            # 현재 usage_list에 모든 arg, attribute 호출중
            # 이거에 맞게 코드 수정 필요


            tmp_info = dict()

            for (arg_name, arg_obj) in arg_info.locals.items() :
                if arg_name in skip_keys : # skip이면 건너뛰기
                    continue

                def getmembers(object, predicate=None): # inspect꺼임 
                    import types
                    """Return all members of an object as (name, value) pairs sorted by name.
                    Optionally, only return members that satisfy a given predicate."""
                    if inspect.isclass(object):
                        mro = (object,) + inspect.getmro(object)
                    else:
                        mro = ()
                    results = []
                    processed = set()
                    try :
                        names = dir(object)
                    except Exception:
                        return [] # pandas interval_index에서 문제가 생겨서 일단 이렇게 처리
                    # :dd any DynamicClassAttributes to the list of names if object is a class;
                    # this may result in duplicate entries if, for example, a virtual
                    # attribute with the same name as a DynamicClassAttribute exists
                    try:
                        for base in object.__bases__:
                            for k, v in base.__dict__.items():
                                if isinstance(v, types.DynamicClassAttribute):
                                    names.append(k)
                    except AttributeError:
                        pass
                    except Exception :
                        pass
                    for key in names:
                        # First try to get the value via getattr.  Some descriptors don't
                        # like calling their __get__ (see bug #1785), so fall back to
                        # looking in the __dict__.
                        try:
                            try :
                                value = getattr(object, key)
                            except Exception as e :
                                continue
                            # handle the duplicate key
                            if key in processed:
                                raise AttributeError
                        except AttributeError:
                            for base in mro:
                                if key in base.__dict__:
                                    value = base.__dict__[key]
                                    break
                            else:
                                # could be a (currently) missing slot member, or a buggy
                                # __dir__; discard and move on
                                continue
                        if not predicate or predicate(value):
                            results.append((key, value))
                        processed.add(key)
                    results.sort(key=lambda pair: pair[0])
                    return results

                def get_all_member(arg_name, arg_obj, usage_obj, result) :
                    attributes = getmembers(arg_obj)

                    for (attr_name, attr_obj) in attributes :
                        is_unique = True
                        for usage in usage_obj :
                            if attr_obj is usage :
                                is_unique = True
                                break
                        if not is_unique :
                            continue
                        if attr_name == "__init__" or attr_name == "__getattribute__":
                            continue
                        if arg_name + '.' + attr_name in usage_list :
                            name = arg_name + '.' + attr_name
                            usage_obj.append(arg_obj)
                            result = get_all_member(name, attr_obj, usage_obj, result)

                            result[name] = attr_obj

                    return result
                result = get_all_member(arg_name, arg_obj, [], dict([]))
                tmp_info.update(result)


            arg_info.locals.update(tmp_info)
            fail_types = prep_select_args(arg_info, skip_keys, subs_list) 

            '''
            arg_info = inspect.getargvalues(cur_frame)
            remove_keys = []
            for arg in arg_info.locals.keys() :
                if arg not in used_arg :
                    remove_keys.append(arg)

            for key in remove_keys :
                arg_info.locals.pop(key)

            attribute_info = cur_frame.f_code.co_names # attribute들
            attribute_info = list(filter(lambda attribute : attribute in used_arg, attribute_info))

            tmp_info = dict()
            
            for (arg_name, arg_obj) in arg_info.locals.items() :
                attributes = inspect.getmembers(arg_obj)

                for (attr_name, attr_obj) in attributes :
                    if attr_name in attribute_info :
                        name = arg_name + '.' + attr_name
                        tmp_info[name] = attr_obj

            arg_info.locals.update(tmp_info)
            fail_types = prep_select_args(arg_info) 
            '''

            pos_args = fail_types[0]
            fail_args = dict()
            

            for key, typ in pos_args.items() :
                name_typ = name_from_type(typ)
                if isinstance(name_typ, bytes) :
                    name_typ = name_typ.decode('utf-8')
                fail_args[key] = [name_typ]

            info = dict()
            info['idx'] = [info_idx-empty_count]
            info['info'] = neg_info
            info['args'] = fail_args

            #print("Before : ", info)
            #global ERROR_LIST
            info_list = add_info(info, info_list)


    global ADDITIONAL_INFO
    ADDITIONAL_INFO['raise'] = raise_info
    ADDITIONAL_INFO['constant'] = constant_dict

    ERROR_LIST = info_list

def add_info(target, info_list) :
    is_equal_info = False 
    for info in info_list :
        target_info = target['info']
        info_info = info['info']

        if target_info == info_info :
            is_equal_info = True
            target_args = target['args']
            info_args = info['args']
            info_idx = info['idx']
            info_idx.extend(target['idx'])

            for target_arg, typ in target_args.items() :
                if target_arg in info_args :
                    info_args[target_arg].extend(typ)
                    info_args[target_arg] = list(set(info_args[target_arg])) # 중복 제거
                else :
                    info_args[target_arg] = typ
            break
    
    if not is_equal_info :
        info_list.append(target)

    return info_list


@contextmanager
def collect():
    # type: () -> Iterator[None]
    start()
    try:
        yield
    finally:
        global CLEAR_FRAME
        global ONLY_FUNC
        if not CLEAR_FRAME and not ONLY_FUNC:
            traceback_manager(reverse=True)

        stop()


def pause():
    # type: () -> None
    """
    Deprecated, replaced by stop().
    """
    # In the future, do: warnings.warn("Function pause() has been replaced by start().", PendingDeprecationWarning)
    return stop()

def stop():
    # type: () -> None
    """
    Start collecting type information.
    """
    global running  # pylint: disable=global-statement
    running = False
    _task_queue.join()


def resume():
    # type: () -> None
    """
    Deprecated, replaced by start().
    """
    # In the future, do: warnings.warn("Function resume() has been replaced by stop().", PendingDeprecationWarning)
    return start()

def start():
    # type: () -> None
    """
    Stop collecting type information.
    """
    global running  # pylint: disable=global-statement
    running = True

    global _pos_args
    _pos_args = dict()
    global ERROR_LIST
    ERROR_LIST = list()
    global ERROR_MSG
    ERROR_MSG = list()
    sampling_counters.clear()


def default_filter_filename(filename):
    # type: (Optional[str]) -> Optional[str]
    """Default filter for filenames.

    Returns either a normalized filename or None.
    You can pass your own filter to init_types_collection().
    """

    #if filename.find("/tests/") != -1 or filename.find("/test/") != -1:
    #    return None
    if filename is None or not filename.startswith('/'):
        return None
    elif filename == CURRENT_FILENAME :
        return None
    elif filename.startswith(TOP_DIR):
        if filename.startswith(TOP_DIR_DOT):
            # Skip subdirectories starting with dot (e.g. .vagrant).
            return None
        else:
            # Strip current directory and following slashes.
            return filename[TOP_DIR_LEN:].lstrip(os.sep)
    elif filename.startswith(str(os.sep)):
        # Skip absolute paths not under current directory.
        return None
    else:
        return filename


_filter_filename = default_filter_filename  # type: Callable[[Optional[str]], Optional[str]]


if sys.version_info[0] == 2:
    RETURN_VALUE_OPCODE = chr(opcode.opmap['RETURN_VALUE'])
    YIELD_VALUE_OPCODE = chr(opcode.opmap['YIELD_VALUE'])
else:
    RETURN_VALUE_OPCODE = opcode.opmap['RETURN_VALUE']
    YIELD_VALUE_OPCODE = opcode.opmap['YIELD_VALUE']


ONLY_FUNC = False
TEST_FUNC = None
TEST_FUNC_ASYNC = None
TEST_FUNC_ENTRY = False
TEST_SETUP_ENTRY = False
NEGATIVE_INFO = None

def async_end(fut) :
    print("Async End")
    global TEST_FUNC_ASYNC
    TEST_FUNC_ASYNC = False


POS_LOCALIZER = dict()
NEG_LOCALIZER = list()

raise_except = False

def pos_args_info(frame, co, neg_args) :
    import copy
    arg_info = inspect.getargvalues(frame)
    #arg_info = copy.deepcopy(arg_info)
    skip_keys = []
    for arg in arg_info.locals.keys() :
        if arg not in neg_args :
            skip_keys.append(arg)

    subs_dict = dict()
    for key in neg_args :
        if '[' in key and ']' == key[-1] :
            split_key = key.split('[')

            value = split_key[0]
            sli = split_key[1][:-1]
            subs_dict[value] = sli


    tmp_info = dict()

    for (arg_name, arg_obj) in arg_info.locals.items() :
        if arg_name in skip_keys : # skip이면 건너뛰기
            continue

        def getmembers(object, predicate=None): # inspect꺼임 
            import types
            """Return all members of an object as (name, value) pairs sorted by name.
            Optionally, only return members that satisfy a given predicate."""
            if inspect.isclass(object):
                mro = (object,) + inspect.getmro(object)
            else:
                mro = ()
            results = []
            processed = set()
            
            try :
                names = dir(object)
            except :
                return []
            # :dd any DynamicClassAttributes to the list of names if object is a class;
            # this may result in duplicate entries if, for example, a virtual
            # attribute with the same name as a DynamicClassAttribute exists
            try:
                for base in object.__bases__:
                    for k, v in base.__dict__.items():
                        if isinstance(v, types.DynamicClassAttribute):
                            names.append(k)
            except AttributeError:
                pass
            for key in names:
                # First try to get the value via getattr.  Some descriptors don't
                # like calling their __get__ (see bug #1785), so fall back to
                # looking in the __dict__.
                try:
                    try :
                        import warnings
                        warnings.filterwarnings(action='ignore') 
                        value = getattr(object, key)
                        warnings.filterwarnings(action='default') 
                    except Exception as e :
                        continue
                    # handle the duplicate key
                    if key in processed:
                        raise AttributeError
                except AttributeError:
                    for base in mro:
                        if key in base.__dict__:
                            value = base.__dict__[key]
                            break
                    else:
                        # could be a (currently) missing slot member, or a buggy
                        # __dir__; discard and move on
                        continue
                if not predicate or predicate(value):
                    results.append((key, value))
                processed.add(key)
            results.sort(key=lambda pair: pair[0])
            return results

        def get_all_member(arg_name, arg_obj, usage_obj, result) :
            attributes = getmembers(arg_obj)

            for (attr_name, attr_obj) in attributes :
                is_unique = True
                for usage in usage_obj :
                    if attr_obj is usage :
                        is_unique = True
                        break
                if not is_unique :
                    continue
                if attr_name == "__init__" or attr_name == "__getattribute__":
                    continue
                if arg_name + '.' + attr_name in neg_args :
                    name = arg_name + '.' + attr_name
                    usage_obj.append(arg_obj)
                    result = get_all_member(name, attr_obj, usage_obj, result)

                    result[name] = attr_obj

            return result
        result = get_all_member(arg_name, arg_obj, [], dict([]))
        tmp_info.update(result)

    arg_info.locals.update(tmp_info)

    arg_types = prep_select_args(arg_info, skip_keys, subs_dict) 

    args = arg_types[0]
    pos_args = dict()

    for key, typ in args.items() :
        pos_args[key] = name_from_type(typ)

    return pos_args


def add_pos_args(neg_filename, neg_line, pos_info) :
    global _pos_args

    if neg_filename in _pos_args : # 이미 파일 정보가 있다면
        if neg_line in _pos_args[neg_filename] : # 이미 라인 정보가 있다면
            isin = False
            for pos_element in _pos_args[neg_filename][neg_line] :
                if pos_element['info'] == pos_info :
                    pos_element['samples'] += 1
                    isin = True
                    break

            if not isin :
                element = dict([])
                element['info'] = pos_info
                element['samples'] = 1
                _pos_args[neg_filename][neg_line].append(element)

        else : # 라인 정보가 없다면
            element = dict([])
            element['info'] = pos_info
            element['samples'] = 1
            _pos_args[neg_filename][neg_line] = [element]

    else :
        element = dict([])
        element['info'] = pos_info
        element['samples'] = 1
        _pos_args[neg_filename] = dict()
        _pos_args[neg_filename][neg_line] = [element]

TEST_SET = set([])

def _trace_local(frame, event, arg) :
    # local def, class 보기
    # 거의 여기만 쓸거임
    # 에러가난 마지막 줄만 localize 하면 됨
    
    #print(frame.f_lineno)
    
    global TEST_FUNC, TEST_FUNC_ENTRY, TEST_SETUP_ENTRY

    assert_isinstance = 'assertIsInstance'
 
    if TEST_FUNC is not None :
        if event == "exception" and (TEST_SETUP_ENTRY == True or TEST_FUNC_ENTRY == True or frame.f_code.co_name in TEST_FUNC):
            global ERROR_INFO
            
            if (arg[2] is not None and arg[0] is TypeError) or frame.f_code.co_name == assert_isinstance:
                #traceback.print_tb(arg[2])
                #print(frame.f_code.co_filename)
                ERROR_INFO['tb'] = arg[2]
                ERROR_INFO['msg'] = arg[1]
                #print("Update", arg[2])
                #traceback.print_tb(arg[2])

    if event != "line" :
        return


    co = frame.f_code
    func_name = co.co_name
    line_no = frame.f_lineno
    filename = co.co_filename

    global NEGATIVE_INFO
    if not NEGATIVE_INFO is None :
        neg_info_list = NEGATIVE_INFO
        #for neg_info_list in NEGATIVE_INFO.values() :
        if not isinstance(neg_info_list, list) :
            neg_info_list = [neg_info_list]
        for neg_info in neg_info_list :
            if neg_info['info'] == 'AssertionError' :
                break
            neg_filename = neg_info['info']['filename']
            neg_line = neg_info['info']['line']

            if filename == neg_filename and line_no == neg_line :
                neg_args = [k for k in neg_info['args'].keys()]
                result = pos_args_info(frame, co, neg_args)
                #print("WOWOWOW")

                add_pos_args(neg_filename, neg_line, result)
                #print(_pos_args)
    
    global ONLY_FUNC
    if not ONLY_FUNC : # all이 아닐 때
        key = '%s %s %s' % (filename, func_name, line_no)
        if TEST_FUNC is None : # pos localize
            global POS_LOCALIZER
            POS_LOCALIZER[key] = POS_LOCALIZER.get(key, 0) + 1

        elif TEST_FUNC_ENTRY or TEST_SETUP_ENTRY : # pos조건, neg조건, all조건
            global NEG_LOCALIZER
            NEG_LOCALIZER.append(key)
    return


def _trace_error(frame, event, arg) :
    if not running:
        return

    if event != 'call' :
        return
    
    if frame.f_code.co_name == "clear_frames" :
        global CLEAR_FRAME
        global ONLY_FUNC
        CLEAR_FRAME = True

        if not ONLY_FUNC:
            traceback_manager()

            #input()

    #global _filter_filename
    
    assert_isinstance = 'assertIsInstance'

    filename = _filter_filename(frame.f_code.co_filename)
    #print(frame.f_code.co_filename)
    if frame.f_code.co_name == assert_isinstance or filename :
        #if filename in frame.f_code.co_filename :
        return _trace_local

    return

key_global = None

def _trace_dispatch(frame, event, arg):
    # type: (Any, str, Optional[Any]) -> None
    """
    This is the main hook passed to setprofile().
    It implement python profiler interface.

    Arguments are described in https://docs.python.org/2/library/sys.html#sys.settrace
    """


    # Bail if we're not tracing.
    if not running:
        return

    global TEST_FUNC
    global TEST_FUNC_ENTRY, TEST_SETUP_ENTRY
    global TEST_FUNC_ASYNC
    global ERROR_INFO_LIST, ERROR_INFO

    if sys.version_info[0] == 3 :
        if isinstance(arg, asyncio.Task) :
            if TEST_FUNC and arg._coro.__name__ in TEST_FUNC :
                arg.add_done_callback(async_end)
                TEST_FUNC_ASYNC = True

        if hasattr(arg, '__class__') and hasattr(arg.__class__, '__name__'):
            
            if TEST_FUNC and arg.__class__.__name__ == 'Deferred' :
                TEST_FUNC_ASYNC = True
                arg.addErrback(async_end)        

    if TEST_FUNC is not None :
        if TEST_FUNC_ASYNC is False :
            #print(ERROR_INFO)
            ERROR_INFO_LIST.append(ERROR_INFO)
            ERROR_INFO = dict()
            TEST_FUNC_ENTRY = False
            TEST_SETUP_ENTRY = False
            TEST_FUNC_ASYNC = None

        if (event == 'call' and frame.f_code.co_name in TEST_FUNC) : # test 진입 완료
            #print(TEST_FUNC, "Entry!")
            TEST_FUNC_ENTRY = True

        if (event == 'return' and frame.f_code.co_name in TEST_FUNC and TEST_FUNC_ENTRY) : # test 끝
            #print("Exit!")
            
            if not (TEST_FUNC_ASYNC is True) :
                ERROR_INFO_LIST.append(ERROR_INFO)
                ERROR_INFO = dict()
            #TEST_FUNC = None
                TEST_FUNC_ENTRY = False

        if (event == 'call' and frame.f_code.co_name == 'setUp') :
            #print(TEST_FUNC, "Entry!")
            TEST_SETUP_ENTRY = True

        if (event == 'return' and frame.f_code.co_name == 'setUp') :
            #print("Exit!")
            TEST_SETUP_ENTRY = False

            if not (TEST_FUNC_ASYNC is True) :
                ERROR_INFO_LIST.append(ERROR_INFO)
                ERROR_INFO = dict()
            #TEST_FUNC = None
                TEST_SETUP_ENTRY = False

    #func_name = get_function_name_from_frame(frame)
    #if func_name == 'convert_to_index_sliceable' :
    #        print(func_name)

    # Get counter for this code object.  Bail if we don't care about this function.
    # An explicit None is stored in the table when we no longer care.
    code = frame.f_code
    key = (id(code), code.co_name) # 왜인지 모르겠지만 다른 함수인데 code id가 같은 경우가 존재
    
    #print(frame.f_lineno)

    #arg_info = inspect.getargvalues(frame)
    #print(arg_info)

    n = sampling_counters.get(key, 0)
    if n is None:
        return
    
    

    if event == 'call':
        # Bump counter and bail depending on sampling sequence.
        sampling_counters[key] = n + 1
        # Each function gets traced at most MAX_SAMPLES_PER_FUNC times per run.
        # NOTE: There's a race condition if two threads call the same function.
        # I don't think we should care, so what if it gets probed an extra time.

        # 이거 얘가 임의로 sample 수를 줄이는 거였음
        if n not in sampling_sequence:
            if n > LAST_SAMPLE:
            #    print("?")
                sampling_counters[key] = None  # We're no longer interested in this function.
            call_pending.discard(key)  # Avoid getting events out of sync
            return
        # Mark that we are looking for a return from this code object.
        call_pending.add(key)
    elif event == 'return':
        if key not in call_pending:
            # No pending call event -- ignore this event. We only collect
            # return events when we know the corresponding call event.
            return
        call_pending.discard(key)  # Avoid race conditions
    else:
        # Ignore other events, such as c_call and c_return.
        return

    # 내가 고쳐야 할 것 #
    '''
    TODO : 일반 변수들도 탐색이 되어야함
    '''


    # Track calls under current directory only.
    #filename = code.co_filename # 외부함수까지 포함
    filename = _filter_filename(code.co_filename) # 외부함수 비포함
    #func_name = get_function_name_from_frame(frame)
    #print(event, filename, func_name)
    
    if filename :
        func_name = get_function_name_from_frame(frame)

        arg_info = inspect.getargvalues(frame)
        #if func_name == 'read_csv' or func_name == '_refine_defaults_read' :
        #    print(arg_info.locals)
        
        if not func_name or func_name[0] == '<':
            # Could be a lambda or a comprehension; we're not interested.
            sampling_counters[key] = None
        else:
            function_key = FunctionKey(filename, code.co_firstlineno, func_name)
            if event == 'call':
                # TODO(guido): Make this faster
                arg_info = inspect.getargvalues(frame)  # type: ArgInfo
                
                resolved_types = prep_args(arg_info)
                #print(resolved_types)
                _task_queue.put(KeyAndTypes(function_key, resolved_types))
            elif event == 'return':
                # This event is also triggered if a function yields or raises an exception.
                # We can tell the difference by looking at the bytecode.
                # (We don't get here for C functions so the bytecode always exists.)
                last_opcode = code.co_code[frame.f_lasti]
                if last_opcode == RETURN_VALUE_OPCODE:
                    if code.co_flags & CO_GENERATOR:
                        # Return from a generator.
                        t = resolve_type(FakeIterator([]))
                    else:
                        t = resolve_type(arg)
                elif last_opcode == YIELD_VALUE_OPCODE:
                    # Yield from a generator.
                    # TODO: Unify generators -- currently each YIELD is turned into
                    # a separate call, so a function yielding ints and strs will be
                    # typed as Union[Iterator[int], Iterator[str]] -- this should be
                    # Iterator[Union[int, str]].
                    t = resolve_type(FakeIterator([arg]))
                else:
                    # This branch is also taken when returning from a generator.
                    # TODO: returning non-trivial values from generators, per PEP 380;
                    # and async def / await stuff.
                    t = NoReturnType
                _task_queue.put(KeyAndReturn(function_key, t))
    else:
        sampling_counters[key] = None  # We're not interested in this function.

    return _trace_local

T = TypeVar('T')

def _filter_types(types_dict):
    # type: (Dict[FunctionKey, T]) -> Dict[FunctionKey, T]
    """Filter type info before dumping it to the file."""
    def exclude(k):
        # type: (FunctionKey) -> bool

        """Exclude filter"""
        return k.path.startswith('<') or k.func_name == '<module>'

    return {k: v for k, v in iteritems(types_dict) if not exclude(k)}


def _dump_impl():
    # type: () -> List[FunctionData]
    """Internal implementation for dump_stats and dumps_stats"""
    #print("collected_signature : ", collected_signatures)
    filtered_signatures = _filter_types(collected_signatures)
    #print("filtered_signatures : ", filtered_signatures)
    sorted_by_file = sorted(iteritems(filtered_signatures),
                            key=(lambda p: (p[0].path, p[0].line, p[0].func_name)))
    res = []  # type: List[FunctionData]

    for function_key, signatures in sorted_by_file:
        comments = [{'type' : _make_type_comment(args, ret_type), 'samples' : type_samples.get((function_key, args), 0)} for args, ret_type in signatures]
        res.append(
            {
                'path': function_key.path,
                'line': function_key.line,
                'func_name': function_key.func_name,
                'type_comments': comments,
                'samples': num_samples.get(function_key, 0),
            }
        )
    return res


def dump_stats(filename):
    # type: (str) -> None
    """
    Write collected information to file.

    Args:
        filename: absolute filename
    """
    
    res = _dump_impl()
    f = open(filename, 'w')
    json.dump(res, f, indent=4)
    f.close()

def my_stats() :
    res = _dump_impl()
    global ERROR_LIST, ERROR_MSG
    global NEG_LOCALIZER, ADDITIONAL_INFO


    return ERROR_LIST, ERROR_MSG, res, NEG_LOCALIZER, ADDITIONAL_INFO

def pos_stats() :
    res = _dump_impl()

    global _pos_args
    global POS_LOCALIZER

    return _pos_args, res, POS_LOCALIZER

def dumps_stats():
    # type: () -> str
    """
    Return collected information as a json string.
    """

    res = _dump_impl()
    return json.dumps(res, indent=4)


def init_types_collection(filter_filename=default_filter_filename, only_func=False, test_option=None, test_func=None, current_filename=None, negative_info=None):
    # type: (Callable[[Optional[str]], Optional[str]]) -> None
    """
    Setup profiler hooks to enable type collection.
    Call this one time from the main thread.

    The optional argument is a filter that maps a filename (from
    code.co_filename) to either a normalized filename or None.
    For the default filter see default_filter_filename().
    """
    global ONLY_FUNC
    ONLY_FUNC = only_func

    global TEST_OPTION
    TEST_OPTION = test_option

    global TEST_FUNC
    TEST_FUNC = test_func

    global CURRENT_FILENAME
    CURRENT_FILENAME = current_filename

    global NEGATIVE_INFO
    NEGATIVE_INFO = negative_info

    global _filter_filename
    _filter_filename = filter_filename

    if not only_func :
        sys.settrace(_trace_error)
        threading.settrace(_trace_error)
    sys.setprofile(_trace_dispatch)
    threading.setprofile(_trace_dispatch)


def stop_types_collection():
    # type: () -> None
    """
    Remove profiler hooks.
    """
    sys.settrace(None)
    threading.settrace(None)  # type: ignore
    sys.setprofile(None)
    threading.setprofile(None)  # type: ignore
