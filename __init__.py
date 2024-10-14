from __future__ import annotations

__all__ = [
    'Chain',
    'SizedChain',

    'iterator_not_empty',
    'SuperIterator',
    'ExtensibleIterator',
    'ExtensibleSizedIterator',
    'SizedIterator',
    'EmptyIterator',
    'DeadIterator',

    'super_iterator',
    'super_iterator_from_raw',
    'sized_iterator',
    'sized_iterator_from_raw',
    'sized_iterator_from_super',
    'sized_iterator_from_raw_unsafelen',
    'sized_iterator_from_super_unsafelen',
    'empty_iterator',
    'empty_iterator_from_raw',
    'empty_iterator_from_super',
    'empty_iterator_from_name',
    'empty_iterator_from_dead',
    'dead_iterator',
    'dead_iterator_from_raw',
    'dead_iterator_from_any_super',
    'dead_iterator_from_name',
    'dead_iterator_from_empty',
    'extensible_iterator',
    'extensible_iterator_from_raw',
    'extensible_iterator_from_raw_no_suffix',
    'extensible_iterator_from_super',
    'extensible_iterator_from_super_no_suffix',
    'extensible_sized_iterator',
    'extensible_sized_iterator_from_raw',
    'extensible_sized_iterator_from_raw_no_suffix',
    'extensible_sized_iterator_from_super',
    'extensible_sized_iterator_from_super_no_suffix',
    'extensible_sized_iterator_from_sized',
    'extensible_sized_iterator_from_sized_no_suffix',

    'map_smart_iterators',
    'map_sized_iterators',
    'map_smart_iterators_longest',
    'map_sized_iterators_longest',

    'LazyCollection',
    'LazySeq',
    'LazyList',
    'LazyDeque',
]

import builtins
import io
import itertools
import types

import collections as col
import collections.abc as abcs
import typing as t



T = t.TypeVar('T')

sentinel = object()

@t.runtime_checkable
class SizedIterable(t.Protocol):
    def __iter__(self): ...
    def __len__(self): ...

def natural_commas(texts:abcs.Iterable[str], delimiter=',', last_delimiter='and') -> str:
    lt = len(texts)
    if lt >= 3: return f"{f'{delimiter} '.join(texts[:-1])}{delimiter} {last_delimiter} {texts[-1]}"
    if lt == 2: return f"{texts[0]} {last_delimiter} {texts[1]}"
    if lt == 1: return texts[0]
    return ''



class Chain(t.Generic[T], abcs.Iterator[T]):
    """Better than itertools.chain when chaining a large amount items that may contain other chains because this unpacks other chains
    [Created 3/31/22]"""
    __slots__ = __match_args__ = ('_its',)
    def __init__(self, *its:abcs.Iterable[T]):
        # self._its:t.Final[col.deque[abcs.Iterator[T]]] = col.deque(itertools.chain.from_iterable((it if isinstance(it, Chain) else [it]) for it in its))
        self._its:col.deque[abcs.Iterable[T]] = col.deque()
        _its_extend = self._its.extend
        _its_append = self._its.append
        for it in its:
            if isinstance(it, Chain):
                _its_extend(it._its)
            elif it: # iterators always are True, empty iterables always are False
                _its_append(it)
    @classmethod
    def blank(cls) -> Chain:
        self = cls.__new__(cls)
        self._its = col.deque()
        return self
    @classmethod
    def from_iterable(cls, its:abcs.Iterable[abcs.Iterable[T]]) -> Chain[T]:
        """Alternative to standard instantiation"""
        self = cls.__new__(cls)
        self._its = col.deque()
        _its_extend = self._its.extend
        _its_append = self._its.append
        for it in its:
            if isinstance(it, Chain):
                _its_extend(it._its)
            elif it: # iterators always are True, empty iterables always are False
                _its_append(it)
        return self
    @classmethod
    def from_its(cls, *its:abcs.Iterable[T]) -> Chain[T]:
        """Use if no iterator is a Chain instance or if Chain optimizations should be ignored"""
        self = cls.__new__(cls)
        self._its = col.deque(its)
        return self
    @classmethod
    def from_iterable_its(cls, its:abcs.Iterable[abcs.Iterable[T]]) -> Chain[T]:
        """Alternative to .from_its"""
        self = cls.__new__(cls)
        self._its = col.deque(its)
        return self
    @classmethod
    def from_chains(cls, *chains:Chain[T]) -> Chain[T]:
        """Use if all iterators are Chain instances"""
        self = cls.__new__(cls)
        self._its = its = col.deque()
        _its_extend = its.extend
        for chain in chains:
            _its_extend(chain._its)
        return self
    @classmethod
    def from_iterable_chains(cls, chains:abcs.Iterable[Chain[T]]) -> Chain[T]:
        """Alternative to .from_chains"""
        self = cls.__new__(cls)
        self._its = its = col.deque()
        _its_extend = its.extend
        for chain in chains:
            _its_extend(chain._its)
        return self
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({', '.join(map(repr, self._its))})"
    def __next__(self) -> T:
        """More preferable to call iter() first if doing this multiple times"""
        its:col.deque[abcs.Iterable[T]] = self._its
        __next = next
        __its_popleft = its.popleft
        __sentinel = sentinel
        while its:
            its[0] = first = iter(its[0])
            res = __next(first, __sentinel)
            if res is __sentinel:
                __its_popleft()
                continue
            return res
        raise StopIteration
    def __bool__(self) -> bool:
        return not not self._its
    if t.TYPE_CHECKING:
        @t.final
        def not_empty(self) -> bool:
            return not not self
    else:
        not_empty = __bool__
    def __iter__(self) -> abcs.Iterator[T]:
        return itertools.chain.from_iterable(self._its)
    def extendleft(self, its:abcs.Reversible[abcs.Iterable[T]]) -> None:
        _its_appendleft = self._its.appendleft
        _its_extendleft = self._its.extendleft
        __reversed = reversed
        for it in reversed(its):
            if isinstance(it, Chain):
                _its_extendleft(__reversed(it._its))
            else:
                _its_appendleft(it)
        return self
    def extendleft_chains(self, its:abcs.Reversible[Chain[T]]) -> None:
        _its_extendleft = self._its.extendleft
        __reversed = reversed
        for c in __reversed(its):
            _its_extendleft(__reversed(c._its))
        return self
    def extendleft_its(self, its:abcs.Reversible[abcs.Iterable[T]]) -> None:
        self._its.extendleft(reversed(its))
        return self
    def extend(self, its:abcs.Iterable[abcs.Iterable[T]]) -> None:
        _its_append = self._its.append
        _its_extend = self._its.extend
        for it in its:
            if isinstance(it, Chain):
                _its_extend(it._its)
            else:
                _its_append(it)
        return self
    def extend_chains(self, it:Chain[T]) -> None:
        self._its.extend(it._its)
        return self
    def extend_its(self, its:abcs.Iterable[abcs.Iterable[T]]) -> None:
        self._its.extend(its)
        return self
    def append(self, it:abcs.Iterable[T]) -> None:
        if isinstance(it, Chain):
            self._its.extend(it._its)
        else:
            self._its.append(it)
        return self
    def append_chain(self, it:Chain[T]) -> None:
        self._its.extend(it._its)
        return self
    def append_it(self, it:abcs.Iterable[T]) -> None:
        """No optimization if `it` is a Chain"""
        self._its.append(it)
        return self
    def appendleft(self, it:abcs.Iterable[T]) -> None:
        if isinstance(it, Chain):
            self._its.extendleft(it._its)
        else:
            self._its.appendleft(it)
        return self
    def appendleft_chain(self, it:Chain[T]) -> None:
        self._its.extendleft(it._its)
        return self
    def appendleft_it(self, it:abcs.Iterable[T]) -> None:
        """No optimization if `it` is a Chain"""
        self._its.appendleft(it)
        return self
    # def remove_empties(self) -> None: # shouldn't be needed
    #     for i in range(len(self._its), -1, -1):
    #         if isinstance(self._its[i], EmptyIterator):
    #             self._its.pop(i)
class SizedChain(t.Generic[T], Chain[T]):
    """[Created 4/2/22]"""
    __slots__ = ()
    # __match_args__ = ('_its',) # defined in parent
    def __next__(self) -> T:
        """More preferable to call iter() first if doing this multiple times"""
        its:col.deque[SizedIterable[T]] = self._its
        __next = next
        __its_popleft = its.popleft
        __sentinel = sentinel
        while its:
            res = __next(its[0], __sentinel)
            if res is __sentinel:
                __its_popleft()
                continue
            return res
        raise StopIteration
    if t.TYPE_CHECKING:
        def __init__(self, *its:SizedIterable[T]):
            super().__init__(*its)
            self._its:col.deque[SizedIterable[T]] = col.deque()
        @classmethod
        def from_iterable(cls, its:SizedIterable[T]) -> SizedChain[T]: ...
        @classmethod
        def from_its(cls, *its:SizedIterable[T]) -> SizedChain[T]: ...
        @classmethod
        def from_iterable_its(cls, its:abcs.Iterable[SizedIterable[T]]) -> SizedChain[T]: ...
        @classmethod
        def from_chains(cls, *chains:SizedChain[T]) -> SizedChain[T]: ...
        @classmethod
        def from_iterable_chains(cls, chains:abcs.Iterable[SizedChain[T]]) -> SizedChain[T]: ...
        def extendleft(self, its:abcs.Reversible[SizedIterable[T]]) -> None: ...
        def extendleft_chains(self, it:SizedChain[T]) -> None: ...
        def extendleft_its(self, its:abcs.Reversible[SizedIterable[T]]) -> None: ...
        def extend(self, its:abcs.Iterable[SizedIterable[T]]) -> None: ...
        def extend_chains(self, it:SizedChain[T]) -> None: ...
        def extend_its(self, its:abcs.Iterable[SizedIterable[T]]) -> None: ...
        def append(self, it:SizedIterable[T]) -> None: ...
        def append_chain(self, it:SizedChain[T]) -> None: ...
        def append_it(self, it:SizedIterable[T]) -> None: ...
        def appendleft(self, it:SizedIterable[T]) -> None: ...
        def appendleft_chain(self, it:SizedChain[T]) -> None: ...
        def appendleft_it(self, it:SizedIterable[T]) -> None: ...
    def __len__(self) -> int:
        its = self._its
        while its and not its[0]: its.popleft() # remove empties
        return sum(map(len, its)) if its else 0
    def __bool__(self) -> bool:
        its = self._its
        while its:
            if its[0]: return True
            else: its.popleft() # remove empties
        return False


def iterator_not_empty(it:abcs.Iterator[T]) -> t.Optional[abcs.Iterator[T]]:
    """Check if an iterator is empty - returns an equivalent version of the iterator if not empty else None
    Since iterators always evaluate to True and None evaluates to False the result can simply be converted to bool
    Be careful if `it` is used by other objects as they will still be using it after it has lost its first item
    [Created 4/1/22]"""
    if not it: return None # EmptyIterator
    __sentinel = sentinel
    if (nxt:=next(it, __sentinel)) is __sentinel:
        return None
    return yyf(nxt, it)

# └ ┌  ┐ ┘ ─ │ ┴ ┬ ┼

                 _SuperIteratorMeta
                        \\│//    └─>──>──>──>──>──>──┬──┬──┐
                 _SuperIteratorBase                  │  │ _EmptySuperIteratorMixinABC
           ┌───────┘      │      └──────────────┐    │ _ExtensibleIteratorMixinABC
           │         SuperIterator              │   _ExtensibleSizedIteratorMixinABC
        ┌──│───────┬──────┴────────┐            │
        │  │  SizedIterator  ExtensibleIterator │
      ┌─│──│──<v>──┘    └─<v>─┐    │            │
      │ │  └──────────┐ ExtensibleSizedIterator │
      │ └──<v>──┐ EmptyIterator                 │
      │         │     │     └───────<v>───────┐ │
      │        EmptySuperIterator             │ │
      │         │     │       └─────<v>─────┐ │ │
EmptySizedIterator  EmptyExtensibleIterator │ │ │
            └─<v>─┐       │           DeadIterator
             EmptyExtensibleSizedIterator

class _SuperIteratorMeta(type):
    if t.TYPE_CHECKING: _saved_mapping_:None|types.MappingProxyType[tuple[_SuperIteratorBase,...], tuple[_SuperIteratorBase,...]] = None
    else: _saved_mapping_ = None
    @property
    def _mapping_(cls) -> types.MappingProxyType[tuple[_SuperIteratorBase,...], tuple[_SuperIteratorBase,...]]:
        """Mapping of classes and their "virtual" subclasses"""
        if (saved:=(thismeta:=cls.__class__)._saved_mapping_):
            return saved
        if super(thismeta.__class__, thismeta) == type:
            raise TypeError(f"{thismeta.__name__} must inherit from {type.__name__}; It inherits from {super(thismeta.__class__, thismeta)}")
        m:types.MappingProxyType[tuple[_SuperIteratorBase,...], tuple[_SuperIteratorBase,...]] = types.MappingProxyType({
            # what we are : what we are saying are virtual subclasses
            (SizedIterator,): (ExtensibleSizedIterator,EmptySizedIterator),
            (SuperIterator,): (EmptySuperIterator,),
            (ExtensibleIterator,): (EmptyExtensibleIterator,),
            (ExtensibleSizedIterator,): (EmptyExtensibleSizedIterator,),
            (EmptyIterator,EmptySuperIterator): (DeadIterator,),
            (EmptySizedIterator,): (EmptyExtensibleSizedIterator,),
        })
        _isinstance = isinstance
        if (bad:={c.__name__ for c in itertools.chain.from_iterable(itertools.chain(m.keys(), m.values())) if not _isinstance(c, thismeta)}):
            raise TypeError(f"{natural_commas(bad)} must have {thismeta.__name__} as their metaclass as they are connected to important virtual inherits")
        thismeta._saved_mapping_ = m
        return m
    def __instancecheck__(cls, instance) -> bool:
        _type___instancecheck__ = type.__instancecheck__
        _any = any
        for k,v in cls._mapping_.items():
            if _any(clss is cls for clss in k) and _any(_type___instancecheck__(c, instance) for c in v): # might want to do THIS check here if this doesn't already work (rather than the type. version)
                return True
        return _type___instancecheck__(cls, instance)
    def __subclasscheck__(cls, subclass) -> bool:
        _type___subclasscheck__ = type.__subclasscheck__
        _any = any
        for k,v in cls._mapping_.items():
            if _any(c is cls for c in k) and _any(_type___subclasscheck__(c, instance) for c in v): # might want to do THIS check here if this doesn't already work (rather than the type. version)
                return True
        return _type___subclasscheck__(cls, instance)
    # def __new__(cls, *args, **kwargs):
    #     return super().__new__(cls, *args, **kwargs)

class SIInitSubclassDict(t.TypedDict):
    __vrs__:dict[str,t.Any]
    __super_called__:bool
    __extensible_called:bool
    __extensiblesized_called__:bool
    __sized_called__:bool
    __empty_called__:bool
    __emptysuper_called__:bool
    __emptysized_called__:bool
    __emptyextensiblesized_called__:bool
    __dead_called__:bool

class _AllBase:
    __slots__ = ('_it','_length', '_suffix', '_previous_cls', '__weakref__')
    __match_args__ = ('_it','_raw')
class _SuperIteratorBase(_AllBase, metaclass=_SuperIteratorMeta):
    __slots__ = ()
    @classmethod
    def _init_all_subclass(cls, subclass, kwargs:dict[str, t.Any], *, virtual:bool=False) -> SIInitSubclassDict:
        """Used to init all subclasses, and their subclasses
        Returns an SIInitSubclassDict"""
        skwargs:SIInitSubclassDict
        if (skwargs:=kwargs.get('__skwargs__')) is None:
            kwargs['__skwargs__'] = skwargs = {}
        if (vrs:=skwargs.get('__vrs__')) is None:
            skwargs['__vrs__'] = vrs = vars(subclass)
        if not skwargs.get('__root_called__'):
            skwargs['__root_called__'] = True
            _vrs_get = vrs.get
            if _vrs_get('__slots__', True): # this is true if `defined || nonempty`
                raise TypeError("'__slots__' is not defined but must be defined and empty")
            _magic_init_subclass(subclass, _vrs_get,
                required_defined=('_init_all_subclass', '__init_subclass__')
            )
        return skwargs
    def __init_subclass__(cls, **kwargs) -> SIInitSubclassDict:
        """Should always return `CLASS_DEFINING_THIS_METHOD._init_all_subclass(cls, kwargs)`"""
        return _SuperIteratorBase._init_all_subclass(cls, kwargs)

class _EmptySuperIteratorMixinABC(_AllBase, metaclass=_SuperIteratorMeta):
    __slots__ = ()
    def _inner_repr(self) -> str:
        return f"{getattr(cls:=self.__class__, '_corresponding_class', cls).__name__}[{it._inner_repr() if isinstance(it:=self._it, EmptyIterator) else it}]"
class _ExtensibleIteratorMixinABC(_AllBase, metaclass=_SuperIteratorMeta):
    __slots__ = ()
    @property
    def suffix(self):
        return self._suffix
class _ExtensibleSizedIteratorMixinABC(_AllBase, metaclass=_SuperIteratorMeta):
    __slots__ = ()
    def __init_subclass__(cls, **kwargs):
        """Raises TypeError if __slots__ is not defined or nonempty, or if new methods are defined that are not classmethods/staticmethods/properties"""
        vrs = vars(cls)
        if not (length_it:=vrs.get('length_it')):
            raise TypeError(f"{length_it.__name__} not defined in {cls.__name__} but must be")
        if not isinstance(length_it, property):
            raise TypeError(f"{length_it.__name__} defined in {cls.__name__} but is not a property")
    if t.TYPE_CHECKING:
        @property
        def length_it(self) -> int:
            """Get the length of the iterator, not the entire structure"""

# if t.TYPE_CHECKING:
#     def init_subclass(parent, subclass, kwargs:dict[str,t.Any]):
#         """Force a class to use its __init_subclass__ method to init a subclass, essentially equal to using super().__init_subclass__, but this allows for using a specific parent
#         [Created 5/8/22]"""
#         return parent.__init_subclass__(**kwargs) # return type inferred
# else:
#     def init_subclass(parent, subclass, kwargs:dict[str,t.Any]):
#         return parent.__getattribute__(parent, '__init_subclass__').__get__(None, subclass)(**kwargs)

def _magic_init_subclass(cls,
    vars_get:None|abcs.Callable[[str], t.Any]=None,
    *,
    required_defined:None|abcs.Sequence[str]=None,
    required_undefined:None|abcs.Sequence[str]=None,
    meta_getattribute:None|abcs.Callable[[str], t.Any]=None,
    required_defined_within_hierarchy:None|abcs.Sequence[str]=None,
    required_undefined_within_hierarchy:None|abcs.Sequence[str]=None,
) -> None:
    """init subclass helper"""
    doing_hierarchry = False
    if (doing_ns:=(required_defined or required_undefined)) or (doing_hierarchry:=(required_defined_within_hierarchy or required_undefined_within_hierarchy)):
        lines = []
        try:
            repr = builtins.repr
            if doing_ns:
                if vars_get is None:
                    lines.append("vars_get must be set if checking within namespace at all")
                else:
                    if required_defined   and (l:=[repr(attr) for attr in required_defined if not vars_get(attr)]):
                        lines.append(f"{natural_commas(l)} {'are' if len(l) > 1 else 'is'} not defined in {cls.__name__} but must be")
                    if required_undefined and (l:=[repr(attr) for attr in required_undefined if   vars_get(attr)]):
                        lines.append(f"{natural_commas(l)} {'are' if len(l) > 1 else 'is'} defined in {cls.__name__} but must not be")
            if doing_hierarchry:
                if meta_getattribute is None:
                    lines.append("meta_getattribute must be passed as kwarg if checking within hierarchy at all")
                else:
                    if required_defined_within_hierarchy   and (l:=[repr(attr) for attr in required_defined_within_hierarchy if not meta_getattribute(attr)]):
                        lines.append(f"{natural_commas(l)} {'are' if len(l) > 1 else 'is'} not defined in {cls.__name__} *or parents* but must be")
                    if required_undefined_within_hierarchy and (l:=[repr(attr) for attr in required_undefined_within_hierarchy if   meta_getattribute(attr)]):
                        lines.append(f"{natural_commas(l)} {'are' if len(l) > 1 else 'is'} defined in {cls.__name__} *or parents* but must not be")
        finally:
            if lines:
                raise TypeError('\n'.join(lines))

class SuperIterator(t.Generic[T], _SuperIteratorBase): # , abcs.Iterator[T]):
    """An intelligent high level iterator implementation
    Be very careful when subclassing
    [Created 4/2/22]"""
    __slots__ = ()
    @classmethod
    def _init_all_subclass(cls, subclass, kwargs:dict[str, t.Any], *, virtual:bool=False) -> SIInitSubclassDict:
        skwargs:SIInitSubclassDict = _SuperIteratorBase._init_all_subclass(subclass, kwargs)
        if not skwargs.get('__super_called__'):
            skwargs['__super_called__'] = True
            _magic_init_subclass(subclass, skwargs['__vrs__'].get,
                required_undefined=('__iter__', '__iter__', 'exhaust', 'flatten'),
                required_defined  =tuple(itertools.chain(('__new__',), () if virtual else ('_COERCE_empty',)))
            )
            si_vrs_keys:frozenset[str] = frozenset(vars(SuperIterator).keys())
            cmsm = (classmethod,staticmethod)
            for k,v in skwargs['__vrs__'].items():
                if isinstance(k, property):
                    continue
                if not k.startswith('_') and isinstance(k, abcs.Callable) and not isinstance(k, cmsm) and k not in si_vrs_keys: # classmethods/staticmethods can be added
                    raise TypeError(f"{SuperIterator.__name__} subclasses may not implement new methods that do not start with '_'; (new classmethods/staticmethods/properties are allowed); Got {k!r}")
        return skwargs
    def __init_subclass__(cls, **kwargs) -> SIInitSubclassDict:
        return SuperIterator._init_all_subclass(cls, kwargs)
    @t.overload
    def __new__(cls, obj:EmptyIterator) -> EmptyIterator: ...
    @t.overload
    def __new__(cls, obj:SizedIterable[T]) -> SizedIterator[T]: ...
    @t.overload
    def __new__(cls, obj:abcs.Iterable[T]) -> SuperIterator[T]: ...
    def __new__(cls, obj:abcs.Iterable[T]) -> SuperIterator[T]:
        """Slowest and safest constructor"""
        if isinstance(obj, SuperIterator): return obj
        if not obj: return EmptyIterator(obj)
        if isinstance(obj, SizedIterable): # we should use abcs.Sized but that annoys type checker so it doesn't matter
            return sized_iterator_from_raw(iter(obj), len(obj))
        return super_iterator_from_raw(iter(obj))
    @property
    def _raw(self) -> abcs.Iterator[T]:
        """Get the root raw iterator; Very unsafe"""
        return self._it
    def _COERCE_empty(self) -> None:
        self.__class__ = EmptySuperIterator
        self._previous_cls = type(self._it)
    @t.final # we always want to keep ourselves the iterator, so we force this to not change
    def __iter__(self):
        return self
    @t.final
    def exhaust(self) -> None:
        """Iterate and empty"""
        for _ in self.iter_unsafe(): pass
    @t.final
    def flatten(self) -> abcs.Iterator[T]:
        return iter(self.as_chain())
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._it})"
    # if t.TYPE_CHECKING: # we don't want subclasses to define __init__ so we have it appear as final
    #     @t.final
    #     def __init__(self, obj): pass
    def map(self, func:abcs.Callable[[T], Y]) -> SuperIterator[Y]:
        """Map inplace"""
        self._it = map(func, self._it)
        return self
    def set_length(self, length:int) -> SizedIterator[T]|EmptyIterator:
        """Coerces to a SizedIterator/EmptyIterator with length `length` and return self"""
        if length < 0:
            raise ValueError(f"Length cannot be below 0; {length=}")
        if length:
            self._length = length
            if not isinstance(self, __SizedIterator:=SizedIterator):
                self.__class__ = __SizedIterator
        else:
            self._COERCE_empty()
        return self
    def __next__(self) -> T:
        __sentinel = sentinel
        if (obj:=next(self._it,__sentinel)) is __sentinel:
            self._COERCE_empty()
            raise StopIteration
        return obj
    def __bool__(self) -> bool:
        if (res:=iterator_not_empty(self._it)): # we don't do a none check, because if we created without checking if it==smartiterator, then this will check size too if its a sizediterator
            self._it = res
            return True
        self._COERCE_empty()
        return False
    def iter_fast(self) -> abcs.Iterator[T]:
        """Faster than __iter__ but slower than iter_unsafe"""
        __sentinel = sentinel
        __next = next
        while (obj:=__next(self._it,__sentinel)) is not __sentinel: # we don't cache self._it because we may call __bool__ mid-iteration
            yield obj
        self._COERCE_empty()
    def iter_unsafe(self) -> abcs.Iterator[T]:
        """Fastest form of iteration; Use if self._it will NOT be touched at all during iteration (it is modified by __bool__ for example)"""
        it = self._it
        self._COERCE_empty()
        return it
    def append(self, obj:T) -> ExtensibleIterator[T]:
        """Coerces to a ExtensibleIterator and returns self after appending"""
        self._suffix = suffix = LazyDeque()
        suffix.append(obj)
        self.__class__ = ExtensibleIterator
        return self
    def extend(self, iterable:abcs.Iterable[T]) -> ExtensibleIterator[T]:
        """Coerces to a ExtensibleIterator and returns self after extension"""
        self._suffix = suffix = LazyDeque(iterable)
        self.__class__ = ExtensibleIterator
        return self
    def as_chain(self) -> Chain[T]:
        return Chain.from_its(self._it)
class ExtensibleIterator(t.Generic[T], SuperIterator[T], _ExtensibleIteratorMixinABC):
    """Allows for an iterator which can have items appended to the end
    When initializing from another instance, the previous instance and new instance will be modified such that the new suffix is appended to the old
    Logically wraps an iterator and a lazy deque, with the iterator in front and the lazy deque in the back
    [Created 3/31/22 // Made a smart iterator 4/18/22]"""
    __slots__ = ()
    __match_args__ = SuperIterator.__match_args__ + ('_suffix',)
    @classmethod
    def _init_all_subclass(cls, subclass, kwargs:dict[str, t.Any]) -> SIInitSubclassDict:
        skwargs:SIInitSubclassDict = SuperIterator._init_all_subclass(subclass, kwargs)
        if not skwargs.get('__extensible_called__'):
            skwargs['__extensible_called__'] = True
            _magic_init_subclass(subclass, skwargs['__vrs__'].get,
                required_defined  =None,
                required_undefined=('_raw','map','iter_fast','iter_unsafe','_iter_suffix','_iter_suffix_no_optimization','append','extend','as_chain')
            )
    def __init_subclass__(cls, **kwargs) -> SIInitSubclassDict:
        return ExtensibleIterator._init_all_subclass(cls, kwargs)
    @t.overload
    def __new__(cls, obj:EmptyIterator, suffix:col.deque[T]|LazyDeque[T]|None=None) -> EmptyIterator: ...
    @t.overload
    def __new__(cls, obj:SizedIterator[T]|ExtensibleSizedIterator[T], suffix:col.deque[T]|LazyDeque[T]|None=None) -> ExtensibleSizedIterator[T]: ...
    @t.overload
    def __new__(cls, obj:SizedIterable[T], suffix:col.deque[T]|LazyDeque[T]|None=None) -> ExtensibleSizedIterator[T]|EmptyIterator: ...
    @t.overload
    def __new__(cls, obj:SuperIterator[T], suffix: col.deque[T] | LazyDeque[T] | None=None) -> ExtensibleIterator[T]: ...
    @t.overload
    def __new__(cls, obj:abcs.Iterable[T], suffix:col.deque[T]|LazyDeque[T]|None=None) -> ExtensibleIterator[T]|EmptyIterator: ...
    def __new__(cls, obj:abcs.Iterable[T], suffix:col.deque[T]|LazyDeque[T]|None=None) -> ExtensibleIterator[T]|EmptyIterator:
        """Slowest and safest constructor"""
        if not obj: return EmptyIterator(obj)
        if isinstance(obj, SizedIterator): # yes we catch ExtensibleSizedIterator as we want
            return extensible_sized_iterator_from_sized(obj, suffix)
        if isinstance(obj, SuperIterator): # yes we catch ExtensibleIterator as we want
            return extensible_iterator_from_super(obj, suffix)
        if isinstance(obj, SizedIterable): # we should use abcs.Sized but that annoys type checker so it doesn't matter
            return extensible_sized_iterator_from_sized(sized_iterator_from_raw(iter(obj), len(obj)), suffix)
        return extensible_iterator_from_raw(iter(obj))
    @property
    @t.final
    def _raw(self) -> abcs.Iterator[T]:
        """Get the root raw iterator; Very unsafe"""
        return self._it._raw
    def _COERCE_empty(self) -> None:
        self.__class__ = EmptyExtensibleIterator
        self._previous_cls = type(self._it)
    if t.TYPE_CHECKING:
        @t.final
        def map(self, func:abcs.Callable[[T], Y]) -> ExtensibleIterator[Y]: ...
    def set_length(self, length:int) -> ExtensibleSizedIterator[T]|EmptyIterator:
        """Coerces to a ExtensibleSizedIterator/EmptyIterator with length `length` and return self"""
        if length < 0: raise ValueError(f"Length cannot be below 0; {length=}")
        if length:
            self._it = SizedIterator(self._it, length)
            self.__class__ = ExtensibleSizedIterator
        else: self._COERCE_empty()
        return self
    def __next__(self) -> T:
        __sentinel = sentinel
        if (obj:=next(it:=self._it,__sentinel)) is __sentinel:
            if suffix:=self.suffix: # if not empty
                return suffix.popleft()
            self._COERCE_empty()
            raise StopIteration
        return obj
    def __bool__(self) -> bool:
        if self._suffix or self._it: return True
        self._COERCE_empty()
        return False
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._it!r}, {self.suffix!r})"
    @t.final
    def iter_fast(self) -> abcs.Iterator[T]:
        """Faster than __iter__ but slower than iter_unsafe"""
        yield from self._it.iter_fast()
        yield from self._iter_suffix()
        self._COERCE_empty()
    @t.final
    def iter_unsafe(self) -> abcs.Iterator[T]:
        """Fastest form of iteration; Use if self._it will NOT be touched at all during iteration (ex: in an asynchronous environment) (it is modified by __bool__ for example)"""
        it = self._it; it_suffix = self._iter_suffix()
        self._COERCE_empty()
        return itertools.chain(it.iter_unsafe(), it_suffix)
    @t.final
    def _iter_suffix(self) -> abcs.Generator[T, t.Any, None]:
        if isinstance(suffix:=self.suffix, LazyDeque): return suffix.popleft_forever()
        else: return self._iter_suffix_no_optimization()
    @t.final
    def _iter_suffix_no_optimization(self) -> abcs.Generator[T, t.Any, None]:
        suffix = self.suffix; __suffix_popleft = suffix.popleft
        while suffix: yield __suffix_popleft()
    @t.final
    def append(self, obj:T) -> None: self.suffix.append(obj)
    @t.final
    def extend(self, iterable:abcs.Iterable[T]) -> None: self.suffix.extend(iterable) # this is just the best move, we won't do something weird like extending _it because it may be used elsewhere
    @t.final
    def as_chain(self) -> Chain[T]:
        return self._it.as_chain().append_it(self._iter_suffix()) # we could make the iter_suffix be a SizedIterator but that would not work well since we could append to it after chain creation
class ExtensibleSizedIterator(t.Generic[T], ExtensibleIterator[T], _ExtensibleSizedIteratorMixinABC): # , abcs.Sized):
    """[Created 4/18/22]"""
    __slots__ = ()
    __match_args__ = ExtensibleIterator.__match_args__ + ('length_it', )
    @classmethod
    def _init_all_subclass(cls, subclass, kwargs:dict[str, t.Any], *, virtual:bool=False) -> SIInitSubclassDict:
        skwargs:SIInitSubclassDict = ExtensibleIterator._init_all_subclass(subclass, kwargs, virtual=virtual)
        if not skwargs.get('__extensiblesized_called__'):
            skwargs['__extensiblesized_called__'] = True
            SizedIterator._init_all_subclass(subclass, kwargs, virtual=True) # virtual inherit
            ...
        return skwargs
    def __init_subclass__(cls, **kwargs) -> SIInitSubclassDict:
        return ExtensibleSizedIterator._init_all_subclass(cls, kwargs)
    def __new__(cls, obj:abcs.Iterable[T], length:int, suffix:col.deque[T]|LazyDeque[T]|None=None) -> ExtensibleSizedIterator[T]:
        """Slowest and safest constructor
        length is the length of the source iterablr `obj` NOT the total length; Slowest constructor"""
        if isinstance(obj, SizedIterator):
            if len(obj)!=length:
                raise ValueError(f"Length of source and length stated are different: {len(obj)=} != {length=}")
            return extensible_sized_iterator_from_sized(obj)
        return extensible_sized_iterator_from_sized(SizedIterator(obj, length))
    def _COERCE_empty(self) -> None:
        self.__class__ = EmptyExtensibleIterator
        self._previous_cls = type(self._it._raw._raw)
    @property
    def length_it(self) -> int:
        return len(self._it)
    if t.TYPE_CHECKING:
        @t.final
        def map(self, func:abcs.Callable[[T], Y]) -> ExtensibleSizedIterator[Y]: ...
    def set_length(self, length:int) -> ExtensibleSizedIterator[T]|EmptyIterator:
        """Set length and return self"""
        if length < 0:
            raise ValueError(f"Length cannot be below 0; {length=}")
        if length:
            self._length = length
        else:
            self._COERCE_empty()
        return self
    def __len__(self) -> int:
        return len(self._it) + len(self._suffix)
    def __bool__(self) -> bool:
        if self._it or self._suffix:
            return True
        self._COERCE_empty()
        return False
class SizedIterator(t.Generic[T], SuperIterator[T]): # , abcs.Sized):
    """Allow for size to be known for an iterator
    When initializing, `length` MUST be exactly correct or unexpected behavior will occur
    If `length` is 0, constructs an EmptyIterator instead of a SizedIterator
    [Created 4/2/22]"""
    __slots__ = ()
    __match_args__ = SuperIterator.__match_args__ + ('length',)
    @classmethod
    def _init_all_subclass(cls, subclass, kwargs:dict[str, t.Any], *, virtual:bool=False) -> SIInitSubclassDict:
        skwargs:SIInitSubclassDict = SuperIterator._init_all_subclass(subclass, kwargs, virtual=virtual)
        if not skwargs.get('__sized_called__'):
            skwargs['__sized_called__'] = True
            if virtual: # virtual subclasses need methods which this has
                _magic_init_subclass(subclass,
                                     meta_getattribute=(meta:=type(subclass)).__getattribute__.__get__(subclass, meta),
                                     required_defined_within_hierarchy=('__len__',)
                                     )
        return skwargs
    def __init_subclass__(cls, **kwargs) -> SIInitSubclassDict:
        return SizedIterator._init_all_subclass(cls, kwargs)
    def __new__(cls, obj:abcs.Iterable[T], length:int) -> SizedIterator[T]:
        """Slowest and safest constructor"""
        if length > 0:
            self:SizedIterator = _SuperIteratorBase.__new__(cls)
            self._it = SuperIterator(obj)
            self._length = length
            return self
        return EmptyIterator(obj)
    @property
    def _raw(self) -> abcs.Iterator[T]:
        """Get the root raw iterator; Very unsafe"""
        return self._it._raw
    def _COERCE_empty(self) -> None:
        self.__class__ = EmptySizedIterator
        del self._length
        self._previous_cls = type(self._it._raw)
    if t.TYPE_CHECKING:
        @t.final
        def map(self, func:abcs.Callable[[T], Y]) -> SizedIterator[Y]:
            """Map inplace"""
    def __repr__(self) -> str: #                         if src IS SuperIterator, we don't need to show that
        return f"{self.__class__.__name__}({self._it._it if type(self._it) is SuperIterator else self._it!r}, {self._length})"
    def iter_fast(self) -> abcs.Iterator[T]:
        """Faster than __iter__ but slower than iter_unsafe"""
        for obj in self._it.iter_fast():
            if (length:=self._length) <= 0:
                raise ValueError(f"{self.__class__.__name__} ended unexpectedly; Stated length was {length} less than it should have been")
            self._length -= 1
            yield obj
        if (length:=self._length):
            raise ValueError(f"{self.__class__.__name__} ended unexpectedly; Stated length was {length} greater than it should have been")
        self._COERCE_empty()
    def iter_unsafe(self) -> abcs.Iterator[T]:
        """Fastest form of iteration; Obtains the raw iterator and sets self to empty"""
        it = self._it
        self._COERCE_empty()
        return it.iter_unsafe()
    def __bool__(self) -> bool:
        return not not self._length
    def __len__(self):
        return self._length
    def __next__(self) -> T:
        __sentinel = sentinel
        if (obj:=next(self._it,__sentinel)) is __sentinel:
            if (length:=self._length):
                raise ValueError(f"{self.__class__.__name__} ended unexpectedly; Stated length was {length} greater than it should have been")
            self._COERCE_empty()
            raise StopIteration
        self._length -= 1
        return obj
    def append(self, obj:T) -> ExtensibleIterator[T]:
        """Coerces to a ExtensibleIterator and returns self after appending; ExtensibleIterator if it used to be sized"""
        self._suffix = suffix = LazyDeque()
        suffix.append(obj)
        self._it = SizedIterator(self._it, self._length) # unsafe because we won't be touching it again, as we are coercing away
        self.__class__ = ExtensibleSizedIterator
        return self
    def extend(self, iterable:abcs.Iterable[T]) -> SizedIterator[T]|ExtensibleSizedIterator[T]:
        """Coerces to a ExtensibleSizedIterator and returns self after extension; If the iterable is empty simply returns self"""
        if iterable:
            self._suffix = suffix = LazyDeque(iterable)
            self._it = SizedIterator(self._it, self._length)
            self.__class__ = ExtensibleSizedIterator
        return self

class EmptyIterator(_SuperIteratorBase):
    """Useful for showing that an iterator is empty in a repr and when checking for iterable lengths (because this evaluates to False)
    Evaluates to False unlike all other iterators
    Also implements Reversible and Container because no additional information is needed to implement them
    [Created 3/31/22]"""
    __slots__ = ()
    __match_args__ = ('_raw','_previous_cls')
    @classmethod
    def _init_all_subclass(cls, subclass, kwargs:dict[str,t.Any], *, virtual:bool=False) -> SIInitSubclassDict:
        skwargs:SIInitSubclassDict = _SuperIteratorBase._init_all_subclass(subclass, kwargs, virtual=virtual)
        if not skwargs.get('__empty_called__'):
            skwargs['__empty_called__'] = True
            SuperIterator._init_all_subclass(subclass, kwargs, virtual=True) # virtual inherit
            _magic_init_subclass(subclass, skwargs['__vrs__'].get,
                required_undefined=('_COERCE_empty', 'flatten', 'as_chain', 'finalize',
                                    'map', '__bool__', '__len__', '__reversed__', '__contains__', '__next__', 'iter_fast', 'iter_unsafe',
                                    'as_chain')
            )
        return skwargs
    def __init_subclass__(cls, **kwargs) -> SIInitSubclassDict:
        return EmptyIterator._init_all_subclass(cls, kwargs)
    @t.overload
    def __new__(cls, source:ExtensibleSizedIterator) -> EmptyExtensibleSizedIterator: ...
    @t.overload
    def __new__(cls, source:ExtensibleIterator) -> EmptyExtensibleIterator: ...
    @t.overload
    def __new__(cls, source:SizedIterator) -> EmptySizedIterator: ...
    @t.overload
    def __new__(cls, source:SuperIterator) -> EmptySuperIterator: ...
    @t.overload
    def __new__(cls, source:None|abcs.Iterable|str=None) -> EmptyIterator: ...
    def __new__(cls, source:None|abcs.Iterable|str=None) -> EmptyIterator:
        if isinstance(source, DeadIterator           ): return empty_iterator_from_dead(source)
        if isinstance(source, EmptyIterator          ): return source # catches all EmptyIterator subclasses
        if isinstance(source, ExtensibleSizedIterator): return empty_extensible_sized_iterator_from_any_iterator(source)
        if isinstance(source, ExtensibleIterator     ): return empty_extensible_iterator_from_any_iterator(source)
        if isinstance(source, SizedIterator          ): return empty_sized_iterator_from_any_iterator(source)
        if isinstance(source, SuperIterator          ): return empty_super_iterator_from_any_iterator(source)
        if source is None:
            return empty_iterator_from_raw(DeadIterator._dead_raw_iterator)
        if isinstance(source, str):
            return empty_iterator_from_name(source) if source else empty_iterator_from_raw(DeadIterator._dead_raw_iterator)
        return empty_iterator_from_raw(iter(source))
    @property
    def _raw(self) -> abcs.Iterator[T]:
        """Returns an empty iterator every time; Subclasses won't always"""
        return DeadIterator._dead_raw_iterator
    def _COERCE_dead(self) -> None:
        self.__class__ = DeadIterator
    if t.TYPE_CHECKING:
        def __iter__(self): return self
    else: __iter__ = SuperIterator.__iter__
    @t.final
    def flatten(self) -> abcs.Iterator[T]:
        """Pointless call"""
        return self
    @t.final
    def as_chain(self) -> Chain[T]:
        return Chain.blank()
    @t.final
    def finalize(self) -> DeadIterator:
        """Coerce into a DeadIterator"""
        self._COERCE_dead()
        return self
    @t.final
    def map(self, *args, **kwargs) -> EmptyIterator:
        """Pointless call"""
        return self
    @t.final
    def __bool__(self) -> t.Literal[False]:
        return False
    @t.final
    def __len__(self) -> t.Literal[0]:
        return 0
    @t.final
    def __reversed__(self):
        return self
    @t.final
    def __contains__(self, item) -> t.Literal[False]:
        return False
    @t.final
    def __next__(self) -> t.NoReturn:
        raise StopIteration
    @t.final
    def iter_fast(self) :
        return self
    @t.final
    def iter_unsafe(self):
        return self
    def _inner_repr(self) -> str: # NOT THE SAME AS EmptySuperIterator Mixin VERSION
        return f"{getattr(cls:=self.__class__, '_corresponding_class', cls).__name__}[{self._previous_cls}]"
    def __repr__(self) -> str:
        return f"<empty {self._previous_cls}>"
    def append(self, obj:T) -> SizedIterator[T]:
        """Coerces to a SizedIterator (or ExtensibleIterator if that's what we used to be) with a single item `obj` and returns self after appending"""
        self._it = iter((obj,))
        self._length = 1
        self.__class__ = SizedIterator
        del self._previous_cls
        return self
    def extend(self, iterable:abcs.Iterable[T]) -> SuperIterator[T]|SizedIterator[T]:
        """Coerces to a SuperIterator/SizedIterator with all items in iterable and returns self after extension"""
        if isinstance(iterable, SizedIterable):
            self._it = iter(iterable)
            self._length = len(iterable)
            self.__class__ = SizedIterator
        else:
            self._it = iter(iterable)
            self.__class__ = SuperIterator
        del self._previous_cls
        return self
class EmptySuperIterator(EmptyIterator, _EmptySuperIteratorMixinABC): # , abcs.Reversible[...], abcs.Container[...], abcs.Sized):
    """Useful for showing that an iterator is empty in a repr and when checking for iterable lengths (because this evaluates to False)
    Evaluates to False unlike all other iterators
    Also implements Reversible and Container because no additional information is needed to implement them
    [Created 3/31/22]"""
    __slots__ = ()
    __match_args__ = ('_it', '_raw')
    _corresponding_class = SuperIterator
    @classmethod
    def _init_all_subclass(cls, subclass, kwargs:dict[str,t.Any], *, virtual:bool=False) -> SIInitSubclassDict:
        skwargs:SIInitSubclassDict = EmptyIterator._init_all_subclass(subclass, kwargs, virtual=virtual)
        if not skwargs.get('__emptysuper_called__'):
            skwargs['__emptysuper_called__'] = True
            _magic_init_subclass(
                subclass, skwargs['__vrs__'].get,
                # required_defined=('_raw',), # not sure why this used to be here
                required_undefined=tuple(itertools.chain(('_inner_repr',), (() if subclass.__name__=='DeadIterator' else ('__repr__',)))),
                meta_getattribute=(meta:=type(subclass)).__getattribute__.__get__(subclass, meta),
                required_defined_within_hierarchy=('_raw',)
            )
        return skwargs
    def __init_subclass__(cls, **kwargs) -> SIInitSubclassDict:
        return EmptySuperIterator._init_all_subclass(cls, kwargs)
    @t.overload
    def __new__(cls, source:SizedIterator) -> EmptySizedIterator: ...
    @t.overload
    def __new__(cls, source:ExtensibleSizedIterator) -> EmptyExtensibleSizedIterator: ...
    @t.overload
    def __new__(cls, source:ExtensibleIterator) -> EmptyExtensibleIterator: ...
    @t.overload
    def __new__(cls, source:None|abcs.Iterable=None) -> EmptySuperIterator: ...
    def __new__(cls, source:None|abcs.Iterable=None) -> EmptySuperIterator:
        if isinstance(source, EmptySuperIterator     ): return source # catches all EmptySuperIterator subclasses
        if isinstance(source, ExtensibleSizedIterator): return empty_extensible_sized_iterator_from_any_iterator(source)
        if isinstance(source, ExtensibleIterator     ): return empty_extensible_iterator_from_any_iterator(source)
        if isinstance(source, EmptyIterator          ): return empty_super_iterator_from_any_iterator(source) # also catches DeadIterator
        if isinstance(source, SizedIterator          ): return empty_sized_iterator_from_any_iterator(source)
        if isinstance(source, SuperIterator          ): return empty_super_iterator_from_any_iterator(source)
        return empty_super_iterator_from_any_iterator(DeadIterator._dead_raw_iterator if source is None else iter(source))
    @property
    def _raw(self) -> abcs.Iterator[T]:
        """Get the root raw iterator; Very unsafe"""
        return it._raw if isinstance(it:=self._it, SuperIterator) else it
    def _COERCE_dead(self) -> None:
        self._previous_cls = f"{self._raw.__class__}"
        self.__class__ = DeadIterator
        del self._it
    @t.final
    def __repr__(self) -> str:
        return f"<empty {self._inner_repr()}>"
    def append(self, obj:T) -> SuperIterator[T]:
        """Coerces to a SizedIterator with a single item `obj` and returns self after appending"""
        self._it = iter([obj])
        self.__class__ = SuperIterator
        return self
    def extend(self, iterable:abcs.Iterable[T]) -> SuperIterator[T]|EmptySuperIterator:
        """Coerces to a SuperIterator/EmptySuperIterator with all items in iterable and returns self after extension"""
        if iterable:
            self._it = iter(iterable)
            self.__class__ = SuperIterator
            del self._previous_cls
        return self
class EmptySizedIterator(EmptySuperIterator):
    """[Created 4/21/22]"""
    __slots__ = ()
    __match_args__ = ('_it', '_raw', '_length')
    _corresponding_class = SizedIterator
    @classmethod
    def _init_all_subclass(cls, subclass, kwargs:dict[str,t.Any], *, virtual:bool=False) -> SIInitSubclassDict:
        skwargs:SIInitSubclassDict = EmptySuperIterator._init_all_subclass(subclass, kwargs, virtual=virtual)
        if not skwargs['__emptysized_called__']:
            skwargs['__emptysized_called__'] = True
            ...
        return skwargs
    def __init_subclass__(cls, **kwargs) -> SIInitSubclassDict:
        return EmptySizedIterator._init_all_subclass(cls, kwargs)
    @t.overload
    def __new__(cls, source:ExtensibleIterator) -> EmptyExtensibleSizedIterator: ...
    @t.overload
    def __new__(cls, source:None|abcs.Iterable=None) -> EmptySizedIterator: ...
    def __new__(cls, source:None|abcs.Iterable=None) -> EmptySizedIterator:
        if isinstance(source, EmptySizedIterator     ): return source # catches all EmptySizedIterator subclasses
        if isinstance(source, ExtensibleIterator     ): return empty_extensible_sized_iterator_from_any_iterator(source) # catches sized and unsized version, both of which we want
        if isinstance(source, (SuperIterator, EmptyIterator)): return empty_sized_iterator_from_any_iterator(source) # catches SuperIterator & SizedIterator & EmptyIterator & DeadIterator
        return empty_sized_iterator_from_any_iterator(DeadIterator._dead_raw_iterator if source is None else iter(source))
    def append(self, obj:T) -> SizedIterator[T]:
        """Coerces back to a SizedIterator with the sole item `obj`"""
        # self._suffix.append(obj)
        self._it = super_iterator_from_raw(iter([obj]))
        self._length = 1
        self.__class__ = SizedIterator
        return self
    def extend(self, iterable:abcs.Iterable[T]) -> SizedIterator[T]|EmptySizedIterator:
        """Coerces back to a SizedIterator with the contents of `iterable` (if `iterable` contains any contents)"""
        # (suffix:=self._suffix).extend(obj)
        # if suffix:
        #     self.__class__ = ExtensibleIterator
        if iterable:
            if isinstance(iterable, SizedIterator):
                self._it = iterable
                self._length = len(iterable)
            elif isinstance(iterable, SizedIterable):
                self._it = super_iterator_from_raw(iter(iterable))
                self._length = len(iterable)
            else:
                if isinstance(iterable, SuperIterator):
                    iterable = iterable.iter_fast()
                iterable = list(iterable) # yea we just have to list it
                self._it = super_iterator_from_raw(iterable)
                self._length = len(iterable)
            self.__class__ = ExtensibleIterator
        return self
class EmptyExtensibleIterator(EmptySuperIterator, _ExtensibleIteratorMixinABC):
    """[Created 4/21/22]"""
    __slots__ = ()
    __match_args__ = ('_it', '_raw', '_suffix')
    _corresponding_class = ExtensibleIterator
    @classmethod
    def _init_all_subclass(cls, subclass, kwargs:dict[str,t.Any], *, virtual:bool=False) -> SIInitSubclassDict:
        skwargs:SIInitSubclassDict = EmptySuperIterator._init_all_subclass(subclass, kwargs, virtual=virtual)
        if not skwargs.get('__emptyextensible_called__'):
            skwargs['__emptyextensible_called__'] = True
            _magic_init_subclass(
                subclass, skwargs['__vrs__'].get,
                required_undefined=('_COERCE_dead', '_inner_repr', '_inner_repr', 'append', 'extend'),
            )
        return skwargs
    def __init_subclass__(cls, **kwargs) -> SIInitSubclassDict:
        return EmptyExtensibleIterator._init_all_subclass(cls, kwargs)
    @t.overload
    def __new__(cls, source:SizedIterator) -> EmptyExtensibleSizedIterator: ...
    @t.overload
    def __new__(cls, source:None|abcs.Iterable=None) -> EmptyExtensibleIterator: ...
    def __new__(cls, source:None|abcs.Iterable=None) -> EmptyExtensibleIterator:
        if isinstance(source, SizedIterator): # we would use SizedIterable but that would catch all Emptys
            return empty_extensible_sized_iterator_from_any_iterator(source)
        return empty_extensible_iterator_from_any_iterator(DeadIterator._dead_raw_iterator if source is None else iter(source))
    @t.final
    def _COERCE_dead(self) -> None:
        self._previous_cls = f"{self._raw.__class__}"
        self.__class__ = DeadIterator
        del self._it
        del self._suffix
    @property
    def suffix(self):
        return self._suffix
    @t.final
    def append(self, obj:T) -> ExtensibleIterator[T]:
        """Coerces back to a ExtensibleIterator, appends, and returns self"""
        # self._suffix.append(obj)
        self._it = super_iterator_from_raw(iter([obj]))
        self.__class__ = self.__class__._corresponding_class
        return self
    @t.final
    def extend(self, iterable:abcs.Iterable[T]) -> ExtensibleIterator[T]|EmptyExtensibleIterator:
        """Coerces back to a ExtensibleIterator, extends, and returns self (If iterable had any contents)"""
        # (suffix:=self._suffix).extend(obj)
        # if suffix:
        #     self.__class__ = self.__class__._corresponding_class
        if iterable:
            self._it = super_iterator_from_raw(iterable.iter_fast() if isinstance(iterable, SuperIterator) else iter(iterable))
            self.__class__ = self.__class__._corresponding_class
        return self
class EmptyExtensibleSizedIterator(EmptyExtensibleIterator, _ExtensibleSizedIteratorMixinABC):
    """[Created 4/21/22]"""
    __slots__ = ()
    _corresponding_class:str = ExtensibleSizedIterator
    @classmethod
    def _init_all_subclass(cls, subclass, kwargs:dict[str,t.Any], *, virtual:bool=False) -> SIInitSubclassDict:
        skwargs:SIInitSubclassDict = EmptyExtensibleIterator._init_all_subclass(subclass, kwargs, virtual=virtual)
        if not skwargs.get('__emptyextensiblesized_called__'):
            skwargs['__emptyextensiblesized_called__'] = True
            SizedIterator._init_all_subclass(subclass, kwargs, virtual=True) # virtual inherit
            ...
        return skwargs
    def __init_subclass__(cls, **kwargs) -> SIInitSubclassDict:
        return EmptyExtensibleSizedIterator._init_all_subclass(cls, kwargs)
    def __new__(cls, source:None|abcs.Iterable=None) -> EmptyExtensibleSizedIterator:
        return empty_extensible_sized_iterator_from_any_iterator(DeadIterator._dead_raw_iterator if source is None else iter(source))
    @property
    def length_it(self) -> int:
        """Pointless call"""
        return 0
    if t.TYPE_CHECKING:
        def append(self, obj:T) -> ExtensibleSizedIterator[T]:
            """Coerces back to a ExtensibleSizedIterator, appends, and returns self"""
        def extend(self, iterable:abcs.Iterable[T]) -> ExtensibleSizedIterator[T]|EmptyExtensibleSizedIterator:
            """Coerces back to a ExtensibleSizedIterator, extends, and returns self"""

@t.final # if we want to subclass in the future, we will remove this deco
class DeadIterator(_SuperIteratorBase, _EmptySuperIteratorMixinABC):
    """An EmptyIterator which will never iterate again; a permanently empty iterator
    [Created 4/21/22]"""
    __slots__ = ()
    __match_args__ = ('_raw','_previous_cls')
    if t.TYPE_CHECKING:
        DEFAULT:t.Final[DeadIterator]
    @t.final
    class iterator:
        def __repr__(self): return 'iterator'
        def __next__(self): raise StopIteration
        def __iter__(self): return self
    _dead_raw_iterator = iterator()
    del iterator
    @classmethod
    def _init_all_subclass(cls, subclass, kwargs:dict[str,t.Any], *, virtual:bool=False) -> SIInitSubclassDict:
        skwargs:SIInitSubclassDict = EmptyIterator._init_all_subclass(subclass, kwargs, virtual=True) # virtual inherit
        if not skwargs.get('__dead_called__'):
            skwargs['__dead_called__'] = True
            _magic_init_subclass(
                subclass, vars_get=skwargs['__vrs__'].get,
                required_defined=('__new__', ),
                required_undefined=('_raw', ),
            )
        return skwargs
    def __init_subclass__(cls, **kwargs) -> SIInitSubclassDict:
        return DeadIterator._init_all_subclass(cls, kwargs)
    def __new__(cls, source:None|abcs.Iterator|str=None) -> DeadIterator:
        if isinstance(source, DeadIterator):
            return source
        if isinstance(source, EmptyIterator):
            return dead_iterator_from_empty(source)
        if isinstance(source, str):
            return dead_iterator_from_name(source) if source else cls.DEFAULT # this is to ensure the string isn't empty
        if isinstance(source, SuperIterator):
            return dead_iterator_from_super(source)
        return dead_iterator_from_raw(source)
    @property
    @t.final
    def _raw(self) -> abcs.Iterator:
        """Returns an empty iterator every time"""
        return self.__class__._dead_raw_iterator
    @t.final
    def finalize(self) -> DeadIterator:
        """Pointless call - we are already dead"""
        return self
    if t.TYPE_CHECKING:
        def map(self, *args, **kwargs) -> DeadIterator:
            """Pointless call"""
    else:
        map = EmptySuperIterator.map
    __bool__ = EmptySuperIterator.__bool__
    __len__ = EmptySuperIterator.__len__
    if t.TYPE_CHECKING:
        def iter_fast(self) -> DeadIterator:
            """Pointless call"""
        def iter_unsafe(self) -> DeadIterator:
            """Pointless call"""
        def __reversed__(self) -> DeadIterator:
            """Pointless call"""
        def __iter__(self) -> DeadIterator:
            """Pointless call"""
    else:
        __reversed__ = EmptySuperIterator.__reversed__
        __iter__ = EmptySuperIterator.__iter__
        iter_fast = EmptySuperIterator.iter_fast
        iter_unsafe = EmptySuperIterator.iter_unsafe
    __contains__ = EmptySuperIterator.__contains__
    __next__ = EmptySuperIterator.__next__
    flatten = EmptySuperIterator.flatten
    as_chain = EmptySuperIterator.as_chain
    @t.final
    def __repr__(self) -> str:
        return f"<dead empty {pcls if isinstance(pcls:=self._previous_cls, str) else pcls.__name__}>"
DeadIterator._corresponding_class = DeadIterator
__DIDefault:DeadIterator = _SuperIteratorBase.__new__(DeadIterator)
__DIDefault._previous_cls = 'iterator'
setattr(DeadIterator, 'DEFAULT', __DIDefault)
del __DIDefault

super_iterator = SuperIterator
def super_iterator_from_raw(it:abcs.Iterator[T]) -> SuperIterator[T]:
    """Create from a normal iterator which is not sized (`it` SHOULD NOT BE A SuperIterator) (Fastest constructor)"""
    self = _SuperIteratorBase.__new__(SuperIterator)
    self._it = it
    return self

extensible_iterator = ExtensibleIterator
def extensible_iterator_from_raw(it:abcs.Iterator[T], suffix:col.deque[T]|LazyDeque[T]|None=None) -> ExtensibleIterator[T]:
    """Create from a raw iterator (Fourth fastest constructor)"""
    self:ExtensibleIterator = _SuperIteratorBase.__new__(ExtensibleIterator)
    self._it = super_iterator_from_raw(it); self._suffix = LazyDeque() if suffix is None else suffix
    return self
def extensible_iterator_from_raw_no_suffix(it:abcs.Iterator[T]) -> ExtensibleIterator[T]:
    """Create from a raw iterator with no suffix (Third fastest constructor)"""
    self:ExtensibleIterator = _SuperIteratorBase.__new__(ExtensibleIterator)
    self._it = super_iterator_from_raw(it); self._suffix = LazyDeque()
    return self
def extensible_iterator_from_super(it:SuperIterator[T], suffix:col.deque[T]|LazyDeque[T]|None=None) -> ExtensibleIterator[T]:
    """Create from a SuperIterator (NOT a subclass) (Second fastest constructor)"""
    self:ExtensibleIterator = _SuperIteratorBase.__new__(ExtensibleIterator)
    self._it = it; self._suffix = LazyDeque() if suffix is None else suffix
    return self
def extensible_iterator_from_super_no_suffix(it:SuperIterator[T]) -> ExtensibleIterator[T]:
    """Create from a SuperIterator (NOT a subclass) with no suffix (Fastest constructor)"""
    self:ExtensibleIterator = _SuperIteratorBase.__new__(ExtensibleIterator)
    self._it = it; self._suffix = LazyDeque()
    return self

extensible_sized_iterator = ExtensibleSizedIterator
def extensible_sized_iterator_from_raw(it:abcs.Iterator[T], length:int, suffix:col.deque[T]|LazyDeque[T]|None=None) -> ExtensibleSizedIterator[T]:
    """Create from a raw iterator (Sixth fastest constructor)"""
    self:ExtensibleSizedIterator = _SuperIteratorBase.__new__(ExtensibleSizedIterator)
    self._it = sized_iterator_from_raw(it, length); self._suffix = LazyDeque() if suffix is None else suffix
    return self
def extensible_sized_iterator_from_raw_no_suffix(it:abcs.Iterator[T], length:int) -> ExtensibleSizedIterator[T]:
    """Create from a raw iterator with no suffix (Fifth fastest constructor)"""
    self:ExtensibleSizedIterator = _SuperIteratorBase.__new__(ExtensibleSizedIterator)
    self._it = sized_iterator_from_raw(it, length); self._suffix = LazyDeque()
    return self
def extensible_sized_iterator_from_super(it:SuperIterator[T], length:int, suffix:col.deque[T]|LazyDeque[T]|None=None) -> ExtensibleIterator[T]:
    """Create from a SuperIterator (NOT a subclass) (Fourth fastest constructor)"""
    self:ExtensibleSizedIterator = _SuperIteratorBase.__new__(ExtensibleSizedIterator)
    self._it = sized_iterator_from_super(it, length); self._suffix = LazyDeque() if suffix is None else suffix
    return self
def extensible_sized_iterator_from_super_no_suffix(it:SuperIterator[T], length:int) -> ExtensibleIterator[T]:
    """Create from a SuperIterator (NOT a subclass) with no suffix (Third fastest constructor)"""
    self:ExtensibleSizedIterator = _SuperIteratorBase.__new__(ExtensibleSizedIterator)
    self._it = sized_iterator_from_super(it, length); self._suffix = LazyDeque()
    return self
def extensible_sized_iterator_from_sized(it:SizedIterator[T], suffix:col.deque[T]|LazyDeque[T]|None=None) -> ExtensibleIterator[T]:
    """Create from a SizedIterator (NOT a subclass) (Second fastest constructor)"""
    self:ExtensibleSizedIterator = _SuperIteratorBase.__new__(ExtensibleSizedIterator)
    self._it = it; self._suffix = LazyDeque() if suffix is None else suffix
    return self
def extensible_sized_iterator_from_sized_no_suffix(it:SizedIterator[T]) -> ExtensibleIterator[T]:
    """Create from a SizedIterator (NOT a subclass) with no suffix (Fastest constructor)"""
    self:ExtensibleSizedIterator = _SuperIteratorBase.__new__(ExtensibleSizedIterator)
    self._it = it; self._suffix = LazyDeque()
    return self

sized_iterator = SizedIterator
def sized_iterator_from_raw_unsafelen(it:abcs.Iterator[T], length:int) -> SizedIterator[T]:
    """Create from a raw iterator which has a known length (`length` MUST be > 0) (Fourth fastest constructor)"""
    self = _SuperIteratorBase.__new__(SizedIterator)
    self._it = super_iterator_from_raw(it)
    self._length = length
    return self
def sized_iterator_from_raw(it:abcs.Iterator[T], length:int) -> SizedIterator[T]|EmptySizedIterator:
    """Create from a raw iterator which has a known length (`it` SHOULD NOT BE A SMART) (Third fastest constructor)"""
    if length > 0:
        return sized_iterator_from_raw_unsafelen(it, length)
    return EmptySizedIterator(it)
def sized_iterator_from_super(it:SuperIterator[T], length:int) -> SizedIterator[T]|EmptySizedIterator:
    """Create from a SuperIterator (NOT a subclass) which has a known length (Second fastest constructor)"""
    if length > 0:
        return sized_iterator_from_super_unsafelen(it, length)
    return EmptySizedIterator(it)
def sized_iterator_from_super_unsafelen(it:SuperIterator[T], length:int) -> SizedIterator[T]:
    """Create from a SuperIterator (NOT a subclass) which has a known length (`length` MUST > 0) (Fastest constructor)"""
    # if isinstance(it, EmptyIterator):
    #     raise TypeError(f"Cannot create {cls.__name__} from {it.__class__.__name__} because it is empty")
    self = _SuperIteratorBase.__new__(SizedIterator)
    self._it = it
    self._length = length
    return self

empty_iterator = EmptyIterator
def empty_iterator_from_raw(it:abcs.Iterator[T]) -> EmptyIterator:
    """Create from a raw iterator (Second fastest constructor)"""
    self:EmptyIterator = _SuperIteratorBase.__new__(EmptyIterator)
    self._previous_cls = f"{it.__class__}"
    return self

def empty_iterator_from_name(name:str) -> EmptyIterator:
    """Fastest way to create from a name"""
    self:EmptyIterator = _SuperIteratorBase.__new__(EmptyIterator)
    self._previous_cls = name
    return self
def empty_iterator_from_dead(it:DeadIterator) -> EmptyIterator:
    """Create from a DeadIterator"""
    self:EmptyIterator = _SuperIteratorBase.__new__(EmptyIterator)
    self._previous_cls = it._inner_repr()
    return self

empty_super_iterator = EmptyIterator
def empty_super_iterator_from_any_iterator(it:abcs.Iterator[T]) -> EmptySuperIterator:
    """Create from any iterator"""
    self:EmptySuperIterator = _SuperIteratorBase.__new__(EmptySuperIterator)
    self._it = it
    return self

empty_sized_iterator = EmptySizedIterator
def empty_sized_iterator_from_any_iterator(it:abcs.Iterator[T]) -> EmptySizedIterator:
    """Create from any iterator"""
    self:EmptySizedIterator = _SuperIteratorBase.__new__(EmptySizedIterator)
    self._it = it
    return self

empty_extensible_iterator = EmptyExtensibleIterator
def empty_extensible_iterator_from_any_iterator(it:abcs.Iterator[T]) -> EmptyExtensibleIterator:
    """Create from any iterator"""
    self:EmptyExtensibleIterator = _SuperIteratorBase.__new__(EmptyExtensibleIterator)
    self._it = it
    return self

empty_extensible_sized_iterator = EmptyExtensibleSizedIterator
def empty_extensible_sized_iterator_from_any_iterator(it:abcs.Iterator[T]) -> EmptyExtensibleSizedIterator:
    """Create from any iterator"""
    self:EmptyExtensibleSizedIterator = _SuperIteratorBase.__new__(EmptyExtensibleSizedIterator)
    self._it = it
    return self

# this is here because they are defined above
empty_iterator_from_super = empty_super_iterator_from_any_iterator
empty_iterator_from_sized = empty_sized_iterator_from_any_iterator
empty_iterator_from_extensible_iterator = empty_sized_iterator_from_any_iterator
empty_iterator_from_extensible_sized_iterator = empty_sized_iterator_from_any_iterator

dead_iterator = DeadIterator
def dead_iterator_from_raw(it:abcs.Iterator[T]) -> DeadIterator:
    """Create from a raw iterator (Second fastest constructor)"""
    self:DeadIterator = _SuperIteratorBase.__new__(DeadIterator)
    self._previous_cls = f"{it.__class__}"
    return self
def dead_iterator_from_any_super(it:SuperIterator[T]) -> DeadIterator:
    """Create from a SuperIterator (same as "safe" version) (Fastest constructor)"""
    self:DeadIterator = _SuperIteratorBase.__new__(DeadIterator)
    self._previous_cls = f"{it._raw.__class__}"
    return self
def dead_iterator_from_name(name:str) -> DeadIterator:
    """Fastest way to create from a name"""
    self:DeadIterator = _SuperIteratorBase.__new__(DeadIterator)
    self._previous_cls = name
    return self
def dead_iterator_from_empty(it:EmptyIterator) -> DeadIterator:
    self:DeadIterator = _SuperIteratorBase.__new__(DeadIterator)
    self._previous_cls = it._inner_repr()
    return self

def _starmap_zip_si(func:abcs.Callable[..., Y], iterator:SuperIterator[T], iterators:tuple[SuperIterator[T],...]) -> itertools.starmap[Y]:
    return itertools.starmap(func, zip(iterator.iter_fast(), *(it.iter_fast() for it in iterators)))
def map_smart_iterators(func:abcs.Callable[..., Y], iterator:SuperIterator[T], /, *iterators:SuperIterator[T]) -> SuperIterator[Y]:
    if iterators:
        if not iterator or not all(iterators): return EmptyIterator.with_name("map")
        if isinstance(iterator, SizedIterator) and all(isinstance(it, SizedIterator) for it in iterators):
            iterators:tuple[SizedIterator] = iterators
            return sized_iterator_from_raw_unsafe(_starmap_zip_si(func, iterator, iterators), min(map(len, iterators)))
        return super_iterator_from_raw(_starmap_zip_si(func, iterator, iterators))
    if not iterator: return empty_iterator_from_name("map")
    if isinstance(iterator, SizedIterator):
        iterators:tuple[SizedIterator] = iterators
        return sized_iterator_from_raw_unsafe(map(func, iterator), len(iterator))
    return super_iterator_from_raw(map(func, iterator))
def map_sized_iterators(func:abcs.Callable[..., Y], iterator:SizedIterator[T], /, *iterators:SizedIterator[T]) -> SizedIterator[Y]:
    if iterators:
        if not iterator or not all(iterators): return empty_iterator_from_name("map")
        return sized_iterator_from_raw_unsafe(_starmap_zip_si(func, iterator, iterators), min(map(len, iterators)))
    if not iterator: return empty_iterator_from_name("map")
    return sized_iterator_from_raw_unsafe(map(func, iterator), len(iterator))
def _starmap_zip_longest_si(func:abcs.Callable[[T], Y], iterator:SuperIterator[T], iterators:tuple[SuperIterator[T],...], *, fillvalue:Y) -> itertools.starmap[Y]:
    return itertools.starmap(func, itertools.zip_longest(iterator.iter_fast(), *(it.iter_fast() for it in iterators), fillvalue=fillvalue))
def map_smart_iterators_longest(func:abcs.Callable[[T], Y], iterator:SuperIterator[T], /, *iterators:SuperIterator[T], fillvalue:Y=None) -> SuperIterator[Y]:
    if iterators:
        if not iterator and not any(iterators): return EmptyIterator.with_name("map")
        if isinstance(iterator, SizedIterator) and all(isinstance(it, SizedIterator) for it in iterators):
            iterators:tuple[SizedIterator] = iterators
            return sized_iterator_from_raw_unsafe(_starmap_zip_longest_si(func, iterator, iterators, fillvalue=fillvalue), min(map(len, iterators)))
        return super_iterator_from_raw(_starmap_zip_longest_si(func, iterator, iterators, fillvalue=fillvalue))
    if not iterator: return EmptyIterator.with_name("map")
    if isinstance(iterator, SizedIterator):
        iterators:tuple[SizedIterator] = iterators
        return sized_iterator_from_raw_unsafe(map(func, iterator.iter_fast()), len(iterator))
    return super_iterator_from_raw(map(func, iterator))
def map_sized_iterators_longest(func:abcs.Callable[[T], Y], iterator:SizedIterator[T], /, *iterators:SizedIterator[T], fillvalue:Y=None) -> SizedIterator[Y]:
    if iterators:
        if not iterator and not any(iterators): return EmptyIterator.with_name("map")
        if not all(iterators): return EmptyIterator.with_name("map")
        return sized_iterator_from_raw_unsafe(_starmap_zip_longest_si(func, iterator, iterators, fillvalue=fillvalue), min(map(len, iterators)))
    if not iterator: return EmptyIterator.with_name("map")
    return sized_iterator_from_raw_unsafe(map(func, iterator.iter_fast()), len(iterator))







# old ideas

# @t.runtime_checkable
# class DequeLike(Protocol[T]):
#     """Deque like object which is used by ExtensibleIterator"""
#     def append(self, obj:T) -> None: ...
#     def extend(self, iterable:abcs.Iterable[T]) -> None: ...
#     def popleft(self) -> T: ...
#     def __iter__(self) -> abcs.Iterator[T]: ...
#     def copy(self) -> t.Union[col.deque[T], LazyDeque[T]: ...

# class ExtensibleIterator(t.Generic[T], abcs.Iterator[T]):
#     """Allows for an iterator which can have items appended to the end
#     When initializing from another instance, the previous instance and new instance will be modified such that the new suffix is appended to the old
#     Logically wraps an iterator and a deque, with the iterator in front and the deque in the back
#     [Created 3/31/22]"""
#     __slots__ = ('_it','suffix','__next')
#     __match_args__ = ('_it', 'suffix', 'it_exhausted')
#     def __init__(self, it:abcs.Iterator[T], suffix:col.deque[T]|LazyDeque[T]|None=None):
#         """Slowest and safest initialization"""
#         if isinstance(it, ExtensibleIterator):
#             self._it:abcs.Iterator[T] = it._it
#             end_suffix = it.suffix
#             if end_suffix:
#                 end_suffix = end_suffix.copy()
#                 if suffix:
#                     end_suffix.extend(suffix)
#             else:
#                 end_suffix = col.deque() if suffix is None else suffix
#             self.suffix:col.deque[T]|LazyDeque[T] = end_suffix
#         else:
#             self._it:abcs.Iterator[T] = it
#             self.suffix:col.deque[T]|LazyDeque[T] = col.deque() if suffix is None else suffix
#         self.__next = self._next_default
#     @property
#     def it_exhausted(self) -> bool:
#         """If this is True then the iterator has completed and the suffix should just be used"""
#         return self.__next is self._next_from_suffix
#     @classmethod
#     def raw_unsafe(cls, it:abcs.Iterator[T]) -> ExtensibleIterator[T]:
#         """If `it` is an ExtensibleIterator, they share the same suffix deque"""
#         self = cls.__new__(cls)
#         if isinstance(it, ExtensibleIterator):
#             self._it = it._it
#             self.suffix = it.suffix
#         else:
#             self._it = it
#             self.suffix = col.deque()
#         self.__next = self._next_default
#         return self
#     @classmethod
#     def raw(cls, it:abcs.Iterator[T]) -> ExtensibleIterator[T]:
#         """Initialize without any suffix"""
#         self = cls.__new__(cls)
#         if isinstance(it, ExtensibleIterator):
#             self._it = it._it
#             self.suffix = it.suffix.copy()
#         else:
#             self._it = it
#             self.suffix = col.deque()
#         self.__next = self._next_default
#         return self
#     @classmethod
#     def unsafe(cls, it:abcs.Iterator[T], suffix:col.deque[T]|LazyDeque[T]|None=None) -> ExtensibleIterator[T]:
#         """If `it` is an ExtensibleIterator, they share the same suffix deque which is modified immediately if suffix is not None"""
#         if isinstance(it, ExtensibleIterator):
#             if suffix:
#                 it.suffix.extend(suffix)
#             return it
#         else:
#             self = cls.__new__(cls)
#             self._it = it
#             self.suffix = col.deque() if suffix is None else suffix
#             self.__next = self._next_default
#             return self
#     @classmethod
#     def from_it(cls, it:abcs.Iterator[T], suffix:col.deque[T]|LazyDeque[T]|None=None) -> ExtensibleIterator[T]:
#         """`it` should NOT gain any new items after __next__ raise StopIteration; For this reason this initializer can cause issues if the iterator which can be emptied, but then iterate again, and then appended"""
#         self = cls.__new__(cls)
#         self._it = it
#         self.suffix = col.deque() if suffix is None else suffix
#         return self
#     if t.TYPE_CHECKING:
#         @t.final
#         def not_empty(self) -> bool:
#             return not not self
#     else:
#         not_empty = __bool__
#     def __repr__(self) -> str:
#         return f"{self.__class__.__name__}.from_it({self._it!r}, {self.suffix!r})"
#     def __iter__(self) -> abcs.Iterator[T]:
#         return itertools.chain(self._it, self._iter_suffix())
#     def as_chain(self) -> Chain[T]:
#         return Chain.from_its(self._it, self._iter_suffix())
#     def _iter_suffix(self) -> abcs.Generator[T, t.Any, None]:
#         suffix = self.suffix
#         if isinstance(suffix, LazyDeque):
#             return suffix.popleft_forever() # optimization
#         else:
#             return self._iter_suffix_no_optimization()
#     def _iter_suffix_no_optimization(self) -> abcs.Generator[T, t.Any, None]:
#         suffix = self.suffix
#         __suffix_popleft = suffix.popleft
#         while suffix:
#             yield __suffix_popleft()
#     def __next__(self):
#         return self.__next()
#     def _next_default(self) -> T:
#         res = next(self._it, __sentinel:=sentinel)
#         if res is __sentinel:
#             self.__next = nxt = self._next_from_suffix # override & optimize
#             self._it = EmptyIterator(self._it)
#             return nxt()
#         return res
#     def _next_from_suffix(self) -> T:
#         if suffix:=self.suffix: # if not empty
#             return suffix.popleft()
#         raise StopIteration
#     def append(self, obj:T) -> None:
#         self.suffix.append(obj)
#     def extend(self, iterable:abcs.Iterable[T]) -> None:
#         self.suffix.extend(iterable)
# class ExtensibleIteratorSized(ExtensibleIterator, abcs.Sized):
#     """[Created 4/18/22]"""
#     def __len__(self) -> int:
#         return len(self._it) + len(self.suffix)
#     def empty(self) -> bool:
#         return     not (self._it or self.suffix)
#     def __bool__(self) -> bool:
#         return not not (self._it or self.suffix)
#     def as_chain(self) -> SizedChain[T]:
#         return SizedChain.from_its(self._it, SizedIterator(self._iter_suffix(), len(self.suffix)))
#     def to_sized_iterator(self) -> SizedIterator[T]:
#         return SizedIterator(self, len(self))

class LazyCollection(t.Generic[T], abcs.Collection[T]):
    """Object that allows for a Collection of lazily generated items
    [Created as LazySeq 2/19/22 - this made 3/31/22]"""
    __slots__ = __match_args__ = ('_it','_collection', '_collection_add_item', '_collection_extend')
    # noinspection PyShadowingBuiltins
    def __init__(self, it:abcs.Iterator[T], collection:abcs.Collection[T], *,
                 collection_add_item:abcs.Callable[[T],t.Any],
                 collection_extend:abcs.Callable[[abcs.Iterable[T]],t.Any] | t.Type[set.update]):
        self._it:SuperIterator[T] = SuperIterator(it)
        self._collection:abcs.Collection[T] = collection
        self._collection_add_item:abcs.Callable[[T],t.Any] = collection_add_item
        self._collection_extend:abcs.Callable[[abcs.Iterable[T]],t.Any] = collection_extend
    @property
    def collection(self) -> abcs.Collection[T]:
        return self._collection
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._it!r}, {self._collection!r}, collection_add_item={self._collection_add_item!r}, _collection_extend={self._collection_extend!r})"
    def iter_it(self) -> abcs.Generator[T, t.Any, None]:
        """Iterator over `it` and save the contents"""
        if not self._it: return
        __append = self._collection_add_item
        for obj in self._it.iter_fast():
            __append(obj)
            yield obj
    def load_am(self, n:int) -> bool:
        """Load a certain amount of objects from `it`
        If `n` is greater than the amount of object that were in `it`, returns False
        Else, returns True"""
        if not self._it: return False
        __sentinel = sentinel
        __append = self._collection_add_item
        __next = next
        it = self._it.iter_fast()
        old_len = len(self._collection) # might want to do an intermediate for async purposes... idk
        self._collection_extend(itertools.islice(self._it.iter_fast(), n))
        return n == len(self._collection)-old_len
    def __iter__(self) -> abcs.Iterator[T]:
        if not self._it:
            return iter(self._collection)
        return itertools.chain(self._collection, self.iter_it())
    def for_chain(self) -> abcs.Collection[T] | Chain[T]:
        """If done will simply be a reference to the collection, if not will be a Chain"""
        if not self._it:
            return self._collection
        return Chain.from_its(iter(self._collection), self.iter_it())
    def __bool__(self) -> bool:
        if self._collection:
            return True
        if not self._it: # no need to attempt to load if we already finished
            return False
        # if isinstance(self._it, SizedIterator) and len(self._it)!=0: return True # this check is already done by the above
        if self.try_load_one(): # we are done if the loading loaded nothing
            return False
        return True
    if t.TYPE_CHECKING:
        @t.final
        def not_empty(self) -> bool:
            return not not self
    else: not_empty = __bool__
    def __len__(self) -> int:
        """Resolves self, no need to use this object then"""
        if isinstance(self._it, SizedIterator):
            return len(self._it) + len(self._collection)
        return len(self.resolve()) # sad we have to do this but we need to
    def __contains__(self, item:T) -> bool:
        return item in self._collection or (self._it and item in self.iter_it())
    def load_one(self) -> T | object:
        """Load a single object from `it` and return it; might be sentinel"""
        if not (it:=self._it): return sentinel
        __sentinel = sentinel
        if (nxt:=next(it,__sentinel)) is not __sentinel:
            self._collection_add_item(nxt)
        return nxt
    def try_load_one(self) -> bool:
        """Try to load a single object from `it` and return whether we are done"""
        if not (it:=self._it): return True
        __sentinel = sentinel
        if (nxt:=next(it,__sentinel)) is __sentinel:
            return True
        else:
            self._collection_add_item(nxt)
            return False
    def resolve(self): # return type will be implied
        if not (it:=self._it):
            return self._collection
        self._collection_extend(it.iter_unsafe())
        return self._collection

class LazySeq(t.Generic[T], LazyCollection[T], abcs.Sequence[T]):
    """Object that allows for a Sequence of lazily generated items"""
    __slots__ = __match_args__ = ('_it','_collection', '_collection_add_item', '_collection_extend')
    @t.overload
    def __init__(self, it:abcs.Iterator[T], /): ...
    # noinspection PyShadowingBuiltins
    @t.overload
    def __init__(self, list:list[T], it:abcs.Iterator[T], /): ...
    # noinspection PyShadowingBuiltins
    def __init__(self, list_or_it1, list_or_it2=None, /):
        if list_or_it2 is None:
            it:abcs.Iterator[T] = list_or_it1 # becomes smart in parent __init__
            list:t.List[T] = []
        else:
            list:t.List[T] = list_or_it1
            it:abcs.Iterator[T] = list_or_it2 # becomes smart in parent __init__
        super().__init__(it=it, collection=list, collection_add_item=list.append, collection_extend=list.extend)
        self._collection:t.List[T] = self._collection
        self._it:SuperIterator[T] = self._it
    @classmethod
    def from_it(cls, it:abcs.Iterator[T], /):
        """Automatically set the collection to an empty list"""
        # noinspection PyShadowingBuiltins
        list:t.List[T] = []
        self = cls.__new__(cls)
        LazyCollection.__init__(self, it=it, collection=list, collection_add_item=list.append, collection_extend=list.extend)
        self._collection:t.List[T] = self._collection
    if t.TYPE_CHECKING:
        # noinspection PyPropertyDefinition
        @property
        def collection(self) -> list[T]: ...
    def __repr__(self) -> str:
        if collection:=self._collection:
            return f"{self.__class__.__name__}({collection!r}, {self._it!r})"
        return f"{self.__class__.__name__}({self._it!r})"
    def __reversed__(self) -> abcs.Iterator[T]:
        return reversed(self.resolve())
    def count(self, obj:T) -> int:
        return self.resolve().count(obj)
    def index(self, obj:T, start:t.Optional[int]=None, end:t.Optional[int]=None, /) -> int:
        if end is not None:
            if end < start: raise ValueError("End cannot come before start")
            if not self._load_indexslice(slice(start, end)):
                raise ValueError("Index out of range")
            return self._collection.index(obj, start, end)
        if start is not None:
            if not self._load_indexslice(start):
                raise ValueError("Index out of range")
            try:
                return self._collection.index(obj, start)
            except ValueError: # means we don't already have it loaded
                pass
        if self._it:
            for i,nxt in zip(itertools.count(len(self._collection)), self.iter_it()):
                if obj==nxt:
                    return i
        raise ValueError("Not found")
    def _load_indexslice(self, key:int|slice) -> bool:
        """Returns whether or not an operation on that key is possible; Negative indices only work after resolving or if self._it is a SizedIterator"""
        typ = type(key)
        if typ is int:
            if not self._it: # we finished, so we can use negative indices
                try: self._collection[key]
                except IndexError:
                    return False
                else: return True
            if key < 0:
                if isinstance(it:=self._it, SizedIterator):
                    if len(_collection) + (len_it:=len(it)) + key < 0: # it was so low that it out of range
                        raise IndexError(f"{key} is out of range (extremely low)")
                    return not (am:=len_it+key+1) or self.load_am(am) # if am==0 (we already have it) or we successfully load it
                raise ValueError(f"Negative indices not supported; If they are necessary, use the .resolve() first or a SizedIterator as a source")
            last_i_plus1:int = 1 + key
        elif typ is slice:
            if not self._it: # we finished, so we can use any slice
                return True
            key:slice
            start:int = key.start
            stop:int = key.stop
            step:int = key.step
            if step is None:
                step = 1
            if stop is not None:
                if empty_range(start, stop, step): # empty slice
                    print('empty')
                    return True # all collections don't care about empty slices
            if isinstance(it:=self._it, SizedIterator):
                _collection = self._collection
                len_self:int = len(_collection)+len(it)
                last_i_self:int = len_self-1
                positive_start_i = (start if start >= 0 else len_self+start)
                if stop is None: # ex [5:]
                    positive_stop_i = last_i_self
                else:
                    positive_stop_i = ( stop if  stop >= 0 else len_self+stop) + 1 # +1 because stop means length, not stop
                positive_step_i  = ( step if  step >= 0 else -step)
                # if not ((0 <= positive_start_i <= last_i_self) and self._load_indexslice(positive_start_i)): # I'm 99% sure we don't need this since the positive stop is always after
                #     return False
                if not ((0 <= positive_stop_i  <= last_i_self) and self._load_indexslice(positive_stop_i)):
                    return False
                # if (am:=len(_collection)+len(it)+(-start if start > 0 else start)) > 0 and not self.load_am(am):
                #     print(1, am)
                #     return False # we needed to load and couldn't load what we needed
                # if (am:=len(_collection)+len(it)+( -stop if  stop > 0 else  stop)) > 0 and not self.load_am(am):
                #     print(2, am)
                #     return False # we needed to load and couldn't load what we needed
                return True
            if start < 0 or stop < 0 or step < 0:
                raise ValueError(f"Negative indices not supported; If they are necessary, use the .resolve() first or a SizedIterator as a source")
            last_i_plus1:int = 1 + last_of_range(start, stop, step) # emptiness checked above, so not ValueError needs to be caught
        else:
            raise TypeError(f"key must be int or slice not {typ!r}")
        if last_i_plus1 <= (ll:=len(self._collection)):
            return True
        if self.load_am(last_i_plus1 - ll):
            return True # guaranteed to work
        return False
    def __getitem__(self, item:int|slice) -> T:
        if self._load_indexslice(item):
            return self._collection[item]
        else: raise IndexError
    if t.TYPE_CHECKING:
        def resolve(self) -> list[T]: ...
        def for_chain(self) -> list[T]|Chain[T]: ...

class LazyList(t.Generic[T], LazySeq[T], abcs.MutableSequence[T]):
    """Object that allows for a MutableSequence of lazily generated items"""
    __slots__ = __match_args__ = ('_it','_collection', '_collection_add_item', '_collection_extend')
    @t.overload
    def __init__(self, /): ...
    @t.overload
    def __init__(self, it:abcs.Iterator[T], /): ...
    # noinspection PyShadowingBuiltins
    @t.overload
    def __init__(self, list:list[T], it:abcs.Iterator[T], /): ...
    # noinspection PyShadowingBuiltins
    def __init__(self, list_or_it1=None, list_or_it2=None, /):
        if list_or_it1 is list_or_it2 is None:
            list_or_it1 = empty_iterator_from_name('null')
        super().__init__(list_or_it1, list_or_it2)
    def __setitem__(self, key:int|slice, value:T) -> None:
        if self._load_indexslice(key):
            self._collection[key] = value
        else: raise IndexError
    def __delitem__(self, key:int|slice) -> None:
        if self._load_indexslice(key):
            del self._collection[key]
        else: raise IndexError
    # noinspection PyShadowingBuiltins
    def insert(self, index:int, object:T) -> None:
        if index==0:
            self._collection.insert(0, object)
        elif self._load_indexslice(index - 1): # if we don't -1, we will generate 1 extra than we need
            self._collection.insert(index, object)
        else: raise IndexError
    def append(self, obj:T) -> None:
        self._it.append(obj)
    def append_asis(self, obj:T) -> None: self._collection.append(obj)
    def reverse(self) -> None:
        self.resolve()
        self._collection.reverse()
    def extend_asis(self, iterable:abcs.Iterable[T]) -> None: self._collection.extend(iterable)
    def extend(self, iterable:abcs.Iterable[T]) -> None:
        """Extends lazily"""
        self._it.extend(iterable)
    def pop(self, i:t.Optional[int]=None) -> T:
        """If no index is provided, pops the last from the collection or first from the iterator
        If an index is provided, attempts to pop from that index"""
        if i is None:
            return self.resolve().pop()
        if self._load_indexslice(i if i >= 0 else len(self.resolve())+i):
            return self._collection.pop(i)
        else: raise IndexError
    def pop_asis(self, i:t.Optional[int]=None) -> T:
        if i is None:
            return self._collection.pop()
        return self._collection.pop(i)
    def remove(self, obj:T) -> None:
        try: self._collection.remove(obj)
        except ValueError:
            for i,x in self.iter_it():
                if obj==x:
                    self._collection.pop() # it will always be the last since we just added it
                    return
            raise
    def remove_asis(self, obj:T) -> None: return self._collection.remove(obj)
    def __iadd__(self, other:abcs.Iterable[T]) -> None:
        self.extend(other)
    def clear(self) -> None:
        self.resolve()
        self._collection.clear()
    def clear_asis(self) -> None: self._collection.clear()

class LazyDeque(t.Generic[T], LazyList[T]):
    """[Created 3/31/22]"""
    __slots__ = __match_args__ = ('_it','_collection', '_collection_add_item', '_collection_extend')
    @t.overload
    def __init__(self, /): ...
    @t.overload
    def __init__(self, it:abcs.Iterator[T], /): ...
    @t.overload
    def __init__(self, deque:col.deque[T], it:abcs.Iterator[T], /): ...
    def __init__(self, it=None, deque=None, /):
        if it is deque is None:
            it:SuperIterator[T] = EmptyIterator()
            deque:col.deque[T] = col.deque()
        elif deque is None:
            it:abcs.Iterator[T] = it # becomes smart in parent __init__
            deque:col.deque[T] = col.deque()
        else:
            deque:col.deque[T] = it
            it:abcs.Iterator[T] = deque # becomes smart in parent __init__
        LazyCollection.__init__(self, it=it, collection=deque, collection_add_item=deque.append, collection_extend=deque.extend) # we dont use the LazySeq init
        self._collection:col.deque[T] = self._collection
    @classmethod
    def from_it(cls, it:abcs.Iterator[T], /):
        """Automatically set the collection to an empty deque"""
        # noinspection PyShadowingBuiltins
        list:t.List[T] = []
        self = cls.__new__(cls)
        LazyCollection.__init__(self, it=it, collection=deque, collection_add_item=deque.append, collection_extend=deque.extend) # we dont use the LazySeq init
        self._collection:col.deque[T] = self._collection
    if t.TYPE_CHECKING:
        # noinspection PyPropertyDefinition
        @property
        def collection(self) -> col.deque[T]: ...
    def appendleft(self, obj:T) -> None:
        self._collection.appendleft(obj)
    def extendleft(self, iterable:abcs.Iterable[T]) -> None:
        """Consider using an ExtensibleIterator with a LazyDeque (self) suffix instead for lazy loading by doing ExtensibleIterator(reversed(iterable), self)"""
        self._collection.extendleft(iterable)
    def popleft(self) -> T:
        if deque:=self._collection:
            return deque.popleft()
        else:
            if self._it:
                return next(self._it)
            raise ValueError("Empty")
    def popleft_forever(self) -> abcs.Generator[T, t.Any, None]:
        """Optimization over `while lazydeque: yield __deque_popleft()` where going over the iterator, the contents are not saved and immediately popped from the deque"""
        deque = self._collection
        __deque_popleft = deque.popleft
        while deque:
            yield __deque_popleft()
        while self._it:
            yield next(self._it)
    if t.TYPE_CHECKING:
        def resolve(self) -> col.deque[T]: ...
        def for_chain(self) -> col.deque[T]|Chain[T]: ...

# K = t.TypeVar('K') # TODO : make this
# V = t.TypeVar('V')
# class LazyMap(t.Generic[K,V], LazyCollection[tuple[K,V]]):
#     __slots__ = __match_args__ = ('_it','_collection', '_collection_add_item', '_collection_extend')
#     @t.overload
#     def __init__(self, it:abcs.Iterator[T], /): ...
#     # noinspection PyShadowingBuiltins
#     @t.overload
#     def __init__(self, mapping:abcs.MutableMapping[K,V], it:abcs.Iterator[T], /): ...
#     # noinspection PyShadowingBuiltins
#     def __init__(self, dict_or_it1, dict_or_it2=None, /):
#         if dict_or_it2 is None:
#             it:abcs.Iterator[T] = dict_or_it1
#             dict:t.Dict[T] = []
#         else:
#             dict:t.Dict[T] = dict_or_it1
#             it:abcs.Iterator[T] = dict_or_it2
#         super().__init__(it=it, collection=list, collection_add_item=list.append, collection_extend=list.extend)
#         self._collection:t.List[T] = self._collection
