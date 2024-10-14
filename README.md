# giter
High level iterators for Python.

- Intelligent iterators which have known size and are extensible.
- Lazy-loading collections (list, deque, etc.).
- Chain iterators


## Lazy Collections
Lazy lists and deques are added,
    which under the hood are just lists and deques
    with an iterator as source.
Data is lazy loaded as needed.
These new types incur a speed loss but are still practical


## Chains
Chain and sized chain objects are added which remember their source iterators. They contain a deque which contains the iterators to be chained.
This makes them useful in two cases:
- When the chain needs to be modified.
- When the chain needs to incorporate other chains (the other chains are unrolled, unlike what `itertools.chain` does).

These are the most practical new objects since iteration uses `itertools.chain`, so iteration has no speed loss over just using `itertools.chain`.


## Super Iterators
"Super Iterators" are high level wrappers around standard iterators which allow for additional functionality.
They automatically coerce between other super iterator types to gain or lose functionality.
The primary additional functions are sized iterators and extensible iterators.
Additionally, all super iterators become empty iterators when exhausted.
Dead iterators are also added which can never become nonempty.

These are useful in cases where speed is not needed at all and ease of coding is of foremost concern.
In all cases though these are completely unneeded, however they are still nice to work with.

Inheritance tree (`<v>` indicates "virtual" inheritance):
```
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
```

For now they still might have bugs due to the convoluted complexity introduced with the dynamic class conversion.
