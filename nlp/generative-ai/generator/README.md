# <center> DATA GENERATOR

## ...

## TODO

- count tokens to prevent overflow in context and estimate inference cost
- similarity ratio to eliminate almost identical generated instances
- incorporate few-shot prompting with examples
- incorporate dynamic prompting based on selected label

concept.txt:

```
User´s responses must naturally follow the previous discussion, and align with the provided knowledge.
User’s questions or statements must be unrelated to the provided knowledge, off-topic or unecpected.
User’s inputs must be nonsensical, repetitive, inappropriate or incomprehensible.
```

examples.txt:

```
The following are examples of valid conversations:
{conversation_examples}
```
