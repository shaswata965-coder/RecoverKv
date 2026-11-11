"""LongGenBench (Wu et al., ICLR 2025): data, judge and scoring, as the paper ran them.

A self-contained harness: it imports nothing from the repository around it, so it
can be copied next to any other generation code and score that code's outputs
the same way. The metric modules need only the standard library; the judge needs
``vllm`` (install ``requirements-judge.lock.txt``). ``README.md`` has the input format and
the commands.
"""

__version__ = "1.0.0"
