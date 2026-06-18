# Ch.11 fine-tuned classifier added as a standalone module, not replacing Ch.4

Ch.4 (frozen encoder + LogReg) is kept intact as the TP1 baseline. Ch.11 (fine-tuned MentalBERT) is added as a new independent module. Both produce the same output schema, enabling a direct before/after comparison in the TP2 report. Ch.11 does not feed into Ch.6 or Ch.7 in TP2.

## Considered Options

- **Replace Ch.4**: simpler pipeline, but loses the TP1 baseline for comparison and risks breaking Ch.6/Ch.7 integration.
- **Add Ch.11 standalone** *(chosen)*: zero risk to existing pipeline, clean before/after experiment, Ch.6/Ch.7 integration deferred to future work.

## Consequences

The `pipeline.py` quick/full mode continues to use Ch.4. A separate `evaluate.py` experiment compares Ch.4 vs Ch.11 directly.
