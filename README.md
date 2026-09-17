# DecisionBench

Measurements of how models return **bounded decisions**, and what changes when you
ask for one a different way.

Maintained alongside [decisionmodels.ai](https://decisionmodels.ai), an independent
field guide to the category. Not affiliated with TypeSafe AI or with any project
measured here.

## Decision Output Paths v0.1

One open model, one classification task, three ways of asking for the same answer:

| Path | What it does | p50 | Output tokens |
|---|---|---|---|
| `generated` | Ask for JSON in free text, parse whatever comes back | 97.7 ms | 11 |
| `constrained` | Server-enforced decoding into a JSON schema | 74.4 ms | 7 |
| `direct` | Read the log-probabilities of the option tokens, decode nothing | **24.6 ms** | 1 |

Schema validity was **100% on all three paths**, so on this task constrained
decoding bought nothing over parsing the generated output.

### The part worth knowing

The paths do not agree with each other:

| Comparison | Agreement |
|---|---|
| constrained vs direct | 76% |
| generated vs constrained | 68% |
| generated vs direct | **48%** |

Same model, same inputs, same four declared options. How you ask changes the answer
roughly a quarter to half of the time. Moving a step from generation to direct
scoring is not a free optimisation; it is a change in behaviour that deserves its own
evaluation.

### The disagreement is not symmetric

Answer distribution across all 75 calls per path:

| Path | technical | sales | other | billing |
|---|---|---|---|---|
| `generated` | 44% | 20% | 32% | 4% |
| `constrained` | 72% | 16% | 12% | 0% |
| `direct` | **92%** | 8% | 0% | **0%** |

The direct readout collapsed onto one label and never once answered `billing`, on a
set that includes *"My card was charged twice for the same order."* and *"Charged in
USD but our contract says AUD."*

So the fastest path was also the most degenerate. The honest reading is that a 0.6B
model is not good at this task by any route, and latency measured on a model that is
answering badly is not latency you can spend. Whether a purpose-trained decision
model avoids this collapse is the open question, and answering it needs labelled data.

### What this does not measure

- **Not accuracy.** There are no ground-truth labels. "Disagreement" means the paths
  differ, not that any one of them is right.
- **Not calibration.** A normalised distribution over declared options is not a
  calibrated probability. Measuring that needs labelled data and a reliability
  analysis. See [Guo et al., 2017](https://arxiv.org/abs/1706.04599).
- **Not hosted APIs.** Latency here is a local model on one laptop. It says nothing
  about Jev or any other hosted service.
- **One task, one model.** Support-ticket categorisation into four options.

## Reproducing it

Needs [Ollama](https://ollama.com) with logprobs support (v0.12.11+) and the model:

```bash
ollama pull qwen3:0.6b
python3 output-paths.py --model qwen3:0.6b --repeats 3
```

No dependencies beyond the Python standard library. Writes `results.jsonl` (one row
per call) and `results-summary.json`.

### The run published here

Qwen3 0.6B (751.63M parameters, Q4_K_M) through Ollama on an Apple Silicon laptop.
25 fixed inputs, 4 declared categories, 3 repeats after 3 warm-up rounds,
temperature 0, 225 timed calls. Raw data in [`results/`](results/).

## Contributing

Corrections and additional runs are welcome, particularly:

- other models, especially encoder-based ones (GLiNER2, GLiGuard, classifier heads)
- other tasks, especially ones with ground-truth labels so accuracy can be measured
- a calibration study, which is the obvious next thing and is not done here

Open an issue or a pull request. If you think a number here is wrong, say so with a
reproduction and it gets fixed.

## Licence

MIT. Do what you like with it.
