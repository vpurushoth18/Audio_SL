# Before submission

## Fix existing inconsistencies
1. MFCC OpenSLR-30 claim: report says 1.13x best encoder, but .443 / .512 = .865 if those scores use identical task samples and AMI. Resolve metric/run/version mismatch.
2. A .443 MFCC versus .848 selected HuBERT comparison does not support 'deep encoder not clearly necessary'.
3. Layer-27 Whisper recommendation cites a preference result measured at the final layer. Supply selected-layer evidence.
4. Speaker classification recommendations cite clustering-selected layers without classification validation.
5. 'Every Whisper ranks worse' and 'every wav2vec2 ranks better' have exceptions in the table.
6. Specify which correlation the MAEB CI column describes, bootstrap unit, repetitions and model-family dependence.

## Supply missing evidence
- Full 13-by-14 score matrix with metrics, denominators, run revisions and seed summaries.
- Full 14-task inventory: capability, corpus, label provenance, split policy and chance/acoustic controls.
- Full 12-model capability-balanced rank table.
- Full content-layer transfer table, with source selection and target evaluation scores.
- Repeated-seed layer gains; uniform layer average baseline and matched MFCC comparison.
- Capability-matched MAEB correlations and uncertainty, rather than speech/non-speech domain labels alone.
- Separate original task aggregation, diagnostic removal, and capability reweighting.
- Clarify exclusion/denominator handling for 282 single-speaker reranking prompts.
- Document unique speakers/prompts/clips for controlled matching and dependencies from reused candidates.
- Freeze immutable checkpoint revisions; correct Whisper encoder-only versus whole-model parameter counts.
- Audit padding/masking/cropping and high unrelated-utterance cosine similarity.

## Interpretation limits
- MFCC competitiveness is not a task-invalidity criterion.
- Same-transcript retrieval is not semantic understanding.
- Non-speech rho .545 versus speech .483 does not establish a difference; estimate uncertainty for the paired difference.
- A task bootstrap measures ranking sensitivity under a chosen resampling scheme, not universal model superiority.
- Distinguish exploratory already-inspected targets from untouched confirmatory data.
