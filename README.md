# SiMTEB-Audio: Sinhala speech embedding benchmark

This repo evaluates frozen speech encoders on Sinhala audio using
[MTEB](https://github.com/embeddings-benchmark/mteb). The task setup follows
[SiMTEB](https://github.com/Sinhala-NLP/SiMTEB).

The tasks subclass MTEB's abstract task classes and models are loaded with
`mteb.get_model`, so all the scoring is done by MTEB itself. Results go into
MTEB's result cache in its normal format.

## Results

12 models x 10 tasks (120 runs). Scores are x100, same as the MTEB leaderboard.

| Rank (Borda) | Model | Params | License | Mean (Task) |
|---|---|---|---|---|
| 1 | microsoft/wavlm-large | 0.32B | mit | 52.50 |
| 2 | facebook/mms-300m | 0.32B | cc-by-nc-4.0 | 44.25 |
| 2 | facebook/w2v-bert-2.0 | 0.58B | mit | 47.12 |
| 4 | facebook/wav2vec2-xls-r-2b | 2.00B | apache-2.0 | 52.61 |
| 5 | openai/whisper-base | 0.07B | mit | 47.69 |
| 6 | openai/whisper-tiny | 0.04B | mit | 46.87 |
| 7 | facebook/wav2vec2-xls-r-1b | 1.00B | apache-2.0 | 47.49 |
| 8 | openai/whisper-small | 0.24B | mit | 49.51 |
| 9 | openai/whisper-medium | 0.77B | mit | 49.38 |
| 10 | openai/whisper-large-v3 | 1.55B | mit | 47.54 |
| 11 | facebook/hubert-large-ls960-ft | 0.32B | mit | 39.86 |
| 12 | facebook/wav2vec2-xls-r-300m | 0.30B | apache-2.0 | 36.53 |

Borda rank and mean don't fully agree. xls-r-2b has the highest mean, but it's
last on retrieval and reranking, which pulls its rank down.

Per-task scores are in `eval/dashboard.html` (open it in a browser) and in
`eval/simteb_audio_results_per_task.tsv`.

Some observations:

- wavlm-large comes first even though it was only pretrained on English. My
  guess is that its speaker-oriented training matters more here than
  multilingual pretraining, since most of the tasks are speaker tasks.
- Whisper models do much better on retrieval/reranking (whisper-medium gets
  36.71 / 12.00) while all the wav2vec2 models are close to zero (0.59-2.09 on
  retrieval). On the speaker tasks it's the other way around. This makes sense
  since Whisper is trained for ASR and keeps content, and wav2vec2 seems to keep
  more speaker info.
- Going from xls-r 300m -> 1b -> 2b, classification goes 25.32 -> 54.15 -> 71.11,
  but clustering barely changes (55.51 -> 58.54 -> 57.69).
- WorldSpeechSinhalaSessionClustering gives every model 91.4-92.9, even though
  the models differ ~50x in size. V-measure isn't chance-corrected and there are
  84 clusters over 924 segments, so I don't think this task is useful as-is.

## Datasets

| Corpus | Used for | Size |
|---|---|---|
| [OpenSLR-52](https://openslr.org/52/) | clustering, classification, pair classification, retrieval, reranking | 185,293 utts, 478 speakers, 224.5 h |
| [OpenSLR-30](https://openslr.org/30/) | clustering, classification, pair classification | 2,064 utts, 12 speakers, 3.4 h |
| [WorldSpeech](https://huggingface.co/datasets/disco-eth/WorldSpeech) `si_lk` | session clustering, quality classification | 20,981 segments (Sri Lankan parliament) |

The eval subsets are balanced per label (max 20 utterances per speaker, 20
segments per session). Sizes above were counted from `data/manifest.tsv`.

The raw data (~30 GB) and prepared datasets (~13 GB) are not in the repo; see
the steps below to rebuild them.

## Tasks

| Task | Type | Metric |
|---|---|---|
| OpenSLR52SpeakerClustering | AudioClustering | V-measure |
| OpenSLR30SpeakerClustering | AudioClustering | V-measure |
| WorldSpeechSinhalaSessionClustering | AudioClustering | V-measure |
| OpenSLR52SpeakerClassification | AudioClassification | accuracy |
| OpenSLR30SpeakerClassification | AudioClassification | accuracy |
| WorldSpeechSinhalaQualityClassification | AudioClassification | accuracy |
| OpenSLR52SpeakerPairClassification | AudioPairClassification | max AP |
| OpenSLR30SpeakerPairClassification | AudioPairClassification | max AP |
| OpenSLR52AudioRetrieval | Any2AnyRetrieval (a2a) | hit rate@5 |
| OpenSLR52AudioReranking | Any2AnyRetrieval (a2a) | MAP@5 |
| OpenSLR52A2TRetrieval | Any2AnyRetrieval (a2t) | hit rate@5 |
| OpenSLR52T2ARetrieval | Any2AnyRetrieval (t2a) | hit rate@5 |

The A2T and T2A tasks are set up but haven't been run yet, because they need a
model with a shared audio-text space and all the models so far are audio-only.

For retrieval and reranking, same-speaker matches are excluded. The correct
answer is the same sentence read by a different speaker, and the hard negatives
for reranking are other utterances from the query speaker. Otherwise a model
could score well just by matching the voice.

## Repo layout

```
simteb_audio/
  src/simteb_audio/
    tasks/            task definitions, grouped by type
    models/           ModelMeta for models not in MTEB
    registry.py       TASK_REGISTRY, CATEGORY_REGISTRY, get_task(s)
    benchmark.py      default task set
  scripts/prepare/    builds the eval datasets from data/manifest.tsv
  evaluate.py         run one model on one or more tasks
  mteb_cache/         MTEB result JSONs
eval/
  extract.py          per-layer pooled embeddings
  cluster.py          cluster-structure metrics
  collect_mteb_results.py   result JSONs -> tables
  build_dashboard.py  tables -> dashboard.html
jobs/
  worker.sh           runs all tasks for a list of models on one GPU
```

## Running it

```bash
# 1. download data (~22 GB, only needed once)
./fetch_openslr52.sh && ./unpack_openslr52.sh && python build_manifest.py

# 2. build eval datasets
cd simteb_audio
python scripts/prepare/prepare_openslr_speaker.py --encoding array --out data_run
python scripts/prepare/prepare_speaker_tasks.py
python scripts/prepare/prepare_retrieval.py
python scripts/prepare/prepare_worldspeech.py

# 3. evaluate (I used two GPUs with different model lists)
cd ..
mkdir -p logs
setsid nohup jobs/worker.sh 0 microsoft/wavlm-large openai/whisper-tiny \
  > logs/gpu0.log 2>&1 < /dev/null &

# 4. make tables and dashboard
python eval/collect_mteb_results.py \
    --cache simteb_audio/mteb_cache --out eval/simteb_audio_results
python eval/build_dashboard.py
```

MTEB skips results that are already in the cache, so an interrupted run can
just be restarted.

## Cluster-specific workarounds

- torchcodec wouldn't install on our cluster (the current build needs CUDA 13
  and older ones need FFmpeg shared libs we don't have). Because of that the
  prepared datasets store decoded float32 arrays (`--encoding array`), and
  `prepare_worldspeech.py` reads the parquet with pyarrow and decodes the
  Ogg/Opus audio with soundfile. There's also `--encoding audio`, which writes a
  normal HF `Audio` column, but I can't decode those on this machine.
- fp16 overflowed in the deeper layers of the wav2vec2-family models (for
  xls-r-300m, layers 21-23; at layer 23 it was 8,980 of 13,219 utterances).
  `eval/extract.py` uses bf16 by default to avoid this.

## Things added on top of MTEB

- `facebook/mms-300m`: MTEB only has the 1B MMS checkpoints.
- `facebook/w2v-bert-2.0`: not supported in MTEB, so I wrote a wrapper (it takes
  filterbank features instead of raw waveform).
- MTEB's `task_subtypes` doesn't have "Speaker Clustering", so I left that field
  empty.

## Missing Sinhala data

- SIB-FLEURS / FLEURS don't include Sinhala (checked `WueNLP/sib-fleurs`, 102
  configs, and `google/fleurs`, 103). So topic classification, topic clustering
  and zero-shot classification aren't possible right now.
- Common Voice 17 and XTREME-S don't have Sinhala either.
- Possible datasets to add later: `facebook/omnilingual-asr-corpus` `sin_Sinh`
  (627 utts, has `speaker_id` and `prompt_id`),
  `jithara/sinhala-emotional-tts-dataset` (499 acted WAVs), VoxLingua107.
