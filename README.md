# SiMTEB-Audio: Sinhala speech embedding benchmark

This repo evaluates frozen speech encoders on Sinhala audio using
[MTEB](https://github.com/embeddings-benchmark/mteb). The task setup follows
[SiMTEB](https://github.com/Sinhala-NLP/SiMTEB).

The tasks subclass MTEB's abstract task classes and models are loaded with
`mteb.get_model`, so all the scoring is done by MTEB itself. Results go into
MTEB's result cache in its normal format.

## Results

12 models x 14 tasks (168 runs) over 5 Sinhala corpora. Scores are x100, same
as the MTEB leaderboard.

| Rank (Borda) | Model | Params | License | Mean (Task) |
|---|---|---|---|---|
| 1 | microsoft/wavlm-large | 0.32B | mit | 56.82 |
| 2 | facebook/w2v-bert-2.0 | 0.58B | mit | 51.82 |
| 3 | facebook/wav2vec2-xls-r-1b | 1.00B | apache-2.0 | 52.21 |
| 4 | facebook/wav2vec2-xls-r-2b | 2.00B | apache-2.0 | 56.20 |
| 5 | openai/whisper-tiny | 0.04B | mit | 51.50 |
| 6 | openai/whisper-small | 0.24B | mit | 53.20 |
| 7 | openai/whisper-base | 0.07B | mit | 51.75 |
| 7 | facebook/mms-300m | 0.32B | cc-by-nc-4.0 | 49.50 |
| 9 | openai/whisper-medium | 0.77B | mit | 53.02 |
| 10 | openai/whisper-large-v3 | 1.55B | mit | 51.79 |
| 11 | facebook/hubert-large-ls960-ft | 0.32B | mit | 45.16 |
| 12 | facebook/wav2vec2-xls-r-300m | 0.30B | apache-2.0 | 40.79 |

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
- Two clustering tasks look saturated. Grouping clips by which recording they
  came from separates the 12 models by under 2.2 points, on two unrelated
  corpora:

  | Task | Clusters | Spread over 12 models |
  |---|---|---|
  | WorldSpeech session | 84 | 1.57 |
  | VoxLingua107 video | 173 | 2.17 |
  | OpenSLR-52 speaker | 476 | 14.05 |
  | Omnilingual speaker | 7 | 17.83 |

  It isn't the number of clusters doing this, since the 7-cluster and
  476-cluster speaker tasks spread the models by 14-18 points. Channel and
  session acoustics just seem easy enough that every encoder picks them up. I'd
  report both of these as saturated rather than as agreement.
- Adding Omnilingual and VoxLingua107 didn't change the ordering, which is
  reassuring since everything before came from OpenSLR. wavlm-large is still
  first, Whisper still wins content retrieval (whisper-medium 10.43 vs 0.53-1.87
  for the wav2vec2 models on Omnilingual), and scale still helps classification
  (xls-r 300m -> 2b goes 46.05 -> 87.66).
- mms-300m wins Omnilingual speaker clustering (79.10) despite sitting mid-table
  overall.

## Datasets

| Corpus | Used for | Size |
|---|---|---|
| [OpenSLR-52](https://openslr.org/52/) | clustering, classification, pair classification, retrieval, reranking | 185,293 utts, 478 speakers, 224.5 h |
| [OpenSLR-30](https://openslr.org/30/) | clustering, classification, pair classification | 2,064 utts, 12 speakers, 3.4 h |
| [WorldSpeech](https://huggingface.co/datasets/disco-eth/WorldSpeech) `si_lk` | session clustering, quality classification | 20,981 segments (Sri Lankan parliament) |
| [Omnilingual ASR](https://huggingface.co/datasets/facebook/omnilingual-asr-corpus) `sin_Sinh` | speaker clustering, classification, content retrieval | 627 utts, **7 speakers**, 12.0 h |
| [VoxLingua107](https://huggingface.co/datasets/TalTechNLP/voxlingua107_wds) `si` | video clustering | 1,439 clips, 173 YouTube videos |

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
| OmnilingualSinhalaSpeakerClustering | AudioClustering | V-measure |
| OmnilingualSinhalaSpeakerClassification | AudioClassification | accuracy |
| OmnilingualSinhalaContentRetrieval | Any2AnyRetrieval (a2a) | hit rate@5 |
| VoxLingua107SinhalaVideoClustering | AudioClustering | V-measure |

The A2T and T2A tasks are set up but haven't been run yet, because they need a
model with a shared audio-text space and all the models so far are audio-only.

Omnilingual only has 7 Sinhala speakers, so its speaker tasks are a lot easier
than the 478-speaker OpenSLR-52 ones. Those columns aren't comparable across
corpora. VoxLingua107's labels are automatic and a video can have more than one
speaker, so its grouping means "same recording", not "same speaker".

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
  worker_new.sh       same, for the Omnilingual + VoxLingua107 tasks
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
python scripts/prepare/prepare_new_corpora.py      # Omnilingual + VoxLingua107

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
- Omnilingual and VoxLingua107 tasks: MTEB has no Sinhala subset for either, so
  both are defined here. Their audio is decoded with `soundfile` straight from
  the cached parquet/tar instead of going through `datasets`' Audio feature.

## Missing Sinhala data

- SIB-FLEURS / FLEURS don't include Sinhala (checked `WueNLP/sib-fleurs`, 102
  configs, and `google/fleurs`, 103). So topic classification, topic clustering
  and zero-shot classification aren't possible right now.
- Common Voice 17 and XTREME-S don't have Sinhala either.
- `ehzawad/sinhala-emotion-dataset` does have audio (2,999 clips, 22.05 kHz) but
  no emotion labels. Its columns are `audio, text, sampling_rate, id`, the ids
  are just sequential, and the transcripts are Buddhist scripture. Fine for ASR,
  not for emotion classification.
- The banking-domain intent dataset behind
  [Karunanayake et al. 2019](https://aclanthology.org/P19-2040/) (7,624 samples,
  7.5 h, 5 intents, 215 speakers) would cover semantic classification, but it
  comes from Buddhika et al. (IALP 2018, University of Moratuwa) and I couldn't
  find a public release. Probably needs an email to the authors.
- Still unused: `jithara/sinhala-emotional-tts-dataset` (499 acted WAVs).
