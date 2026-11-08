# Benchmark audit



## 2. Duplicate audio

                    dataset  n_clips  n_unique  exact_duplicates  pct_duplicate
    openslr30_speaker_pairs     1920       240              1680      87.500000
    openslr52_speaker_pairs     8000      5411              2589      32.362500
  omnilingual_sin_retrieval      374       373                 1       0.267380
    omnilingual_sin_speaker      415       414                 1       0.240964
omnilingual_sin_speaker_clf      415       414                 1       0.240964
          openslr30_speaker      240       240                 0       0.000000
      openslr30_speaker_clf      240       240                 0       0.000000
           openslr52_rerank     2384      2384                 0       0.000000
          openslr52_speaker     9557      9557                 0       0.000000
        openslr52_retrieval     3056      3056                 0       0.000000
      openslr52_speaker_clf     9557      9557                 0       0.000000
        voxlingua_sin_video     1439      1439                 0       0.000000
    worldspeech_sin_quality     1050      1050                 0       0.000000
    worldspeech_sin_session      924       924                 0       0.000000

  clips present in more than one prepared dataset: 13423
      5329  openslr52_speaker + openslr52_speaker_clf + openslr52_speaker_pairs
      4082  openslr52_speaker + openslr52_speaker_clf
      2274  openslr52_rerank + openslr52_retrieval
       924  worldspeech_sin_quality + worldspeech_sin_session
       240  openslr30_speaker + openslr30_speaker_clf + openslr30_speaker_pairs
       188  omnilingual_sin_retrieval + omnilingual_sin_speaker + omnilingual_sin_speaker_clf
       100  omnilingual_sin_speaker + omnilingual_sin_speaker_clf
        71  omnilingual_sin_retrieval + omnilingual_sin_speaker_clf
        69  omnilingual_sin_retrieval + omnilingual_sin_speaker
        65  openslr52_rerank + openslr52_retrieval + openslr52_speaker + openslr52_speaker_clf + openslr52_speaker_pairs
  Overlap BETWEEN datasets treated as separate corpora would
  undermine cross-corpus claims. Overlap WITHIN one dataset is
  expected where a pair task reuses clips by design.


## 3+4. MFCC shortcut and chance baselines

                                   task           kind    metric    n   k  mfcc_score  mfcc_ami   chance  chance_sd
             OpenSLR52SpeakerClustering     clustering v_measure 9557 478    0.628862  0.265036 0.494919   0.000335
             OpenSLR30SpeakerClustering     clustering v_measure  240  12    0.533890  0.471815 0.117143   0.011769
    OmnilingualSinhalaSpeakerClustering     clustering v_measure  415   7    0.547119  0.535858 0.022826   0.004883
    WorldSpeechSinhalaSessionClustering     clustering v_measure  924  84    0.529349  0.149556 0.446108   0.002082
     VoxLingua107SinhalaVideoClustering     clustering v_measure 1439 173    0.790270  0.518229 0.564153   0.000869
         OpenSLR52SpeakerClassification classification  accuracy 2867 478    0.826299       NaN 0.002093        NaN
         OpenSLR30SpeakerClassification classification  accuracy   72  12    0.972222       NaN 0.083333        NaN
OmnilingualSinhalaSpeakerClassification classification  accuracy  124   7    0.991935       NaN 0.144330        NaN
WorldSpeechSinhalaQualityClassification classification  accuracy  315   3    0.606349       NaN 0.337415        NaN
     OpenSLR52SpeakerPairClassification           pair    max_ap 4000   2    0.831982       NaN 0.500000        NaN
     OpenSLR30SpeakerPairClassification           pair    max_ap  960   2    0.819282       NaN 0.500000        NaN

  mfcc_score uses a 40-dim pooled-MFCC feature with no learned
  component. Close to the encoders means the task is solvable from
  recording characteristics. `chance` is label-permuted V-measure
  (clustering), the majority class (classification), or the positive
  rate (pairs).


## 6. Verdict per task

                                   task    provenance  best_encoder     mfcc   chance    verdict                                                           reason
     OmnilingualSinhalaContentRetrieval         human        0.1043      NaN      NaN       core                                                           passes
    OmnilingualSinhalaSpeakerClustering         human        0.7910 0.547119 0.022826       core                                                           passes
                OpenSLR52AudioReranking       derived        0.1200      NaN      NaN       core                                                           passes
                OpenSLR52AudioRetrieval       derived        0.3671      NaN      NaN       core                                                           passes
OmnilingualSinhalaSpeakerClassification         human        0.9306 0.991935 0.144330 diagnostic                                MFCC reaches 107% of best encoder
         OpenSLR30SpeakerClassification         human        0.9111 0.972222 0.083333 diagnostic                                MFCC reaches 107% of best encoder
             OpenSLR30SpeakerClustering         human        0.3930 0.533890 0.117143 diagnostic                                MFCC reaches 136% of best encoder
     OpenSLR30SpeakerPairClassification       derived        0.8163 0.819282 0.500000 diagnostic                                MFCC reaches 100% of best encoder
         OpenSLR52SpeakerClassification         human        0.7756 0.826299 0.002093 diagnostic                                MFCC reaches 107% of best encoder
             OpenSLR52SpeakerClustering         human        0.6096 0.628862 0.494919 diagnostic                                MFCC reaches 103% of best encoder
     OpenSLR52SpeakerPairClassification       derived        0.8734 0.831982 0.500000 diagnostic                                 MFCC reaches 95% of best encoder
     VoxLingua107SinhalaVideoClustering     automatic        0.9740 0.790270 0.564153 diagnostic                        automatic label: not a speaker capability
WorldSpeechSinhalaQualityClassification derived-by-us        0.4667 0.606349 0.337415 diagnostic MFCC reaches 130% of best encoder; label thresholds chosen by us
    WorldSpeechSinhalaSessionClustering      metadata        0.9294 0.529349 0.446108 diagnostic                         metadata label: not a speaker capability

  core 4 | diagnostic 10 | exclude 0