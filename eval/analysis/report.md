# Analysis: 12 models x 14 tasks (168 results)

Within-cell standard deviations available for 117 cells (clustering and classification only).

Excluded from model comparisons and held out as controls: baseline/mfcc-40. See K_baseline_rows.tsv and the audit for the task-validity comparison.

## A. Ranking depends on the aggregation scheme

                                mean_score  mean_rank  borda  mean_rank_by_type
model                                                                          
microsoft/wavlm-large                    1          1      1                  1
facebook/w2v-bert-2.0                    6          2      2                  7
facebook/wav2vec2-xls-r-1b               5          3      3                  9
facebook/wav2vec2-xls-r-2b               2          4      4                  8
openai/whisper-tiny                      9          5      5                  5
openai/whisper-small                     3          6      6                  2
openai/whisper-base                      8          7      7                  4
facebook/mms-300m                       10          7      7                 10
openai/whisper-medium                    4          9      9                  3
openai/whisper-large-v3                  7         10     10                  6
facebook/hubert-large-ls960-ft          11         11     11                 11
facebook/wav2vec2-xls-r-300m            12         12     12                 12

Agreement between schemes (Kendall tau):
  mean_score           vs mean_rank            tau = +0.534
  mean_score           vs borda                tau = +0.534
  mean_score           vs mean_rank_by_type    tau = +0.545
  mean_rank            vs borda                tau = +1.000
  mean_rank            vs mean_rank_by_type    tau = +0.321
  borda                vs mean_rank_by_type    tau = +0.321

Rank stability, resampling the 14 tasks (B=10000):
                                borda_rank  rank_ci_lo  rank_ci_hi  p_is_best
microsoft/wavlm-large                    1         1.0         2.0     0.9680
facebook/w2v-bert-2.0                    2         2.0        11.0     0.0037
facebook/wav2vec2-xls-r-1b               3         2.0        10.0     0.0002
facebook/wav2vec2-xls-r-2b               4         2.0        11.0     0.0048
openai/whisper-tiny                      5         2.0         9.0     0.0000
openai/whisper-small                     6         2.0        10.0     0.0079
openai/whisper-base                      7         3.0         9.0     0.0000
facebook/mms-300m                        7         2.0        11.0     0.0005
openai/whisper-medium                    9         2.0        11.0     0.0145
openai/whisper-large-v3                 10         3.0        11.0     0.0004
facebook/hubert-large-ls960-ft          11         4.0        11.0     0.0000
facebook/wav2vec2-xls-r-300m            12        12.0        12.0     0.0000

  Read this as the headline caveat: with 14 tasks the 95% rank
  interval spans several positions for most models, so only large
  rank gaps are interpretable. P(is best) is the honest statement of
  which model leads.


## A2. Capability-balanced aggregate

Mean RANK within each capability (1 = best); channel tasks are
reported but excluded from the aggregate.

                                channel  content  speaker  capability_balanced_rank  plain_borda_rank
model                                                                                                
openai/whisper-small               8.00     2.50     6.88                         1                 6
microsoft/wavlm-large              2.33     7.00     2.50                         2                 1
openai/whisper-medium              6.17     1.00     8.88                         3                 9
openai/whisper-large-v3            8.50     2.67     7.81                         4                10
openai/whisper-base                7.33     4.17     6.69                         5                 7
openai/whisper-tiny                6.00     4.67     6.62                         6                 5
facebook/wav2vec2-xls-r-2b         8.00     8.67     4.25                         7                 4
facebook/w2v-bert-2.0              5.67    10.33     4.12                         8                 2
facebook/wav2vec2-xls-r-1b         5.00     9.50     5.00                         9                 3
facebook/hubert-large-ls960-ft     5.00     6.00     9.25                        10                11
facebook/mms-300m                  4.67    10.33     5.38                        11                 7
facebook/wav2vec2-xls-r-300m      11.33    11.17    10.62                        12                12

  Kendall tau vs plain Borda = +0.168. 8 of 14 tasks are speaker tasks,
  so a plain aggregate is largely a speaker ranking.


Leave-one-corpus-out:
dropped_corpus  n_tasks_dropped  kendall_vs_full              new_best  max_rank_shift
    OpenSLR-30                3         0.542652 microsoft/wavlm-large               6
  VoxLingua107                1         0.707692 microsoft/wavlm-large               4
    OpenSLR-52                5         0.718838 microsoft/wavlm-large               4
   Omnilingual                3         0.748113 microsoft/wavlm-large               5
   WorldSpeech                2         0.748113 microsoft/wavlm-large               3

  If new_best changes when a corpus is dropped, the headline
  recommendation is corpus-dependent and should be a shortlist.


## B. Task redundancy

Most correlated task pairs (Spearman over 12 models, 95% CI):
  +0.95   [+0.70, +1.00]  OpenSLR30SpeakerClassification ~ OpenSLR52SpeakerClassification
  +0.94   [+0.72, +1.00]  OmnilingualSinhalaContentRetrieval ~ OpenSLR52AudioRetrieval
  +0.92   [+0.66, +1.00]  OpenSLR52AudioReranking ~ OpenSLR52AudioRetrieval
  +0.90   [+0.52, +1.00]  OmnilingualSinhalaContentRetrieval ~ OpenSLR52AudioReranking
  +0.85   [+0.47, +0.98]  OmnilingualSinhalaSpeakerClustering ~ OpenSLR30SpeakerPairClassification
  +0.77   [+0.23, +0.98]  OmnilingualSinhalaSpeakerClassification ~ OpenSLR30SpeakerClassification
  +0.70   [+0.11, +0.96]  OmnilingualSinhalaSpeakerClustering ~ WorldSpeechSinhalaQualityClassification
  +0.70   [+0.20, +0.96]  OmnilingualSinhalaSpeakerClassification ~ OpenSLR30SpeakerPairClassification

Least correlated:
  -0.55   [-0.94, +0.07]  OmnilingualSinhalaContentRetrieval ~ OmnilingualSinhalaSpeakerClustering
  -0.63   [-1.00, +0.07]  OmnilingualSinhalaSpeakerClustering ~ OpenSLR52AudioReranking
  -0.88   [-1.00, -0.55]  OpenSLR52AudioReranking ~ OpenSLR52SpeakerClustering
  -0.91   [-1.00, -0.63]  OmnilingualSinhalaContentRetrieval ~ OpenSLR52SpeakerClustering
  -0.94   [-0.99, -0.72]  OpenSLR52AudioRetrieval ~ OpenSLR52SpeakerClustering

  Median CI width is 1.22. With 12 models these correlations cannot
  separate 0.6 from 0.95, so they support 'these tasks behave similarly',
  not a precise redundancy estimate.

Leave-one-task-out influence on the overall ranking:
                                   task               task_type  kendall_vs_full  max_rank_shift
     OpenSLR30SpeakerPairClassification AudioPairClassification         0.707692               4
     VoxLingua107SinhalaVideoClustering         AudioClustering         0.707692               4
             OpenSLR52SpeakerClustering         AudioClustering         0.708859               4
OmnilingualSinhalaSpeakerClassification     AudioClassification         0.717578               4
    OmnilingualSinhalaSpeakerClustering         AudioClustering         0.717578               4
         OpenSLR52SpeakerClassification     AudioClassification         0.748113               3
WorldSpeechSinhalaQualityClassification     AudioClassification         0.748113               3
         OpenSLR30SpeakerClassification     AudioClassification         0.748113               4
             OpenSLR30SpeakerClustering         AudioClustering         0.809184               4
     OpenSLR52SpeakerPairClassification AudioPairClassification         0.870254               3
    WorldSpeechSinhalaSessionClustering         AudioClustering         0.870254               2
     OmnilingualSinhalaContentRetrieval          AudioRetrieval         0.900790               2
                OpenSLR52AudioRetrieval          AudioRetrieval         0.900790               2
                OpenSLR52AudioReranking          AudioReranking         0.900790               2

4 task groups (average linkage on 1-|rho|):
  group 1: VoxLingua107SinhalaVideoClustering, WorldSpeechSinhalaSessionClustering
  group 2: OmnilingualSinhalaContentRetrieval, OpenSLR52AudioReranking, OpenSLR52AudioRetrieval, OpenSLR52SpeakerClustering
  group 3: OmnilingualSinhalaSpeakerClassification, OpenSLR30SpeakerClassification, OpenSLR52SpeakerClassification
  group 4: OmnilingualSinhalaSpeakerClustering, OpenSLR30SpeakerClustering, OpenSLR30SpeakerPairClassification, OpenSLR52SpeakerPairClassification, WorldSpeechSinhalaQualityClassification

5 task groups (average linkage on 1-|rho|):
  group 1: VoxLingua107SinhalaVideoClustering
  group 2: WorldSpeechSinhalaSessionClustering
  group 3: OmnilingualSinhalaContentRetrieval, OpenSLR52AudioReranking, OpenSLR52AudioRetrieval, OpenSLR52SpeakerClustering
  group 4: OmnilingualSinhalaSpeakerClassification, OpenSLR30SpeakerClassification, OpenSLR52SpeakerClassification
  group 5: OmnilingualSinhalaSpeakerClustering, OpenSLR30SpeakerClustering, OpenSLR30SpeakerPairClassification, OpenSLR52SpeakerPairClassification, WorldSpeechSinhalaQualityClassification

6 task groups (average linkage on 1-|rho|):
  group 1: VoxLingua107SinhalaVideoClustering
  group 2: WorldSpeechSinhalaSessionClustering
  group 3: OmnilingualSinhalaContentRetrieval, OpenSLR52AudioReranking, OpenSLR52AudioRetrieval, OpenSLR52SpeakerClustering
  group 4: OmnilingualSinhalaSpeakerClassification, OpenSLR30SpeakerClassification, OpenSLR52SpeakerClassification
  group 5: OmnilingualSinhalaSpeakerClustering, OpenSLR30SpeakerPairClassification, WorldSpeechSinhalaQualityClassification
  group 6: OpenSLR30SpeakerClustering, OpenSLR52SpeakerPairClassification


## C. Task saturation

                                   task               task_type   spread  median_within_cell_sd  spread_over_noise
    WorldSpeechSinhalaSessionClustering         AudioClustering 0.015628               0.005837           2.677403
     VoxLingua107SinhalaVideoClustering         AudioClustering 0.021668               0.001865          11.618231
     OmnilingualSinhalaContentRetrieval          AudioRetrieval 0.098930                    NaN                NaN
                OpenSLR52AudioReranking          AudioReranking 0.119830                    NaN                NaN
WorldSpeechSinhalaQualityClassification     AudioClassification 0.122857               0.040624           3.024241
             OpenSLR52SpeakerClustering         AudioClustering 0.140460               0.003427          40.980306
    OmnilingualSinhalaSpeakerClustering         AudioClustering 0.178298               0.052346           3.406176
             OpenSLR30SpeakerClustering         AudioClustering 0.188622               0.015535          12.141744
     OpenSLR30SpeakerPairClassification AudioPairClassification 0.222531                    NaN                NaN
     OpenSLR52SpeakerPairClassification AudioPairClassification 0.251196                    NaN                NaN
                OpenSLR52AudioRetrieval          AudioRetrieval 0.361260                    NaN                NaN
OmnilingualSinhalaSpeakerClassification     AudioClassification 0.470161               0.034113          13.782527
         OpenSLR30SpeakerClassification     AudioClassification 0.563889               0.046577          12.106514
         OpenSLR52SpeakerClassification     AudioClassification 0.706976               0.005584         126.616779

V-measure vs AMI on clustering tasks (chance correction):
                               task  v_measure      ami  inflation  v_spread  ami_spread
         OpenSLR52SpeakerClustering   0.522266 0.276810   0.245456  0.140460    0.131643
         OpenSLR30SpeakerClustering   0.312666 0.311602   0.001064  0.188622    0.188907
 VoxLingua107SinhalaVideoClustering   0.962574 0.961820   0.000754  0.021668    0.022108
WorldSpeechSinhalaSessionClustering   0.919521 0.918926   0.000595  0.015628    0.015745
OmnilingualSinhalaSpeakerClustering   0.693945 0.693755   0.000190  0.178298    0.178418

  AMI is chance-corrected, V-measure is not. A large gap
  means the V-measure headline is driven by cluster count
  rather than by the embedding.


## D. Model scale

 family               task_type  n_sizes  spearman_params_vs_score  smallest  largest     delta
Whisper     AudioClassification        5                       0.6  0.520521 0.539491  0.018969
Whisper         AudioClustering        5                      -0.7  0.672695 0.663956 -0.008739
Whisper AudioPairClassification        5                      -0.5  0.686279 0.662717 -0.023563
Whisper          AudioReranking        5                       0.7  0.075410 0.084010  0.008600
Whisper          AudioRetrieval        5                       0.8  0.158230 0.181605  0.023375
  XLS-R     AudioClassification        3                       1.0  0.305040 0.752502  0.447463
  XLS-R         AudioClustering        3                       0.5  0.650885 0.688316  0.037431
  XLS-R AudioPairClassification        3                       1.0  0.610677 0.689717  0.079040
  XLS-R          AudioReranking        3                       1.0  0.001030 0.002870  0.001840
  XLS-R          AudioRetrieval        3                       1.0  0.006955 0.016725  0.009770

  Whisper has 5 sizes and XLS-R has 3. A rank correlation on 3-5
  points cannot be significant, so these are reported as direction and
  magnitude only. The interpretable claim is whether the SIGN is
  consistent across task types within a family, not the coefficient.


## E. Pre-training language

English-only: 2 models ['microsoft/wavlm-large', 'facebook/hubert-large-ls960-ft']
Multilingual: 10 models
  The groups are 2 vs 10. No two-sample test on n=2 is
  informative, so this is reported as where the English-only
  models RANK inside each task type.

              task_type                          model  rank  of_models    score  best_multilingual  gap_to_best_multi
    AudioClassification          microsoft/wavlm-large     2         12 0.653520           0.752502          -0.098982
    AudioClassification facebook/hubert-large-ls960-ft    11         12 0.362413           0.752502          -0.390089
        AudioClustering          microsoft/wavlm-large     3         12 0.720828           0.725195          -0.004367
        AudioClustering facebook/hubert-large-ls960-ft     6         12 0.674567           0.725195          -0.050628
AudioPairClassification          microsoft/wavlm-large     2         12 0.754984           0.844885          -0.089901
AudioPairClassification facebook/hubert-large-ls960-ft    11         12 0.627738           0.844885          -0.217147
         AudioReranking          microsoft/wavlm-large     7         12 0.041640           0.119960          -0.078320
         AudioReranking facebook/hubert-large-ls960-ft     6         12 0.044190           0.119960          -0.075770
         AudioRetrieval          microsoft/wavlm-large     7         12 0.092615           0.235715          -0.143100
         AudioRetrieval facebook/hubert-large-ls960-ft     6         12 0.099925           0.235715          -0.135790

speaker tasks: English-only ranks ['wavlm-large=2', 'hubert-large-ls960-ft=11'] of 12

content tasks: English-only ranks ['wavlm-large=7', 'hubert-large-ls960-ft=6'] of 12


## F. Speaker vs content trade-off

                                speaker_mean_rank  content_mean_rank
model                                                               
microsoft/wavlm-large                      2.5000           7.000000
facebook/w2v-bert-2.0                      4.1250          10.333333
facebook/wav2vec2-xls-r-2b                 4.2500           8.666667
facebook/wav2vec2-xls-r-1b                 5.0000           9.500000
facebook/mms-300m                          5.3750          10.333333
openai/whisper-tiny                        6.6250           4.666667
openai/whisper-base                        6.6875           4.166667
openai/whisper-small                       6.8750           2.500000
openai/whisper-large-v3                    7.8125           2.666667
openai/whisper-medium                      8.8750           1.000000
facebook/hubert-large-ls960-ft             9.2500           6.000000
facebook/wav2vec2-xls-r-300m              10.6250          11.166667

Spearman(speaker rank, content rank) = -0.319 [-0.91, +0.40], permutation p = 0.3076
  The CI includes zero. With 12 models this analysis cannot
  establish a trade-off either way; report it as inconclusive
  and lean on the per-family pattern instead.


## G. Cross-dataset robustness

                            task_a                                  task_b  n_models  spearman     ci_lo    ci_hi
        OpenSLR52SpeakerClustering              OpenSLR30SpeakerClustering        12  0.209790 -0.597207 0.789148
        OpenSLR52SpeakerClustering     OmnilingualSinhalaSpeakerClustering        12  0.622378  0.032252 0.971631
        OpenSLR30SpeakerClustering     OmnilingualSinhalaSpeakerClustering        12  0.475524 -0.254613 0.879910
    OpenSLR52SpeakerClassification          OpenSLR30SpeakerClassification        12  0.951049  0.743590 1.000000
    OpenSLR52SpeakerClassification OmnilingualSinhalaSpeakerClassification        12  0.647987  0.010769 0.957299
OpenSLR52SpeakerPairClassification      OpenSLR30SpeakerPairClassification        12  0.671329  0.057078 0.955232
           OpenSLR52AudioRetrieval      OmnilingualSinhalaContentRetrieval        12  0.936842  0.744992 1.000000

  Same task family, different corpus. High correlation means a
  model's standing transfers; low means the corpus matters more
  than the model, which would undermine single-corpus claims.