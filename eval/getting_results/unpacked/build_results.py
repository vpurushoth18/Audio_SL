"""Rebuild figures/tables from displayed values in analysis report (3).
No raw predictions, new confidence intervals, or missing results are invented.
Run: python3 build_results.py
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau

ROOT=Path(__file__).resolve().parent
for d in ['figures','tables','data']:(ROOT/d).mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
 'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'axes.labelsize':10,
 'savefig.dpi':300})
BLUE='#23689B'; ORANGE='#C76C22'; GREY='#9AA4AD'; GREEN='#287D64'

def save(fig,name):
 fig.savefig(ROOT/'figures'/f'{name}.pdf',bbox_inches='tight')
 fig.savefig(ROOT/'figures'/f'{name}.png',bbox_inches='tight',dpi=300)
 plt.close(fig)
def csvwrite(name,header,rows):
 with (ROOT/'data'/f'{name}.csv').open('w',newline='') as f:
  w=csv.writer(f); w.writerow(header); w.writerows(rows)
def texwrite(name,text): (ROOT/'tables'/f'{name}.tex').write_text(text)

layer=[('HuBERT-large ASR-ft',3,24,.188,.848),('XLS-R-300M',4,24,.087,.733),
 ('Whisper-large-v3',13,32,.218,.807),('MMS-300M',4,24,.229,.784),
 ('WavLM-large',3,24,.315,.846),('w2v-BERT 2.0',17,24,.512,.810)]
reported_gains=[.660,.646,.589,.554,.531,.298]
csvwrite('layer_transfer',['model','selected_layer','total_layers','final_ami','selected_ami','reported_gain','gain_from_rounded_scores'],
 [(*r,g,round(r[4]-r[3],3)) for r,g in zip(layer,reported_gains)])
fig,ax=plt.subplots(figsize=(6.6,3.05))
y=np.arange(len(layer))[::-1]
for yi,(name,l,n,final,picked) in zip(y,layer):
 ax.plot([final,picked],[yi,yi],color='#CDD4DA',lw=3,zorder=1)
 ax.scatter(final,yi,color=GREY,s=42,zorder=2)
 ax.scatter(picked,yi,color=BLUE,s=42,zorder=2)
 ax.text(picked+.017,yi,f'+{reported_gains[len(layer)-1-yi]:.3f}',va='center',fontsize=8)
ax.scatter([],[],color=GREY,label='Final layer');ax.scatter([],[],color=BLUE,label='Dev-selected layer')
ax.set_yticks(y,[f'{r[0]} (L{r[1]}/{r[2]})' for r in layer])
ax.set_xlim(0,1.04);ax.set_xlabel('Speaker clustering AMI on OpenSLR-30')
ax.legend(loc='upper center',bbox_to_anchor=(.5,1.19),ncol=2,frameon=False,fontsize=8);ax.grid(axis='x',alpha=.15)
fig.tight_layout();save(fig,'fig1_layer_transfer')

pref=[('Whisper-large-v3',.609,.593,.625),('HuBERT-large ASR-ft',.566,.553,.581),
 ('WavLM-large',.431,.416,.446),('XLS-R-300M',.395,.384,.406),
 ('MMS-300M',.282,.271,.294),('w2v-BERT 2.0',.176,.165,.186)]
csvwrite('controlled_matching',['model','probability','reported_ci_low','reported_ci_high'],pref)
fig,ax=plt.subplots(figsize=(6.6,3.05));y=np.arange(len(pref))[::-1]
for yi,(name,p,lo,hi) in zip(y,pref):
 ax.errorbar(p,yi,xerr=[[p-lo],[hi-p]],fmt='o',color=BLUE if p>.5 else ORANGE,capsize=3,ms=5)
 ax.text(hi+.018,yi,f'{p:.3f}',va='center',fontsize=8)
ax.axvline(.5,color='#555555',ls='--',lw=1)
ax.set_yticks(y,[r[0] for r in pref]);ax.set_xlim(0,1)
ax.set_xlabel('P(content match ranks above speaker match)')
ax.text(.13,1.02,'Speaker preference',transform=ax.transAxes,color=ORANGE,ha='center',fontsize=8)
ax.text(.76,1.02,'Content preference',transform=ax.transAxes,color=BLUE,ha='center',fontsize=8)
ax.grid(axis='x',alpha=.15);fig.tight_layout();save(fig,'fig2_controlled_matching')

maeb=[('WavLM-large','SSL',43.58,39.95,56.82,5,1),('XLS-R-2B','SSL',37.80,31.61,56.19,7,2),
 ('Whisper-small','Whisper',44.40,40.67,53.20,3,3),('Whisper-medium','Whisper',46.69,43.03,53.01,1,4),
 ('XLS-R-1B','SSL',34.76,30.11,52.21,11,5),('w2v-BERT 2.0','SSL',35.40,30.28,51.82,9,6),
 ('Whisper-large-v3','Whisper',44.43,40.78,51.79,2,7),('Whisper-base','Whisper',43.69,39.68,51.75,4,8),
 ('Whisper-tiny','Whisper',42.19,38.22,51.50,6,9),('MMS-300M','SSL',34.97,29.03,49.50,10,10),
 ('HuBERT-large ASR-ft','ASR-ft',36.25,35.29,45.16,8,11),('XLS-R-300M','SSL',31.56,27.95,40.79,12,12)]
csvwrite('maeb_comparison',['model','group','maeb_overall_mean','maeb_speech_mean','sinhala_mean','maeb_mean_rank','sinhala_mean_rank'],maeb)
fig,axes=plt.subplots(1,2,figsize=(7.1,3.2),sharey=True)
for ax,idx,title in zip(axes,[2,3],['MAEB overall (19 tasks)','MAEB speech (12 tasks)']):
 for r in maeb:
  color=ORANGE if r[1]=='Whisper' else BLUE
  ax.scatter(r[idx],r[4],color=color,s=35,edgecolor='white',linewidth=.5)
 # Label only separated/high-value points; full values remain in Table 2.
 for name,dx,dy in [('WavLM-large',-5,5),('XLS-R-2B',-5,-13),('XLS-R-300M',5,5),('HuBERT-large ASR-ft',5,-9)]:
  r=next(r for r in maeb if r[0]==name)
  short=name.replace('-large ASR-ft',' ASR-ft')
  ax.annotate(short,(r[idx],r[4]),xytext=(dx,dy),textcoords='offset points',fontsize=6.5)
 rho=spearmanr([r[idx] for r in maeb],[r[4] for r in maeb]).statistic
 ax.set_title(f'{title}\nSpearman rho = {rho:.3f}',fontsize=9)
 ax.set_xlabel('MAEB mean score');ax.grid(alpha=.15);ax.set_ylim(37,61)
 ax.margins(x=.18)
axes[0].set_ylabel('Sinhala mean score (14-task aggregate)')
axes[1].scatter([],[],color=ORANGE,label='Whisper');axes[1].scatter([],[],color=BLUE,label='Other encoders')
axes[1].legend(frameon=False,fontsize=7,loc='lower right')
fig.tight_layout();save(fig,'fig3_maeb_agreement')

fig,ax=plt.subplots(figsize=(7.1,4.6))
for r in maeb:
 color=ORANGE if r[1]=='Whisper' else BLUE
 ax.plot([0,1],[r[5],r[6]],color=color,lw=1.4,alpha=.75)
 ax.scatter([0,1],[r[5],r[6]],color=color,s=22)
 ax.text(-.06,r[5],f'{r[5]}  {r[0]}',ha='right',va='center',fontsize=8)
 ax.text(1.06,r[6],f'{r[6]}  {r[0]}',ha='left',va='center',fontsize=8)
ax.set_xlim(-.03,1.03);ax.set_ylim(12.7,.3)
ax.set_xticks([0,1],['MAEB mean-score rank','Sinhala mean-score rank'])
ax.set_yticks([])
ax.spines['bottom'].set_visible(False);ax.spines['left'].set_visible(False)
fig.subplots_adjust(left=.30,right=.70,bottom=.10,top=.98);save(fig,'fig4_rank_changes')

cross=[('Speaker classification','OpenSLR-52 / OpenSLR-30',.951,.744,1),
 ('Content retrieval','OpenSLR-52 / Omnilingual',.937,.745,1),
 ('Speaker pair classification','OpenSLR-52 / OpenSLR-30',.671,.057,.955),
 ('Speaker clustering','OpenSLR-52 / OpenSLR-30',.210,-.597,.789)]
csvwrite('cross_corpus',['task_family','corpus_pair','rho','reported_ci_low','reported_ci_high'],cross)
fig,ax=plt.subplots(figsize=(6.6,2.7));y=np.arange(len(cross))[::-1]
for yi,r in zip(y,cross):ax.errorbar(r[2],yi,xerr=[[r[2]-r[3]],[r[4]-r[2]]],fmt='o',color=GREEN,capsize=3)
ax.axvline(0,color='#666666',ls='--',lw=1);ax.set_xlim(-1.05,1.05)
ax.set_yticks(y,[r[0] for r in cross]);ax.set_xlabel('Cross-corpus Spearman correlation')
ax.grid(axis='x',alpha=.15);fig.tight_layout();save(fig,'fig5_cross_corpus')

rows='\n'.join(f'{name} & {l}/{n} & {final:.3f} & {pick:.3f} & +{gain:.3f} \\\\' for (name,l,n,final,pick),gain in zip(layer,reported_gains))
texwrite('table1_layer_transfer',r'''\begin{table*}[t]
\centering\small
\begin{tabular}{lrrrr}\toprule
Encoder & Layer & Final AMI & Selected AMI & Gain \\\midrule
'''+rows+r'''
\bottomrule\end{tabular}
\caption{Speaker clustering on OpenSLR-30 after layer selection on OpenSLR-52 development data. Values are transcribed from the report; repeated-seed uncertainty is not supplied. Median absolute improvement is 0.572 AMI. Gains reproduce reported values and can differ from subtraction of rounded scores.}
\label{tab:layer-transfer}\end{table*}
''')
rows='\n'.join(f'{r[0]} & {r[2]:.2f} & {r[3]:.2f} & {r[4]:.2f} & {r[5]} & {r[6]} \\\\' for r in maeb)
texwrite('table2_maeb',r'''\begin{table*}[t]
\centering\small
\begin{tabular}{lrrrrr}\toprule
Encoder & MAEB all & MAEB speech & Sinhala & MAEB rank & Sinhala rank \\\midrule
'''+rows+r'''
\bottomrule\end{tabular}
\caption{Mean scores and mean-score ranks for twelve encoders. MAEB uses the evaluated 19-task beta audio-only configuration; Sinhala uses the original 14-task aggregate. These aggregates have different task composition and are not balanced capability comparisons.}
\label{tab:maeb}\end{table*}
''')
balanced=[('Whisper-small',2.50,6.88,1,6),('WavLM-large',7,2.50,2,1),('Whisper-medium',1,8.88,3,9),
 ('Whisper-large-v3',2.67,7.81,4,10),('w2v-BERT 2.0',10.33,4.12,8,2),('XLS-R-300M',11.17,10.62,12,12)]
csvwrite('balanced_ranking_subset',['model','content_mean_rank','speaker_mean_rank','balanced_rank','plain_borda_rank'],balanced)
rows='\n'.join(f'{r[0]} & {r[1]:.2f} & {r[2]:.2f} & {r[3]} & {r[4]} \\\\' for r in balanced)
texwrite('table3_aggregation_subset',r'''\begin{table}[htbp]
\centering\small
\begin{tabular}{lrrrr}\toprule
Encoder & Content rank & Speaker rank & Balanced rank & Plain Borda \\\midrule
'''+rows+r'''
\bottomrule\end{tabular}
\caption{Selected rows from the reported aggregation comparison (six of twelve encoders). Capability-balanced aggregation excludes recording diagnostics; plain Borda uses the original task set. This comparison combines task removal and reweighting. Complete the full matrix and isolate these effects before final submission.}
\label{tab:aggregation}\end{table}
''')
summary={'overall_rho':spearmanr([r[2] for r in maeb],[r[4] for r in maeb]).statistic,
 'overall_tau':kendalltau([r[2] for r in maeb],[r[4] for r in maeb]).statistic,
 'speech_rho':spearmanr([r[3] for r in maeb],[r[4] for r in maeb]).statistic,
 'median_layer_gain':np.median([r[4]-r[3] for r in layer])}
(ROOT/'data'/'recomputed_statistics.txt').write_text('\n'.join(f'{k}: {v:.6f}' for k,v in summary.items())+'\n')
print('Built 5 figures (PDF + PNG), 3 LaTeX tables, 5 CSV data files, and recomputed statistics.')
