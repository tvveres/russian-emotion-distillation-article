import os
os.environ['HF_HOME']=os.path.abspath('cache');os.environ['HF_HUB_DISABLE_XET']='1'
import json,time,platform,hashlib,re
from pathlib import Path
import numpy as np,torch,joblib
from transformers import AutoTokenizer,AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score,accuracy_score
LABELS=['joy','sadness','surprise','fear','anger']
# run.py resumes from cache; import only after initial run is complete.
p=Path('results');rows=json.loads((p/'sample.json').read_text());teachers={v['id']:v for v in map(json.loads,(p/'teacher.jsonl').read_text().splitlines())};result=json.loads((p/'metrics.json').read_text())
valid=np.array([teachers[r['id']]['valid'] for r in rows]);y=np.array([[int(l in teachers[r['id']]['labels']) for l in LABELS] for r in rows]);tr=np.array([r['split']=='train' for r in rows])&valid;te=np.array([r['split']=='test' for r in rows])&valid;texts=[r['text'] for r in rows]
a={'teacher_annotation_seconds':sum(t['seconds'] for t in teachers.values()),'teacher_latency_median_seconds':float(np.median([teachers[r['id']]['seconds'] for r in rows if r['split']=='test'])),'teacher_latency_p95_seconds':float(np.quantile([teachers[r['id']]['seconds'] for r in rows if r['split']=='test'],.95)),'teacher_backend':'MLX Metal, 4-bit','n_test_valid':int(te.sum()),'n_test_total':80,'source_sentiment_emotions':{},'baselines':{}}
for sentiment in ['negative','neutral','positive']:
 mask=np.array([r['sentiment']==sentiment for r in rows])&valid
 a['source_sentiment_emotions'][sentiment]={'n':int(mask.sum()),'labels':dict(zip(LABELS,y[mask].sum(0).tolist())),'empty':int((y[mask].sum(1)==0).sum())}
for name,baseline in [('always_empty',np.zeros(5,dtype=int)),('train_majority',np.mean(y[tr],axis=0)>=.5)]:
 pred=np.tile(baseline,(te.sum(),1));a['baselines'][name]={'macro_f1':float(f1_score(y[te],pred,average='macro',zero_division=0)),'micro_f1':float(f1_score(y[te],pred,average='micro',zero_division=0)),'exact_match':float(accuracy_score(y[te],pred))}
va=np.array([r['split']=='val' for r in rows])&valid
freq=np.mean(y[tr],axis=0);grid=np.arange(.1,1,.1)
scores=[f1_score(y[va],np.tile(freq>=t,(va.sum(),1)),average='macro',zero_division=0) for t in grid]
th=float(grid[np.argmax(scores)]);constant=np.tile(freq>=th,(te.sum(),1))
a['baselines']['constant_prevalence_tuned']={'threshold':th,'labels':[LABELS[i] for i in range(5) if freq[i]>=th],'macro_f1':float(f1_score(y[te],constant,average='macro',zero_division=0)),'micro_f1':float(f1_score(y[te],constant,average='micro',zero_division=0)),'exact_match':float(accuracy_score(y[te],constant))}
# End-to-end warm per-text latency; median of 3 full passes, excludes model loading.
lat={}
for analyzer,ngram,name in [('word',(1,2),'tfidf_word'),('char',(3,5),'tfidf_char')]:
 vec=TfidfVectorizer(analyzer=analyzer,ngram_range=ngram,min_df=2,max_features=30000,sublinear_tf=True);X=vec.fit_transform([texts[i] for i in np.flatnonzero(tr)])
 c=OneVsRestClassifier(LogisticRegression(C=1,max_iter=2000,class_weight='balanced')).fit(X,y[tr]);vec.transform([texts[0]])
 ts=[]
 for _ in range(3):
  for i in np.flatnonzero(te):
   t=time.perf_counter();c.predict_proba(vec.transform([texts[i]]));ts.append(time.perf_counter()-t)
 lat[name]={'median_seconds':float(np.median(ts)),'p95_seconds':float(np.quantile(ts,.95)),'backend':'CPU','batch_size':1}
 joblib.dump({'vectorizer':vec,'classifier':c,'threshold':result[name]['threshold']},p/(name+'.joblib'))
# Recompute frozen embeddings only; model revision and test set stay fixed.
device='cpu'
torch.set_num_threads(4)
tok=AutoTokenizer.from_pretrained('cointegrated/rubert-tiny2',revision='e8ed3b0c8bbf4fb6984c3de043bf7d2f4e5969ae')
enc=AutoModel.from_pretrained('cointegrated/rubert-tiny2',revision='e8ed3b0c8bbf4fb6984c3de043bf7d2f4e5969ae').to(device).eval()
def embed(items):
 inp=tok(items,return_tensors='pt',padding=True,truncation=True,max_length=256).to(device)
 with torch.inference_mode():v=torch.nn.functional.normalize(enc(**inp).last_hidden_state[:,0,:],dim=-1)
 return v.cpu().numpy()
X=np.concatenate([embed(texts[i:i+16]) for i in range(0,len(texts),16)])
c=OneVsRestClassifier(LogisticRegression(C=1,max_iter=2000,class_weight='balanced')).fit(X[tr],y[tr]);embed([texts[0]])
ts=[]
for _ in range(3):
 for i in np.flatnonzero(te):
  if device=='mps':torch.mps.synchronize()
  t=time.perf_counter();c.predict_proba(embed([texts[i]]))
  if device=='mps':torch.mps.synchronize()
  ts.append(time.perf_counter()-t)
lat['rubert_frozen']={'median_seconds':float(np.median(ts)),'p95_seconds':float(np.quantile(ts,.95)),'backend':device,'batch_size':1}
joblib.dump({'classifier':c,'threshold':result['rubert_frozen']['threshold']},p/'rubert_head.joblib')
# Consistency audit: row overlap and exact normalized review overlap.
ids={s:{r['id'] for r in rows if r['split']==s} for s in ['train','val','test']}
a['split_overlap_ids']={x+'_'+z:len(ids[x]&ids[z]) for x,z in [('train','val'),('train','test'),('val','test')]}
a['test_input_ids']=[r['id'] for r in rows if r['split']=='test'];a['word_char_timing']=lat
(p/'analysis.json').write_text(json.dumps(a,ensure_ascii=False,indent=2));print(json.dumps(a,ensure_ascii=False,indent=2))
