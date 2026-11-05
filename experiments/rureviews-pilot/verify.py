"""Проверяет целостность выборки и пересчитывает опубликованные метрики из предсказаний."""
import json,re
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score,accuracy_score
p=Path('results');labels=['joy','sadness','surprise','fear','anger']
rows=json.loads((p/'sample.json').read_text());preds=json.loads((p/'test_predictions.json').read_text());metrics=json.loads((p/'metrics.json').read_text())
assert len(rows)==480 and len({r['id'] for r in rows})==480
norm=lambda x:re.sub(r'\s+',' ',x.lower()).strip()
assert len({norm(r['text']) for r in rows})==480
assert {s:sum(r['split']==s for r in rows) for s in ['train','val','test']}=={'train':320,'val':80,'test':80}
t={x['id']:x for x in map(json.loads,(p/'teacher.jsonl').read_text().splitlines())};assert set(t)=={r['id'] for r in rows}
assert {r['id'] for r in preds}=={r['id'] for r in rows if r['split']=='test' and t[r['id']]['valid']}
y=np.array([[l in r['teacher']['labels'] for l in labels] for r in preds])
for name,m in metrics.items():
 z=np.array([[l in r[name] for l in labels] for r in preds])
 for key,value in [('macro_f1',f1_score(y,z,average='macro',zero_division=0)),('micro_f1',f1_score(y,z,average='micro',zero_division=0)),('exact_match',accuracy_score(y,z))]:
  assert abs(m[key]-value)<1e-12,(name,key,m[key],value)
print('OK: 480 разных отзывов, разделение без пересечения, сырые ответы всех моделей, метрики пересчитаны независимо.')
