import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
p=Path('results');m=json.loads((p/'metrics.json').read_text());a=json.loads((p/'analysis.json').read_text());audit=json.loads((p/'label_audit.json').read_text());records=json.loads((p/'test_predictions.json').read_text())
labels=['joy','sadness','surprise','fear','anger'];names=['Радость','Грусть','Удивление','Страх','Гнев']
y=np.array([[l in r['teacher']['labels'] for l in labels] for r in records]);pw=np.array([[l in r['tfidf_word'] for l in labels] for r in records]);pb=np.array([[l in r['rubert_frozen'] for l in labels] for r in records]);rng=np.random.default_rng(42);d=[]
for _ in range(2000):
 idx=rng.integers(0,len(y),len(y));d.append(f1_score(y[idx],pb[idx],average='macro',zero_division=0)-f1_score(y[idx],pw[idx],average='macro',zero_division=0))
a['paired_bootstrap_rubert_minus_word_95ci']=np.quantile(d,[.025,.975]).tolist();(p/'analysis.json').write_text(json.dumps(a,ensure_ascii=False,indent=2))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(9,4.6));keys=['tfidf_word','tfidf_char','rubert_frozen'];titles=['TF-IDF слова + LR','TF-IDF символы + LR','ruBERT (заморожен) + LR'];values=[m[k]['macro_f1'] for k in keys];bars=ax.barh(titles,values,color=['#4075a8','#8da8be','#326b56']);ax.invert_yaxis();ax.set_xlim(0,.85);ax.set_xlabel('Macro-F1 относительно псевдометок LLM, 65 тестовых отзывов');ax.set_title('Совпадение классификаторов с метками Qwen',loc='left',pad=20)
for bar,v,k in zip(bars,values,keys):
 lo,hi=m[k]['bootstrap_macro_f1_95ci'];ax.errorbar(v,bar.get_y()+bar.get_height()/2,xerr=[[v-lo],[hi-v]],fmt='none',ecolor='#222',capsize=4);ax.text(hi+.015,bar.get_y()+bar.get_height()/2,f'{v:.3f}',va='center')
ax.axvline(a['baselines']['constant_prevalence_tuned']['macro_f1'],ls='--',c='#a35737',label='Постоянные метки, порог выбран на validation');ax.legend(loc='lower right',fontsize=9);fig.text(.02,.015,'Полосы: bootstrap 95% по объектам теста. Один запуск обучения; метки получены от Qwen.',fontsize=9,color='#444');fig.tight_layout(rect=(0,.05,1,1));fig.savefig(p/'agreement.png',dpi=170);plt.close(fig)
fig,ax=plt.subplots(figsize=(9,4.8));x=np.arange(5);width=.24
for j,k in enumerate(keys):ax.bar(x+(j-1)*width,[m[k]['per_label'][l]['f1'] for l in labels],width,label=titles[j])
ax.set_xticks(x,[f'{name}\nn={audit["per_split"]["test"]["counts"][l]}' for name,l in zip(names,labels)]);ax.set_ylim(0,1);ax.set_ylabel('F1 согласия с LLM');ax.set_title('Редкие эмоции: в тесте только 4 метки страха и 6 — гнева',loc='left',pad=16);ax.legend(fontsize=9);fig.tight_layout();fig.savefig(p/'per_emotion.png',dpi=170);plt.close(fig)
print('Разность ruBERT - TF-IDF:',m['rubert_frozen']['macro_f1']-m['tfidf_word']['macro_f1'],'CI:',a['paired_bootstrap_rubert_minus_word_95ci'])

# Intermediate stages: strict answer filtering and the resulting sample bias.
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
colors = ['#326b56', '#ba6753']
for ax, categories, good, bad, title in [
    (axes[0], ['Train', 'Validation', 'Test'], [265, 65, 65], [55, 15, 15], 'После проверки ответов Qwen'),
    (axes[1], ['Положительные', 'Нейтральные', 'Отрицательные'], [108, 149, 138], [52, 11, 22], 'Фильтрация меняет состав выборки'),
]:
    ax.bar(categories, good, label='Приняты', color=colors[0])
    ax.bar(categories, bad, bottom=good, label='Исключены', color=colors[1])
    for i, (g, b) in enumerate(zip(good, bad)):
        ax.text(i, g / 2, str(g), ha='center', va='center', color='white')
        ax.text(i, g + b / 2, str(b), ha='center', va='center', color='white')
    ax.set_title(title, fontsize=12, loc='left', pad=15)
    ax.set_ylabel('Число отзывов')
    ax.set_ylim(0, max(g+b for g,b in zip(good,bad))*1.12)
    ax.tick_params(axis='x', labelsize=10)
axes[0].legend(loc='upper right', fontsize=9)
fig.text(.02, .015, 'Всего 480 отзывов: 395 приняты, 85 исключены. Исходная тональность — не эталон эмоций.', fontsize=10)
fig.tight_layout(rect=(0,.05,1,1)); fig.savefig(p/'filtering.png', dpi=170); plt.close(fig)

# Compare proportions rather than counts: splits have different sizes.
fig, ax = plt.subplots(figsize=(9, 4.8)); x = np.arange(5)
for j, (split, title, color) in enumerate([('train','Train, n=265','#4075a8'),('val','Validation, n=65','#8da8be'),('test','Test, n=65','#326b56')]):
    part = audit['per_split'][split]
    values = [100*part['counts'][l]/part['valid'] for l in labels]
    ax.bar(x+(j-1)*.25, values, .25, label=title, color=color)
ax.set_xticks(x, names); ax.set_ylim(0,100); ax.set_ylabel('Доля отзывов с меткой, %')
ax.set_title('Qwen редко выбирает страх и гнев', loc='left', pad=18)
ax.legend(fontsize=9)
fig.text(.02,.015,'Псевдометки Qwen: у одного отзыва может быть несколько эмоций; сумма долей не равна 100%.',fontsize=9)
fig.tight_layout(rect=(0,.05,1,1)); fig.savefig(p/'label_distribution.png',dpi=170);plt.close(fig)

fig, ax = plt.subplots(figsize=(9,4.8))
titles_t = ['TF-IDF слова + LR','TF-IDF символы + LR','ruBERT + LR (CPU)','Qwen (MLX / Metal)']
med = [a['word_char_timing'][k]['median_seconds']*1000 for k in keys]+[a['teacher_latency_median_seconds']*1000]
p95 = [a['word_char_timing'][k]['p95_seconds']*1000 for k in keys]+[a['teacher_latency_p95_seconds']*1000]
ypos=np.arange(4)
ax.hlines(ypos,med,p95,color='#777',lw=2)
ax.scatter(med,ypos,color='#326b56',label='Медиана',s=50,zorder=3)
ax.scatter(p95,ypos,color='#ba6753',marker='|',s=130,label='95-й перцентиль',zorder=3)
for i,v in enumerate(med):ax.annotate(f'{v:.2f} мс',(v,i),xytext=(0,12),textcoords='offset points',ha='center',fontsize=10)
ax.set_yticks(ypos,titles_t);ax.invert_yaxis();ax.set_xscale('log');ax.set_xlim(.2,1200);ax.set_ylim(3.6,-.7)
ax.set_xlabel('Время на отзыв, мс — логарифмическая шкала');ax.set_title('Задержка локального запуска после прогрева',loc='left',pad=18)
ax.legend(loc='upper right',fontsize=9)
fig.text(.02,.015,'Apple M1, 8 ГБ; разные среды исполнения. Ученики: 3 × 65 текстов; Qwen: 80 запросов.',fontsize=9)
fig.tight_layout(rect=(0,.05,1,1));fig.savefig(p/'latency.png',dpi=170);plt.close(fig)
