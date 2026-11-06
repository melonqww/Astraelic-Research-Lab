"""Optional static figures: requires matplotlib."""
import json
from analyze import ROOT,compute
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
_,rows=compute(json.loads((ROOT/'evidence/measurements.json').read_text(encoding='utf-8')))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none'})
fig,axes=plt.subplots(1,3,figsize=(12,4.5),sharey=True)
for ax,(split,metric,title) in zip(axes,[('selection_ood','recall_passed','Верные RECALL OOD ↑'),('selection_iid','changed_answers','Изменившийся ответ IID ↓'),('selection_ood','changed_answers','Изменившийся ответ OOD ↓')]):
    for cohort,label,style,color in [('baseline','Контроль','o--','#8b612d'),('permuted_training','Перестановки','s-','#225e89')]:
        group=sorted([x for x in rows if x['split']==split and x['cohort']==cohort],key=lambda x:x['seed'])
        ax.plot([x['seed'] for x in group],[x[metric] for x in group],style,color=color,label=label)
    ax.set(title=title,xlabel='Seed',xticks=[17,29,43],ylim=(0,128));ax.grid(alpha=.15)
axes[0].set_ylabel('Число случаев из 128');axes[0].legend(frameon=False)
fig.suptitle('AR-2 · Правильность и чувствительность к порядку фактов',fontsize=15)
fig.text(.5,.025,'Одна модель, три парных обучения, новая development-панель. Линии соединяют отдельные seed; это не интервалы неопределённости.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.08,1,.92));(ROOT/'figures').mkdir(exist_ok=True)
for ext in ['png','svg']:fig.savefig(ROOT/'figures'/('results.'+ext),dpi=160,facecolor='white')
