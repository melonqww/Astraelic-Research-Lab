"""Plot saved metrics; optional matplotlib dependency, no model inference."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
def main():
    scores=list(csv.DictReader((ROOT/'metrics/scores.csv').open(encoding='utf-8')))
    pairs=list(csv.DictReader((ROOT/'metrics/pairs.csv').open(encoding='utf-8')))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'AR-3'})
    fig,axes=plt.subplots(1,2,figsize=(11,4.6))
    for ax,rows,field,title,ylabel in [(axes[0],[r for r in scores if r['panel']=='ordinary' and r['family']=='all'],'correct','Обычная OOD-панель','Изменение верных ответов / 384'),(axes[1],pairs,'both_correct','Пары с изменённым фактом: OOD','Изменение полностью верных пар / 192')]:
        for offset,cohort,color in [(-.12,'transformer','#4169a1'),(.12,'hybrid','#ba6125')]:
            changes=[]
            for i,seed in enumerate(('17','29','43')):
                subset={r['mode']:int(r[field]) for r in rows if r['split']=='selection_ood' and r['cohort']==cohort and r['seed']==seed}
                changes.append(subset['memory']-subset['base'])
            ax.scatter([i+offset for i in range(3)],changes,c=color,s=65,label=cohort,zorder=3)
        ax.axhline(0,c='#777777',lw=1);ax.set_xticks(range(3),['17','29','43']);ax.set_xlabel('Seed модели');ax.set_ylabel(ylabel);ax.set_title(title,pad=14);ax.grid(axis='y',alpha=.18);ax.legend(frameon=False)
    fig.suptitle('AR-3 · kNN-память: выигрыш зависит от проверки',fontsize=16,y=1.02)
    fig.tight_layout();(ROOT/'figures').mkdir(exist_ok=True)
    for ext in ('png','svg'):
        output=ROOT/f'figures/results.{ext}'
        fig.savefig(output,dpi=180,bbox_inches='tight',metadata={'Date':None} if ext=='svg' else None)
        if ext=='svg':output.write_text('\n'.join(line.rstrip() for line in output.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
if __name__=='__main__':main()
