"""Render the two comparisons from reviewed CSV, without model dependencies."""
from pathlib import Path
import csv,math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
ROOT=Path(__file__).resolve().parents[1]
BLUE='#4169a1';ORANGE='#ba6125'

def finish(fig,name):
    # Small research mark anchored to the header; no separate asset dependency.
    for angle in range(0,360,72):
        rad=math.radians(angle)
        fig.add_artist(Circle((.975+.006*math.cos(rad),.956+.011*math.sin(rad)),.004,transform=fig.transFigure,color=ORANGE,clip_on=False))
    fig.add_artist(Circle((.975,.956),.002,transform=fig.transFigure,color='#444444',clip_on=False))
    (ROOT/'figures').mkdir(exist_ok=True)
    for ext in ('png','svg'):
        p=ROOT/f'figures/{name}.{ext}'
        fig.savefig(p,dpi=180,bbox_inches='tight',metadata={'Date':None} if ext=='svg' else None)
        if ext=='svg':p.write_text('\n'.join(s.rstrip() for s in p.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    plt.close(fig)

def main():
    rows=list(csv.DictReader((ROOT/'metrics/criteria.csv').open(encoding='utf-8')))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.hashsalt':'AR-4'})
    fig,axes=plt.subplots(1,2,figsize=(11,4.9));fig.subplots_adjust(top=.72,bottom=.17,wspace=.35)
    subset=[r for r in rows if r['screen']=='gem64'];names=['SGD','GEM8','GEM64'];y=list(range(3))
    for ax,key,scale,title,xlabel in [(axes[0],'worst_old_delta',1e4,'Ухудшение худшего старого домена','ΔNLL × 10⁴ (меньше — лучше)'),(axes[1],'arithmetic_delta',-1.,'Улучшение арифметики','−ΔNLL (больше — лучше)')]:
        values=[float(r[key])*scale for r in subset]
        ax.barh(y,values,color=BLUE,height=.52);ax.set_yticks(y,names);ax.invert_yaxis();ax.set_xlim(0,max(values)*(1.35 if scale==-1. else 1.23))
        for i,v in enumerate(values):ax.text(v+max(values)*.025,i,f'{v:.6f}' if scale==-1. else f'{v:.3f}',va='center',fontsize=11)
        ax.set_xlabel(xlabel);ax.set_title(title,pad=13);ax.grid(axis='x',alpha=.16);ax.set_axisbelow(True)
    axes[0].axvline(1,c='#555555',ls='--',lw=1.2);axes[0].text(1.1,1.5,'Допуск: 1',fontsize=10,va='center')
    fig.suptitle('AR-4 · увеличение памяти GEM: 8 → 64 окна',x=.05,y=.96,ha='left',fontsize=16)
    fig.text(.05,.84,'Одна панель seed173030 · 8 окон на домен · худший старый домен во всех ветках — English',fontsize=10)
    finish(fig,'gem-memory')

    fig,axes=plt.subplots(1,2,figsize=(11,5.1));fig.subplots_adjust(top=.69,bottom=.16,wspace=.34)
    for offset,variant,color,marker,label in [(-.12,'full',BLUE,'s','Все параметры'),(.12,'LayerNorm_only',ORANGE,'o','Только LayerNorm')]:
        selected=[next(r for r in rows if r['screen']=='scope_validation' and r['variant']==variant and r['panel_seed']==seed) for seed in ('176030','177030','178030')]
        for ax,key,scale,title,xlabel in [(axes[0],'worst_old_delta',1e4,'Худший старый домен','ΔNLL × 10⁴ (меньше — лучше)'),(axes[1],'arithmetic_delta',-1e3,'Улучшение арифметики','−ΔNLL × 10³ (больше — лучше)')]:
            values=[float(r[key])*scale for r in selected];positions=[i+offset for i in range(3)]
            ax.scatter(positions,values,c=color,marker=marker,s=62,label=label,zorder=3)
            for pos,value in zip(positions,values):ax.annotate(f'{value:.2f}',(pos,value),xytext=(0,9),textcoords='offset points',ha='center',fontsize=10)
            ax.set_xticks(range(3),['176030','177030','178030']);ax.set_xlabel('Seed проверочной панели');ax.set_ylabel(xlabel);ax.set_title(title,pad=12)
            ax.grid(axis='y',alpha=.16);ax.set_xlim(-.5,2.5)
    axes[0].axhline(1,c='#555555',ls='--',lw=1.2,label='Допуск старого домена');axes[0].set_ylim(0,6.4);axes[1].set_ylim(0,132)
    fig.suptitle('AR-4 · ограничение параметров на трёх панелях',x=.05,y=.96,ha='left',fontsize=16)
    fig.text(.05,.86,'Одна обученная модель · full: 2 630 408 параметров · LayerNorm: 3456 · 8 окон на домен',fontsize=10)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper left',bbox_to_anchor=(.045,.825),ncol=3,frameon=False,fontsize=10)
    finish(fig,'scope-panels')
if __name__=='__main__':main()
