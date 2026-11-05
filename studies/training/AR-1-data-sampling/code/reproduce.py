"""Recompute tables and the Russian report from bundled evidence; no model calls."""
from pathlib import Path
from collections import defaultdict
import argparse
import csv
import json
import math
import statistics

ROOT = Path(__file__).resolve().parents[1]
NAMES = {'en':'Английский','ru':'Русский','code':'Python-код'}

def table(name, rows):
    p=ROOT/'metrics'/name; p.parent.mkdir(exist_ok=True)
    with p.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

def calculate():
    e=json.loads((ROOT/'evidence/measurements.json').read_text(encoding='utf-8'))
    pairs=defaultdict(dict)
    for run in e['runs']: pairs[run['seed']][run['recipe']]=run
    results=[]; budget=[]; exposure=[]; coverage=[]
    for seed,pair in sorted(pairs.items()):
        for recipe,run in pair.items():
            v=run['result']; assert v['training_tokens']==2097152
            budget.append(dict(seed=seed,recipe=recipe,tokens=v['training_tokens'],bytes=v['training_bytes'],session_seconds=v['CPU_seconds']))
        for domain in NAMES:
            a=pair['files']['result']['final'][domain]; b=pair['windows']['result']['final'][domain]
            for metric in (a,b):
                assert math.isclose(metric['bits_per_byte'],metric['nll_per_token']*metric['evaluated_tokens']/metric['evaluated_bytes']/math.log(2),rel_tol=1e-12)
            results.append(dict(seed=seed,domain=domain,files_BPB=a['bits_per_byte'],windows_BPB=b['bits_per_byte'],reduction_percent=100*(1-b['bits_per_byte']/a['bits_per_byte'])))
    summary={}
    for domain in NAMES:
        rows=[x for x in results if x['domain']==domain]
        effect=100*(1-statistics.mean(x['windows_BPB'] for x in rows)/statistics.mean(x['files_BPB'] for x in rows))
        assert math.isclose(effect,e['reported_reductions'][domain],abs_tol=1e-10)
        summary[domain]=dict(reduction_percent=effect,min_seed=min(x['reduction_percent'] for x in rows),max_seed=max(x['reduction_percent'] for x in rows))
        train=[x for x in e['file_statistics'] if x['split']=='train' and x['domain']==domain]
        total=sum(x['tokens']-256 for x in train)
        for x in train:
            w=x['tokens']-256; assert w>0
            exposure.append(dict(domain=domain,file=x['file'],tokens=x['tokens'],bytes=x['bytes'],valid_windows=w,file_probability_uniform_files=1/len(train),file_probability_uniform_windows=w/total,per_window_exposure_ratio=total/(len(train)*w)))
        assert math.isclose(sum(x['file_probability_uniform_windows'] for x in exposure if x['domain']==domain),1)
        validation=[x for x in e['file_statistics'] if x['split']=='validation' and x['domain']==domain]
        intervals=defaultdict(list); draws=0
        for receipt in e['evaluation_receipts']:
            if receipt['domain']!=domain:continue
            for offset in receipt['offsets']:
                i,start=offset['file_index'],offset['offset']; assert 0<=start<validation[i]['tokens']-256
                # Target positions, not input positions.
                intervals[i].append((start+1,start+257)); draws+=1
        unique=0
        for values in intervals.values():
            end=-1
            for lo,hi in sorted(values):unique+=max(0,hi-max(lo,end)); end=max(end,hi)
        coverage.append(dict(domain=domain,validation_files=len(validation),drawn_windows=draws,evaluated_token_positions=draws*256,unique_target_positions=unique,available_target_positions=sum(x['tokens']-1 for x in validation)))
    return e,results,budget,exposure,coverage,summary

def figures(results,exposure):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    out=ROOT/'figures';out.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(12,4.4),sharey=True)
    markers=['o','s','^']
    for ax,domain in zip(axes,NAMES):
        rows=[x for x in results if x['domain']==domain]
        for x,marker in zip(rows,markers):
            ax.scatter(x['seed'],x['reduction_percent'],s=60,marker=marker,color='#1d5a8a')
            ax.annotate(f"{x['reduction_percent']:.2f}%",(x['seed'],x['reduction_percent']),xytext=(0,10),textcoords='offset points',ha='center')
        ax.axhline(0,color='#475569',linewidth=1);ax.set(title=NAMES[domain],xlabel='Seed',xticks=[17,29,43],ylim=(-.3,6),xlim=(10,50));ax.grid(axis='y',alpha=.18)
    axes[0].set_ylabel('Снижение BPB относительно выборки файлов, %')
    fig.suptitle('AR-1 · Результаты трёх парных сравнений',fontsize=16)
    fig.text(.5,.02,'Одна архитектура, технический корпус, фиксированные validation-окна. Точки — отдельные seed, не доверительные интервалы.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.08,1,.92))
    for ext in ['png','svg']:fig.savefig(out/f'effects.{ext}',dpi=160,facecolor='white')
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4.4),sharey=True)
    for ax,domain in zip(axes,NAMES):
        rows=sorted([x for x in exposure if x['domain']==domain],key=lambda x:x['valid_windows'])
        ax.scatter([x['valid_windows'] for x in rows],[x['per_window_exposure_ratio'] for x in rows],color='#976023',marker='o',s=35)
        ax.axhline(1,color='#475569',linestyle='--');ax.set_xscale('log');ax.set_yscale('log');ax.set(title=NAMES[domain],xlabel='Число допустимых окон в файле');ax.grid(alpha=.15)
    axes[0].set_ylabel('Вероятность конкретного окна: файлы / окна')
    fig.suptitle('AR-1 · Как длина файла меняет частоту выбора окна',fontsize=16)
    fig.text(.5,.02,'Аналитические вероятности по сохранённым длинам токенизации. Линия 1 — одинаковая вероятность; это не результат обучения.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.08,1,.92))
    for ext in ['png','svg']:fig.savefig(out/f'exposure.{ext}',dpi=160,facecolor='white')
    plt.close(fig)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--figures',action='store_true');args=parser.parse_args()
    e,results,budget,exposure,coverage,summary=calculate()
    for name,rows in [('results.csv',results),('budgets.csv',budget),('file_exposure.csv',exposure),('evaluation_coverage.csv',coverage)]:table(name,rows)
    (ROOT/'metrics/summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if args.figures:figures(results,exposure)
    print(json.dumps({'paired_results':len(results),'seeds':3,'summary':summary,'coverage':coverage},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
