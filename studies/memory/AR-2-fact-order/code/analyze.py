"""Regrade saved synthetic responses and summarize paired model comparisons."""
from pathlib import Path
import json,csv,re,statistics,argparse
ROOT=Path(__file__).resolve().parents[1]

def oracle(prompt):
    lines=prompt.splitlines(); task=lines[0]
    def pairs(s):return [(k,int(v)) for k,v in (x.split('=') for x in s.split(';'))]
    if task=='TASK RECALL':
        facts=pairs(lines[1][6:]); assert len(dict(facts))==len(facts)
        return str(dict(facts)[lines[2][4:]])
    if task=='TASK UPDATE':
        state=dict(pairs(lines[1][5:]));state.update(pairs(lines[2][6:]));return str(state[lines[3][4:]])
    if task=='TASK COUNTER':
        value=int(lines[1][6:])
        for op in lines[2].split(';'):
            kind,n=op.split();assert kind in ('ADD','SUB');value+=int(n)*(1 if kind=='ADD' else -1)
        return str(value)
    raise ValueError(task)

def signature(prompt):
    lines=prompt.splitlines()
    for i,line in enumerate(lines):
        if line.startswith(('FACTS ','INIT ')):
            head,body=line.split(' ',1);lines[i]=head+' '+';'.join(sorted(body.split(';')))
    return '\n'.join(lines)

def compute(e):
    rows=[]; changes=[];total=0
    for run in e['runs']:
        answers=run['answers'];maps={}
        for split,cases in e['cases'].items():
            a={x['id']:x for x in answers[split]};assert set(a)=={x['id'] for x in cases};maps[split]=a
            for c in cases:
                x=a[c['id']];assert oracle(c['prompt'])==c['answer']==x['expected']
                ids=x['generated_token_ids'];natural=bool(ids and ids[-1]==259)
                body=ids[:-1] if natural else ids
                assert natural==x['natural_eos'] and all(0<=n<128 for n in body)
                assert ''.join(map(chr,body))==x['answer']
                passed=natural and x['decode_error'] is None and x['answer']==c['answer']
                assert passed==x['passed'];total+=1
            rows.append(dict(seed=run['seed'],cohort=run['cohort'],split=split,cases=len(cases),passed=sum(x['passed'] for x in a.values())))
        for split,rev in [('selection_iid','reversed_iid'),('selection_ood','reversed_ood')]:
            original={x['id']:x for x in e['cases'][split]}; changed=lost=both=0
            for c in e['cases'][rev]:
                old=original[c['source_case_id']];assert signature(old['prompt'])==signature(c['prompt'])
                a=maps[split][old['id']];b=maps[rev][c['id']]
                changed+=a['answer']!=b['answer'];lost+=a['passed'] and not b['passed'];both+=a['passed'] and b['passed']
            recall=sum(x['passed'] for x in maps[split].values() if x['family']=='memory_recall')
            changes.append(dict(seed=run['seed'],cohort=run['cohort'],split=split,pairs=128,recall_passed=recall,changed_answers=changed,lost_after_reversal=lost,both_correct=both))
    assert total==6144
    return rows,changes

def main():
    e=json.loads((ROOT/'evidence/measurements.json').read_text(encoding='utf-8'));rows,changes=compute(e)
    (ROOT/'metrics').mkdir(exist_ok=True)
    for name,values in [('scores.csv',rows),('order_sensitivity.csv',changes),('training.csv',e['training'])]:
        with (ROOT/'metrics'/name).open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    for cohort in ('baseline','permuted_training'):
        print(cohort,{s:statistics.mean(x['passed'] for x in rows if x['cohort']==cohort and x['split']==s) for s in e['cases']})
    print('Regraded 6144 responses; oracle, ASCII decoding, EOS and paired reversal checks passed.')

if __name__=='__main__':main()
