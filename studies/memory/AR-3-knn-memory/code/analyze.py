"""Regrade frozen kNN responses. Python standard library only."""
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

MODES=('base','memory','shuffled')
def check_intervention(a,b):
    left=a['prompt'].splitlines();right=b['prompt'].splitlines()
    assert a['family']==b['family'] and len(left)==len(right)
    index={'memory_recall':1,'memory_update':2,'execute_counter':1}[a['family']]
    assert [i for i,(x,y) in enumerate(zip(left,right)) if x!=y]==[index]
    if a['family']=='execute_counter':return
    x=left[index].split(' ',1)[1].split(';');y=right[index].split(' ',1)[1].split(';')
    assert len(x)==len(y)
    query=left[-2].removeprefix('ASK ')
    differences=[i for i,(u,v) in enumerate(zip(x,y)) if u!=v]
    assert len(differences)==1
    changed=differences[0]; assert x[changed].split('=')[0]==y[changed].split('=')[0]==query
    if a['family']=='memory_update':assert changed==max(i for i,item in enumerate(x) if item.split('=')[0]==query)

def grade(case, answer):
    assert oracle(case['prompt'])==case['answer']==answer['expected']
    ids=answer['generated_token_ids']; eos=bool(ids and ids[-1]==259)
    body=ids[:-1] if eos else ids
    assert eos==answer['natural_eos'] and all(type(n) is int and 0<=n<128 for n in body)
    assert ''.join(map(chr,body))==answer['answer']
    passed=eos and answer['decode_error'] is None and answer['answer']==case['answer']
    assert passed==answer['passed']
    return passed

def compute(e):
    scores=[]; pairs=[]; total=0
    for label,panel in e['panels'].items():
        for run in panel['runs']:
            for split,cases in panel['cases'].items():
                lookup={(a['id'],a['mode']):a for a in run['answers'][split]}
                assert len(lookup)==len(run['answers'][split])==len(cases)*3
                assert set(lookup)=={(c['id'],m) for c in cases for m in MODES}
                for mode in MODES:
                    good={c['id']:grade(c,lookup[c['id'],mode]) for c in cases};total+=len(cases)
                    common=dict(panel=label,cohort=run['cohort'],seed=run['seed'],split=split,mode=mode)
                    for family in ['all']+sorted({c['family'] for c in cases}):
                        subset=[c for c in cases if family=='all' or c['family']==family]
                        scores.append(dict(**common,family=family,cases=len(subset),correct=sum(good[c['id']] for c in subset)))
                    if label=='counterfactual':
                        grouped={}
                        for c in cases:
                            group=grouped.setdefault(c['pair_id'],{});assert c['variant'] not in group;group[c['variant']]=c
                        both=baseline=stale=0
                        for group in grouped.values():
                            assert set(group)=={'baseline','changed'}
                            a,b=group['baseline'],group['changed']; assert a['answer']!=b['answer']
                            check_intervention(a,b)
                            both+=good[a['id']] and good[b['id']]; baseline+=good[a['id']]
                            stale+=good[a['id']] and lookup[b['id'],mode]['answer']==a['answer']
                        pairs.append(dict(**common,pairs=len(grouped),both_correct=both,baseline_correct=baseline,stale_after_correct_baseline=stale))
    assert total==27648,total
    return scores,pairs

def main():
    evidence=json.loads((ROOT/'evidence/measurements.json').read_text(encoding='utf-8'))
    scores,pairs=compute(evidence)
    provenance=json.loads((ROOT/'evidence/provenance.json').read_text(encoding='utf-8'))
    for label,rows in provenance['original_review_rows'].items():
        for row in rows:
            for split in ('selection_iid','selection_ood'):
                for mode in MODES:
                    score=next(x for x in scores if x['panel']==label and x['seed']==row['seed'] and x['cohort']==row['cohort'] and x['split']==split and x['mode']==mode and x['family']=='all')
                    expected=row['counts'][split][mode] if label=='ordinary' else row['stats'][split]['marginal_correct'][mode]
                    assert score['correct']==expected
                    if label=='counterfactual':
                        pair=next(x for x in pairs if x['seed']==row['seed'] and x['cohort']==row['cohort'] and x['split']==split and x['mode']==mode)
                        for key,old in [('both_correct','pairs_both_correct'),('baseline_correct','baseline_correct'),('stale_after_correct_baseline','stale_after_correct_baseline')]:assert pair[key]==row['stats'][split][old][mode]
    (ROOT/'metrics').mkdir(exist_ok=True)
    for name,rows in [('scores.csv',scores),('pairs.csv',pairs)]:
        with (ROOT/'metrics'/name).open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('Regraded 27648 responses; exact oracle, token decoding, EOS and original review totals agree.')
if __name__=='__main__':main()
