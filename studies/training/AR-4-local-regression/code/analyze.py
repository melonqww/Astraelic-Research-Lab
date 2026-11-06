"""Recompute public AR-4 metrics from numeric evidence. Standard library only."""
from pathlib import Path
import csv,hashlib,json,math
ROOT=Path(__file__).resolve().parents[1]
DOMAINS=('stories','en','ru','code','arithmetic')
VARIANTS={'gem8':['plain_SGD','GEM_SGD'],'gem64':['plain_SGD','GEM8_SGD','GEM64_SGD'],
          'scopes':['full','backbone_only','attention_only','SSM_only','LayerNorm_only'],
          'scope_validation':['full','LayerNorm_only']}

def criterion(d,old_limit=1e-4,new_decrease=.001):
    if set(d)!=set(DOMAINS) or not all(math.isfinite(v) for v in d.values()):raise ValueError('five finite domain deltas required')
    return all(d[k]<=old_limit for k in DOMAINS[:-1]) and d['arithmetic']<=-new_decrease

def deltas(branch):
    assert set(branch['before'])==set(branch['after'])==set(branch['NLL_delta'])
    result={k:branch['after'][k]-v for k,v in branch['before'].items()}
    assert all(math.isfinite(v) for d in ('before','after') for v in branch[d].values())
    assert result==branch['NLL_delta']
    return result

def compute(e):
    assert e['schema']=='AR-4-numeric-v1' and set(e['screens'])==set(VARIANTS)
    rules=e['rules'];assert rules['old_limit']==1e-4 and rules['new_decrease']==.001
    scores=[];decisions=[];comparison=[]
    for label,s in e['screens'].items():
        assert [r['variant'] for r in s['rows']]==VARIANTS[label]
        for b in s['rows']:
            assert b['before']==s['rows'][0]['before'];d=deltas(b)
            assert set(d)==set(s['windows'])
            for seed in s['panel_seeds']:
                prefix=f'{seed}:' if s['panels_are_separate'] else ''
                domain_delta={k:d[prefix+k] for k in DOMAINS}
                passed=criterion(domain_delta,rules['old_limit'],rules['new_decrease'])
                expected=b['per_seed_criterion'][str(seed)] if s['panels_are_separate'] else b['local_criterion_met']
                assert passed==expected
                decisions.append(dict(screen=label,variant=b['variant'],panel_seed=seed,passed=passed,
                                      worst_old_delta=max(domain_delta[k] for k in DOMAINS[:-1]),arithmetic_delta=domain_delta['arithmetic']))
                for domain in DOMAINS:
                    key=prefix+domain;w=s['windows'][key];assert w['windows']==8 and w['targets']>0
                    scores.append(dict(screen=label,variant=b['variant'],panel_seed=seed,domain=domain,windows=w['windows'],targets=w['targets'],before=b['before'][key],after=b['after'][key],NLL_delta=d[key]))
    assert len(scores)==80 and len(decisions)==16
    assert set(e['replay'])=={'smoke','gem8','gem64','scopes','scope_validation'}
    for job,replay in e['replay'].items():
        label='scopes' if job=='smoke' else job;s=e['screens'][label]
        assert replay['status']=='complete' and replay['rollback_exact'] and replay['GPU_seconds']==0
        expected=['full'] if job=='smoke' else VARIANTS[label]
        assert [r['variant'] for r in replay['rows']]==expected
        for b in replay['rows']:
            old=next(r for r in s['rows'] if r['variant']==b['variant']);d=deltas(b)
            assert set(d)==set(old['before'])
            for key in d:
                before_error=abs(b['before'][key]-old['before'][key]);after_error=abs(b['after'][key]-old['after'][key]);error=abs(d[key]-old['NLL_delta'][key])
                assert before_error<=rules['NLL_absolute_tolerance'] and after_error<=rules['NLL_absolute_tolerance'] and error<=rules['delta_absolute_tolerance']
                comparison.append(dict(job=job,variant=b['variant'],panel_domain=key,before=b['before'][key],after=b['after'][key],NLL_delta=d[key],original_delta=old['NLL_delta'][key],before_error=before_error,after_error=after_error,delta_error=error))
            for seed in s['panel_seeds']:
                prefix=f'{seed}:' if s['panels_are_separate'] else ''
                assert criterion({k:d[prefix+k] for k in DOMAINS})==criterion({k:old['NLL_delta'][prefix+k] for k in DOMAINS})
    assert len(comparison)==85
    return scores,decisions,comparison

def main():
    f=ROOT/'evidence/measurements.json';e=json.loads(f.read_text(encoding='utf-8'))
    provenance=json.loads((ROOT/'evidence/provenance.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(f.read_bytes()).hexdigest()==provenance['measurements_sha256']
    scores,decisions,comparison=compute(e)
    (ROOT/'metrics').mkdir(exist_ok=True)
    for name,rows in [('results.csv',scores),('criteria.csv',decisions),('reproduction_comparison.csv',comparison)]:
        with (ROOT/'metrics'/name).open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('Recomputed 80 domain deltas, 16 decisions and 85 replay comparisons; max replay error:',max(r['delta_error'] for r in comparison))
if __name__=='__main__':main()
