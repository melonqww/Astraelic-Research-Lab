import unittest
from analyze import criterion,deltas,DOMAINS
class AnalysisTests(unittest.TestCase):
    def test_all_domains_must_pass(self):
        d={k:0. for k in DOMAINS};d.update(en=.00012,arithmetic=-1.,ru=-1.)
        self.assertFalse(criterion(d))
        d['en']=1e-4;self.assertTrue(criterion(d))
    def test_new_task_improvement_is_required(self):
        d={k:0. for k in DOMAINS};d['arithmetic']=-.0009
        self.assertFalse(criterion(d))
        d['arithmetic']=-.001;self.assertTrue(criterion(d))
    def test_missing_or_nonfinite_is_not_success(self):
        with self.assertRaises(ValueError):criterion({'arithmetic':-1.})
        d={k:0. for k in DOMAINS};d['en']=float('nan')
        with self.assertRaises(ValueError):criterion(d)
    def test_stored_delta_is_not_trusted(self):
        b=dict(before={'en':1.},after={'en':1.01},NLL_delta={'en':0.})
        with self.assertRaises(AssertionError):deltas(b)
        b['NLL_delta']['en']=b['after']['en']-b['before']['en']
        self.assertGreater(deltas(b)['en'],0.)
if __name__=='__main__':unittest.main()
