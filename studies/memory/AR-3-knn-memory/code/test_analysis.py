import unittest
from analyze import oracle,grade
class AnalysisTests(unittest.TestCase):
    def test_last_write_and_counter(self):
        self.assertEqual(oracle('TASK UPDATE\nINIT a=1;b=2\nWRITE a=4;a=9\nASK a\nANSWER:'),'9')
        self.assertEqual(oracle('TASK COUNTER\nSTART 2\nADD 3;SUB 7\nANSWER:'),'-2')
    def test_eos_is_required(self):
        c={'prompt':'TASK RECALL\nFACTS a=2;b=3\nASK a\nANSWER:','answer':'2'}
        a=dict(expected='2',answer='2',generated_token_ids=[50],natural_eos=False,decode_error=None,passed=False)
        self.assertFalse(grade(c,a))
        a.update(generated_token_ids=[50,259],natural_eos=True,passed=True)
        self.assertTrue(grade(c,a))
        a.update(generated_token_ids=[51,259],answer='3',passed=False)
        self.assertFalse(grade(c,a))
if __name__=='__main__':unittest.main()
