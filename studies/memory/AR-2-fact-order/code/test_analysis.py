import unittest,json
from analyze import oracle,compute,ROOT
class Checks(unittest.TestCase):
    def test_oracles(self):
        self.assertEqual(oracle('TASK RECALL\nFACTS a=3;b=5\nASK a\nANSWER:'),'3')
        self.assertEqual(oracle('TASK UPDATE\nINIT a=3\nWRITE a=4;a=9\nASK a\nANSWER:'),'9')
        self.assertEqual(oracle('TASK COUNTER\nSTART -3\nADD 4;SUB 2\nANSWER:'),'-1')
    def test_saved_answers(self):
        rows,changes=compute(json.loads((ROOT/'evidence/measurements.json').read_text(encoding='utf-8')))
        self.assertEqual(len(rows),24);self.assertEqual(len(changes),12)
if __name__=='__main__':unittest.main()
