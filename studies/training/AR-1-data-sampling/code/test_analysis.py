import unittest
from reproduce import calculate
from window_sampling import WindowIndex

class MeasurementChecks(unittest.TestCase):
    def test_window_bijection(self):
        for lengths,context in [([5,6,8],4),([2],1),([257,257,260],256)]:
            sampler=WindowIndex(lengths,context)
            self.assertEqual([sampler.locate(i) for i in range(sampler.total)],[(f,j) for f,n in enumerate(lengths) for j in range(n-context)])
    def test_reject_invalid_inputs(self):
        for lengths,context in [([],1),([3],3),([True],1),([3],0)]:
            with self.assertRaises(ValueError):WindowIndex(lengths,context)
        sampler=WindowIndex([5,6,8],4)
        for draw in [-1,sampler.total,True]:
            with self.assertRaises(ValueError):sampler.locate(draw)
    def test_recorded_comparison(self):
        _,rows,budget,exposure,coverage,summary=calculate()
        self.assertEqual(len(rows),9)
        self.assertEqual(len(budget),6)
        self.assertTrue(all(x['reduction_percent']>0 for x in rows))
        self.assertTrue(all(x['unique_target_positions']<=x['evaluated_token_positions'] for x in coverage))
        self.assertEqual(len(exposure),40)

if __name__=='__main__':unittest.main()
