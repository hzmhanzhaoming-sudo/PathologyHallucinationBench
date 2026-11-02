import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from score_stress_run import v1_score,t1_score,validate_report_target

class FormulaContracts(unittest.TestCase):
    def test_v1_mixed_option_chance(self):
        refs={'a':{'options':['A','B']},'b':{'options':['A','B','C','D']}}
        base={k:{'correct':True} for k in refs};no={'a':{'correct':True},'b':{'correct':False}}
        s=v1_score(base,no,refs)
        self.assertEqual(s['C_chance'],.75)
        self.assertEqual(s['score_percent'],80.)
    def test_v1_chance_or_worse_is_undefined(self):
        refs={'a':{'options':['A','B']}}
        base={'a':{'correct':False}};no={'a':{'correct':False}}
        self.assertIsNone(v1_score(base,no,refs)['score_percent'])
    def test_v1_clips(self):
        refs={k:{'options':['A','B']} for k in ('a','b','c')}
        base={k:{'correct':True} for k in refs};no={k:{'correct':False} for k in refs}
        self.assertEqual(v1_score(base,no,refs)['score_percent'],100.)
    def fixture(self):
        base={'400-9':{'correct':True},'500-1':{'correct':False}}
        refs={'T1_6choice':{},'T1_4choice':{}};variants={k:{} for k in refs}
        for l in (1,2):
            for condition,num,origin,value in [('T1_6choice',f'200-9-unknown{l}','400-9',False),('T1_4choice',f'500-1-unknown{l}','500-1',True)]:
                refs[condition][num]={'t3_original_num':origin,'t3_unknown_count':l}
                variants[condition][num]={'correct':value}
        return base,variants,refs,set(base)
    def test_t1_authoritative_origin_not_prefix(self):
        s=t1_score(*self.fixture())
        self.assertEqual(s['N'],4);self.assertEqual(s['coverage'],1)
        self.assertEqual(s['score_percent'],0.)
    def test_t1_duplicate_origin_rejected(self):
        b,v,r,e=self.fixture();r['T1_6choice']['x']=dict(r['T1_6choice']['200-9-unknown1']);v['T1_6choice']['x']={'correct':True}
        with self.assertRaises(ValueError):t1_score(b,v,r,e)
    def test_t1_no_wrong_opportunities_undefined(self):
        b,v,r,e=self.fixture();b['500-1']['correct']=True
        self.assertIsNone(t1_score(b,v,r,e)['score_percent'])
    def test_protected_output_locations(self):
        root=Path(__file__).resolve().parent/'nonexistent_test_root';run=root/'CODE'/'01_stress_tests'/'results'/'new'
        self.assertFalse(root.exists())
        for name in ('Draw_picture','Conclusion','Benchmark_T1_data','Benchmark_T7_results'):
            with self.assertRaises(ValueError):validate_report_target(root,root/name/'new_report',run)
        validate_report_target(root,root/'CODE'/'02_scoring'/'reports'/'new',run)
        with self.assertRaises(ValueError):validate_report_target(root,run/'new_report',run)
        with self.assertRaises(ValueError):validate_report_target(root,root,run)

if __name__=='__main__':unittest.main()
