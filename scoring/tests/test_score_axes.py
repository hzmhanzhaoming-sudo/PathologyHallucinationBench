import copy,unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from score_axes import compute_axes,pareto_layers

class AxisContract(unittest.TestCase):
    def setUp(self):
        self.weights=[{'domain':d,'component':k,'weight':w} for d,parts in [('visual',[('V1',.2),('V2',.3),('V3',.5)]),('text',[('T1',.4),('T2',.2),('T3',.4)]),('dataset',[(x,.25) for x in ('QuiltVQA','PathVQA','PathMMU','PatchVQA')])] for k,w in parts]
        self.scores=[dict(model_key='a',V1=10,V2=20,V3=30,T1=80,T2=50,T3=60)]
        self.counts=[dict(model_key='a',dataset=d,correct=c,total=n) for d,c,n in [('QuiltVQA',5,10),('PathVQA',20,20),('PathMMU',25,50),('PatchVQA',0,40)]]
    def test_weights_counts_not_percentages(self):
        r=compute_axes(self.scores,self.counts,self.weights)[0]
        self.assertEqual(r['visual_resistance'],23)
        self.assertEqual(r['text_resistance'],66)
        self.assertAlmostEqual(r['weighted_accuracy'],100*50/120)
        self.assertEqual(r['pareto_3d_layer'],1)
    def test_equal_points_not_dominate_each_other(self):
        rows=[dict(visual_resistance=x,text_resistance=y,weighted_accuracy=z) for x,y,z in [(3,3,3),(3,3,3),(2,2,2),(1,4,1)]]
        self.assertEqual(pareto_layers(rows),[1,1,2,1])
    def test_missing_dataset_rejected(self):
        with self.assertRaises(ValueError):compute_axes(self.scores,self.counts[:-1],self.weights)
    def test_duplicate_model_rejected(self):
        with self.assertRaises(ValueError):compute_axes(self.scores*2,self.counts,self.weights)
    def test_duplicate_count_rejected(self):
        with self.assertRaises(ValueError):compute_axes(self.scores,self.counts+[self.counts[0]],self.weights)
    def test_non_normalized_weights_rejected(self):
        weights=copy.deepcopy(self.weights);weights[0]['weight']=.8
        with self.assertRaises(ValueError):compute_axes(self.scores,self.counts,weights)
    def test_undefined_task_rejected(self):
        scores=copy.deepcopy(self.scores);scores[0]['V1']=float('nan')
        with self.assertRaises(ValueError):compute_axes(scores,self.counts,self.weights)
    def test_no_input_mutations(self):
        before=copy.deepcopy((self.scores,self.counts,self.weights));compute_axes(self.scores,self.counts,self.weights)
        self.assertEqual(before,(self.scores,self.counts,self.weights))

if __name__=='__main__':unittest.main()
