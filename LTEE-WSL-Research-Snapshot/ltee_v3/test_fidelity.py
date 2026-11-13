import tempfile,unittest
from pathlib import Path
from audit_simulation_fidelity import parse_gd, simulation_table
class Tests(unittest.TestCase):
 def test_gd(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'a.gd';p.write_text('#=TIME 100\nSNP 1 . REL606 4 A\nDEL 2 . REL606 5 3\n')
   m,s,t,b=parse_gd(p)
   self.assertEqual(m['TIME'],'100');self.assertEqual(s,[(4,'A')]);self.assertEqual(t['DEL'],1);self.assertFalse(b)
 def test_sim(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'neutral/seed_42';p.mkdir(parents=True)
   (p/'observed_genomes.csv').write_text('generation,sequence\n0,0011\n0,0111\n10,1111\n')
   x,e=simulation_table(Path(d));self.assertEqual(len(x),2);self.assertFalse(e)
 def test_missing_columns(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'neutral/seed_42';p.mkdir(parents=True)
   (p/'observed_genomes.csv').write_text('generation,dna\n0,0011\n')
   x,e=simulation_table(Path(d));self.assertTrue(x.empty);self.assertTrue(e)
if __name__=='__main__':unittest.main()
