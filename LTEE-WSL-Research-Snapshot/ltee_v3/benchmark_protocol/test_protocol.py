import json,tempfile,unittest
from pathlib import Path
from build_protocol import build
from validate_protocol import validate
class ProtocolTests(unittest.TestCase):
    def test_valid(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);build(p,[11,23],[42],[1001]);self.assertEqual(validate(p)['status'],'PASS_PROTOCOL_STRUCTURE_ONLY')
    def test_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(ValueError):build(Path(t),[11],[11],[1001])
    def test_inspected_test_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(ValueError):build(Path(t),[11],[42],[67])
    def test_tampering_detected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);build(p,[11],[42],[1001]);d=json.loads((p/'protocol.json').read_text());d['primary_metric']['name']='brier';(p/'protocol.json').write_text(json.dumps(d));self.assertEqual(validate(p)['status'],'FAIL')
if __name__=='__main__':unittest.main()
