import json
import pandas as pd
from backend.replay import ReplayController


def test_replay_uses_loaded_dataset_dates(tmp_path):
    labels=tmp_path/'labels'
    labels.mkdir()
    pd.DataFrame({'T':['2031-04-12 09:00:00','2031-04-12 10:00:00']}).to_csv(labels/'labels_test.csv',index=False)
    c=ReplayController(tmp_path,'test',lambda *a:None,lambda:None,lambda t:None)
    try:
        assert str(c.start)=='2031-04-12 09:00:00'
        assert str(c.end)=='2031-04-12 10:15:00'
        c.control('seek',timestamp='2031-04-12 09:30:00')
    finally:c.close()


def test_packaged_replay_uses_manifest_window(tmp_path):
    (tmp_path/'manifest.json').write_text(json.dumps({'start':'2032-05-01 11:00:00','end':'2032-05-01 12:00:00'}))
    c=ReplayController(tmp_path,'test',lambda *a:None,lambda:None,lambda t:None)
    try:
        assert str(c.start)=='2032-05-01 11:00:00'
        assert str(c.end)=='2032-05-01 12:00:00'
    finally:c.close()
