"""Small train-CV-only search around the selected v3 model, reusing its cache."""
import argparse,copy,json
from pathlib import Path
from ml.kaggle_train import train,json_write

def run(data,source,output):
    if output.exists():raise ValueError('Choose a fresh output directory to preserve previous runs')
    provenance=json.loads((source/'artifacts/provenance.json').read_text())
    metadata=json.loads((source/'models/metadata.json').read_text())
    config=copy.deepcopy(provenance['config'])
    selected=metadata['metrics']['candidate']
    base=next(c for c in config['candidates'] if c['name']==selected)
    candidates=[]
    def add(name,**changes):
        cfg={**base,**changes,'name':name};candidates.append(cfg)
    add('winning_config_control')
    add('shallower',depth=max(2,base['depth']-1),l2_leaf_reg=max(5,base['l2_leaf_reg']))
    add('deeper_regularized',depth=min(8,base['depth']+1),l2_leaf_reg=max(5,base['l2_leaf_reg']))
    add('synthetic_weight_variant',synthetic_weight=.1 if base['synthetic_weight']==0 else 0)
    config['candidates']=candidates
    output.mkdir(parents=True)
    json_write(output/'refine_config.json',config)
    print('Previous leaderboard model is preserved in',source)
    print('Selection uses purged forward train folds, not test or leaderboard scores.')
    return train(data,output,config,cache_run=source)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();run(args.data,args.source,args.output)
