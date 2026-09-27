"""Causal v4 search: new features, diverse learners, OOF blends, fast export.

Only purged forward TRAIN folds select models/blends. Holdout/test are reports.
"""
import argparse,copy,itertools,json,platform,time
from pathlib import Path
import catboost
import numpy as np
import pandas as pd
from catboost import Pool,CatBoostError
from ml import kaggle_train as kt
from ml.features import CAT_FEATURES as V3_CATS
from ml.features_v4 import FEATURE_VERSION,FEATURES,CAT_FEATURES,augment_frame
from ml.model import Predictor

CATS=V3_CATS+CAT_FEATURES

def candidates(base):
    base=copy.deepcopy(base)
    def variant(name,**kw):return {**base,'name':name,**kw}
    return [variant('v3_control',features='v3'),
            variant('v4_mae',features='v4'),
            variant('v4_direct',features='v4',residual=False),
            variant('v4_rmse',features='v4',loss_function='RMSE',l2_leaf_reg=8),
            variant('v4_ordered',features='v4',task_type='CPU',boosting_type='Ordered',depth=5,l2_leaf_reg=5),
            variant('v4_recent',features='v4',recency_half_life_hours=6),
            variant('v4_compact',features='numeric',depth=4,border_count=64,l2_leaf_reg=5)]

def feature_columns(frame,cfg,base_columns):
    if cfg['features']=='v3':return list(base_columns)
    cols=[c for c in frame if c not in kt.META]
    return [c for c in cols if c not in CATS] if cfg['features']=='numeric' else cols

def pool(frame,cols,cfg,config,training=False):
    weight=None
    if training:
        weight=kt.weights(frame,cfg,config)
        if cfg.get('recency_half_life_hours'):
            age=(frame['T'].max()-frame['T']).dt.total_seconds().to_numpy()/3600
            weight=weight*np.exp2(-age/cfg['recency_half_life_hours'])
        keep=weight>0;frame=frame.loc[keep];weight=weight[keep]
    return Pool(frame[cols],kt.target(frame,cfg['residual'],cfg.get('target_scale',1)),cat_features=[c for c in cols if c in CATS],weight=weight)

def fit(frame,cfg,config,cols,iterations,seed=42,val=None):
    model=kt.estimator(cfg,config,iterations,seed)
    model.set_params(boosting_type=cfg.get('boosting_type','Plain'),border_count=cfg.get('border_count',254))
    kwargs={}
    if val is not None:kwargs=dict(eval_set=pool(val,cols,cfg,config),early_stopping_rounds=config['early_stopping_rounds'])
    model.fit(pool(frame,cols,cfg,config,training=True),**kwargs)
    return model

def predict(model,frame,cols,cfg):
    return np.asarray(model.predict(frame[cols],thread_count=1))*cfg.get('target_scale',1)+(frame.cur_dev_s.to_numpy() if cfg['residual'] else 0)

def choose_strategies(records,oof,y):
    ranking=sorted(records,key=lambda r:r['cv_mae'])
    best=ranking[0];quality=[(best['candidate'],1.)];quality_mae=best['cv_mae']
    for a,b in itertools.combinations(ranking[:4],2):
        for weight in [.25,.5,.75]:
            p=weight*oof[a['candidate']]+(1-weight)*oof[b['candidate']]
            score=kt.mae(y,p)
            if score<best['cv_mae']-.25 and score<quality_mae:
                quality=[(a['candidate'],weight),(b['candidate'],1-weight)];quality_mae=score
    # At most +1 second on train OOF in exchange for a smaller single model.
    eligible=[r for r in ranking if r['cv_mae']<=best['cv_mae']+1.]
    fast=min(eligible,key=lambda r:r['complexity'])
    return {'quality':quality,'fast':[(fast['candidate'],1.)]}, {'quality':quality_mae,'fast':fast['cv_mae']}

def export(directory,strategy,seeds,training,frame,configs,records,config,base_cols,fit_cache):
    directory.mkdir(parents=True,exist_ok=True);members=[];used=[]
    for name,weight in strategy:
        cfg=configs[name];cols=feature_columns(frame,cfg,base_cols)
        for seed in seeds:
            key=(name,seed)
            if key not in fit_cache:
                fit_cache[key]=fit(training,cfg,config,cols,records[name]['iterations'],seed)
            model=fit_cache[key]
            filename='best_model.cbm' if not members else f'member_{len(members)}.cbm'
            model.save_model(str(directory/filename))
            members.append(dict(file=filename,weight=weight/len(seeds),residual=cfg['residual'],target_scale=cfg.get('target_scale',1),features=cols,sha256=kt.sha(directory/filename),candidate=name,seed=seed,training_device=model.get_all_params()['task_type'],trees=model.tree_count_))
            used.extend(c for c in cols if c not in used)
    import hashlib
    identity=json.dumps(members,sort_keys=True)
    metadata=dict(feature_version=FEATURE_VERSION,model_version=hashlib.sha256(identity.encode()).hexdigest()[:12],features=used,categorical_features=[c for c in used if c in CATS],members=members,residual=False,target_scale=1,inference_threads=1,horizon_seconds=[600,900],training_source='labels_train; purged forward CV; OOF-only blend selection')
    kt.json_write(directory/'metadata.json',metadata)
    return Predictor(directory)

def run(data,source,output,task_type=None,quick=False):
    if output.exists():raise ValueError('Use a new output directory; previous runs are preserved')
    start=time.perf_counter()
    provenance=json.loads((source/'artifacts/provenance.json').read_text())
    metadata=json.loads((source/'models/metadata.json').read_text())
    if provenance['feature_version']!='causal-v3':raise ValueError('SOURCE must be a completed v3 run')
    if not all(provenance['rules'].values()):raise ValueError('Causal cache provenance required')
    for name,digest in provenance['input_sha256'].items():
        if kt.sha(data/name)!=digest:raise ValueError('Changed raw input: '+name)
    config=copy.deepcopy(provenance['config'])
    if task_type:config['task_type']=task_type
    if quick:config.update(iterations=30,early_stopping_rounds=8)
    base=next(c for c in config['candidates'] if c['name']==metadata['metrics']['candidate'])
    config['candidates']=candidates(base)
    base_cols=metadata['features']
    output.mkdir(parents=True);art=output/'artifacts';art.mkdir()
    kt.json_write(art/'config.json',config)
    def dataset(split):
        print('Causal v3 cache + new v4 transforms:',split,flush=True)
        frame=augment_frame(pd.read_parquet(source/'artifacts'/f'{split}_dataset.parquet'))
        if not frame.sample_id.is_unique:raise ValueError('Duplicate sample IDs')
        return frame
    frame=dataset('train')
    hold=frame['T'].quantile(config['holdout_quantile'])
    bounds=[frame['T'].quantile(q) for q in config['cv_quantiles']]
    if bounds[-1]!=hold:raise ValueError('CV must end before final holdout')
    folds=[(kt.past(frame,a),frame[(frame['T']>=a)&(frame['T']<b)&kt.real_mask(frame,config)]) for a,b in zip(bounds[:-1],bounds[1:])]
    for tr,val in folds:
        if len(tr)<100 or len(val)<20:raise ValueError('Insufficient fold data')
    oof_frame=pd.concat([v[['sample_id','T','target_delay_s']] for _,v in folds],ignore_index=True)
    oof={};records=[];fold_records=[];failures=[]
    for cfg in config['candidates']:
        cols=feature_columns(frame,cfg,base_cols);preds=[];trees=[]
        for i,(tr,val) in enumerate(folds):
            try:model=fit(tr,cfg,config,cols,config['iterations'],val=val)
            except CatBoostError as exc:
                failures.append(dict(candidate=cfg['name'],fold=i,error=str(exc)));print(failures[-1],flush=True);break
            p=predict(model,val,cols,cfg);preds.append(p);trees.append(model.tree_count_)
            result=dict(candidate=cfg['name'],fold=i,mae=kt.mae(val.target_delay_s,p),trees=model.tree_count_,task_type=model.get_all_params()['task_type'])
            fold_records.append(result);print(json.dumps(result),flush=True)
        if len(preds)!=len(folds):continue
        oof[cfg['name']]=np.concatenate(preds);iterations=max(20,int(np.median(trees)))
        records.append(dict(candidate=cfg['name'],cv_mae=kt.mae(oof_frame.target_delay_s,oof[cfg['name']]),iterations=iterations,complexity=iterations*2**cfg['depth']*(2 if any(c in CATS for c in cols) else 1)))
    if not records:raise RuntimeError('All candidates failed')
    pd.DataFrame(records).sort_values('cv_mae').to_csv(art/'experiments.csv',index=False)
    pd.DataFrame(fold_records).to_csv(art/'fold_metrics.csv',index=False)
    kt.json_write(art/'failed_candidates.json',failures)
    for name,p in oof.items():oof_frame[name]=p
    oof_frame.to_parquet(art/'oof_predictions.parquet',index=False)
    strategies,cv=choose_strategies(records,oof,oof_frame.target_delay_s)
    kt.json_write(art/'selection.json',dict(strategies=strategies,cv_mae=cv,fast_max_cv_degradation_s=1,selection='train OOF only'))
    print('Selected BEFORE holdout/test:',strategies,cv,flush=True)
    cfgs={c['name']:c for c in config['candidates']};recs={r['candidate']:r for r in records}
    tr=kt.past(frame,hold);val=frame[(frame['T']>=hold)&kt.real_mask(frame,config)]
    hold_cache={};final_cache={};metrics={};runtimes={}
    for mode,strategy in strategies.items():
        seeds=[42,17] if mode=='quality' and not quick else [42]
        temporal=export(output/f'holdout_models_{mode}',strategy,seeds,tr,frame,cfgs,recs,config,base_cols,hold_cache)
        hp=temporal.predict_frame(val)
        metrics[mode]=dict(cv_mae=cv[mode],holdout_mae=kt.mae(val.target_delay_s,hp),holdout_baseline_mae=kt.mae(val.target_delay_s,val.cur_dev_s),members=len(temporal.members))
        hv=val[['sample_id','T','target_delay_s']].copy();hv['prediction']=hp;hv.to_parquet(art/f'holdout_{mode}.parquet',index=False)
        runtimes[mode]=export(output/f'models_{mode}',strategy,seeds,frame,frame,cfgs,recs,config,base_cols,final_cache)
    # All model choices and fits are complete before opening diagnostic test.
    test=dataset('test');validate=dataset('validate')
    template=pd.read_csv(data/'sample_submission.csv',sep=';')
    if list(template.columns)!=['sample_id','prediction'] or not template.sample_id.is_unique or set(template.sample_id)!=set(validate.sample_id):raise ValueError('Invalid submission coverage')
    old=pd.read_csv(source/'artifacts/submission.csv',sep=';').set_index('sample_id').prediction
    for mode,runtime in runtimes.items():
        tp=runtime.predict_frame(test);vp=runtime.predict_frame(validate)
        if not np.isfinite(vp).all():raise ValueError('Non-finite prediction')
        submission=template.copy();submission['prediction']=submission.sample_id.map(dict(zip(validate.sample_id,vp)))
        submission.to_csv(art/f'submission_v4_{mode}.csv',sep=';',index=False)
        td=test[['sample_id','target_delay_s']].copy();td['prediction']=tp;td.to_parquet(art/f'test_{mode}.parquet',index=False)
        delta=np.abs(submission.prediction.to_numpy()-old.reindex(submission.sample_id).to_numpy())
        # Per-row backend predictor benchmark, including v4 transforms, excluding GPS extraction/HTTP.
        times=[]
        for row in validate.drop(columns=FEATURES).head(50).to_dict('records'):
            before=time.perf_counter();single=runtime.predict(row);times.append((time.perf_counter()-before)*1000)
        sample=validate.head(25)
        singles=np.array([runtime.predict(r) for r in sample.drop(columns=FEATURES).to_dict('records')])
        np.testing.assert_allclose(singles,runtime.predict_frame(sample),rtol=1e-9,atol=1e-8)
        metrics[mode].update(test_mae=kt.mae(test.target_delay_s,tp),test_baseline_mae=kt.mae(test.target_delay_s,test.cur_dev_s),max_change_vs_v3_s=float(delta.max()),same_submission_as_v3=bool(np.allclose(delta,0,rtol=0,atol=1e-8)),predict_p50_ms=float(np.median(times)),predict_p95_ms=float(np.percentile(times,95)))
    report=dict(results=metrics,elapsed_s=time.perf_counter()-start,holdout_cutoff=str(hold),leaderboard_score='unknown until submitted',quick_smoke_test=quick)
    kt.json_write(art/'metrics.json',report)
    kt.json_write(art/'provenance.json',dict(feature_version=FEATURE_VERSION,config=config,source_provenance_sha256=kt.sha(source/'artifacts/provenance.json'),input_sha256=provenance['input_sha256'],cache_sha256={s:kt.sha(source/'artifacts'/f'{s}_dataset.parquet') for s in ['train','test','validate']},code_sha256={p:kt.sha(Path(__file__).parent/p) for p in ['features.py','features_v4.py','train_v4.py','model.py','kaggle_train.py']},python=platform.python_version(),catboost=catboost.__version__,rules=dict(no_test_selection=True,no_validate_labels=True,holdout_not_used_for_selection=True,telemetry_event_lte_T=True,receive_and_gps_lte_T=True,facts_not_features=True)))
    print(json.dumps(report,indent=2),flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--task-type',choices=['CPU','GPU']);p.add_argument('--quick',action='store_true')
    a=p.parse_args();run(a.data,a.source,a.output,a.task_type,a.quick)
