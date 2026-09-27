"""Forward CV, separate final holdout, train-only final models and provenance.

Test labels are read only AFTER model selection and fitting. Validate facts are
never opened. The official test is diagnostic, not a model-selection dataset.
"""
import argparse,hashlib,json,platform,time
from pathlib import Path
import numpy as np
import pandas as pd
import catboost
from catboost import CatBoostError
from catboost import CatBoostRegressor,CatBoostClassifier,Pool
from ml.features import FEATURE_VERSION,EXTENDED,CAT_FEATURES
from ml.build_dataset import build

META={'sample_id','tr_id','T','target_time_begin','target_event_time','prediction_horizon_seconds','target_delay_s','target_stop_id'}
CONTEXT=['cur_dev_s','hour','weekday','is_weekend','scheduled_time_to_target']
def mae(y,p):return float(np.abs(np.asarray(y)-np.asarray(p)).mean())
def real_mask(frame,config):
    lo,hi=config['synthetic_id_range']
    return ~pd.to_numeric(frame.tr_id).between(lo,hi)
def columns(frame,kind):
    cols=[c for c in frame if c not in META]
    if kind=='context':return CONTEXT
    if kind=='legacy':return [c for c in cols if c not in EXTENDED]
    if kind=='numeric':return [c for c in cols if c not in CAT_FEATURES]
    if kind!='extended':raise ValueError('Unknown feature group')
    return cols
def past(frame,cutoff):
    # A label is eligible only when both its plan and fact have occurred.
    return frame[(frame['T']<cutoff)&(frame.target_time_begin<cutoff)&(frame.target_event_time<cutoff)]
def weights(frame,cfg,config):return np.where(real_mask(frame,config),1.0,cfg['synthetic_weight'])
def estimator(cfg,config,iterations,seed=42):
    device='CPU' if config['task_type']=='CPU' else cfg.get('task_type',config['task_type'])
    loss=cfg.get('loss_function','MAE')
    args=dict(iterations=int(iterations),depth=cfg['depth'],learning_rate=cfg['learning_rate'],l2_leaf_reg=cfg['l2_leaf_reg'],loss_function=loss,eval_metric='MAE',random_seed=seed,thread_count=config['thread_count'],allow_writing_files=False,verbose=False,task_type=device,boosting_type='Plain',border_count=254,metric_period=1,boost_from_average=True)
    if loss=='MAE':args['leaf_estimation_method']='Exact'
    if device=='GPU':args['devices']=config.get('devices','0')
    return CatBoostRegressor(**args)
def target(frame,residual,scale=1):return (frame.target_delay_s.to_numpy()-(frame.cur_dev_s.to_numpy() if residual else 0))/scale
def prediction(model,frame,cols,residual,scale=1):return model.predict(frame[cols])*scale+(frame.cur_dev_s.to_numpy() if residual else 0)
def pool(frame,cols,residual,scale=1,weight=None):
    cats=[c for c in cols if c in CAT_FEATURES]
    return Pool(frame[cols],target(frame,residual,scale),cat_features=cats,weight=weight)
def fit_one(train,val,cfg,config):
    cols=columns(train,cfg['features']);w=weights(train,cfg,config);keep=w>0
    model=estimator(cfg,config,config['iterations'])
    scale=cfg.get('target_scale',1)
    model.fit(pool(train.loc[keep],cols,cfg['residual'],scale,w[keep]),eval_set=pool(val,cols,cfg['residual'],scale),early_stopping_rounds=config['early_stopping_rounds'])
    return model,cols
def json_write(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def train(data,output,config,cache_run=None):
    output.mkdir(parents=True,exist_ok=True);art=output/'artifacts';models=output/'models'
    art.mkdir(exist_ok=True);models.mkdir(exist_ok=True)
    start=time.perf_counter()
    cache_provenance=None
    if cache_run is not None:
        if cache_run.resolve()==output.resolve():raise ValueError('Use a new output directory')
        cache_provenance=json.loads((cache_run/'artifacts/provenance.json').read_text())
        if cache_provenance['feature_version']!=FEATURE_VERSION:raise ValueError('Incompatible feature cache')
        for name,digest in cache_provenance['input_sha256'].items():
            if sha(data/name)!=digest:raise ValueError('Changed source data: '+name)
        if not all(cache_provenance['rules'].values()):raise ValueError('Cache lacks causal provenance')
    def dataset(split):
        if cache_run is None:return build(data,split)
        print('Reusing verified causal cache:',split,flush=True)
        return pd.read_parquet(cache_run/'artifacts'/f'{split}_dataset.parquet')
    # Cache scoped to this run/configuration; never trust a stale shared cache.
    print('Building causal train features (CPU; raw CSV -> cutoff at each T)...',flush=True)
    frame=dataset('train');frame.to_parquet(art/'train_dataset.parquet',index=False)
    if frame.sample_id.duplicated().any():raise ValueError('Duplicate training samples')
    hold=frame['T'].quantile(config['holdout_quantile'])
    quantiles=[frame['T'].quantile(q) for q in config['cv_quantiles']]
    if quantiles[-1]!=hold:raise ValueError('CV must end at the start of final holdout')
    results=[];fold_records=[];oof=[];failed=[]
    for cfg in config['candidates']:
        errors=[];trees=[]
        for fold,(begin,end) in enumerate(zip(quantiles[:-1],quantiles[1:])):
            tr=past(frame,begin);val=frame[(frame['T']>=begin)&(frame['T']<end)&real_mask(frame,config)]
            if len(tr)<100 or len(val)<20:raise ValueError('Insufficient fold data')
            try:
                model,cols=fit_one(tr,val,cfg,config)
            except CatBoostError as exc:
                device='CPU' if config['task_type']=='CPU' else cfg.get('task_type',config['task_type'])
                if device!='GPU':raise
                failed.append({'candidate':cfg['name'],'fold':fold,'error':str(exc)})
                print('GPU candidate failed and is excluded:',failed[-1],flush=True)
                break
            p=prediction(model,val,cols,cfg['residual'],cfg.get('target_scale',1))
            errors.extend(np.abs(val.target_delay_s.to_numpy()-p));trees.append(model.tree_count_)
            record={'candidate':cfg['name'],'fold':fold,'cutoff':str(begin),'end':str(end),'train_n':int((weights(tr,cfg,config)>0).sum()),'real_validation_n':len(val),'mae':mae(val.target_delay_s,p),'baseline_mae':mae(val.target_delay_s,val.cur_dev_s),'trees':model.tree_count_,'fitted_params':{k:model.get_all_params().get(k) for k in ['task_type','boosting_type','leaf_estimation_method','loss_function','learning_rate']}}
            fold_records.append(record);print(json.dumps(record),flush=True)
            saved=val[['sample_id','tr_id','T','target_time_begin','target_delay_s','cur_dev_s']].copy();saved['prediction']=p;saved['candidate']=cfg['name'];saved['fold']=fold;oof.append(saved)
        if len(trees)==len(quantiles)-1:
            results.append({'candidate':cfg['name'],'cv_mae':float(np.mean(errors)),'iterations':int(np.median(trees))})
    json_write(art/'failed_candidates.json',failed)
    if not results:raise RuntimeError('No candidate completed all folds')
    pd.DataFrame(fold_records).to_csv(art/'fold_metrics.csv',index=False)
    pd.DataFrame(results).to_csv(art/'experiments.csv',index=False)
    pd.concat(oof).to_parquet(art/'oof_predictions.parquet',index=False)
    winner=min(results,key=lambda r:r['cv_mae']);cfg=next(c for c in config['candidates'] if c['name']==winner['candidate']);cols=columns(frame,cfg['features'])
    # Final holdout is never used to choose candidate, iterations, seed or blend.
    tr=past(frame,hold);val=frame[(frame['T']>=hold)&real_mask(frame,config)]
    model=estimator(cfg,config,max(20,winner['iterations']));w=weights(tr,cfg,config);keep=w>0
    scale=cfg.get('target_scale',1)
    model.fit(pool(tr.loc[keep],cols,cfg['residual'],scale,w[keep]))
    model.save_model(str(models/'temporal_model.cbm'))
    hp=prediction(model,val,cols,cfg['residual'],scale)
    actual_device=model.get_all_params().get('task_type','CPU')
    metrics={'selection_cv_mae':winner['cv_mae'],'candidate':cfg['name'],'holdout_cutoff':str(hold),'holdout_real_n':len(val),'holdout_mae':mae(val.target_delay_s,hp),'holdout_baseline_mae':mae(val.target_delay_s,val.cur_dev_s),'task_type':actual_device,'gpu_executed':actual_device=='GPU'}
    vh=val[['sample_id','tr_id','T','target_delay_s','cur_dev_s']].copy();vh['prediction']=hp;vh.to_parquet(art/'holdout_predictions.parquet',index=False)
    pd.DataFrame({'feature':cols,'importance':model.feature_importances_}).sort_values('importance',ascending=False).to_csv(art/'feature_importance.csv',index=False)
    members=[];final_models=[]
    for i,seed in enumerate(config['seeds']):
        final=estimator(cfg,config,max(20,winner['iterations']),seed);w=weights(frame,cfg,config);keep=w>0
        final.fit(pool(frame.loc[keep],cols,cfg['residual'],scale,w[keep]))
        name='best_model.cbm' if i==0 else f'ensemble_{seed}.cbm';final.save_model(str(models/name));final_models.append(final)
        members.append({'file':name,'weight':1/len(config['seeds']),'residual':cfg['residual'],'target_scale':scale,'sha256':sha(models/name)})
    metadata={'model_version':hashlib.sha256(''.join(m['sha256'] for m in members).encode()).hexdigest()[:12],'feature_version':FEATURE_VERSION,'features':cols,'categorical_features':[c for c in cols if c in CAT_FEATURES],'residual':cfg['residual'],'target_scale':scale,'members':members,'horizon_seconds':[600,900],'training_source':'labels_train only; purged forward CV; test diagnostic only','metrics':metrics}
    json_write(models/'metadata.json',metadata)
    # Only now load test: no test metric can influence the selected model.
    print('Selected model:',winner,'; building diagnostic test features...',flush=True)
    test=dataset('test');test.to_parquet(art/'test_dataset.parquet',index=False)
    tp=np.mean([prediction(m,test,cols,cfg['residual'],scale) for m in final_models],axis=0)
    metrics.update(test_n=len(test),test_mae=mae(test.target_delay_s,tp),test_baseline_mae=mae(test.target_delay_s,test.cur_dev_s),test_zero_mae=mae(test.target_delay_s,np.zeros(len(test))))
    td=test[['sample_id','tr_id','T','target_delay_s','cur_dev_s']].copy();td['prediction']=tp;td.to_parquet(art/'test_predictions.parquet',index=False)
    print('Building validate features and submission (no target facts)...',flush=True)
    validate=dataset('validate');validate.to_parquet(art/'validate_dataset.parquet',index=False)
    vp=np.mean([prediction(m,validate,cols,cfg['residual'],scale) for m in final_models],axis=0)
    template=pd.read_csv(data/'sample_submission.csv',sep=';')
    if list(template.columns)!=['sample_id','prediction'] or not template.sample_id.is_unique or set(template.sample_id)!=set(validate.sample_id):raise ValueError('Invalid submission template/coverage')
    template['prediction']=template.sample_id.map(dict(zip(validate.sample_id,vp)))
    if not np.isfinite(template.prediction).all():raise ValueError('Non-finite submission')
    template.to_csv(art/'submission.csv',sep=';',index=False)
    metrics['elapsed_s']=time.perf_counter()-start;json_write(art/'metrics.json',metrics)
    metadata['metrics']=metrics;json_write(models/'metadata.json',metadata)
    inputs=['train/traffic.csv','train/schedule.csv','labels/labels_train.csv','test/traffic.csv','test/schedule.csv','labels/labels_test.csv','validate/traffic.csv','validate/schedule_plan.csv','validate/points.csv']
    provenance={'config':config,'python':platform.python_version(),'catboost':catboost.__version__,'numpy':np.__version__,'pandas':pd.__version__,'feature_version':FEATURE_VERSION,'feature_columns':cols,'input_sha256':{p:sha(data/p) for p in inputs},'code_sha256':{p:sha(Path(__file__).parent/p) for p in ['features.py','kaggle_train.py','build_dataset.py','model.py']},'rules':{'telemetry_event_lte_T':True,'receive_and_gps_lte_T':True,'facts_not_features':True,'no_test_selection':True,'no_validate_labels':True,'holdout_not_used_for_selection':True}}
    json_write(art/'provenance.json',provenance);print(json.dumps(metrics,indent=2),flush=True)
    if cache_run is not None:
        provenance['feature_cache_source']=str(cache_run)
        provenance['feature_cache_provenance_sha256']=sha(cache_run/'artifacts/provenance.json')
        provenance['feature_cache_sha256']={split:sha(cache_run/'artifacts'/f'{split}_dataset.parquet') for split in ['train','test','validate']}
        json_write(art/'provenance.json',provenance)
    return metrics

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path('data/raw'));p.add_argument('--output',type=Path,default=Path('runs/kaggle'));p.add_argument('--config',type=Path,default=Path('configs/kaggle.json'));p.add_argument('--task-type',choices=['CPU','GPU']);p.add_argument('--quick',action='store_true');args=p.parse_args()
    config=json.loads(args.config.read_text())
    if args.task_type:config['task_type']=args.task_type
    if args.quick:config.update(iterations=300,early_stopping_rounds=50,seeds=[42])
    train(args.data,args.output,config)
