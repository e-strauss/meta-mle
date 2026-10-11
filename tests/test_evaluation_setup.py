"""Realistic multi-table, derived-label setup with an agent-defined splitter."""
import pandas as pd
from meta_mle.execution import run_plan
from meta_mle.contracts import create_contract
from meta_mle.models import EvaluationSpec
from meta_mle.plans import evaluation_source


def test_custom_population_labels_and_cutoff_cv(tmp_path):
    lots = tmp_path / 'lots.csv'
    events = tmp_path / 'events.csv'
    pd.DataFrame({'bbl':range(1,13),'units':[2,4,5]*4,'cutoff':[0]*4+[1]*4+[2]*4}).to_csv(lots,index=False)
    pd.DataFrame({'bbl':range(1,13),'violations':range(12)}).to_csv(events,index=False)
    source = f'''import pandas as pd
import numpy as np
import skrub
class CutoffSplit:
    def split(self, X, y=None):
        cutoffs = X['cutoff'].to_numpy()
        for cutoff in [1, 2]:
            yield np.flatnonzero(cutoffs < cutoff), np.flatnonzero(cutoffs == cutoff)
    def get_n_splits(self, X=None, y=None):
        return 2

def read_table(path):
    return skrub.as_data_op(path).skb.apply_func(pd.read_csv)

def build():
    lots = read_table({str(lots)!r})
    events = read_table({str(events)!r})
    eligible = lots[lots['units'] >= 3]
    population = eligible.merge(events, on='bbl', how='left')
    X = population.drop(columns=['violations']).skb.mark_as_X(cv=CutoffSplit())
    y = population['violations'].fillna(0).skb.mark_as_y()
    return {{'X':X, 'y':y, 'row_keys':population[['bbl','cutoff']], 'scoring':'neg_mean_absolute_error',
            'audit': population.groupby('cutoff').size()}}
'''
    setup = run_plan(tmp_path/'setup',source,{'sources': {'tmp': {'path': str(tmp_path)}}, 'kind':'evaluation'},60)
    assert setup['status'] == 'ok', setup
    assert setup['snapshot']['rows'] == 8
    assert setup['snapshot']['audit']['folds'] == 2 and (tmp_path / 'setup' / 'folds.npz').exists()
    assert 'audit' in setup['outputs']  # a single audit DataOp is accepted
    contract = create_contract(setup['snapshot'], evaluation_source(source), EvaluationSpec(scoring='neg_mean_absolute_error',rationale='Future cutoffs'))
    candidate = contract['setup_source'] + '''
from sklearn.linear_model import Ridge
def build():
    setup = build_evaluation()
    pred = setup['X'].drop(columns=['bbl']).skb.apply(Ridge(), y=setup['y']).skb.set_name('model')
    return {'pred':pred, 'scoring':setup['scoring'], 'row_keys':setup['row_keys']}
'''
    scored = run_plan(tmp_path/'candidate',candidate,{'sources': {'tmp': {'path': str(tmp_path)}}, 'kind':'pipeline','contract':contract,'folds_path':str(tmp_path/'setup'/'folds.npz'),'remaining_evaluations':1},60)
    assert scored['status'] == 'ok', scored
    assert scored['evaluation_count'] == 1


def test_attempt_timings_cover_lazy_build_audit_and_search(tmp_path):
    data = tmp_path / 'train.csv'
    pd.DataFrame({'a': range(30), 'target': [i * 0.5 for i in range(30)]}).to_csv(data, index=False)
    source = f'''import pandas as pd
import skrub
from sklearn.model_selection import KFold

def build():
    data = skrub.as_data_op({str(data)!r}).skb.apply_func(pd.read_csv)
    X = data.drop(columns=['target']).skb.mark_as_X(cv=KFold(3, shuffle=True, random_state=0), split_kwargs={{}})
    y = data['target'].skb.mark_as_y()
    return {{'X': X, 'y': y, 'scoring': 'neg_mean_absolute_error'}}
'''
    setup = run_plan(tmp_path / 'setup', source, {'sources': {'tmp': {'path': str(tmp_path)}}, 'kind': 'evaluation'}, 60)
    assert setup['status'] == 'ok', setup
    contract = create_contract(setup['snapshot'], evaluation_source(source),
                               EvaluationSpec(scoring='neg_mean_absolute_error', rationale='iid rows'))
    candidate = contract['setup_source'] + '''
from sklearn.linear_model import Ridge
def build():
    setup = build_evaluation()
    model = Ridge(alpha=skrub.choose_from([0.1, 1.0], name='alpha'))
    return {'pred': setup['X'].skb.apply(model, y=setup['y']).skb.set_name('model'), 'scoring': setup['scoring']}
'''
    scored = run_plan(tmp_path / 'candidate', candidate,
                      {'sources': {'tmp': {'path': str(tmp_path)}}, 'kind': 'pipeline', 'contract': contract, 'folds_path': str(tmp_path / 'setup' / 'folds.npz'), 'remaining_evaluations': 2}, 60)
    assert scored['status'] == 'ok', scored
    timings = scored['timings']
    names = [p['phase'] for p in timings['phases']]
    for phase in ('imports', 'load_request', 'verify_contract', 'build_graph', 'audit_fingerprint',
                  'audit_eval_xy', 'audit_checks', 'learner_and_grid', 'grid_search',
                  'grid_search.fold_fit', 'grid_search.fold_score', 'grid_search.overhead'):
        assert phase in names, (phase, names)
    assert timings['in_progress'] is None
    search = next(p for p in timings['phases'] if p['phase'] == 'grid_search')
    assert search['variants'] == 2 and search['folds'] == 3
    # Fold times are compute time summed over the parallel jobs.
    seconds = {p['phase']: p['seconds'] for p in timings['phases'] if p.get('part_of') == 'grid_search'}
    compute = seconds['grid_search.fold_fit'] + seconds['grid_search.fold_score']
    assert search['n_jobs'] == 2
    assert compute / search['n_jobs'] + seconds['grid_search.overhead'] <= search['seconds'] + 1e-6
    assert scored['wall_s'] >= timings['elapsed_s']
    assert (tmp_path / 'candidate' / 'timings.json').exists()


def test_timeout_reports_phase_in_progress(tmp_path):
    source = "import numpy as np\ndef build():\n    while True:\n        np.zeros(1)\n"
    result = run_plan(tmp_path / 'slow', source, {'sources': {'tmp': {'path': str(tmp_path)}}, 'kind': 'exploration'}, 8)
    assert result['status'] == 'failed' and 'timed out' in result['error']
    assert result['timings']['in_progress'] == 'build_graph'


def test_custom_scorer_is_locked_with_the_setup(tmp_path):
    data = tmp_path / 'pairs.csv'
    rows = [{'g': g, 't': t, 'a': (g * 7 + t * 3) % 5, 'target': int((g + t) % 3 == 0)}
            for g in range(24) for t in range(5)]
    pd.DataFrame(rows).to_csv(data, index=False)
    source = f'''import numpy as np
import pandas as pd
import skrub
from sklearn.model_selection import GroupKFold

K = 2

def top_k_recall(estimator, X, y):
    scores = estimator.predict_proba(X)[:, 1]
    frame = pd.DataFrame({{'g': X['g'].to_numpy(), 'y': np.asarray(y), 's': scores}})
    top = frame.sort_values(['g', 's'], ascending=[True, False]).groupby('g').head(K)
    hits = top.groupby('g')['y'].sum()
    total = frame.groupby('g')['y'].sum()
    return float((hits / total.clip(lower=1)).mean())

def build():
    data = skrub.as_data_op({str(data)!r}).skb.apply_func(pd.read_csv)
    X = data[['g', 't', 'a']].skb.mark_as_X(cv=GroupKFold(3), split_kwargs={{'groups': data['g']}})
    y = data['target'].skb.mark_as_y()
    return {{'X': X, 'y': y, 'scoring': top_k_recall}}
'''
    setup = run_plan(tmp_path / 'setup', source, {'sources': {'tmp': {'path': str(tmp_path)}}, 'kind': 'evaluation', 'expected_scoring': 'recall@2'}, 60)
    assert setup['status'] == 'ok', setup
    assert setup['snapshot']['scoring'] == 'custom:top_k_recall'
    contract = create_contract(setup['snapshot'], evaluation_source(source),
                               EvaluationSpec(scoring='recall@2', rationale='per-group top-k'))
    pipeline = '''
from sklearn.linear_model import LogisticRegression
def build():
    setup = build_evaluation()
    pred = setup['X'][['t', 'a']].skb.apply(LogisticRegression(), y=setup['y']).skb.set_name('model')
    return {'pred': pred, 'scoring': setup['scoring']}
'''
    scored = run_plan(tmp_path / 'candidate', contract['setup_source'] + pipeline,
                      {'sources': {'tmp': {'path': str(tmp_path)}}, 'kind': 'pipeline', 'contract': contract, 'folds_path': str(tmp_path / 'setup' / 'folds.npz'), 'remaining_evaluations': 1}, 60)
    assert scored['status'] == 'ok', scored
    assert 0 <= scored['variants'][0]['score'] <= 1
    # Run by the harness before the plan instead of pasted, the scorer is locked alike.
    prelude = run_plan(tmp_path / 'prelude', pipeline,
                       {'sources': {'tmp': {'path': str(tmp_path)}}, 'kind': 'pipeline', 'contract': contract, 'setup_source': contract['setup_source'], 'folds_path': str(tmp_path / 'setup' / 'folds.npz'), 'remaining_evaluations': 1}, 60)
    assert prelude['status'] == 'ok', prelude
    assert prelude['variants'][0]['score'] == scored['variants'][0]['score']
    drifted = contract['setup_source'].replace('K = 2', 'K = 3') + pipeline
    rejected = run_plan(tmp_path / 'drift', drifted,
                        {'sources': {'tmp': {'path': str(tmp_path)}}, 'kind': 'pipeline', 'contract': contract, 'folds_path': str(tmp_path / 'setup' / 'folds.npz'), 'remaining_evaluations': 1}, 60)
    assert rejected['status'] == 'failed' and rejected.get('warning') == 'contract_drift'
    assert 'scoring' in rejected['changed_components']


def test_probe_returns_out_of_fold_probabilities_with_row_keys(tmp_path):
    data = tmp_path / 'pairs.csv'
    rows = [{'g': g, 't': t, 'a': (g * 7 + t * 3) % 5, 'target': int((g + t) % 3 == 0)}
            for g in range(24) for t in range(5)]
    pd.DataFrame(rows).to_csv(data, index=False)
    source = f'''import pandas as pd
import skrub
from sklearn.model_selection import GroupKFold

def build():
    data = skrub.as_data_op({str(data)!r}).skb.apply_func(pd.read_csv)
    X = data[['g', 't', 'a']].skb.mark_as_X(cv=GroupKFold(3), split_kwargs={{'groups': data['g']}})
    y = data['target'].skb.mark_as_y()
    keys = data['g'].astype(str) + ':' + data['t'].astype(str)
    return {{'X': X, 'y': y, 'scoring': 'roc_auc', 'row_keys': keys}}
'''
    setup = run_plan(tmp_path / 'setup', source, {'sources': {'tmp': {'path': str(tmp_path)}}, 'kind': 'evaluation'}, 60)
    assert setup['status'] == 'ok', setup
    contract = create_contract(setup['snapshot'], evaluation_source(source),
                               EvaluationSpec(scoring='roc_auc', rationale='grouped'))
    body = '''
from sklearn.linear_model import LogisticRegression
def build():
    setup = build_evaluation()
    pred = setup['X'][['t', 'a']].skb.apply(LogisticRegression(C={c}), y=setup['y']).skb.set_name('model')
    return {{'pred': pred, 'scoring': setup['scoring'], 'row_keys': setup['row_keys']}}
'''
    probed = run_plan(tmp_path / 'probe', contract['setup_source'] + body.format(c='1.0'),
                      {'sources': {'tmp': {'path': str(tmp_path)}}, 'kind': 'probe', 'contract': contract, 'folds_path': str(tmp_path / 'setup' / 'folds.npz')}, 60)
    assert probed['status'] == 'ok', probed
    oof = pd.read_parquet(probed['probe']['path'])
    assert list(oof.columns) == ['row', 'row_key', 'fold', 'y', 'prediction']
    assert sorted(oof['row']) == list(range(120)) and oof['prediction'].between(0, 1).all()
    assert oof.loc[oof['row'] == 7, 'row_key'].item() == '1:2'
    assert probed['probe']['summary']['kind'] == 'binary probability'
    assert probed['evaluation_count'] == 0
    grid = run_plan(tmp_path / 'grid', contract['setup_source'] + body.format(c="skrub.choose_from([0.1, 1.0], name='C')"),
                    {'sources': {'tmp': {'path': str(tmp_path)}}, 'kind': 'probe', 'contract': contract, 'folds_path': str(tmp_path / 'setup' / 'folds.npz')}, 60)
    assert grid['status'] == 'failed' and 'single value' in grid['error']


def test_pipelines_run_after_the_locked_setup_and_keep_their_predictions(tmp_path, monkeypatch):
    data = tmp_path / 'pairs.csv'
    rows = [{'g': g, 't': t, 'a': (g * 7 + t * 3) % 5, 'target': int((g + t) % 3 == 0)}
            for g in range(24) for t in range(5)]
    pd.DataFrame(rows).to_csv(data, index=False)
    # Two-column row keys, as a setup that keys rows by entity and time declares them.
    source = f'''import pandas as pd
import skrub
from sklearn.model_selection import GroupKFold

def build():
    data = skrub.as_data_op({str(data)!r}).skb.apply_func(pd.read_csv)
    X = data[['g', 't', 'a']].skb.mark_as_X(cv=GroupKFold(3), split_kwargs={{'groups': data['g']}})
    y = data['target'].skb.mark_as_y()
    return {{'X': X, 'y': y, 'scoring': 'roc_auc', 'row_keys': data[['g', 't']]}}
'''
    setup = run_plan(tmp_path / 'setup', source, {'sources': {'tmp': {'path': str(tmp_path)}}, 'kind': 'evaluation'}, 60)
    assert setup['status'] == 'ok', setup
    contract = create_contract(setup['snapshot'], evaluation_source(source),
                               EvaluationSpec(scoring='roc_auc', rationale='grouped'))
    # The plan does not paste the setup: the harness runs it first.
    body = '''
from sklearn.linear_model import LogisticRegression
def build():
    setup = build_evaluation()
    pred = setup['X'][['t', 'a']].skb.apply(LogisticRegression(C={c}), y=setup['y']).skb.set_name('model')
    return {{'pred': pred, 'scoring': setup['scoring'], 'row_keys': setup['row_keys']}}
'''
    request = {'sources': {'tmp': {'path': str(tmp_path)}}, 'contract': contract,
               'setup_source': contract['setup_source'], 'folds_path': str(tmp_path / 'setup' / 'folds.npz')}
    probed = run_plan(tmp_path / 'probe', body.format(c='1.0'), {**request, 'kind': 'probe'}, 60)
    assert probed['status'] == 'ok', probed
    oof = pd.read_parquet(probed['probe']['path'])
    assert list(oof.columns) == ['row', 'row_key_g', 'row_key_t', 'fold', 'y', 'prediction']
    assert oof.loc[oof['row'] == 7, ['row_key_g', 'row_key_t']].values.tolist() == [[1, 2]]
    grid = body.format(c="skrub.choose_from([0.1, 1.0], name='C')")
    parallel = run_plan(tmp_path / 'parallel', grid, {**request, 'kind': 'pipeline', 'remaining_evaluations': 2}, 60)
    # Folds scored in parallel arrive in no fixed order, so no predictions are kept.
    assert parallel['status'] == 'ok' and all('oof_file' not in v for v in parallel['variants'])
    config = tmp_path / 'sequential.toml'
    config.write_text('[execution]\ncpu_threads = 2\ngrid_n_jobs = 1\n')
    monkeypatch.setenv('META_MLE_CONFIG', str(config))
    scored = run_plan(tmp_path / 'candidate', grid, {**request, 'kind': 'pipeline', 'remaining_evaluations': 2}, 60)
    assert scored['status'] == 'ok', scored
    for variant in scored['variants']:
        kept = pd.read_parquet(tmp_path / 'candidate' / variant['oof_file'])
        assert sorted(kept['row']) == list(range(120))
    # The second variant (C=1.0) is the probed configuration: same predictions.
    kept = pd.read_parquet(tmp_path / 'candidate' / scored['variants'][1]['oof_file'])
    assert kept['prediction'].tolist() == oof['prediction'].tolist()


def test_contract_references_folds_instead_of_inlining_them(tmp_path):
    import json
    from meta_mle.contracts import contract_folds

    data = tmp_path / 'train.csv'
    pd.DataFrame({'a': range(30), 'target': range(30)}).to_csv(data, index=False)
    source = f'''import pandas as pd
import skrub
from sklearn.model_selection import KFold

def build():
    data = skrub.as_data_op({str(data)!r}).skb.apply_func(pd.read_csv)
    X = data.drop(columns=['target']).skb.mark_as_X(cv=KFold(3), split_kwargs={{}})
    return {{'X': X, 'y': data['target'].skb.mark_as_y(), 'scoring': 'r2'}}
'''
    setup = run_plan(tmp_path / 'setup', source, {'sources': {'tmp': {'path': str(tmp_path)}}, 'kind': 'evaluation'}, 60)
    contract = create_contract(setup['snapshot'], evaluation_source(source),
                               EvaluationSpec(scoring='r2', rationale='iid'), folds_file='folds.npz')
    assert 'splits' not in contract and len(json.dumps(contract)) < 20_000
    folds = contract_folds(contract, tmp_path / 'setup' / 'folds.npz')
    assert sorted(int(i) for f in folds for i in f['test']) == list(range(30))
    corrupted = tmp_path / 'other.npz'
    import numpy as np
    np.savez(corrupted, train_0=np.arange(10), test_0=np.arange(10, 20))
    import pytest
    with pytest.raises(ValueError, match='does not match'):
        contract_folds(contract, corrupted)


def test_multi_output_labels_lock_and_probe(tmp_path):
    data = tmp_path / 'multi.csv'
    pd.DataFrame([{'a': i % 7, 'b': i % 5, 'l1': int(i % 7 > 3), 'l2': int(i % 5 > 1)}
                  for i in range(60)]).to_csv(data, index=False)
    source = f'''import pandas as pd
import skrub
from sklearn.model_selection import KFold

def build():
    data = skrub.as_data_op({str(data)!r}).skb.apply_func(pd.read_csv)
    X = data[['a', 'b']].skb.mark_as_X(cv=KFold(3), split_kwargs={{}})
    y = data[['l1', 'l2']].skb.mark_as_y()
    return {{'X': X, 'y': y, 'scoring': 'accuracy'}}
'''
    setup = run_plan(tmp_path / 'setup', source, {'sources': {'tmp': {'path': str(tmp_path)}}, 'kind': 'evaluation'}, 60)
    assert setup['status'] == 'ok', setup
    contract = create_contract(setup['snapshot'], evaluation_source(source),
                               EvaluationSpec(scoring='accuracy', rationale='multi-output'))
    body = '''
from sklearn.tree import DecisionTreeClassifier
def build():
    setup = build_evaluation()
    return {'pred': setup['X'].skb.apply(DecisionTreeClassifier(random_state=0), y=setup['y']).skb.set_name('model'),
            'scoring': setup['scoring']}
'''
    probed = run_plan(tmp_path / 'probe', contract['setup_source'] + body,
                      {'sources': {'tmp': {'path': str(tmp_path)}}, 'kind': 'probe', 'contract': contract, 'folds_path': str(tmp_path / 'setup' / 'folds.npz')}, 60)
    assert probed['status'] == 'ok', probed
    oof = pd.read_parquet(probed['probe']['path'])
    assert len(oof) == 60 and len(oof['y'][0]) == 2 and len(oof['prediction'][0]) == 2


def test_pipeline_with_custom_transformer_and_estimator(tmp_path):
    data = tmp_path / 'rows.csv'
    pd.DataFrame([{'a': i % 7, 'b': i % 5, 'target': int(i % 7 > 3)} for i in range(60)]).to_csv(data, index=False)
    source = f'''import pandas as pd
import skrub
from sklearn.model_selection import KFold

def build():
    data = skrub.as_data_op({str(data)!r}).skb.apply_func(pd.read_csv)
    X = data[['a', 'b']].skb.mark_as_X(cv=KFold(3), split_kwargs={{}})
    return {{'X': X, 'y': data['target'].skb.mark_as_y(), 'scoring': 'roc_auc'}}
'''
    setup = run_plan(tmp_path / 'setup', source, {'sources': {'tmp': {'path': str(tmp_path)}}, 'kind': 'evaluation'}, 60)
    assert setup['status'] == 'ok', setup
    contract = create_contract(setup['snapshot'], evaluation_source(source),
                               EvaluationSpec(scoring='roc_auc', rationale='kfold'))
    body = '''
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.linear_model import LogisticRegression

class Center(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.means_ = X.mean()
        return self
    def transform(self, X):
        return X - self.means_

class Wrapped(ClassifierMixin, BaseEstimator):
    def __init__(self, C=1.0):
        self.C = C
    def fit(self, X, y):
        self._model = LogisticRegression(C=self.C).fit(X, y)
        self.classes_ = self._model.classes_
        return self
    def predict_proba(self, X):
        return self._model.predict_proba(X)

def build():
    setup = build_evaluation()
    features = setup['X'].skb.apply(Center())
    model = skrub.choose_from({'weak': Wrapped(C=0.01), 'strong': Wrapped(C=10.0)}, name='model')
    return {'pred': features.skb.apply(model, y=setup['y']).skb.set_name('classifier'), 'scoring': setup['scoring']}
'''
    result = run_plan(tmp_path / 'pipe', contract['setup_source'] + body,
                      {'sources': {'tmp': {'path': str(tmp_path)}}, 'kind': 'pipeline', 'contract': contract, 'remaining_evaluations': 4,
                       'folds_path': str(tmp_path / 'setup' / 'folds.npz')}, 120)
    assert result['status'] == 'ok', result
    assert result['evaluation_count'] == 2


def test_multiclass_probe_labels_probabilities_consistently(tmp_path):
    data = tmp_path / 'classes.csv'
    labels = [1] * 20 + [2] * 20 + [3] * 19 + [5]  # one row: absent from one fold's training data
    pd.DataFrame({'a': [i % 7 for i in range(60)], 'target': labels}).to_csv(data, index=False)
    source = f'''import pandas as pd
import skrub
from sklearn.model_selection import KFold

def build():
    data = skrub.as_data_op({str(data)!r}).skb.apply_func(pd.read_csv)
    X = data[['a']].skb.mark_as_X(cv=KFold(3, shuffle=True, random_state=0), split_kwargs={{}})
    return {{'X': X, 'y': data['target'].skb.mark_as_y(), 'scoring': 'accuracy'}}
'''
    setup = run_plan(tmp_path / 'setup', source, {'sources': {'tmp': {'path': str(tmp_path)}}, 'kind': 'evaluation'}, 60)
    assert setup['status'] == 'ok', setup
    contract = create_contract(setup['snapshot'], evaluation_source(source),
                               EvaluationSpec(scoring='accuracy', rationale='multiclass'))
    body = '''
from sklearn.linear_model import LogisticRegression
def build():
    setup = build_evaluation()
    return {'pred': setup['X'].skb.apply(LogisticRegression(max_iter=500), y=setup['y']).skb.set_name('model'),
            'scoring': setup['scoring']}
'''
    probed = run_plan(tmp_path / 'probe', contract['setup_source'] + body,
                      {'sources': {'tmp': {'path': str(tmp_path)}}, 'kind': 'probe', 'contract': contract,
                       'folds_path': str(tmp_path / 'setup' / 'folds.npz')}, 60)
    assert probed['status'] == 'ok', probed
    oof = pd.read_parquet(probed['probe']['path'])
    assert [c for c in oof.columns if c.startswith('proba_')] == ['proba_1', 'proba_2', 'proba_3', 'proba_5']
    assert set(oof['prediction']) <= {1, 2, 3, 5}
    sums = oof[['proba_1', 'proba_2', 'proba_3', 'proba_5']].sum(axis=1)
    assert ((sums - 1).abs() < 1e-6).all()
    missing = oof.groupby('fold')['proba_5'].max()
    assert (missing == 0).sum() == 1  # the fold whose training rows lack class 5
    summary = probed['probe']['summary']
    assert summary['kind'] == 'classification' and 'recall' in summary['per_class']
    assert 'confusion' in ' '.join(summary)
