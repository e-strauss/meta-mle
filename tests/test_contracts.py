import pandas as pd
import pytest
import skrub
from sklearn.model_selection import KFold, GroupKFold
from sklearn.linear_model import Ridge
from meta_mle.contracts import ContractDrift, create_contract, source_manifest, verify_contract
from meta_mle.evaluation import audit_boundary
from meta_mle.models import EvaluationSpec, Task


def boundary(path, cv=None, filtered=False, shifted=False):
    table = skrub.as_data_op(str(path)).skb.apply_func(pd.read_csv)
    if filtered:
        table = table[table['a'] > 0]
    X = table.drop(columns=['target']).skb.mark_as_X(cv=cv or KFold(3))
    y = (table['target'] + 1 if shifted else table['target']).skb.mark_as_y()
    return {'X': X, 'y': y, 'scoring': 'r2'}


@pytest.fixture
def data(tmp_path):
    path = tmp_path / 'rows.csv'
    pd.DataFrame({'a': range(12), 'group': [0,1,2]*4, 'target': range(12)}).to_csv(path,index=False)
    return path


def lock(result):
    return create_contract(audit_boundary(result), 'locked code', EvaluationSpec(scoring='r2', rationale='test'))


def test_equivalent_graph_and_downstream_features(data):
    contract = lock(boundary(data))
    verify_contract(contract)
    rebuilt = boundary(data)
    pred = rebuilt['X'].drop(columns=['group']).skb.apply(Ridge(), y=rebuilt['y']).skb.set_name('model')
    assert audit_boundary({'pred': pred, 'scoring':'r2'}, contract)['fold_fingerprint'] == contract['fold_fingerprint']


@pytest.mark.parametrize('kwargs,component', [({'filtered':True},'X'),({'shifted':True},'y'),({'cv':KFold(4)},'cv')])
def test_drift_rejected_before_materialization(data, kwargs, component):
    contract = lock(boundary(data))
    changed = boundary(data, **kwargs)
    data.unlink()  # graph mismatch is detected before even reading
    with pytest.raises(ContractDrift, match=component):
        audit_boundary(changed, contract)


def test_metadata_integrity_but_no_source_content_check(data):
    contract = lock(boundary(data))
    with pytest.raises(ValueError, match='modified'):
        verify_contract({**contract,'rows':100})
    data.write_text('changed')
    verify_contract(contract)
    manifest = source_manifest(Task(description='test',sources={'remote':'gs://bucket/table.parquet'}))
    assert manifest == {'remote': {'path':'gs://bucket/table.parquet'}}


def test_groups_are_graph_defined(data):
    table = skrub.as_data_op(str(data)).skb.apply_func(pd.read_csv)
    X = table.drop(columns=['target']).skb.mark_as_X(cv=GroupKFold(3), split_kwargs={'groups':table['group']})
    snapshot = audit_boundary({'X':X,'y':table['target'].skb.mark_as_y(),'scoring':'r2'})
    frame = pd.read_csv(data)
    for fold in snapshot['splits']:
        assert set(frame.iloc[fold['train']]['group']).isdisjoint(frame.iloc[fold['test']]['group'])
