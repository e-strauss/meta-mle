import numpy as np
import pandas as pd
import pytest
import skrub
from nano_mle.graphs import validate_graph
from nano_mle.plans import resolve_source, validate_source

@pytest.mark.parametrize('body', ["return open('features.csv')", "return data.to_csv('features.csv')", "return data.skb.eval()"])
def test_forbidden_execution(body):
    with pytest.raises(ValueError):
        validate_source('def build():\n    '+body+'\n')


def test_opaque_callbacks_and_materialized_data_rejected(tmp_path):
    with skrub.config_context(eager_data_ops=False):
        table = skrub.as_data_op(str(tmp_path / 'rows.csv')).skb.apply_func(pd.read_csv)
        for plan in [table.apply(lambda x:x), table.skb.apply_func(lambda x:x), table['a'].map(lambda x:x), skrub.as_data_op(pd.DataFrame({'a':[1]}))]:
            with pytest.raises(ValueError):
                validate_graph(plan)
        validate_graph(table['a'].map({1:2}))


def test_parent_grid_is_resolved_before_implementation():
    source = """import skrub
from sklearn.linear_model import Ridge
def build():
    return Ridge(alpha=skrub.choose_from([0.1, 10.0], name='alpha'))
"""
    resolved = resolve_source(source, {'alpha':1})
    assert 'choose_from' not in resolved
    assert 'alpha=10.0' in resolved


def test_unnamed_choices_rejected():
    with pytest.raises(ValueError, match='name'):
        validate_source('import skrub\ndef build():\n    return skrub.choose_from([1,2])\n')


def test_transformer_alias_cannot_hide_function_nodes():
    with pytest.raises(ValueError, match='function transformers'):
        validate_source('from sklearn.preprocessing import FunctionTransformer as F\ndef build():\n    return F()\n')


def test_graph_building_dictionary_helpers_allowed():
    validate_source('''def helper(data):
    outputs = {}
    for col in ['a', 'b']:
        outputs[col] = data[col].isna().sum()
    return outputs

def build():
    return helper(data)
''')


def test_eager_reader_alias_rejected():
    with pytest.raises(ValueError, match='Record readers'):
        validate_source('from pandas import read_csv as reader\ndef build():\n    return reader("data.csv")\n')


def test_locked_setup_export_preserves_existing_helper():
    from nano_mle.plans import evaluation_source
    source = '''def build_evaluation():
    return {'scoring': 'r2'}

def locked_setup_entry():
    return None

def build():
    return build_evaluation()
'''
    exported = evaluation_source(source)
    namespace = {}
    exec(exported, namespace)
    assert namespace['build_evaluation']() == {'scoring': 'r2'}


def test_lint_errors_name_the_rejected_construct():
    with pytest.raises(ValueError, match="Import of os is not allowed"):
        validate_source("import os\ndef build():\n    return {}\n")
    helper = "class Helper:\n    def run(self, x):\n        return x\ndef build():\n    return {}\n"
    with pytest.raises(ValueError, match="Class Helper.*fine-grained DataOps"):
        validate_source(helper)


def test_custom_estimators_and_torch_modules_are_allowed():
    validate_source('''
import numpy as np
import torch
from torch import nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression

class Net(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layer = nn.Linear(width, 1)
    def forward(self, x):
        return self.layer(x)

class Wrapped(ClassifierMixin, BaseEstimator):
    def __init__(self, C=1.0):
        self.C = C
    def fit(self, X, y):
        self._model = LogisticRegression(C=self.C).fit(X, y)
        self.classes_ = self._model.classes_
        net = Net(X.shape[1])
        net.eval()
        return self
    def predict_proba(self, X):
        return self._model.predict_proba(X)

def build():
    return {}
''')
    for body in ["        self.__dict__['a'] = 1", "        X.to_csv('x.csv')", "        torch.save(self, 'm.pt')"]:
        with pytest.raises(ValueError):
            validate_source("class E:\n    def fit(self, X, y=None):\n" + body + "\n        return self\n"
                            "def build():\n    return {}\n")
    with pytest.raises(ValueError, match="harness-owned"):
        validate_source("def build():\n    return model.fit(X)\n")

def test_installed_libraries_import_and_missing_ones_are_named():
    from nano_mle.plans import MissingLibrary

    validate_source("import lightgbm\nfrom xgboost import XGBClassifier\nimport polars as pl\n"
                    "def build():\n    return {}\n")
    with pytest.raises(MissingLibrary) as error:
        validate_source("import definitely_not_installed_pkg\ndef build():\n    return {}\n")
    assert error.value.modules == ["definitely_not_installed_pkg"]


def test_dtype_arguments_are_not_opaque_callables():
    data = skrub.var("data")  # no value: no eager preview
    validate_graph(data.assign(b=data["a"].astype(str), c=data["a"].astype(float)))
    validate_graph(data["a"].astype(np.float64))
    with pytest.raises(ValueError, match="'helper'.*clip"):
        def helper(x):
            return x
        helper.__module__ = "generated_plan"  # as if defined in the plan
        validate_graph(data["a"].clip(upper=helper))


def test_local_reader_helpers_are_not_eager_reads():
    helper = ("import pandas as pd\nimport skrub\n"
              "def read_parquet(name):\n"
              "    return skrub.as_data_op(name).skb.apply_func(pd.read_parquet)\n"
              "def build():\n    return {'t': read_parquet('t.parquet')}\n")
    validate_source(helper)
    for eager in ("import pandas as pd\ndef build():\n    return pd.read_parquet('t')\n",
                  "import pandas\ndef build():\n    return pandas.read_csv('t')\n",
                  "from pandas import read_csv as rc\ndef build():\n    return rc('t')\n"):
        with pytest.raises(ValueError, match="Record readers"):
            validate_source(eager)


def test_getattr_only_with_literal_public_name():
    validate_source('def build():\n    return getattr(est, "classes_", None)\n')
    for call in ['getattr(est, name)', 'getattr(est, "__class__")', 'getattr(est, "_private")', 'getattr(est, "fit")']:
        with pytest.raises(ValueError):
            validate_source('def build():\n    return ' + call + '\n')


def test_library_functions_allowed_unless_primitives_are_restricted(tmp_path, monkeypatch):
    import scipy.special
    from nano_mle import graphs

    with skrub.config_context(eager_data_ops=False):
        column = skrub.as_data_op(str(tmp_path / 'rows.csv')).skb.apply_func(pd.read_csv)['a']
        validate_graph(column.skb.apply_func(scipy.special.expit))
        validate_graph(column.skb.apply_func(np.log1p))
        planned = lambda x: x
        with pytest.raises(ValueError, match="not functions defined in the plan"):
            validate_graph(column.skb.apply_func(planned))
        monkeypatch.setattr(graphs, "load_config", lambda: {"plans": {"restrict_primitives": True}})
        validate_graph(column.skb.apply_func(np.log1p))
        with pytest.raises(ValueError):
            validate_graph(column.skb.apply_func(scipy.special.expit))


def test_config_defaults_and_unknown_keys(tmp_path, monkeypatch):
    from nano_mle import config

    monkeypatch.setenv("NANO_MLE_CONFIG", str(tmp_path / "missing.toml"))
    config.load_config.cache_clear()
    assert config.load_config() == config.DEFAULTS
    (tmp_path / "bad.toml").write_text("[execution]\nthreads = 4\n")
    monkeypatch.setenv("NANO_MLE_CONFIG", str(tmp_path / "bad.toml"))
    config.load_config.cache_clear()
    with pytest.raises(ValueError, match="Unknown keys"):
        config.load_config()
    config.load_config.cache_clear()


def test_data_volume_study_prompt_follows_config(tmp_path, monkeypatch):
    from nano_mle import config, prompts

    # The older flat key [prompts] data_volume_study still works.
    for text in ("[prompts]\ndata_volume_study = {}\n", "[prompts.default]\ndata_volume_study = {}\n"):
        for enabled in (True, False):
            path = tmp_path / "prompts.toml"
            path.write_text(text.format(str(enabled).lower()))
            monkeypatch.setenv("NANO_MLE_CONFIG", str(path))
            config.load_config.cache_clear()
            prompt_set = prompts.prompts("default", config.load_config()["prompts"]["default"])
            assert ("Data volume study" in prompt_set.writer()) == enabled
            assert ("Data volume study" in prompt_set.control()) == enabled
            # The population check is not optional: every role that shapes the lock sees it.
            for instructions in prompt_set.rendered().values():
                assert "Population check" in instructions
    monkeypatch.undo()
    config.load_config.cache_clear()


def test_plans_read_only_task_sources(tmp_path):
    from nano_mle.graphs import check_reads

    task = tmp_path / "task"
    task.mkdir()
    (task / "rows.csv").write_text("a\n1\n")
    earlier = tmp_path / "workspace" / "artifacts" / "out.csv"
    earlier.parent.mkdir(parents=True)
    earlier.write_text("a\n1\n")
    sources = {"task": {"path": str(task)}, "remote": {"path": "gs://bucket/data"}}
    with skrub.config_context(eager_data_ops=False):
        rows = skrub.as_data_op(str(task / "rows.csv")).skb.apply_func(pd.read_csv)
        hosts = rows["a"].astype(str).str.extract(r"^[a-zA-Z][a-zA-Z0-9+.-]*://([^/]+)", expand=False)
        parts = rows["a"].astype(str).str.split("/").str[0] + "/" + rows["a"].astype(str)
        remote = skrub.as_data_op("gs://bucket/data/part.parquet").skb.apply_func(pd.read_parquet)
        check_reads([rows, hosts, parts, remote], sources)
        reused = skrub.as_data_op(str(earlier)).skb.apply_func(pd.read_csv)
        with pytest.raises(ValueError, match="read only task sources"):
            check_reads([reused], sources)
        with pytest.raises(ValueError, match="read only task sources"):
            check_reads([skrub.as_data_op("gs://other/x.parquet").skb.apply_func(pd.read_parquet)], sources)


def test_choice_outcomes_may_live_in_a_variable_assigned_once():
    source = ("import skrub\nfrom sklearn.linear_model import Ridge\n\ndef build():\n"
              "    models = {'weak': Ridge(alpha=10.0), 'strong': Ridge(alpha=0.1)}\n"
              "    return skrub.choose_from(models, name='model')\n")
    resolved = resolve_source(source, {"model": 1})
    assert "choose_from" not in resolved and "alpha=0.1" in resolved
    for outcomes in ("make_models()", "models", "{**models}"):
        with pytest.raises(ValueError, match="outcomes must be a literal"):
            validate_source("import skrub\n\ndef build():\n    models = {}\n    models = {'a': 1}\n"
                            f"    return skrub.choose_from({outcomes}, name='m')\n")


def test_estimator_parameters_on_self_may_share_forbidden_names():
    cls = '''from sklearn.base import BaseEstimator, ClassifierMixin

class Model(BaseEstimator, ClassifierMixin):
    def __init__(self, subsample=0.8):
        self.subsample = subsample

    def fit(self, X, y):
        return self

def build():
    return Model()
'''
    validate_source(cls)
    with pytest.raises(ValueError, match="Unsupported operation in class"):
        validate_source(cls.replace("return self\n", "return X.skb.subsample(n=10)\n", 1))
    with pytest.raises(ValueError, match="Unsupported operation in class"):
        validate_source(cls.replace("self.subsample = subsample", "self.__dict__['subsample'] = subsample"))


def test_eager_reads_from_any_library_are_rejected():
    for eager in ("import pyarrow as pa\ndef build():\n    return pa.parquet.ParquetFile('t')\n",
                  "import pyarrow.parquet as pq\ndef build():\n    return pq.read_table('t')\n",
                  "import polars as pl\ndef build():\n    return pl.scan_parquet('t')\n",
                  "import numpy as np\ndef build():\n    return np.loadtxt('t')\n"):
        with pytest.raises(ValueError, match="never eager reads|record reads"):
            validate_source(eager)


def test_plans_keep_no_state_between_fits():
    head = "import numpy as np\nfrom sklearn.base import BaseEstimator\n_CACHE = {}\n"
    cached = head + ("class E(BaseEstimator):\n    def fit(self, X, y):\n"
                     "        _CACHE['graph'] = np.zeros(3)\n        return self\n"
                     "def build():\n    return {}\n")
    updated = head + "def helper():\n    _CACHE.update(a=1)\ndef build():\n    return {}\n"
    globalled = head + "def build():\n    global _CACHE\n    return {}\n"
    class_dict = ("from sklearn.base import BaseEstimator\nclass E(BaseEstimator):\n    store = {}\n"
                  "    def fit(self, X, y):\n        return self\ndef build():\n    return {}\n")
    for source in (cached, updated, globalled, class_dict):
        with pytest.raises(ValueError, match="no state between fits"):
            validate_source(source)
    # Module constants and locals that shadow them are fine.
    validate_source("FIT = {'rows': 'train.csv'}\nPREDICT = {'rows': 'test.csv'}\n"
                    "def build():\n    FIT = {}\n    FIT['x'] = 1\n    return {}\n")


def test_torch_estimators_may_log_and_set_flags():
    validate_source('''
import numpy as np
import torch
from torch import nn
from sklearn.base import BaseEstimator, ClassifierMixin


class Net(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(4, width), nn.ReLU(), nn.Linear(width, 3))

    def forward(self, x):
        return self.layers(x)


class MLP(ClassifierMixin, BaseEstimator):
    def __init__(self, width=8):
        self.width = width

    def fit(self, X, y):
        torch.backends.cudnn.benchmark = True
        self.net_ = Net(self.width)
        for p in self.net_.parameters():
            p.requires_grad = True
        print("epoch 0 loss", 0.0)
        self.net_.eval()
        return self

def build():
    return {}
''')
    with pytest.raises(ValueError, match="Unsupported name: print"):
        validate_source("def build():\n    print('x')\n    return {}\n")


def test_hardware_comes_from_the_config_only(tmp_path, monkeypatch):
    from nano_mle import config

    path = tmp_path / "hw.toml"
    path.write_text('[execution]\ncpu_threads = 16\ngrid_n_jobs = 2\n'
                    '[hardware]\nmemory_gb = 512\ngpus = ["NVIDIA A100-SXM4-80GB, 80 GB"]\n')
    monkeypatch.setenv("NANO_MLE_CONFIG", str(path))
    config.load_config.cache_clear()
    stated = config.hardware()
    assert stated["cpu_threads"] == 16 and stated["threads_per_pipeline_fit"] == 8
    assert stated["memory_gb"] == 512 and stated["gpus"] == ["NVIDIA A100-SXM4-80GB, 80 GB"]
    assert "GPU is idle" in stated["note"]
    # Unset values are reported as not stated, never probed from the machine.
    path.write_text('[execution]\ncpu_threads = 0\n')
    config.load_config.cache_clear()
    stated = config.hardware()
    assert stated["cpu_threads"] == stated["memory_gb"] == "not stated" and stated["gpus"] == []
    monkeypatch.undo()
    config.load_config.cache_clear()


def test_model_choice_conventions_reach_controller_and_planner():
    from nano_mle import prompts

    for instructions in prompts.prompts().rendered().values():
        assert "Model families" in instructions and "context.hardware" in instructions
        assert "Training pool" in instructions
