"""Generated plans own their reads and X/y construction; graph checks enforce style.

Source lint is not a security sandbox or a proof of no leakage.
"""

import ast

from .config import load_config


_GUIDE = """
Plan contract (checked by the harness):
- Standalone Python with imports, constants, graph-building helpers and one
  zero-argument build(). The harness builds the graph lazily and evaluates it.
- Record reads: skrub.as_data_op(path).skb.apply_func(pd.read_csv | pd.read_parquet, ...)
  with columns/filters/sep as needed. Plans read only the task sources; files
  written by earlier steps (outputs, predictions, manifests) are never inputs, so
  recompute anything you want to reuse in the plan.
- Compute with fine-grained DataOps: indexing, filters, joins, groupby/agg, assign,
  string/date operations, .skb.concat. Build-time helpers and loops are fine when they
  emit explicit graph nodes.
  {apply_func}
  No deferred, UDFs (functions or lambdas defined in the plan), callable
  apply/map/transform, eager reads, materialised data or files. Dictionary and
  Series maps are allowed. Estimators are applied with .skb.apply(...).
- Files no library reader handles (images, audio, documents of file-backed rows) are
  read by a plan-defined reader: a function named read_* that takes paths (and
  parameters) and only reads and decodes them into a DataFrame or array, applied
  with paths.skb.apply_func(read_images, size=...). It reads only through a
  library's file reader (e.g. PIL.Image.open, soundfile.read), never the built-in
  open; take metadata such as dimensions or duration from that reader's result.
  For remote paths, fetch the files in batched calls (the filesystem's cat on a
  list of paths, e.g. fsspec.filesystem('gs').cat(paths)) rather than opening them
  one by one, and decode the bytes in parallel (concurrent.futures), e.g. from
  io.BytesIO; io allows only BytesIO and StringIO.
  Build the paths with string DataOps from a source directory constant, e.g.
  DIR + X['key'] + '.ext'. Everything after the read (statistics, filters, joins,
  models) is DataOps and library calls, e.g. a quick check in an exploration with
  skrub.as_data_op(estimator).skb.apply_func(cross_val_score, X, y, cv=..., scoring=...)
  (apply_func passes the DataOp it is called on as the first argument). Pipelines use the same reader, so explorations and
  pipelines decode identically.
- Prefer fine-grained DataOps whenever the step can be written with them. Write a
  custom estimator or transformer class (sklearn API, defining fit) only when the
  step needs fitted state or is a genuinely new model, e.g. a torch network.
- Exploration returns a dict of named DataOps; all are evaluated in one pass.
  Explorations produce findings, never candidates. They may fit models for quick
  evidence (own splits or subsamples, setups the lock does not allow), e.g. inside a
  custom estimator; such numbers are not comparable with candidate scores.
- Evaluation setup returns {'X', 'y', 'scoring', optional 'row_keys', optional
  'audit' (a DataOp or dict of named DataOps; fixed values such as numbers from
  explorations go in plain dicts or lists, since DataFrames, Series and arrays
  placed directly into the graph are rejected)}: X marked with
  mark_as_X(cv=..., split_kwargs=...) and the raw y marked with mark_as_y(). scoring is a sklearn scorer string or a plain function
  scorer(estimator, X, y) -> float defined in the setup (higher is better), called on
  each test fold with that fold's marked X. CV is explicit and deterministic: sklearn
  splitters, custom splitter classes or explicit folds. For KFold use split_kwargs={}
  (n_splits/shuffle/random_state go to the constructor); split_kwargs are arguments
  to split(), e.g. groups for GroupKFold.
- Pipelines start from locked_evaluation_source, which the harness runs before the
  plan in the same module: call build_evaluation() and use the setup's imports,
  constants and functions by name, without pasting or redefining them. Build
  features and the model downstream, and return {'pred', 'scoring':
  setup scoring, 'row_keys': the same DataOp if declared}. The harness compares the
  X/y/CV/scoring/row-key graphs and the folds with the lock before fitting; drift is
  rejected. Keep the row count and order of X.
- Pipelines name their blocks with .skb.set_name('<block>'): the last step of each
  feature block (e.g. 'dates', 'text_tfidf', 'target_encoding') and the model, which
  is pred itself, e.g. pred = features.skb.apply(model, y=y).skb.set_name('model').
  Blocks are how experiments refer to components (to change, remove or combine
  them), so give each its own descriptive name, distinct from choose_from names; a
  parent's blocks keep their names.
- A probe is written like a pipeline with a single configuration (no choose_from).
  The harness fits it on the locked folds and reports its fold scores, an error
  summary of its out-of-fold predictions (per-class recall and confusion, calibration
  by decile, or error quantiles) and a preview. Probes are evidence, not candidates.
- A final plan turns a scored candidate (its resolved source is given) into a
  submission; it is fitted once and never scored. Keep the candidate's features and
  model, but read the rows that change between fitting and predicting through a
  skrub.var holding a path, e.g. skrub.var("rows").skb.apply_func(pd.read_csv), and
  define FIT and PREDICT dicts mapping each var name to a task source path, plus
  FORMAT (the sample submission path, or None). Mark y as usual; drop columns missing
  at prediction time (the target) with errors="ignore". build() returns
  {'submission': DataOp}: the predictions formatted as the submission table, with
  FORMAT's columns. No CV, no choose_from, no scoring.
  When the rows to fit on and the rows to predict come from different sources or
  formats, read both as recorded sources and select them by mode:
  rows = skrub.eval_mode().skb.match({'predict': predict_rows}, default=fit_rows),
  derive X and y from rows (y is evaluated only when fitting) and set FIT = PREDICT = {}.
  Without a FORMAT file, set OUTPUT_SEP = '\t' when the task asks for TSV output.
  Steps after X.skb.apply(model, y=y) also run while fitting, where a predictor
  passes on the fitted estimator itself, not predictions; indexing it fails ("object
  is not subscriptable"). To post-process predictions, keep predict and also give
  the estimator transform (and fit_transform) returning the same predictions. They
  are aligned with X's rows, with no key columns: attach keys with
  rows[['key']].assign(...) or return a DataFrame with the key. For top-k outputs (rankings, multi-label), predict k columns per row
  and reshape them into the long table with DataOps (one slice per rank,
  .skb.concat, a merge to map ids).
- Named skrub.choose_from grids become one candidate per variant (within the
  evaluation budget). Children receive the parent's resolved configuration.
- Return code without markdown fences. No credentials, no caching. Pipelines and
  probes never fit or score manually: the harness scores them on the locked folds.

Skrub and library notes (exact signatures; do not guess other keywords):
- a.skb.concat([b, c], axis=0) takes only a list and axis; no ignore_index. It works on
  DataFrames, not Series: concatenate df[['col']] rather than df['col'].
  Chain .reset_index(drop=True) afterwards if a fresh index is needed.
- skrub.choose_from(outcomes, name='...'): outcomes (list or dict) first, name as a
  keyword. It returns a Choice, not a DataOp: pass it as an argument (an
  estimator, a column list, a parameter) or wrap it with skrub.as_data_op(choice)
  before using .skb or DataFrame methods on it.
  Candidates are labelled by their outcomes: choose between estimators or settings
  with a dict whose keys name them ({'lr0.03_750trees': est_a, ...}), or put the
  choice on the parameter itself; a list of estimators shows only 'Classifier(...)'.
- X.skb.apply(estimator, y=y) for the final model; there is no method= argument.
  Scorers call predict_proba/decision_function on the learner themselves.
- skrub.TableVectorizer(cardinality_threshold=40, low_cardinality=..., high_cardinality=...,
  numeric=..., datetime=..., specific_transformers=..., drop_null_fraction=...) takes only
  these keyword arguments; there is no categorical= argument.
- Casts use dtypes: .astype("string"), .astype("float64") or .astype(str) are fine.
- .skb.apply keeps DataFrame output, so sklearn encoders must produce dense output:
  OneHotEncoder(sparse_output=False, handle_unknown="ignore").
- Any installed library can be imported; nothing is installed on demand. Installed
  ML libraries include scikit-learn, lightgbm, xgboost, catboost, torch (CUDA),
  skorch, sentence-transformers, polars, faiss and rank_bm25. Modules for processes,
  files or network access (os, subprocess, pathlib, requests, ...) are not allowed.
- A torch model is a custom estimator: fit seeds torch, builds an nn.Module (a class
  with forward), moves it and each batch to the device ('cuda' when
  torch.cuda.is_available(), else 'cpu') and trains over a DataLoader of tensors;
  predict_proba runs under torch.no_grad() in batches and returns a numpy array of
  shape (n_rows, n_classes), one column per label for multi-label targets. Keep the
  fitted network on self (e.g. self.net_), and print training losses if useful.
- The training loop may be delegated to skorch: the estimator builds a skorch
  NeuralNet in fit (module, criterion, optimizer, LR schedule and per-epoch scoring as
  callbacks) and keeps only the glue: rows to a torch Dataset, network output to
  predictions. Subclass the net or a skorch Callback for what skorch lacks, e.g. mixed
  precision in train_step, or keeping the best epoch's weights in memory (copy the
  state_dict) rather than in Checkpoint files.
- Training on file-backed rows: when the decoded files fit in context.hardware's
  memory, read them in the graph at the size the model uses,
  X['path'].skb.apply_func(read_images, size=...), and train on the in-memory arrays
  (augmentation inside the estimator), so each file is read once per fit rather than
  once per epoch. Only when they do not fit, decode lazily in a torch Dataset (a class
  with __len__ and __getitem__ calling the reader on one path) with DataLoader
  workers.
- A validation set for early stopping or epoch selection is carved from the fold's
  own training rows, as its own node: a transformer whose fit_transform(X, y) splits
  them and returns {'X': ..., 'y': ..., 'X_val': ..., 'y_val': ...}, and whose
  transform(X) returns {'X': X, 'y': None, 'X_val': None, 'y_val': None}. Apply it
  with parts = X.skb.apply(splitter, y=y, no_wrap=True), then
  parts['X'].skb.apply(model, y=parts['y'], fit_kwargs={'X_val': parts['X_val'],
  'y_val': parts['y_val']}) with fit(self, X, y, X_val=None, y_val=None). The locked
  test fold never steers training.
- Custom classes are estimators/transformers (fit), torch modules (forward), torch
  Datasets (__getitem__), skorch nets or callbacks (train_step, infer, on_* hooks) or
  CV splitters (split(self, X, y=None, groups=None) and get_n_splits(self, X=None,
  y=None, groups=None)). Estimators subclass BaseEstimator and keep __init__ to
  storing parameters; classifier wrappers list ClassifierMixin before BaseEstimator.
  Plans keep no state between fits: no module-level caches, globals or mutable class
  attributes, and no file reads outside the recorded readers and read_* readers.
- Boolean masks over nullable columns must not contain NA: use .fillna(False) or
  str methods with na=False before combining masks with & or |.
- Output names are dict keys that must be valid Python identifiers.
"""

APPLY_FUNC = {
    True: "apply_func is limited to curated library primitives (pandas readers,\n"
          "  to_datetime, to_numeric, concat, cut; numpy where/select/log1p/clip/...; len).",
    False: "apply_func may call library functions.",
}
GUIDE = _GUIDE.replace("{apply_func}", APPLY_FUNC[load_config()["plans"]["restrict_primitives"]])

# Any installed library may be imported, except modules that reach outside the plan
# (processes, files, network, interpreter internals). Plans that import a library
# which is not installed fail with MissingLibrary; the runner records it.
DENIED_MODULES = {"os", "sys", "subprocess", "shutil", "pathlib", "socket", "ctypes", "importlib",
                  "builtins", "multiprocessing", "threading", "pickle", "dill", "joblib", "marshal",
                  "requests", "urllib", "http", "ftplib", "smtplib", "tempfile", "glob", "gc"}
# io is allowed only for in-memory buffers, e.g. decoding fetched bytes.
IO_ALLOWED = {"BytesIO", "StringIO"}


class MissingLibrary(ValueError):
    def __init__(self, modules):
        self.modules = sorted(modules)
        super().__init__(f"Library not installed: {', '.join(self.modules)}. The harness does not install "
                         "packages; use an installed library instead (see the plan guide).")


def installed(module):
    import importlib.util

    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


EAGER_READERS = {"ParquetFile", "ParquetDataset", "open_dataset", "iter_batches", "loadtxt", "genfromtxt",
                 "fromfile", "memmap"}
# File-reading submodules: their readers bypass the recorded graph.
DENIED_SUBMODULES = {"pyarrow.parquet", "pyarrow.dataset", "pyarrow.csv", "pyarrow.feather", "pyarrow.json",
                     "pyarrow.orc"}


def is_reader_name(name):
    return name.startswith(("read_", "scan_")) or name in EAGER_READERS


MUTATORS = {"update", "append", "extend", "setdefault", "add", "insert", "pop", "popitem", "clear",
            "remove", "discard"}


# Inside custom estimator classes, fitting and torch's model.eval() are allowed.
CLASS_ALLOWED = {"fit", "fit_transform", "eval"}
OPAQUE_FUNCTION = ("Opaque function node: use fine-grained DataOps and library functions, not functions "
                   "defined in the plan (except file readers named read_*); put training and other "
                   "fitted logic in an estimator or module class")
FORBIDDEN = {"deferred", "eval", "exec", "preview", "get_data", "set_data", "get_vars",
             "make_grid_search", "make_randomized_search", "make_learner", "with_scoring",
             "cross_validate", "iter_cv_splits", "train_test_split", "subsample",
             "choose_float", "choose_int", "choose_bool", "optional", "fit", "fit_transform",
             "to_csv", "to_parquet", "to_pickle", "to_json", "to_sql", "save", "dump"}


def validate_source(source):
    tree = ast.parse(source)
    builders = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build"]
    if len(builders) != 1:
        raise ValueError("Plan must define exactly one zero-argument build() function")
    build = builders[0]
    if (build.args.args or build.args.posonlyargs or build.args.kwonlyargs or build.args.vararg
            or build.args.kwarg or build.decorator_list):
        raise ValueError("Use undecorated build() without arguments")
    # Eager reads are calls of file readers from any library: pandas/polars read_* and
    # scan_*, pyarrow's read_table and ParquetFile, numpy's loadtxt and fromfile. A
    # plan's own helper that happens to be called read_parquet (and records the read)
    # is not one; passing a reader to apply_func is not a call.
    local_functions = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    imported_readers = {a.asname or a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
                        for a in n.names if is_reader_name(a.name)}

    def is_eager_read(call):
        if isinstance(call.func, ast.Attribute):
            return is_reader_name(call.func.attr)
        name = getattr(call.func, "id", "")
        return name in imported_readers or (name not in local_functions and is_reader_name(name))
    choice_names = set()
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        if isinstance(node, ast.ImportFrom):
            if node.level or not node.module:
                raise ValueError("Relative imports are not supported")
            if any(a.name == "*" for a in node.names):
                raise ValueError(f"Wildcard import from {node.module} is not supported")
            imported.add(node.module)
    denied = sorted(m for m in imported if m.split(".")[0] in DENIED_MODULES)
    readers = sorted(m for m in imported if m in DENIED_SUBMODULES)
    if readers:
        raise ValueError(f"Import of {', '.join(readers)} is not allowed: record reads with "
                         "skrub.as_data_op(path).skb.apply_func(pd.read_parquet | pd.read_csv, ...)")
    if denied:
        raise ValueError(f"Import of {', '.join(denied)} is not allowed: plans may not access processes, "
                         "files, the network or interpreter internals")
    missing = {m.split(".")[0] for m in imported if not installed(m.split(".")[0])}
    if missing:
        raise MissingLibrary(missing)
    # getattr with a literal public attribute name is plain attribute access.
    literal_getattr = {id(n.func) for n in ast.walk(tree)
                       if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "getattr"
                       and 2 <= len(n.args) <= 3 and not n.keywords and isinstance(n.args[1], ast.Constant)
                       and isinstance(n.args[1].value, str) and n.args[1].value.isidentifier()
                       and not n.args[1].value.startswith("_") and n.args[1].value not in FORBIDDEN}
    in_class = {id(n) for c in ast.walk(tree) if isinstance(c, ast.ClassDef)
                for m in c.body for n in ast.walk(m)}
    check_no_shared_state(tree)
    # Plan-defined functions run in the graph only as file readers; checked first because
    # it decides the plan's structure.
    module_functions = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"apply_func", "apply"}):
            for arg in node.args[:1]:
                if isinstance(arg, ast.Name) and arg.id in module_functions and not arg.id.startswith("read_"):
                    raise ValueError(f"{node.func.attr}({arg.id}): {OPAQUE_FUNCTION}")
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == "io" and any(a.name not in IO_ALLOWED for a in node.names):
                raise ValueError("From io only BytesIO and StringIO may be imported (in-memory buffers)")
            if any(a.name == "FunctionTransformer" for a in node.names):
                raise ValueError("Custom function transformers are not supported")
        if isinstance(node, ast.ClassDef):
            methods = {m.name for m in node.body if isinstance(m, ast.FunctionDef)}
            if not (methods & {"fit", "forward", "__getitem__", "train_step", "infer"}
                    or any(m.startswith("on_") for m in methods) or {"split", "get_n_splits"} <= methods):
                raise ValueError(f"Class {node.name}: custom classes are estimators/transformers (fit), "
                                 "torch modules (forward), torch Datasets (__getitem__), skorch nets or "
                                 "callbacks (train_step, infer, on_* hooks) or CV splitters (split, "
                                 "get_n_splits); "
                                 "write other logic as fine-grained DataOps")
            if node.keywords or node.decorator_list or not all(
                    isinstance(b, (ast.Name, ast.Attribute)) for b in node.bases):
                raise ValueError(f"Class {node.name}: custom classes take plain base classes "
                                 "(e.g. BaseEstimator, TransformerMixin, nn.Module), no metaclass or decorators")
        if isinstance(node, ast.Attribute):
            if isinstance(node.value, ast.Name) and node.value.id == "io" and node.attr not in IO_ALLOWED:
                raise ValueError("Only io.BytesIO and io.StringIO are allowed (in-memory buffers)")
            if id(node) in in_class:
                # Estimator internals: fitting, private state, torch's model.eval() and
                # public parameters stored on self (e.g. LightGBM's self.subsample).
                on_self = isinstance(node.value, ast.Name) and node.value.id == "self"
                allowed = (node.attr in CLASS_ALLOWED or (node.attr.startswith("_") and not node.attr.startswith("__"))
                           or (on_self and not node.attr.startswith("_"))
                           or (node.attr == "__init__" and isinstance(node.value, ast.Call)
                               and getattr(node.value.func, "id", "") == "super"))
                if not allowed and (node.attr.startswith("_") or node.attr in FORBIDDEN):
                    raise ValueError(f"Unsupported operation in class: {node.attr}")
            elif node.attr.startswith("_") or node.attr in FORBIDDEN:
                raise ValueError(f"Unsupported opaque or harness-owned operation: {node.attr}" + (
                    "; training and inference (including torch's model.eval()) belong inside an "
                    "estimator or module class" if node.attr in CLASS_ALLOWED else ""))
        if isinstance(node, ast.Name) and id(node) not in literal_getattr and (node.id.startswith("__") or node.id in
                {"open", "eval", "exec", "compile", "getattr", "setattr", "globals", "locals", "vars", "FunctionTransformer"}
                or (node.id == "print" and id(node) not in in_class)):
            raise ValueError(f"Unsupported name: {node.id}" + (
                "; set attributes inside an estimator or module class" if node.id == "setattr" else ""))
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            if is_eager_read(node):
                raise ValueError("Record readers with as_data_op(path).skb.apply_func(reader), never eager reads")
            if name == "choose_from":
                names = [k.value for k in node.keywords if k.arg == "name"]
                if (len(names) != 1 or not isinstance(names[0], ast.Constant)
                        or not isinstance(names[0].value, str) or not names[0].value):
                    raise ValueError("Every choose_from needs a literal nonempty name, passed as a keyword: "
                                     "skrub.choose_from(outcomes, name='...') with the outcomes (list or "
                                     "dict) first; the name is not positional")
                if names[0].value in choice_names:
                    raise ValueError("Choice names must be unique")
                if choice_outcomes(node, tree) is None:
                    raise ValueError(f"choose_from {names[0].value!r}: outcomes must be a literal list or dict, "
                                     "inline or in a variable assigned once, so children can resolve the "
                                     "selected variant")
                choice_names.add(names[0].value)
        if isinstance(node, ast.Assign):
            # Dictionary construction is allowed in graph-building helpers.
            # DataOp item mutation fails at construction; use recorded assign().
            # Inside classes, attributes of other objects may be set (torch flags such as
            # p.requires_grad = False); there are no DataOps there.
            for target in node.targets:
                if (isinstance(target, ast.Attribute) and id(target) not in in_class
                        and not (isinstance(target.value, ast.Name) and target.value.id == "self")):
                    raise ValueError("Set DataFrame columns with recorded .assign(col=...), not attribute "
                                     "assignment (df.col = ...). Attribute assignment on other objects "
                                     "(e.g. replacing a torch model's layer) belongs inside an estimator "
                                     "or module class")
    return tree


def check_no_shared_state(tree):
    """Plans keep no state between fits and folds: no global/nonlocal, no mutation of
    module-level names from functions or classes, no mutable class attributes. Module
    constants (paths, FIT/PREDICT dicts) stay allowed."""
    module_names = {t.id for n in tree.body if isinstance(n, (ast.Assign, ast.AnnAssign))
                    for t in (n.targets if isinstance(n, ast.Assign) else [n.target]) if isinstance(t, ast.Name)}
    message = ("Plans keep no state between fits: no module-level caches or global variables; "
               "keep fitted state on self and recompute inputs in the graph")
    for function in ast.walk(tree):
        if isinstance(function, ast.ClassDef):
            for item in function.body:
                if isinstance(item, (ast.Assign, ast.AnnAssign)) and isinstance(
                        item.value, (ast.Dict, ast.List, ast.Set, ast.DictComp, ast.ListComp, ast.SetComp)):
                    raise ValueError(f"Class {function.name}: no mutable class attributes; {message}")
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            continue
        local = {t.id for n in ast.walk(function) if isinstance(n, (ast.Assign, ast.AnnAssign, ast.For))
                 for t in ([n.target] if not isinstance(n, ast.Assign) else n.targets) if isinstance(t, ast.Name)}
        local |= {a.arg for a in ast.walk(function) if isinstance(a, ast.arg)}
        shared = module_names - local
        for node in ast.walk(function):
            if isinstance(node, (ast.Global, ast.Nonlocal)):
                raise ValueError(message)
            targets = (node.targets if isinstance(node, ast.Assign) else
                       [node.target] if isinstance(node, (ast.AugAssign, ast.AnnAssign)) else
                       node.targets if isinstance(node, ast.Delete) else [])
            for target in targets:
                if (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
                        and target.value.id in shared):
                    raise ValueError(message)
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in MUTATORS
                    and isinstance(node.func.value, ast.Name) and node.func.value.id in shared):
                raise ValueError(message)


def evaluation_source(source):
    tree = validate_source(source)
    # Preserve existing graph helpers, including a helper called build_evaluation.
    # Rename the entry point to a fresh name, then expose a stable wrapper.
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    names.update(n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef)))
    entry = "locked_setup_entry"
    while entry in names:
        entry += "_"
    helper = "locked_setup_helper"
    while helper in names or helper == entry:
        helper += "_"

    class Rename(ast.NodeTransformer):
        def visit_Name(self, node):
            if node.id == "build":
                node.id = entry
            elif node.id == "build_evaluation":
                node.id = helper
            return node

        def visit_FunctionDef(self, node):
            if node.name == "build":
                node.name = entry
            elif node.name == "build_evaluation":
                node.name = helper
            return self.generic_visit(node)

    tree = Rename().visit(tree)
    tree.body.extend(ast.parse(f"def build_evaluation():\n    return {entry}()\n").body)
    return ast.unparse(tree) + "\n"


def with_setup(setup_source, source):
    """A pipeline as one self-contained program: the locked setup, which the harness
    runs before a pipeline, followed by the pipeline itself."""
    return setup_source + "\n\n" + source


LITERAL_OUTCOMES = (ast.List, ast.Tuple, ast.Dict)


def choice_outcomes(node, tree):
    """The literal outcomes of a choose_from call: written inline, or a variable assigned
    exactly once to a literal list, tuple or dict. None when they cannot be resolved."""
    values = node.args[0] if node.args else next((k.value for k in node.keywords if k.arg == "outcomes"), None)
    if isinstance(values, ast.Name):
        assigned = [a.value for a in ast.walk(tree) if isinstance(a, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == values.id for t in a.targets)]
        values = assigned[0] if len(assigned) == 1 else None
    if not isinstance(values, LITERAL_OUTCOMES):
        return None
    if isinstance(values, ast.Dict):
        return None if None in values.keys else values.values  # no **unpacking
    return None if any(isinstance(e, ast.Starred) for e in values.elts) else values.elts


def resolve_source(source, configuration):
    tree = validate_source(source)

    class Resolve(ast.NodeTransformer):
        def visit_Call(self, node):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
            if name == "choose_from":
                choice = next(k.value.value for k in node.keywords if k.arg == "name")
                if choice in configuration:
                    return self.visit(choice_outcomes(node, tree)[int(configuration[choice])])
            return self.generic_visit(node)

    return ast.unparse(ast.fix_missing_locations(Resolve().visit(tree))) + "\n"
