from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import json
import numpy as np
from scipy.io import loadmat

if TYPE_CHECKING:
    import h5py

# Lazy import handle
_h5py_lib = None

def _get_h5py() -> Optional[Any]:
    """Lazy load h5py to avoid crashing if binary incompatibility exists."""
    global _h5py_lib
    if _h5py_lib is not None:
        return _h5py_lib
    try:
        import h5py
        _h5py_lib = h5py
        return h5py
    except (ImportError, ValueError) as e:
        print(f"Warning: h5py could not be imported ({e}). HDF5 .jab files will not be readable.")
        return None


@dataclass
class WeakLearner:
    dim: int
    thr: float
    direction: int  # typically +1 or -1
    alpha: float


@dataclass
class JabModel:
    behavior: str
    score_filename: str
    timestamp: float
    feature_names: Optional[List[Any]]  # JAABA-style nested tokens; None if unavailable
    learners: List[WeakLearner]
    post_params: Optional[Dict[str, Any]] = None
    score_norm: Optional[float] = None


def _safe_get_dict(d: Dict[str, Any], key: str) -> Any:
    for k in d.keys():
        if k.lower() == key.lower():
            return d[k]
    return None


def _try_loadmat(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = loadmat(str(path), squeeze_me=True, struct_as_record=False)
        return data
    except Exception:
        return None


def _try_h5(path: Path) -> Optional['h5py.File']:
    h5 = _get_h5py()
    if h5 is None:
        return None
    try:
        f = h5.File(str(path), "r")
        return f
    except Exception:
        return None


def _mat_to_py(obj: Any) -> Any:
    """Convert MATLAB objects (mat_structs) into Python nested dict/lists where possible."""
    if hasattr(obj, "_fieldnames"):
        out = {}
        for fn in obj._fieldnames:
            out[fn] = _mat_to_py(getattr(obj, fn))
        return out
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        return [ _mat_to_py(x) for x in obj.tolist() ]
    return obj


def _extract_behaviors_from_mat(data: Dict[str, Any]) -> List[str]:
    behs: List[str] = []
    behaviors = _safe_get_dict(data, "behaviors")
    if behaviors is None:
        return behs
    behaviors = _mat_to_py(behaviors)
    names = None
    if isinstance(behaviors, dict):
        names = behaviors.get("names")
    if names is None:
        return behs
    if isinstance(names, (list, tuple)):
        return [str(n) for n in names]
    return [str(names)]


def _looks_like_tokens_list(obj: Any) -> bool:
    # Heuristically detect a list of token lists: [ ['stat','mean',...], ['stat','max',...] ]
    if not isinstance(obj, (list, tuple)):
        return False
    if not obj:
        return False
    first = obj[0]
    if not isinstance(first, (list, tuple)):
        return False
    toks = list(first)
    # Look for 'stat' or 'trans' or 'radius' keys
    keys = set()
    for i in range(0, len(toks)-1, 2):
        k = toks[i]
        if isinstance(k, (bytes, bytearray)):
            k = k.decode(errors='ignore')
        if isinstance(k, str):
            keys.add(k.lower())
    return any(k in keys for k in ('stat','trans','radius','offset'))


def _extract_classifiers_from_mat(data: Dict[str, Any]) -> Tuple[List[List[WeakLearner]], List[float], List[str], List[Optional[List[Any]]], List[Optional[Dict[str, Any]]], List[Optional[float]]]:
    """Return (classifiers_per_behavior, timestamps, scorefilenames, feature_names)
    classifiers_per_behavior: list over behaviors; each is a list of WeakLearner for that behavior.
    feature_names may be None if not found.
    """
    def _find_nested_with_keys(obj: Any, keys: List[str], depth: int = 0, max_depth: int = 5) -> Optional[Dict[str, Any]]:
        if depth > max_depth:
            return None
        if isinstance(obj, dict):
            lower_keys = set(k.lower() for k in obj.keys())
            if any(k.lower() in lower_keys for k in keys):
                return obj
            for v in obj.values():
                found = _find_nested_with_keys(_mat_to_py(v), keys, depth+1, max_depth)
                if found is not None:
                    return found
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                found = _find_nested_with_keys(_mat_to_py(v), keys, depth+1, max_depth)
                if found is not None:
                    return found
        return None

    def _find_any_token_lists(obj: Any, depth: int = 0, max_depth: int = 5) -> Optional[List[Any]]:
        if depth > max_depth:
            return None
        try:
            p = _mat_to_py(obj)
        except Exception:
            return None
        if _looks_like_tokens_list(p):
            return p
        if isinstance(p, dict):
            for v in p.values():
                res = _find_any_token_lists(v, depth+1, max_depth)
                if res is not None:
                    return res
        elif isinstance(p, (list, tuple)):
            for v in p:
                res = _find_any_token_lists(v, depth+1, max_depth)
                if res is not None:
                    return res
        return None

    # Try direct extraction; if missing, search nested structs (Macguffin object content)
    cs = _safe_get_dict(data, "classifierStuff")
    fileinfo = _safe_get_dict(data, "file")
    if cs is None:
        found = _find_nested_with_keys(data, ["classifierStuff"]) or _find_nested_with_keys(data, ["ClassifierStuff"]) 
        if found is not None:
            data = found
            cs = _safe_get_dict(data, "classifierStuff") or _safe_get_dict(data, "ClassifierStuff")
            fileinfo = _safe_get_dict(data, "file")
    feature_names_list: List[Optional[List[Any]]] = []
    timestamps: List[float] = []
    scorefilenames: List[str] = []
    all_learners: List[List[WeakLearner]] = []
    post_params_list: List[Optional[Dict[str, Any]]] = []
    score_norm_list: List[Optional[float]] = []

    if cs is None:
        return [], [], [], [], [], []
    cs = _mat_to_py(cs)

    # Handle multiple behaviors or single
    if isinstance(cs, list):
        css = cs
    elif isinstance(cs, dict):
        css = [cs]
    else:
        css = []

    # score filenames
    if fileinfo is not None:
        fileinfo = _mat_to_py(fileinfo)
        scorefn = fileinfo.get("scorefilename") if isinstance(fileinfo, dict) else None
        if isinstance(scorefn, list):
            scorefilenames = [str(x) for x in scorefn]
        elif isinstance(scorefn, str):
            scorefilenames = [scorefn]

    for i, c in enumerate(css):
        learners: List[WeakLearner] = []
        ts = None
        cur_feature_names: Optional[List[Any]] = None
        pp: Optional[Dict[str, Any]] = None
        scorenorm: Optional[float] = None
        if isinstance(c, dict):
            ts = c.get("timeStamp")
            scorenorm = c.get("scoreNorm")
            params = c.get("params")
            if params is not None:
                params = _mat_to_py(params)
                # 'params' may be a list of structs with fields dim, tr, dir, alpha
                if isinstance(params, list):
                    for p in params:
                        try:
                            learners.append(WeakLearner(
                                dim=int(p.get("dim")),
                                thr=float(p.get("tr")),
                                direction=int(p.get("dir", 1)),
                                alpha=float(p.get("alpha", 1.0)),
                            ))
                        except Exception:
                            continue
                elif isinstance(params, dict):
                    # Single weak learner case
                    try:
                        learners.append(WeakLearner(
                            dim=int(params.get("dim")),
                            thr=float(params.get("tr")),
                            direction=int(params.get("dir", 1)),
                            alpha=float(params.get("alpha", 1.0)),
                        ))
                    except Exception:
                        pass

            # Try to get feature names if present
            # Feature names (per classifier)
            # Direct fields common in some JAABA jabs
            for key in ("featureNames","feature_names","windowfeatures","windowfeaturescellparams"):
                fns = c.get(key)
                if fns is not None:
                    fns_py = _mat_to_py(fns)
                    if _looks_like_tokens_list(fns_py):
                        cur_feature_names = fns_py
                        break
            if cur_feature_names is None:
                for k, v in c.items():
                    vv = _mat_to_py(v)
                    if _looks_like_tokens_list(vv):
                        cur_feature_names = vv
                        break
            # Post-process params
            pp_raw = c.get('postProcessParams')
            if pp_raw is not None:
                pp = _mat_to_py(pp_raw)
        timestamps.append(float(ts) if ts is not None else 0.0)
        all_learners.append(learners)
        feature_names_list.append(cur_feature_names)
        post_params_list.append(pp)
        score_norm_list.append(float(scorenorm) if scorenorm is not None else None)

    return all_learners, timestamps, scorefilenames, feature_names_list, post_params_list, score_norm_list


def _extract_from_h5(f: 'h5py.File') -> Tuple[List[str], List[List[WeakLearner]], List[float], List[str], Optional[List[Any]]]:
    """Best-effort extraction from v7.3 HDF5 .jab files.
    Attempts to find classifier params (dim/tr/dir/alpha) and featureNames anywhere in the tree.
    """
    h5 = _get_h5py() # ensure loaded for types if needed, though we have 'f' passed in.
    behaviors: List[str] = []
    learners_all: List[List[WeakLearner]] = []
    timestamps: List[float] = []
    scorefilenames: List[str] = []
    feature_names: Optional[List[Any]] = None

    # Behaviors
    try:
        if "behaviors" in f:
            grp = f["behaviors"]
            if "names" in grp:
                names_ds = grp["names"]
                try:
                    if hasattr(names_ds, "asstr"):
                        behaviors = [s for s in names_ds.asstr()[:]]
                    else:
                        behaviors = [bytes(x).decode("utf-8", errors="ignore") for x in names_ds[:]]
                except Exception:
                    pass
    except Exception:
        pass

    # Helper to read token cell arrays recursively
    def _read_tokens_obj(obj) -> Any:
        if isinstance(obj, h5.Dataset):
            dt = obj.dtype
            if hasattr(obj, 'asstr') and dt.kind in ('S','O','U'):
                try:
                    return obj.asstr()[()].tolist()
                except Exception:
                    pass
            try:
                val = obj[()]
            except Exception:
                return None
            if isinstance(val, (bytes, bytearray)):
                return val.decode(errors='ignore')
            if isinstance(val, np.ndarray) and val.dtype == object:
                out = []
                for ref in val.flat:
                    if isinstance(ref, h5.Reference) and ref:
                        out.append(_read_tokens_obj(f[ref]))
                    else:
                        out.append(ref)
                return out
            if isinstance(val, np.ndarray) and val.dtype.kind in ('S','U'):
                try:
                    return [x.decode(errors='ignore') if isinstance(x, (bytes, bytearray)) else str(x) for x in val.flat.tolist()]
                except Exception:
                    return val.tolist()
            if isinstance(val, np.ndarray):
                return val.tolist()
            return val
        elif isinstance(obj, h5.Group):
            # Try to find datasets directly under group
            out = []
            for k in obj.keys():
                out.append(_read_tokens_obj(obj[k]))
            return out
        return None

    # Find params groups and feature names
    params_groups: List['h5py.Group'] = []
    feature_ds: Optional['h5py.Dataset'] = None

    def _visit(name, obj):
        nonlocal feature_ds
        if isinstance(obj, h5.Group):
            # params may be this group or its child 'params'
            g = obj
            if 'params' in g and isinstance(g['params'], h5.Group):
                pg = g['params']
                if all(k in pg for k in ('dim','tr')):
                    params_groups.append(pg)
            elif all(k in g for k in ('dim','tr')):
                params_groups.append(g)
        elif isinstance(obj, h5.Dataset):
            base = name.split('/')[-1].lower()
            if 'featurenames' in base and feature_ds is None:
                feature_ds = obj
    f.visititems(_visit)

    # Read learners lists per params group
    for pg in params_groups:
        try:
            dims = np.array(pg['dim'][()]).astype(int).ravel()
            trs = np.array(pg['tr'][()]).astype(float).ravel()
            dirs = np.array(pg['dir'][()]).astype(int).ravel() if 'dir' in pg else np.ones_like(dims)
            alphas = np.array(pg['alpha'][()]).astype(float).ravel() if 'alpha' in pg else np.ones_like(dims, dtype=float)
            n = min(len(dims), len(trs), len(dirs), len(alphas))
            learners: List[WeakLearner] = []
            for i in range(n):
                learners.append(WeakLearner(dim=int(dims[i]), thr=float(trs[i]), direction=int(dirs[i]), alpha=float(alphas[i])))
            if learners:
                learners_all.append(learners)
        except Exception:
            continue

    # Read feature names if possible
    if feature_ds is not None:
        try:
            raw = _read_tokens_obj(feature_ds)
            # Expect list of tokens; ensure type
            if isinstance(raw, list):
                feature_names = raw
        except Exception:
            pass

    return behaviors, learners_all, timestamps, scorefilenames, feature_names


def _norm(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def load_jab_models(jab_dir: Path, only_behaviors: Optional[List[str]] = None) -> List[JabModel]:
    """Load all .jab models in jab_dir. Returns a list of JabModel objects.
    Best-effort parser for both v7 and v7.3 files.
    Also supports sidecar tokens JSON files placed next to .jab files or in a tokens.json file.
    """
    models: List[JabModel] = []
    only_norm = set(_norm(b) for b in (only_behaviors or []))

    # Load a directory-level tokens.json if present
    dir_tokens_map: Dict[str, List[Any]] = {}
    dir_tokens_file = jab_dir / "tokens.json"
    if dir_tokens_file.exists():
        try:
            d = json.loads(dir_tokens_file.read_text())
            # Expect mapping: behavior -> tokens list
            if isinstance(d, dict):
                for k, v in d.items():
                    dir_tokens_map[_norm(k)] = v
        except Exception:
            pass

    for jab_path in sorted(jab_dir.glob("*.jab")):
        data = _try_loadmat(jab_path)
        behaviors: List[str] = []
        learners_all: List[List[WeakLearner]] = []
        timestamps: List[float] = []
        scorefns: List[str] = []
        feature_names_list: List[Optional[List[Any]]] = []
        post_params_list: List[Optional[Dict[str, Any]]] = []
        score_norm_list: List[Optional[float]] = []
        if data is not None:
            behaviors = _extract_behaviors_from_mat(data)
            (learners_all, timestamps, scorefns,
             feature_names_list, post_params_list, score_norm_list) = _extract_classifiers_from_mat(data)
        else:
            f = _try_h5(jab_path)
            if f is not None:
                try:
                    behaviors, learners_all, timestamps, scorefns, feature_names = _extract_from_h5(f)
                    # Fallback for H5: replicate feature_names across classifiers
                    feature_names_list = [feature_names for _ in behaviors]
                    post_params_list = [None for _ in behaviors]
                    score_norm_list = [None for _ in behaviors]
                finally:
                    try:
                        f.close()
                    except Exception:
                        pass

        # Sidecar tokens file support: <jab>.tokens.json or dir-level tokens.json
        sidecar = jab_path.with_suffix('.tokens.json')
        side_tokens_map: Dict[str, List[Any]] = {}
        if sidecar.exists():
            try:
                dd = json.loads(sidecar.read_text())
                if isinstance(dd, dict):
                    # Map behavior -> tokens
                    for k, v in dd.items():
                        side_tokens_map[_norm(k)] = v
            except Exception:
                pass

        if not behaviors:
            # Fallback: derive behavior from filename
            behaviors = [jab_path.stem]
        # If classifiers list is empty, create placeholder
        if not learners_all:
            learners_all = [[] for _ in behaviors]
        if not timestamps:
            timestamps = [0.0 for _ in behaviors]
        if not scorefns:
            scorefns = [f"scores_{beh}.mat" for beh in behaviors]

        for i, beh in enumerate(behaviors):
            # Normalize both sides to avoid underscore/space/case mismatches
            if only_norm and _norm(beh) not in only_norm and _norm(jab_path.stem) not in only_norm:
                continue
            # Prefer feature_names from reader; if missing, try sidecar or dir-level tokens
            fns = feature_names_list[min(i, len(feature_names_list)-1)] if feature_names_list else None
            if not fns:
                key = _norm(beh)
                if key in side_tokens_map:
                    fns = side_tokens_map[key]
                elif key in dir_tokens_map:
                    fns = dir_tokens_map[key]
            models.append(JabModel(
                behavior=beh,
                score_filename=scorefns[min(i, len(scorefns)-1)],
                timestamp=timestamps[min(i, len(timestamps)-1)],
                feature_names=fns,
                learners=learners_all[min(i, len(learners_all)-1)] or [],
                post_params=post_params_list[min(i, len(post_params_list)-1)] if post_params_list else None,
                score_norm=score_norm_list[min(i, len(score_norm_list)-1)] if score_norm_list else None,
            ))
    return models
