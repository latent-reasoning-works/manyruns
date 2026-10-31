"""The catalog: recipes and datasets are YAML files loaded as plain dicts.

There is ONE representation. An earlier pass wrapped these in dataclasses, which meant a
recipe went `YAML → dict → Recipe → .to_dict() → dict → runner` — the object round-tripped
to a dict purely to be consumed, because every layer below (pipeline, serving, session) was
already written against dicts. That is two forms of one thing, which is the failure this
codebase keeps repeating. Nothing has shipped, so there is no compatibility reason to keep
the second form: the dict is canonical and the types are gone.

Validation is a function, not a type:
  `load_*`  is strict — it raises, so a bad file cannot flow into a run;
  `check_*` returns a list of problems, so `manyruns check` can report EVERY bad file
            instead of dying on the first.

The DIRECTORY is the registry for both — adding one is adding a file. `$MANYRUNS_RECIPE_DIR`
and `$MANYRUNS_DATASET_DIR` override the bundled sets, which is how a sweep's many
experimental arms stay out of the shipped product.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional
from urllib.parse import urlsplit

from manyruns import env
from manyruns.vocab import (
    HANDLE_KINDS,
    SHAPES,
    SOURCE_KINDS,
    STEP_GROUPS,
    check_claims,
    check_provides,
    check_topology,
)

#: A recipe's `id:` — a short canonical handle, dotted and lowercase, `<domain>.<verb>.<method>`.
#: Separate from `name` because `name` is bound to the FILENAME by `check_recipe`, so it cannot
#: be renamed without breaking every reference to it; `id` survives a file rename and is what a
#: record, a paper or a URL can quote.
_ID_RE = re.compile(r"^[a-z0-9]+(\.[a-z0-9_]+)+$")


def _dir(env_name: str, kind: str, override: Optional[Path | str] = None) -> Path:
    if override:
        return Path(override)
    value = env.get(env_name)
    if value:
        return Path(value)
    from importlib import resources  # never __file__-relative: survives `uv tool install`

    return Path(str(resources.files("manyruns") / "configs" / kind))


def recipe_dir(config_dir: Optional[Path | str] = None) -> Path:
    return _dir("RECIPE_DIR", "recipe", config_dir)


def dataset_dir(config_dir: Optional[Path | str] = None) -> Path:
    return _dir("DATASET_DIR", "dataset", config_dir)


def metrics_dir(config_dir: Optional[Path | str] = None) -> Path:
    return _dir("METRICS_DIR", "metrics", config_dir)


def load_suite(name: str = "default", config_dir: Optional[Path | str] = None) -> list[str]:
    """The declared metric suite, as a plain list of names.

    A list, not a type — the suite is data the engine consumes, and `check_suite` already
    validates it against manylatents' live registry. Returns `[]` when the file is absent so
    a stackless run falls back to engine defaults rather than failing.
    """
    d = metrics_dir(config_dir)
    path = d / f"{name}.yaml"
    if not path.is_file():
        return []
    from omegaconf import OmegaConf

    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    return list((cfg or {}).get("metrics") or [])


def _discover(d: Path) -> list[str]:
    return sorted(p.stem for p in d.glob("*.yaml")) if d.is_dir() else []


def discover_recipes(config_dir: Optional[Path | str] = None) -> list[str]:
    """Every recipe that exists, by name."""
    return _discover(recipe_dir(config_dir))


def discover_datasets(config_dir: Optional[Path | str] = None) -> list[str]:
    """Every dataset that exists, by name."""
    return _discover(dataset_dir(config_dir))


def _read(d: Path, name: str, what: str, known: list[str]) -> dict:
    from omegaconf import OmegaConf

    path = d / f"{name}.yaml"
    if not path.is_file():
        raise ValueError(f"no {what} named {name!r} at {path}. Known: {known or '(none)'}")
    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)  # type: ignore[return-value]


def load_recipe(name: str, config_dir: Optional[Path | str] = None) -> dict:
    """Load a recipe by name. Strict — an invalid recipe raises rather than reaching a run."""
    d = recipe_dir(config_dir)
    cfg = _read(d, name, "recipe", discover_recipes(config_dir))
    problems = check_recipe(cfg, name)
    if problems:
        raise ValueError(f"invalid recipe {name!r}: " + "; ".join(problems))
    return cfg


def load_dataset(name: str, config_dir: Optional[Path | str] = None) -> dict:
    """Load a dataset by name. Strict, for the same reason."""
    d = dataset_dir(config_dir)
    try:
        cfg = _read(d, name, "dataset", discover_datasets(config_dir))
    except OSError as error:
        # OmegaConf rejects numeric/bool YAML roots before check_dataset can inspect them.
        raise ValueError(f"invalid dataset {name!r}: {error}") from error
    problems = check_dataset(cfg, name)
    if problems:
        raise ValueError(f"invalid dataset {name!r}: " + "; ".join(problems))
    return cfg


def load_recipes(
    names: Optional[Iterable[str]] = None, config_dir: Optional[Path | str] = None
) -> list[dict]:
    return [load_recipe(n, config_dir) for n in (names if names is not None
                                                 else discover_recipes(config_dir))]


def load_datasets(
    names: Optional[Iterable[str]] = None, config_dir: Optional[Path | str] = None
) -> list[dict]:
    return [load_dataset(n, config_dir) for n in (names if names is not None
                                                  else discover_datasets(config_dir))]


def known_steps(config_dir: Optional[Path | str] = None) -> dict[str, dict]:
    """Every step the catalog declares: ``name -> {"name", "group", "params"[, "via"]}``.

    DERIVED, because there were three hand-kept copies of this list — `session._ACTIONS`
    (5 aliases → 4 steps), `session.available_actions()` (the same 4 as a second literal) and
    the `step` half of `commands.COMMANDS` — kept in sync by a test that fails AFTER the
    omission rather than by construction. The directory is already the registry for recipes
    (see this module's header); this reads it rather than writing a fourth table.

    The `params` are the catalog's own declared defaults, which is what makes an offered step
    runnable as-is: `phate` arrives carrying `{"n_components": 3}` from `embed.yaml`, so the
    session issues the same step dict the recipe would, and `final_dim` is 3 on both paths.
    First declaration wins when two recipes declare one name differently, and THAT IS NOW LIVE
    ON THE BUNDLED SET rather than the hypothetical this paragraph used to call it. The cutover
    gave `cflows`/`contrast`/`embed`/`pseudotime` a `pca n_components: 50` while
    `archetypes`/`cluster`/`markers` keep their own `10`; `discover_recipes()` is alphabetical,
    so `archetypes` wins here and this function answers 10 for `pca`.

    That is correct for what this function IS — one catalog-wide answer per name, for `--help`
    and for a session with no recipe in hand — and wrong as the answer to "what does the next
    step of MY recipe mean". `session.resolve_step` takes the session's recipe and prefers its
    declaration, falling back here; see manyruns#65 for the measurement, and this docstring's
    old claim that "no bundled recipe is affected" for how long a stated non-problem survives
    after it stops being one.

    `via` rides along for the same reason `params` does: a step issued one-at-a-time in a
    session must be the same step dict the recipe would have issued, or the two paths report
    differently about one run. Without it, `vocab.standins` on the session path could say only
    "not mioflow itself" where the recipe path says "not the manylatents mioflow the recipe
    names". Included ONLY when declared — an unconditional ``"via": None`` would make every
    step dict claim a field it does not have.

    Never raises: a broken file in `$MANYRUNS_RECIPE_DIR` drops out of the menu instead of
    taking the menu down with it. `load_recipe` stays strict — a recipe that is actually being
    RUN must still refuse to load — which is the same split `_live_suite` draws in `runner`.
    """
    out: dict[str, dict] = {}
    for recipe_name in discover_recipes(config_dir):
        try:
            cfg = load_recipe(recipe_name, config_dir)
        except (ValueError, OSError):  # a bad file is not offerable; it is not fatal either
            continue
        for step in cfg.get("steps") or []:
            name = step.get("name")
            if not name or name in out:
                continue
            out[name] = {"name": name, "group": step.get("group"),
                         "params": dict(step.get("params") or {})}
            if step.get("via"):
                out[name]["via"] = step["via"]
    return out


# ── validation: return problems, don't raise, so every file gets reported ────
def _check_limits(step: Any) -> list[str]:
    """One step's `limits:` block, checked. Imported locally so this module — which the base
    install imports to draw the ledger — does not pull the pipeline package to validate a
    recipe. Same argument `vocab`'s header makes about `protocols` reaching into `pipeline`."""
    from manyruns.pipeline.bounds import check_limits

    return check_limits(step)


def _check_source(source: Any, *, require_kind: bool = True) -> list[str]:
    """Problems with `source:` — where a workflow or dataset comes from.

    A MAPPING (kind / cite / doi / url / verified / note), not a bare `cite:` string: a scalar
    cannot carry the kind, cannot carry a URL when no DOI exists, and cannot record when the
    identifier was last resolved. Some works have no DOI, so "URL, and say so" has to be
    expressible or the pressure is to invent one. An earlier recipe carried a
    CellChat DOI guessed one digit wrong, which resolved to nothing.

    `doi` is NOT network-validated here. A strict loader that needs the internet is a loader
    that fails on a plane; resolution belongs in the `admit` verbs (`manyruns check` /
    `audit`), reporting rather than raising (§3.4).

    Dataset citations may omit `kind`; when supplied, the same provenance rules apply.
    """
    if not isinstance(source, Mapping):
        return [f"`source` must be a mapping (kind/cite/doi/url/verified/note), got {source!r}"]
    if not require_kind and "kind" not in source:
        return []
    bad: list[str] = []
    kind = source.get("kind")
    if kind not in SOURCE_KINDS:
        bad.append(f"source kind must be one of {SOURCE_KINDS}, got {kind!r}")
    # `none` is the escape hatch for a workflow with no publication behind it, and it is the
    # ONLY one: every other kind names something a reader can go and check, so it must carry
    # an identifier. Without this the convention is decorative.
    if kind != "none" and not (source.get("doi") or source.get("url")):
        bad.append("source needs a `doi` or a `url` unless `kind: none` — a citation nobody "
                   "can resolve is not provenance")
    return bad


def check_recipe(cfg: Mapping[str, Any], name: Optional[str] = None) -> list[str]:
    """Problems with a recipe dict; empty means valid.

    The four metadata checks — `id`, `source`, `claims`,
    `suits` — are all **absent-is-legal**, because the convention is additive and this
    function is what `load_recipe` raises on: making any of them mandatory would refuse every
    recipe written before they existed, including a user's own in `$MANYRUNS_RECIPE_DIR`.
    Present-but-wrong is refused, which is the half that stops the fields from being
    decorative. `suits` and `claims` were previously unvalidated entirely — a typo in either
    fell through to a silently missing recommendation (`suits`) or a silently unclaimed rung
    (`claims`, via `score.claimed_rungs`).
    """
    bad: list[str] = []
    if name and cfg.get("name") != name:
        bad.append(f"name is {cfg.get('name')!r} but the file is {name}.yaml — they must match")
    rid = cfg.get("id")
    if rid is not None and not (isinstance(rid, str) and _ID_RE.match(rid)):
        bad.append(f"id must be dotted lowercase, <domain>.<verb>.<method> "
                   f"(matching {_ID_RE.pattern}), got {rid!r}")
    if cfg.get("source") is not None:
        bad.extend(_check_source(cfg["source"]))
    bad.extend(check_claims(cfg.get("claims")))
    suits = cfg.get("suits")
    if suits is not None:
        if isinstance(suits, str) or not isinstance(suits, (list, tuple)):
            bad.append(f"`suits` must be a list of data shapes, got {suits!r}")
        else:
            unknown = [s for s in suits if s not in SHAPES]
            if unknown:
                bad.append(f"unknown suits {unknown} — must be from {SHAPES}")
    steps = cfg.get("steps")
    if not steps:
        bad.append("no steps")
        return bad
    for i, s in enumerate(steps):
        if not isinstance(s, Mapping):
            bad.append(f"step {i} is not a mapping")
            continue
        if not s.get("name"):
            bad.append(f"step {i} has no name")
        if s.get("group") not in STEP_GROUPS:
            bad.append(f"step {s.get('name', i)!r}: group must be one of {STEP_GROUPS}, "
                       f"got {s.get('group')!r}")
        # `via` is advisory — it never picks an executor — but `vocab.standins` compares it to
        # the engine name, so a list or a number there would make a real caveat silently
        # unreachable rather than loudly wrong. Not in §3.4's four; added because the field is
        # only useful if it is a name.
        # `tool:` — the step's implementation lives in another interpreter
        # (`pipeline/external.py`). Validated for EXISTENCE of the manifest and never for
        # whether the environment is built: a recipe is a declaration and must stay loadable on
        # a machine where the tool has not been installed. Availability is a RUN-time question,
        # answered by `toolchain.resolve` with the command that fixes it.
        tool = s.get("tool")
        if tool is not None:
            from manyruns import toolchain as _toolchain

            if not isinstance(tool, str) or not tool.strip():
                bad.append(f"step {s.get('name', i)!r}: `tool` must be a tool name, got {tool!r}")
            elif tool not in _toolchain.discover_tools():
                bad.append(f"step {s.get('name', i)!r}: no tool manifest named {tool!r} "
                           f"(known: {_toolchain.discover_tools() or '(none)'})")
        # `layers:` — which second matrices this step reads (`app.declared_layers` unions them,
        # `loading.layers_of` supplies exactly those). A string here would be iterated as
        # characters and load nothing, silently.
        want = s.get("layers")
        if want is not None and (isinstance(want, str)
                                 or not isinstance(want, (list, tuple))
                                 or not all(isinstance(x, str) for x in want)):
            bad.append(f"step {s.get('name', i)!r}: `layers` must be a list of layer names, "
                       f"got {want!r}")
        via = s.get("via")
        if via is not None and not (isinstance(via, str) and via.strip()):
            bad.append(f"step {s.get('name', i)!r}: `via` must be the name of the canonical "
                       f"implementation, got {via!r}")
        # `limits` is where a step declares what its own parameters are capped by, in terms of
        # the data's shape — the field that keeps the METHOD's contract out of manyruns's
        # source (see `pipeline/bounds.py`). Absent-is-legal like the four above; a typo'd
        # shape name is refused, because it would resolve to nothing, cap nothing, and read on
        # the page as though it had.
        bad.extend(_check_limits(s))
    return bad


def _check_download_handle(handle: Mapping[str, Any]) -> list[str]:
    """Validate download declarations without resolving a path or reading any data bytes.

    Pins alone can describe manually obtained data. A URL commits to a complete download
    contract: an HTTPS source and a relative file ref materialised by basename in the drop
    folder. Shared with the downloader so a malformed declaration is never fetchable.
    """
    bad: list[str] = []
    if "sha256" in handle:
        digest = handle["sha256"]
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            bad.append("handle `sha256` must be 64 lowercase hexadecimal characters")
    if "bytes" in handle:
        size = handle["bytes"]
        if type(size) is not int or size <= 0:
            bad.append("handle `bytes` must be a positive integer (not bool)")
    if "url" in handle:
        url = handle["url"]
        valid_url = False
        if (isinstance(url, str) and url.isascii()
                and not any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in url)):
            try:
                parsed = urlsplit(url)
                valid_url = (parsed.scheme == "https" and bool(parsed.hostname)
                             and parsed.username is None and parsed.password is None)
                if parsed.hostname:
                    # urllib encodes the host this way before connecting. Reject empty
                    # or oversized DNS labels here instead of crashing on selection.
                    parsed.hostname.encode("idna")
                # Access also validates malformed/out-of-range ports.
                if parsed.port == 0:
                    valid_url = False
            except ValueError:
                valid_url = False
        if not valid_url:
            bad.append("handle `url` must be ASCII HTTPS with a valid hostname and no credentials; "
                       "percent-encode non-ASCII paths")
        for pin in ("sha256", "bytes"):
            if pin not in handle:
                bad.append(f"handle `url` requires `{pin}`")
        ref = handle.get("ref")
        if (handle.get("kind") != "path" or not isinstance(ref, str) or not ref
                or ref.startswith(("/", "~")) or ref.endswith("/")
                or ":" in ref or "\\" in ref or "\x00" in ref
                or ".." in Path(ref).parts or Path(ref).name in ("", ".", "..")):
            bad.append("handle `url` requires a relative, non-generated path to a file")
    return bad


def check_dataset(cfg: Any, name: Optional[str] = None) -> list[str]:
    """Return problems for any YAML value used as a dataset; empty means valid.

    `handle.provides` is checked HERE and nowhere else, because `vocab.dataset_provides` reads
    it on a hot path and deliberately ignores names it does not recognise — see
    `vocab.check_provides` for the measured cost of that being the only reader.

    Optional fields may be omitted; present fields must have the structure their readers
    expect. Reject a non-mapping root before inspecting any fields.
    """
    if not isinstance(cfg, Mapping):
        return [f"dataset must be a mapping, got {cfg!r}"]
    bad: list[str] = []
    if name and cfg.get("name") != name:
        bad.append(f"name is {cfg.get('name')!r} but the file is {name}.yaml — they must match")
    h = cfg.get("handle")
    if not isinstance(h, Mapping):
        bad.append("`handle` must be a mapping with `kind` and `ref`")
    else:
        if h.get("kind") not in HANDLE_KINDS:
            bad.append(f"handle kind must be one of {HANDLE_KINDS}, got {h.get('kind')!r}")
        ref = h.get("ref")
        if not isinstance(ref, str) or not ref.strip():
            bad.append("handle `ref` must be a non-empty string")
        if "provides" in h and not isinstance(h["provides"], list):
            bad.append("handle `provides` must be a list of dataset facts")
        else:
            bad.extend(check_provides(h.get("provides")))
        bad.extend(_check_download_handle(h))
    if "dims" in cfg:
        dims = cfg["dims"]
        if not isinstance(h, Mapping) or h.get("kind") != "manylatents":
            bad.append("`dims` is only meaningful on a manylatents handle")
        if not isinstance(dims, Mapping):
            bad.append("`dims` must be a mapping with n_samples and n_features")
        else:
            for key in ("n_samples", "n_features"):
                value = dims.get(key)
                if type(value) is not int or value <= 0:
                    bad.append(f"dims `{key}` must be a positive integer (not bool)")
    if not isinstance(cfg.get("shape"), str) or cfg.get("shape") not in SHAPES:
        bad.append(f"shape must be one of {SHAPES}, got {cfg.get('shape')!r} — this is the "
                   "ground-truth label the metric-separation experiment scores against")
    if "topology" in cfg and not isinstance(cfg["topology"], list):
        bad.append("`topology` must be a list of topology labels")
    else:
        bad.extend(check_topology(cfg.get("topology")))
    if "params" in cfg and not isinstance(cfg["params"], Mapping):
        bad.append(f"`params` must be a mapping of generator kwargs, got {cfg.get('params')!r}")
    if "source" in cfg:
        bad.extend(_check_source(cfg["source"], require_kind=False))
    return bad


def check_suite(names: Iterable[str]) -> list[str]:
    """Problems with a metric suite. Checks against manylatents' live registry when it is
    importable; when it is not, the name check is DEFERRED rather than faked."""
    names = list(names)
    bad: list[str] = []
    if not names:
        bad.append("a suite must declare at least one metric")
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        bad.append(f"duplicate metrics: {dupes}")
    known = available_metrics()
    if known is not None:
        unknown = [n for n in names if n not in known]
        if unknown:
            bad.append(f"unknown metric(s) {unknown} — not in manylatents' registry")
    return bad


def available_metrics() -> Optional[set]:
    """manylatents' live metric registry, or None when it is not importable."""
    try:
        from manylatents.metrics.registry import list_metrics
    except Exception:  # noqa: BLE001 - stackless path: defer, never fake
        return None
    try:
        return set(list_metrics())
    except Exception:  # noqa: BLE001
        return None
