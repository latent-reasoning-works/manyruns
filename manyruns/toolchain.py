"""The toolchain: calling a tool manyruns CANNOT install, reproducibly.

THE PROBLEM, measured rather than assumed. `pyrovelocity==0.4.5` is not a pin. It resolves, it
installs — exit 0, 148 packages — and then `import pyrovelocity` raises `ModuleNotFoundError:
scvi.model.base._utils`, because it declares `scvi-tools>=1.1.1` and imports a private module
that exists only in 1.1.x. Reaching a working interpreter took TEN interventions and nine
further constraints, terminating at `numpy==1.26.4` — against manyruns's 2.2.6. Not a heavier
extra: a CONTRADICTION. There is no version of that tool which lives in this wheel.

So the unit of reproducibility is not a version. It is the **fully-pinned, hashed transitive
closure** — 211 packages and 3070 hashes for that one tool (measured) — plus the interpreter it
was resolved for. This module is the layer that declares, locates, verifies and RECORDS that.

  declared   `configs/tool/<name>.yaml`      the constraints, with the reason for each
  locked     `configs/tool/locks/<name>-<version>-<platform>.lock`   `uv pip compile --generate-hashes`
  realized   `$MANYRUNS_TOOL_HOME/<name>/<digest>/`                 `uv pip sync --require-hashes`
  recorded   `<step>.tool_*` keys in the g-vector                    what actually ran

**PER-PLATFORM LOCKS, NOT UNIVERSAL, and it is not a style choice.** A `--universal` resolve of
that same tool adds 22 packages — `nvidia-cublas`, `cuda-toolkit`, `nvidia-cudnn-cu13` and the
rest — because jax takes CUDA wheels on linux and none on macOS (measured: 211 packages against
233). One lock for both platforms therefore means either 2.4 GB of CUDA nobody asked for or a
resolution that is a lie on one of them. Two locks, and the platform is part of the identity.

**NOTHING HERE EVER BUILDS AN ENVIRONMENT.** `resolve()` reports; it does not install. The
realized env for that one tool is **2.4 GB** (measured), and a recipe step that quietly
downloads 2.4 GB mid-run is not a step, it is an outage. A run that needs a missing tool is
REFUSED — at plan time where possible — with the exact command that fixes it. That is the same
rule `vocab.unmet` follows for data and `catalog.load_recipe` follows for a bad file: refuse
early, say what is wrong, never improvise.

**Identity is the lock digest, not the version string.** Two runs are comparable when they name
the same digest. A tool re-locked for a security patch is a DIFFERENT tool as far as the record
is concerned, which is the honest answer — that is what re-locking means.
"""
from __future__ import annotations

import os
from manyruns import env as _env
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional

#: Fields a tool manifest may declare. Closed, because a typo'd key that silently does nothing
#: is the failure this whole vocabulary style exists to prevent.
TOOL_FIELDS = frozenset({
    "name", "version", "summary", "source", "licence", "produces", "needs",
    "python", "requires", "entrypoint", "selftest", "input", "output", "params",
    "mode", "call", "commercial_use", "distribution",
})

#: How manyruns reaches a tool. `run` (the default) starts the manifest's own adapter in the
#: tool's interpreter. `call` starts manyruns's generic call adapter there, which makes the one
#: call the manifest's `call` block describes: a library is usable with a manifest and no code.
#: `precomputed` starts nothing: the result file already exists, and manyruns reads and checks it.
MODES = ("run", "call", "precomputed")

#: The `output.schema` of a model manifest: the tool writes a version 1 `result.json`, which
#: `pipeline.external.check_result` checks before anything is merged.
RESULT_SCHEMA = "result-v1"

#: The keys of a call block. Closed: a block that needs more is a program, and a program is a
#: pinned adapter, which is testable as code.
CALL_KEYS = frozenset({"target", "method", "options", "args", "kwargs", "params", "outputs"})

#: The named inputs the call adapter can pass to the target.
SLOTS = ("matrix", "time", "seed")

#: The facts one selector can express. A graph is four arrays with cross-references and class
#: probabilities need a fixed key set, so both need a pinned adapter.
CALL_FACTS = ("embedding", "clusters", "pseudotime", "trajectories")

#: A selector picks one output value after the call.
SELECTOR_PATTERN = r'^(return(\[-?\d+\]|\.[A-Za-z_]\w*|\["[^"]+"\])?|instance\.[A-Za-z_]\w*)$'

#: Licences that forbid commercial use. A consistency check in ONE direction only: such a
#: licence cannot claim `commercial_use: true`. `commercial_use` is declared, never derived
#: from `licence`, because a derived rule would call an unknown or misspelled licence commercial.
NON_COMMERCIAL_LICENCES = frozenset({
    "LicenseRef-Yale-Non-Commercial", "CC-BY-NC-4.0", "CC-BY-NC-SA-4.0", "CC-BY-NC-ND-4.0",
    "PolyForm-Noncommercial-1.0.0",
})

#: How the tool's inputs are assembled. A CLOSED vocabulary, because the two differ in what they
#: are allowed to see and getting that wrong is silent.
#:
#:   state_matrix — the working matrix as the run currently holds it, narrowing included. The
#:                  ordinary case: a tool that takes numbers and returns numbers.
#:   state_frame  — the run's CURRENT VIEW as an `.h5ad`: narrowed counts, narrowed layers,
#:                  surviving genes. For a tool needing a second matrix over the same cells —
#:                  every velocity method wants spliced/unspliced — and the reason the frame
#:                  learned to carry layers at all. The tool sees the cells this run holds and
#:                  no others, so it composes AFTER a prep block instead of being refused by it.
#:   source_file  — the ORIGINAL `.h5ad`, untouched, for a tool needing something the frame
#:                  still does not carry (`obsp`, `raw`, `uns`). REFUSED once the selection has
#:                  narrowed: the file still holds cells a filter removed, so the tool would
#:                  report on a population this run rejected. Prefer `state_frame`; this is the
#:                  escape hatch, and its refusal is the price of being one.
INPUT_KINDS = ("state_matrix", "state_frame", "source_file")


def tool_dir(config_dir: Optional[Path | str] = None) -> Path:
    """Where tool manifests live — `$MANYRUNS_TOOL_DIR`, else the bundled set."""
    if config_dir:
        return Path(config_dir)
    if _env.is_set("TOOL_DIR"):
        return Path(_env.get("TOOL_DIR"))
    from importlib import resources  # never __file__-relative — survives `uv tool install`

    return Path(str(resources.files("manyruns") / "configs" / "tool"))


def discover_tools(config_dir: Optional[Path | str] = None) -> list[str]:
    d = tool_dir(config_dir)
    return sorted(p.stem for p in d.glob("*.yaml")) if d.is_dir() else []


def platform_tag() -> str:
    """`linux-x86_64`, `macos-arm64`, … — the half of a tool's identity that is the machine.

    Part of the lock filename because the resolutions genuinely differ (see the module header),
    so a lock built elsewhere must not be silently accepted here.
    """
    system = {"Darwin": "macos", "Linux": "linux", "Windows": "windows"}.get(
        platform.system(), platform.system().lower())
    machine = {"AMD64": "x86_64", "arm64": "arm64", "aarch64": "arm64"}.get(
        platform.machine(), platform.machine())
    return f"{system}-{machine}"


def lock_name(cfg: dict, tag: Optional[str] = None) -> str:
    return f"{cfg['name']}-{cfg['version']}-{tag or platform_tag()}.lock"


def lock_path(cfg: dict, config_dir: Optional[Path | str] = None,
              tag: Optional[str] = None) -> Path:
    return tool_dir(config_dir) / "locks" / lock_name(cfg, tag)


def lock_digest(path: Path) -> Optional[str]:
    """The first 16 hex of sha256(lock). THE IDENTITY — see the module header.

    Short on purpose: it is written into every g-vector that used the tool and read by humans
    comparing two runs. 16 hex is 64 bits, which is not a security boundary and is not asked to
    be one — the *hashes inside the lock* are what `uv pip sync --require-hashes` verifies.
    """
    import hashlib

    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def env_home() -> Path:
    """Where realized tool environments live. Outside the repo and outside `outputs/`, because
    they are a machine-level cache: 2.4 GB for one tool (measured), shared across projects, and
    reproducible from the lock, so nothing here is precious."""
    if _env.is_set("TOOL_HOME"):
        return Path(_env.get("TOOL_HOME"))
    return Path(os.path.expanduser("~/.cache/manyruns/tools"))


def env_path(cfg: dict, digest: str) -> Path:
    return env_home() / cfg["name"] / digest


def _override_python(name: str) -> Optional[Path]:
    """`MANYRUNS_TOOL_<NAME>_PYTHON` — an interpreter the operator built themselves.

    The escape hatch, and it exists for two real cases rather than as a courtesy: a cluster
    where the env is a module load rather than a venv, and CI, where the tool may be baked into
    an image. Its cost is stated where it is used — an overridden env has NO lock digest, so the
    record cannot claim reproducibility for it and says so instead of inventing one.
    """
    raw = _env.get(f"TOOL_{name.upper().replace('-', '_')}_PYTHON")
    return Path(raw) if raw else None


def install_command(cfg: dict, config_dir: Optional[Path | str] = None) -> str:
    """The exact command that makes this tool runnable. Printed in every refusal, because a
    refusal that does not say how to fix it is a dead end rather than a next step."""
    digest = lock_digest(lock_path(cfg, config_dir)) or "<lock-missing>"
    return (f"manyruns tools install {cfg['name']}"
            f"   # builds {env_path(cfg, digest)} from {lock_name(cfg)}")


def resolve(name: str, config_dir: Optional[Path | str] = None) -> dict:
    """Everything the runner needs to decide whether this tool can be called — and NO side
    effects. Never creates, never downloads, never mutates.

    Returns `{"status": …, "python": …, "digest": …, "reason": …, "fix": …}` where `status` is:

      ready        an interpreter exists for this exact lock; `python` is it.
      overridden   `MANYRUNS_TOOL_<NAME>_PYTHON` points at an interpreter. Honoured, and the
                   record will say the environment was NOT the locked one.
      no-manifest  no such tool is declared.
      no-lock      declared, but not locked for THIS platform — the tool has never been pinned
                   here, so there is nothing to reproduce.
      not-built    locked and not realized. The ordinary first-run state; `fix` is the command.
      precomputed  `mode: precomputed`: the results already exist under `output.root`. Nothing
                   runs, so there is no interpreter, and `MANYRUNS_TOOL_<NAME>_PYTHON` has no
                   effect on it.
    """
    try:
        cfg = load_tool(name, config_dir)
    except (ValueError, FileNotFoundError) as e:
        return {"status": "no-manifest", "python": None, "digest": None,
                "reason": str(e), "fix": None}

    if tool_mode(cfg) == "precomputed":
        return {"status": "precomputed", "python": None, "digest": None, "cfg": cfg,
                "reason": f"precomputed results under {precomputed_root(cfg, config_dir)}",
                "fix": None}

    override = _override_python(name)
    if override is not None:
        ok = override.is_file() and os.access(override, os.X_OK)
        return {"status": "overridden" if ok else "no-lock", "python": override if ok else None,
                "digest": None, "cfg": cfg,
                "reason": (f"using MANYRUNS_TOOL_{name.upper()}_PYTHON={override}" if ok else
                           f"MANYRUNS_TOOL_{name.upper()}_PYTHON={override} is not an executable"),
                "fix": None}

    path = lock_path(cfg, config_dir)
    digest = lock_digest(path)
    if digest is None:
        return {"status": "no-lock", "python": None, "digest": None, "cfg": cfg,
                "reason": (f"{cfg['name']} {cfg['version']} has no lock for {platform_tag()} "
                           f"({path} is absent) — it has never been pinned on this platform, so "
                           f"there is no environment to reproduce"),
                "fix": f"manyruns tools lock {cfg['name']}"}

    env = env_path(cfg, digest)
    python = env / "bin" / "python"
    if python.is_file():
        return {"status": "ready", "python": python, "digest": digest, "cfg": cfg,
                "reason": f"{cfg['name']} {cfg['version']} @ {digest}", "fix": None}
    return {"status": "not-built", "python": None, "digest": digest, "cfg": cfg,
            "reason": (f"{cfg['name']} {cfg['version']} is locked for {platform_tag()} "
                       f"(lock {digest}) but not built at {env}"),
            "fix": install_command(cfg, config_dir)}


def provenance(name: str, resolved: dict) -> dict:
    """What the RECORD keeps about the environment a tool ran in.

    Deliberately not the whole lock: 211 pinned packages do not belong on one line of
    `index.jsonl`. The digest is the handle — it names the lock file in the repo, which is the
    thing that reproduces the environment — plus the two facts a reader needs to know whether
    two runs are comparable at all.

    `reproducible` is FALSE for an overridden interpreter and that is the point of recording it:
    an env somebody assembled by hand may be perfect, and the record still cannot vouch for it.

    Every tool step also records where its output came from and under which licence:
    `tool_origin` is `ran` or `precomputed`, and `tool_licence` and `tool_commercial_use` copy
    the manifest. A precomputed step has no platform and no lock, because manyruns does not
    know where the model ran — the key is absent rather than a guess.
    """
    cfg = resolved.get("cfg") or {}
    precomputed = resolved.get("status") == "precomputed"
    out = {f"{name}.tool_version": cfg.get("version")}
    if not precomputed:
        out[f"{name}.tool_platform"] = platform_tag()
    out[f"{name}.tool_reproducible"] = resolved.get("status") == "ready"
    if resolved.get("digest") and not precomputed:
        out[f"{name}.tool_lock"] = resolved["digest"]
    out[f"{name}.tool_origin"] = "precomputed" if precomputed else "ran"
    out[f"{name}.tool_licence"] = cfg.get("licence")
    out[f"{name}.tool_commercial_use"] = cfg.get("commercial_use")
    return out


def manifest_facts(name: str, config_dir: Optional[Path | str] = None) -> tuple:
    """`(needs, produces)` of the manifest `name`, as frozensets. Legality reads a tool step's
    facts here, because the declaration the runner enforces is the declaration legality reads.
    Raises `ValueError` when no valid manifest has that name."""
    cfg = load_tool(name, config_dir)
    return frozenset(cfg.get("needs") or ()), frozenset(cfg.get("produces") or ())


def precomputed_root(cfg: dict, config_dir: Optional[Path | str] = None) -> Path:
    """The folder of a precomputed manifest's results; relative to the manifest's folder."""
    root = Path(str((cfg.get("output") or {}).get("root") or ""))
    return root if root.is_absolute() else tool_dir(config_dir) / root


def _requirement_name(item: str) -> str:
    import re

    m = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", item)
    return _normalize_dist(m.group(1)) if m else ""


def _normalize_dist(name: str) -> str:
    import re

    return re.sub(r"[-_.]+", "-", str(name)).lower()


def _pin_problems(cfg: dict) -> list[str]:
    """THE PIN RULE. Exactly one item of `requires` names the distribution, with `==` and an
    exact version. The pin is the contract between manyruns and the tool: a new upstream
    release changes nothing until a person moves the pin and runs `manyruns tools verify`."""
    import re

    req = cfg.get("requires")
    if not req or not isinstance(req, (list, tuple)):
        return ["`requires` must be a non-empty list of PEP 508 constraints"]
    dist = str(cfg.get("distribution") or cfg.get("name"))
    named = [str(r).strip() for r in req if _requirement_name(str(r)) == _normalize_dist(dist)]
    if not named:
        return [f"`requires` has no pin for {dist!r}: one item must be `{dist}==<exact version>` "
                f"(set `distribution` when the PyPI name differs from the tool's name)"]
    if len(named) > 1:
        return [f"`requires` names {dist!r} {len(named)} times ({named}); the pin is one item"]
    pattern = rf"^{re.escape(dist)}(\[[a-z0-9_,-]+\])?==[0-9][0-9A-Za-z.+!-]*$"
    if not re.match(pattern, named[0], flags=re.IGNORECASE):
        return [f"`requires` item {named[0]!r} is not a pin: name {dist!r} with `==` and an "
                f"exact version, so an upstream release cannot change a result unannounced"]
    return []


def _json_value(value: Any) -> bool:
    if value is None or isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_json_value(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _json_value(v) for k, v in value.items())
    return False


def check_call(call: Any, produces: Any) -> list[str]:
    """Problems with a call block; empty means valid. Checked before any call is made.

    The grammar is closed (`CALL_KEYS`): one target, constant options, slot bindings, the
    options a recipe step may override, and one selector per output."""
    import re

    if not isinstance(call, dict):
        return [f"call must be a mapping of {sorted(CALL_KEYS)}, got {type(call).__name__}"]
    bad = [f"call: unknown key {k!r} (known: {sorted(CALL_KEYS)})" for k in call
           if k not in CALL_KEYS]
    target = call.get("target")
    if not isinstance(target, str) or not re.match(
            r"^[A-Za-z_][\w.]*:[A-Za-z_][\w.]*$", target):
        bad.append(f"call.target must be '<module>:<qualname>', got {target!r}")
    method = call.get("method")
    if method is not None and not (isinstance(method, str) and method.isidentifier()):
        bad.append(f"call.method must be an identifier, got {method!r}")
    options = call.get("options") or {}
    if not isinstance(options, dict) or not _json_value(options):
        bad.append("call.options must be a map of JSON scalars, lists and maps")
        options = {}
    args = call.get("args") or []
    kwargs = call.get("kwargs") or {}
    if not isinstance(args, (list, tuple)):
        bad.append(f"call.args must be a list of slots {list(SLOTS)}")
        args = []
    if not isinstance(kwargs, dict):
        bad.append(f"call.kwargs must be a map from argument name to a slot {list(SLOTS)}")
        kwargs = {}
    for slot in [*args, *kwargs.values()]:
        if slot not in SLOTS:
            bad.append(f"call: unknown slot {slot!r} (slots: {list(SLOTS)})")
    params = call.get("params") or []
    if not isinstance(params, (list, tuple)):
        bad.append("call.params must be a list of `options` keys")
        params = []
    for item in params:
        if item not in options:
            bad.append(f"call.params: {item!r} is not a key of call.options "
                       f"{sorted(options)}")
    outputs = call.get("outputs")
    if not isinstance(outputs, dict) or not outputs:
        bad.append(f"call.outputs must be a non-empty map from a fact in {list(CALL_FACTS)} "
                   f"to a selector")
        outputs = {}
    for fact, selector in outputs.items():
        if fact not in CALL_FACTS:
            bad.append(f"call.outputs: {fact!r} cannot be selected (selectable: "
                       f"{list(CALL_FACTS)}); a {fact} needs a pinned adapter")
        elif not isinstance(selector, str) or not re.match(SELECTOR_PATTERN, selector):
            bad.append(f"call.outputs.{fact}: {selector!r} is not a selector (return, "
                       f"return[<int>], return.<attr>, return[\"<key>\"], instance.<attr>)")
        elif selector.startswith("instance.") and not method:
            bad.append(f"call.outputs.{fact}: the instance selector {selector!r} needs "
                       f"call.method, because without it no instance is built")
    if outputs and set(outputs) != set(produces or ()):
        bad.append(f"call.outputs {sorted(outputs)} must equal the manifest's produces "
                   f"{sorted(produces or ())}")
    return bad


def tool_mode(cfg: dict) -> str:
    return str(cfg.get("mode") or "run")


def selftest_command(cfg: dict) -> Optional[str]:
    """The selftest a tool's interpreter runs. A call-mode manifest without one imports the
    module of its target, because that import is the first thing the call adapter does."""
    if cfg.get("selftest"):
        return str(cfg["selftest"])
    target = (cfg.get("call") or {}).get("target")
    if tool_mode(cfg) == "call" and isinstance(target, str) and ":" in target:
        return f"import {target.split(':', 1)[0]}"
    return None


def check_tool(cfg: Any, name: Optional[str] = None) -> list[str]:
    """Problems with a tool manifest; empty means valid. Same contract as `catalog.check_*`.

    The rules depend on the mode (`MODES`). A model manifest (`output.schema: result-v1`)
    returns version 1 results and may produce only seam facts. Every manifest declares its
    `licence` and its `commercial_use`."""
    from manyruns import vocab

    bad: list[str] = []
    if not isinstance(cfg, dict):
        return [f"tool {name!r} is not a mapping"]
    unknown = sorted(set(cfg) - TOOL_FIELDS)
    if unknown:
        bad.append(f"unknown field(s) {unknown} — known: {sorted(TOOL_FIELDS)}")
    if name and cfg.get("name") != name:
        bad.append(f"name is {cfg.get('name')!r} but the file is {name}.yaml — they must match")
    if not cfg.get("version"):
        bad.append("no `version` — a tool with no version cannot be pinned, and an unpinned "
                   "tool cannot be reproduced or compared across runs")
    licence = cfg.get("licence")
    if not licence or not isinstance(licence, str):
        bad.append("no `licence` — every tool declares its SPDX licence expression, because a "
                   "licence is often the reason a tool runs as a separate program")
    if not isinstance(cfg.get("commercial_use"), bool):
        bad.append(f"commercial_use must be true or false, got {cfg.get('commercial_use')!r} — "
                   f"it is declared, never derived from the licence")
    elif licence in NON_COMMERCIAL_LICENCES and cfg["commercial_use"] is True:
        bad.append(f"licence {licence} forbids commercial use, so commercial_use cannot be true")

    mode = tool_mode(cfg)
    if mode not in MODES:
        bad.append(f"mode must be one of {MODES}, got {mode!r}")
    output = cfg.get("output") or {}
    if not isinstance(output, dict):
        bad.append("output must be a mapping")
        output = {}
    schema = output.get("schema")
    if schema is not None and schema != RESULT_SCHEMA:
        bad.append(f"output.schema must be {RESULT_SCHEMA!r}, got {schema!r}")
    model = schema == RESULT_SCHEMA
    produces = cfg.get("produces") or []
    for fact in produces:
        if not isinstance(fact, str):
            bad.append(f"produces must be a list of fact names, got {fact!r}")
    if model:
        for key in ("obsm", "obs"):
            if key in output:
                bad.append(f"output.{key} belongs to the .h5ad path; a {RESULT_SCHEMA} manifest "
                           f"returns its facts in result.json")
        allowed = [f for f in vocab.TOOL_OUTPUT_FACTS if f != "velocity"]
        for fact in produces:
            if isinstance(fact, str) and fact not in allowed:
                bad.append(f"a {RESULT_SCHEMA} manifest cannot produce {fact!r} — seam facts are "
                           f"{allowed}; velocity keeps the .h5ad path")
    known = vocab.facts()
    for fact in cfg.get("needs") or []:
        if fact not in known:
            bad.append(f"needs: {fact!r} is not a fact the calculus names ({sorted(known)})")

    if mode == "precomputed":
        for field in ("requires", "entrypoint", "selftest", "python", "input", "distribution",
                      "call"):
            if field in cfg:
                bad.append(f"`{field}` is not allowed in mode: precomputed — manyruns starts "
                           f"no interpreter for it and only reads its files")
        root = output.get("root")
        if not isinstance(root, str) or not root:
            bad.append("output.root must name the folder of the precomputed results")
        if not model:
            bad.append(f"mode: precomputed needs output.schema: {RESULT_SCHEMA}")
        return bad

    if "root" in output:
        bad.append(f"output.root is only for mode: precomputed, and this manifest is mode: {mode}")
    if mode == "call":
        if not model:
            bad.append(f"mode: call needs output.schema: {RESULT_SCHEMA}")
        if "entrypoint" in cfg:
            bad.append("`entrypoint` is not allowed in mode: call — manyruns's call adapter "
                       "makes the call the `call` block describes")
        if "call" not in cfg:
            bad.append("mode: call needs a `call` block")
        else:
            bad.extend(check_call(cfg["call"], produces))
    elif mode == "run":
        if "call" in cfg:
            bad.append("a `call` block needs mode: call")
        entry = cfg.get("entrypoint")
        if not entry or not isinstance(entry, str):
            bad.append("`entrypoint` must be the adapter module/script run INSIDE the tool's env")
        if not cfg.get("selftest"):
            # Not decoration: `pyrovelocity==0.4.5` installs and does not import. A pin nobody
            # exercised is a pin nobody has checked.
            bad.append("no `selftest` — a lock that has never been executed is not a working "
                       "pin (measured: pyrovelocity 0.4.5 installs with exit 0 and fails to "
                       "import)")
    bad.extend(_pin_problems(cfg))
    io = cfg.get("input") or {}
    if not isinstance(io, dict) or io.get("kind") not in INPUT_KINDS:
        kind = io.get("kind") if isinstance(io, dict) else io
        bad.append(f"input.kind must be one of {INPUT_KINDS}, got {kind!r}")
    return bad


def load_tool(name: str, config_dir: Optional[Path | str] = None) -> dict:
    """Load a tool manifest by name. Strict — an invalid manifest never reaches a run."""
    from omegaconf import OmegaConf

    path = tool_dir(config_dir) / f"{name}.yaml"
    if not path.is_file():
        raise ValueError(f"no tool named {name!r} at {path}. "
                         f"Known: {discover_tools(config_dir) or '(none)'}")
    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    problems = check_tool(cfg, name)
    if problems:
        raise ValueError(f"invalid tool manifest {name!r}: " + "; ".join(problems))
    return cfg  # type: ignore[return-value]


def uv() -> Optional[str]:
    """`uv`, or None. Every build path here needs it and the refusal should say so by name."""
    return shutil.which("uv")


def build(name: str, config_dir: Optional[Path | str] = None,
          timeout: int = 3600) -> dict:
    """Realize the locked environment. THE ONLY function here that touches the disk.

    Called by `manyruns tools install` and by nothing on the run path — see the module header
    for why (2.4 GB, measured). `--require-hashes` is what makes this a reproduction rather than
    a fresh resolve: every wheel is verified against the lock, so a yanked or re-uploaded
    artifact fails loudly instead of quietly changing the tool.
    """
    cfg = load_tool(name, config_dir)
    path = lock_path(cfg, config_dir)
    digest = lock_digest(path)
    if digest is None:
        raise RuntimeError(f"no lock for {name} on {platform_tag()} at {path}")
    exe = uv()
    if exe is None:
        raise RuntimeError("`uv` is not on PATH, and it is what builds a tool environment")
    env = env_path(cfg, digest)
    env.parent.mkdir(parents=True, exist_ok=True)
    py = cfg.get("python") or "3.12"
    subprocess.run([exe, "venv", "--python", str(py), str(env)], check=True, timeout=timeout)
    subprocess.run([exe, "pip", "sync", "--require-hashes", str(path)],
                   check=True, timeout=timeout, env={**os.environ, "VIRTUAL_ENV": str(env)})
    return {"env": env, "digest": digest, **selftest(cfg, env / "bin" / "python")}


def selftest(cfg: dict, python: Path, timeout: int = 600) -> dict:
    """Run the manifest's `selftest` in the built env. A pin nobody executed is not a pin.

    Returns rather than raises: `tools check` wants to report every tool, and one broken
    environment must not hide the state of the others.
    """
    cmd = selftest_command(cfg)
    if not cmd:
        return {"selftest": "declared none", "ok": False}
    try:
        proc = subprocess.run([str(python), "-c", str(cmd)], capture_output=True,
                              text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as e:
        return {"selftest": f"could not run: {e}", "ok": False}
    tail = (proc.stderr or proc.stdout or "").strip().splitlines()
    return {"selftest": "ok" if proc.returncode == 0 else (tail[-1] if tail else "failed"),
            "ok": proc.returncode == 0}
