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
    """
    try:
        cfg = load_tool(name, config_dir)
    except (ValueError, FileNotFoundError) as e:
        return {"status": "no-manifest", "python": None, "digest": None,
                "reason": str(e), "fix": None}

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
    """
    cfg = resolved.get("cfg") or {}
    out = {
        f"{name}.tool_version": cfg.get("version"),
        f"{name}.tool_platform": platform_tag(),
        f"{name}.tool_reproducible": resolved.get("status") == "ready",
    }
    if resolved.get("digest"):
        out[f"{name}.tool_lock"] = resolved["digest"]
    return out


def check_tool(cfg: Any, name: Optional[str] = None) -> list[str]:
    """Problems with a tool manifest; empty means valid. Same contract as `catalog.check_*`."""
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
    req = cfg.get("requires")
    if not req or not isinstance(req, (list, tuple)):
        bad.append("`requires` must be a non-empty list of PEP 508 constraints")
    elif not any(str(r).startswith(str(cfg.get("name"))) for r in req):
        bad.append(f"`requires` never names {cfg.get('name')!r} itself")
    entry = cfg.get("entrypoint")
    if not entry or not isinstance(entry, str):
        bad.append("`entrypoint` must be the adapter module/script run INSIDE the tool's env")
    if not cfg.get("selftest"):
        # Not decoration: `pyrovelocity==0.4.5` installs and does not import. A pin nobody
        # exercised is a pin nobody has checked.
        bad.append("no `selftest` — a lock that has never been executed is not a working pin "
                   "(measured: pyrovelocity 0.4.5 installs with exit 0 and fails to import)")
    io = cfg.get("input") or {}
    if io.get("kind") not in INPUT_KINDS:
        bad.append(f"input.kind must be one of {INPUT_KINDS}, got {io.get('kind')!r}")
    for fact in cfg.get("produces") or []:
        if not isinstance(fact, str):
            bad.append(f"produces must be a list of fact names, got {fact!r}")
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
    cmd = cfg.get("selftest")
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
