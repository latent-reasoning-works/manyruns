# manyruns/harness/__main__.py
"""CLI entry point for the DR Workflow Labeling Harness.

Usage:
    python -m manyruns.harness run --preset pca_umap --dataset swissroll
    python -m manyruns.harness label --dataset swissroll --output labeled.json
    python -m manyruns.harness rlhf --dataset labeled.json
    python -m manyruns.harness prompt --dataset swissroll

This harness uses:
    - manyLatents: DR algorithm execution (PCA, UMAP, PHATE, etc.)
"""

from __future__ import annotations

import logging
import signal
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()
log = logging.getLogger(__name__)

HARNESS_VERSION = "0.1.0"

EPILOG = """
\b
Powered by the LRW ecosystem:
  • manyLatents - DR algorithms (required for run --backend dr)

Install dependencies: uv sync --extra harness
"""


def _check_manylatents() -> bool:
    try:
        import manylatents  # noqa: F401

        return True
    except ImportError:
        return False


def _show_banner() -> None:
    ml_status = "[green]✓[/green]" if _check_manylatents() else "[red]✗[/red]"

    console.print(
        Panel(
            f"[bold]DR Workflow Labeling Harness[/bold] v{HARNESS_VERSION}\n\n"
            f"  manyLatents: {ml_status}  [dim](DR algorithms)[/dim]",
            title="manyruns.harness",
            border_style="blue",
        )
    )


def _handle_interrupt(*_) -> None:
    console.print("\n[dim]Interrupted.[/dim]")
    sys.exit(0)


signal.signal(signal.SIGINT, _handle_interrupt)


@click.group(epilog=EPILOG)
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging")
@click.option("--no-banner", is_flag=True, help="Skip startup banner")
def cli(verbose: bool, no_banner: bool) -> None:
    """DR Workflow Labeling Harness.

    Tools for running DR workflows, labeling results, collecting feedback,
    and managing datasets. Uses manyLatents for DR execution; the workflow ends at the store.
    """
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

    if not no_banner:
        _show_banner()


@cli.command()
@click.option("--preset", "-p", help="Named preset workflow (pca_umap, phate, etc.)")
@click.option("--config", "-c", type=click.Path(exists=True), help="YAML workflow config file")
@click.option(
    "--manylatents-config", type=click.Path(exists=True), help="manyLatents experiment config"
)
@click.option("--algorithm", "-a", help="DR algorithm (PCA, UMAP, etc.) - legacy, prefer --preset")
@click.option("--n-components", "-n", type=int, help="Number of output dimensions - legacy")
@click.option("--dataset", "-d", required=True, help="Dataset name (swissroll, embryoid, etc.)")
@click.option("--metric", "-m", multiple=True, help="Metrics to compute (can specify multiple)")
@click.option(
    "--viz",
    type=click.Choice(["auto", "browser", "terminal", "file", "none"]),
    default="file",
    help="Visualization mode (default: file — saves PNG and prints path)",
)
@click.option("--viz-output", type=click.Path(), help="Output path for viz (file mode)")
@click.option(
    "--backend",
    type=click.Choice(["mock", "dr"]),
    default="mock",
    help="Backend to use: mock (canned, no deps) or dr (manyLatents)",
)
def run(
    preset: Optional[str],
    config: Optional[str],
    manylatents_config: Optional[str],
    algorithm: Optional[str],
    n_components: Optional[int],
    dataset: str,
    metric: tuple[str, ...],
    viz: str,
    viz_output: Optional[str],
    backend: str,
) -> None:
    """Run a DR workflow and display results.

    Config sources (in priority order):
    \b
      --preset            Named preset (pca_umap, phate, deep_pca_umap, etc.)
      --config            Custom YAML workflow file
      --manylatents-config  manyLatents experiment config
      --algorithm/-a      Single algorithm (legacy)

    Example:
    \b
        python -m manyruns.harness run --preset pca_umap -d swissroll
        python -m manyruns.harness run --preset pca_umap -d swissroll --backend dr
        python -m manyruns.harness run -c my_workflow.yaml -d embryoid --viz browser
    """
    from manyruns.harness.configs import resolve_config
    from manyruns.harness.display import display_results, visualize_embedding
    from manyruns.harness.interface import Request
    from manyruns.harness.interface import run as harness_run
    from manyruns.harness.runner import run_workflow  # noqa: F401

    try:
        cfg = resolve_config(
            preset=preset,
            config_file=config,
            manylatents_config=manylatents_config,
            algorithm=algorithm,
            n_components=n_components,
            metrics=list(metric) if metric else None,
        )
    except (KeyError, ValueError, FileNotFoundError) as e:
        console.print(f"[red]Config Error:[/red] {e}")
        raise SystemExit(1)

    workflow_str = " → ".join(
        f"{s['algorithm']}({s.get('params', {}).get('n_components', '?')})" for s in cfg["workflow"]
    )
    console.print(
        f"[bold]Running {workflow_str} on {dataset}[/bold] [dim](backend: {backend})[/dim]"
    )

    request = Request(
        dataset=dataset,
        workflow=cfg["workflow"],
        metrics=cfg["metrics"] or [],
    )

    try:
        response = harness_run(request, backend=backend)
    except ImportError as e:
        if "manylatents" in str(e).lower():
            console.print(
                "[red]Error:[/red] manyLatents not installed.\n"
                "[dim]Install with: uv sync --extra harness[/dim]"
            )
        else:
            console.print(f"[red]Import Error:[/red] {e}")
        raise SystemExit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)

    display_results(response.as_dict())

    if viz != "none":
        output_path = Path(viz_output) if viz_output else None
        viz_path = visualize_embedding(
            response.embeddings,
            mode=viz,
            output_path=output_path,
        )
        if viz_path:
            console.print(f"[dim]Plot saved to: {viz_path}[/dim]")


@cli.command()
@click.option("--dataset", "-d", required=True, help="Dataset name")
@click.option("--output", "-o", required=True, type=click.Path(), help="Output dataset file (JSON)")
@click.option("--labeler", "-l", default="anonymous", help="Your name/ID")
def label(dataset: str, output: str, labeler: str) -> None:
    """Label a DR workflow result interactively and save to dataset.

    \b
    Example:
        python -m manyruns.harness label -d swissroll -o labeled.json
    """
    console.print("[yellow]label:[/yellow] not yet implemented — stub")
    raise SystemExit(0)


@cli.command()
@click.option(
    "--dataset", "-d", required=True, type=click.Path(exists=True), help="Labeled dataset (JSON)"
)
@click.option("--rounds", "-r", type=int, default=10, help="Number of feedback rounds")
def rlhf(dataset: str, rounds: int) -> None:
    """Collect reinforcement feedback on labeled embeddings (RLHF loop).

    \b
    Example:
        python -m manyruns.harness rlhf -d labeled.json --rounds 5
    """
    console.print("[yellow]rlhf:[/yellow] not yet implemented — stub")
    raise SystemExit(0)


def _ensure_api_key(agent: str, api_key: Optional[str]) -> None:
    """Hosted backends (claude) need a key. Resolve from flag/env, else prompt
    once, and place it in the environment so the manyAgents adapter reads it.
    """
    import os

    if agent != "claude":
        return
    if not api_key:
        api_key = click.prompt("Anthropic API key", hide_input=True)
    os.environ["ANTHROPIC_API_KEY"] = api_key


@cli.command()
@click.argument("prompt_text")
@click.option(
    "--agent",
    default="vllm",
    help="Agent backend: vllm (GPU+trace) | hf (MPS+trace) | ollama | claude | mock",
)
@click.option("--model", default=None, help="Model override (vllm default: Qwen/Qwen3-0.6B)")
@click.option(
    "--api-key",
    default=None,
    envvar="ANTHROPIC_API_KEY",
    help="API key for hosted backends (claude). Falls back to ANTHROPIC_API_KEY; "
    "prompts securely if needed.",
)
def prompt(prompt_text: str, agent: str, model: Optional[str], api_key: Optional[str]) -> None:
    """Send a prompt to manyAgents and print the response.

    \b
    Examples:
        manyruns prompt "Compute the gene graph" --agent ollama --model qwen3:30b
        manyruns prompt "Describe a swiss roll" --agent claude
    """
    from manyruns.harness.interface import Request
    from manyruns.harness.interface import run as harness_run

    _ensure_api_key(agent, api_key)

    try:
        response = harness_run(Request(prompt=prompt_text, agent=agent, model=model))
    except ImportError as e:
        console.print(
            f"[red]Error:[/red] {e}\n"
            "[dim]Install the prompt stream with: uv sync --extra agents[/dim]"
        )
        raise SystemExit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise SystemExit(1)

    console.print(response.answer or "[dim](no text returned)[/dim]")
    if response.trace_path:
        console.print(f"[dim]trace: {response.trace_path}[/dim]")


MANYRUNS_SYSTEM = (
    "You are Manyruns, an assistant for geometric analysis of scientific data. "
    "When the user asks to embed, reduce, visualize, or analyze the structure of a "
    "dataset, call the run_dr_workflow tool, then explain the results plainly."
)


@cli.command()
@click.option(
    "--agent",
    default="ollama",
    help="Agent backend: ollama (laptop) | claude (frontier API) | vllm (GPU) | hf (MPS) | mock",
)
@click.option(
    "--model",
    default=None,
    help="Model override (ollama: qwen3:30b · claude: claude-opus-4-8)",
)
@click.option(
    "--api-key",
    default=None,
    envvar="ANTHROPIC_API_KEY",
    help="API key for hosted backends (claude). Falls back to ANTHROPIC_API_KEY; "
    "prompts securely if needed.",
)
@click.option("--max-steps", default=8, show_default=True, help="Max tool-loop steps per turn")
def chat(agent: str, model: Optional[str], api_key: Optional[str], max_steps: int) -> None:
    """Interactive agentic REPL — the model can call DR compute and answer.

    \b
    Examples:
        manyruns chat --agent ollama --model qwen3:30b
        manyruns chat --agent claude   # uses ANTHROPIC_API_KEY, or prompts
        > Run a UMAP embedding on swissroll and tell me the trustworthiness.
    """
    import asyncio

    try:
        from manyagents.agent_loop import run_agent_loop

        from manyruns.harness.agent_tools import make_dr_tool
    except ImportError as e:
        console.print(
            f"[red]Error:[/red] {e}\n"
            "[dim]Install the agent stream with: uv sync --extra agents[/dim]"
        )
        raise SystemExit(1)

    _ensure_api_key(agent, api_key)

    tools = [make_dr_tool()]

    def render(kind: str, payload: dict) -> None:
        if kind == "tool_call":
            args = ", ".join(f"{k}={v!r}" for k, v in payload["arguments"].items())
            console.print(f"[dim]🔧 {payload['name']}({args})[/dim]")
        elif kind == "tool_result":
            console.print(f"[dim]↳ {payload['result']}[/dim]")

    console.print("[bold]Manyruns chat[/bold] [dim](ctrl-D or 'exit' to quit)[/dim]")
    history: list = []
    while True:
        try:
            user = console.input("[bold cyan]> [/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye[/dim]")
            break
        if user.lower() in {"exit", "quit"}:
            console.print("[dim]bye[/dim]")
            break
        if not user:
            continue
        try:
            result = asyncio.run(
                run_agent_loop(
                    user,
                    agent=agent,
                    model=model,
                    tools=tools,
                    system_prompt=MANYRUNS_SYSTEM,
                    history=history,
                    max_steps=max_steps,
                    on_event=render,
                )
            )
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            continue
        history = result.messages
        console.print(f"[green]{result.answer}[/green]")


@cli.command("list")
@click.option(
    "--dataset", "-d", required=True, type=click.Path(exists=True), help="Dataset file (JSON)"
)
def list_cmd(dataset: str) -> None:
    """List entries in a labeled dataset."""
    from manyruns.harness.storage import dataset_summary, load_labeled

    entries = load_labeled(dataset)

    if len(entries) == 0:
        console.print("[dim]Dataset is empty (0 entries)[/dim]")
        return

    summary = dataset_summary(dataset)
    console.print(f"[bold]Dataset:[/bold] {dataset}")
    console.print(f"[bold]Entries:[/bold] {summary['total_entries']}")
    console.print(f"[bold]Created:[/bold] {summary['created']}")
    console.print()

    if summary["labels"]:
        console.print("[bold]Labels:[/bold]")
        for lbl, count in summary["labels"].items():
            console.print(f"  {lbl}: {count}")
        console.print()

    table = Table(title="Entries")
    table.add_column("ID", style="dim", width=8)
    table.add_column("Dataset")
    table.add_column("Workflow")
    table.add_column("Label", style="cyan")
    table.add_column("Labeler")

    for entry in entries:
        workflow_str = " → ".join(
            f"{s['algorithm']}({s.get('params', {}).get('n_components', '?')})"
            for s in entry["workflow"]
        )
        table.add_row(
            entry["id"][:8],
            entry["dataset"],
            workflow_str,
            entry["label"],
            entry["labeler"],
        )

    console.print(table)


@cli.command("presets")
def presets_cmd() -> None:
    """List available workflow presets."""
    from manyruns.harness.configs import PRESETS

    console.print("[bold]Available Workflow Presets[/bold]\n")

    table = Table()
    table.add_column("Name", style="cyan")
    table.add_column("Description")
    table.add_column("Workflow")
    table.add_column("Metrics", style="dim")

    for key, preset in PRESETS.items():
        workflow_str = " → ".join(
            f"{s['algorithm']}({s.get('params', {}).get('n_components', '?')})"
            for s in preset["workflow"]
        )
        metrics_str = ", ".join(preset.get("metrics", []))
        table.add_row(key, preset["description"], workflow_str, metrics_str)

    console.print(table)
    console.print()
    console.print("[dim]Use with: python -m manyruns.harness run --preset <name> -d <dataset>[/dim]")


def main() -> None:
    """Entry point for python -m manyruns.harness."""
    cli()


if __name__ == "__main__":
    main()
