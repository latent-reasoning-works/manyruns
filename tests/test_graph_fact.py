"""A graph-valued step provides only the facts its name declares.

`vocab.GROUP_PROVIDES["latent"]` says every latent step provides `embedding`, because every
latent executor replaces `state["emb"]`. `reeb_graph` does not: it leaves the coordinates in
place and hands back a graph. So the calculus must not credit it with an embedding, or
`[reeb_graph, dpt]` is legal at plan time and `dpt` skips at run time.
"""
from __future__ import annotations

from manyruns import vocab


def test_the_reeb_step_does_not_provide_an_embedding():
    assert vocab.STEP_PRODUCES["reeb_graph"] == ("graph",)
    assert vocab.step_provides({"name": "reeb_graph", "group": "latent"}) == frozenset({"graph"})


def test_a_readout_that_needs_coordinates_is_illegal_after_a_lone_reeb_step():
    recipe = {"name": "r", "steps": [{"name": "reeb_graph", "group": "latent"},
                                     {"name": "dpt", "group": "analysis"}]}
    assert vocab.unmet(recipe, "manifold") == frozenset({"embedding"})


def test_other_latent_steps_are_unchanged():
    assert vocab.step_provides({"name": "phate", "group": "latent"}) == frozenset({"embedding"})
    assert vocab.step_provides({"name": "leiden", "group": "latent"}) == frozenset(
        {"embedding", "clusters"})
