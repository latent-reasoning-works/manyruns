from manyruns import catalog

LATENT_ON_INPUT = {"pca", "phate", "umap", "diffusionmap", "leiden"}
EXEMPT = {"sandbox"}  # a dev fixture with no analysis, see its yaml


def test_every_bundled_recipe_normalizes_before_its_first_latent_step():
    """`loading._anndata_matrix` returns `adata.X` as stored and its docstring says every bundled
    recipe declares the preamble. A recipe that does not embeds raw counts."""
    for name in sorted(set(catalog.discover_recipes()) - EXEMPT):
        steps = [s["name"] for s in catalog.load_recipe(name)["steps"]]
        first = next((i for i, s in enumerate(steps) if s in LATENT_ON_INPUT), None)
        if first is None:
            continue
        assert "normalize" in steps[:first] and "transform" in steps[:first], (name, steps)


def test_traced_matches_cflows_up_to_phate():
    def head(name):
        steps = catalog.load_recipe(name)["steps"]
        stop = [s["name"] for s in steps].index("phate") + 1
        return [(s["name"], s.get("params")) for s in steps[:stop]]

    assert head("traced") == head("cflows")
