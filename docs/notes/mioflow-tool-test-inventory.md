# MIOFlow commercial flag and declared time: test inventory

This file is on the branch only. It re-classifies, against the change
`add-commercial-flag-and-declare-mioflow-time`, every row of the inventory that commit `6879eb4`
wrote for the withdrawn tool-route plan (tasks.md, task 1). It is deleted in task 9.3, so it
stays in the branch history for review.

The in-process MIOFlow route stays, so most rows are `keep`. The verdicts are MEASURED: the full
suite ran with the change applied and the tests unchanged, and every test that failed is listed
below with what changed it.

| Verdict | Meaning |
|---|---|
| keep | the behaviour still exists; the test is unchanged |
| adjust | the behaviour still exists, and the test changes one expectation of this change: `non_commercial_steps`, the `time` need, or the probe slot |
| retire | this change removes the behaviour; the column "Requirement" names the requirement in `spec.md` that removes it |

"time-course" in a note means the test's shape moves from a shape with no time axis to
`time-course`, so `mioflow`'s new `time` need is met and the test keeps its subject (an
embedding need, a narrowing, an order). "re-pinned" means a measured table gains the `time`
rows of `cflows`, `traced` and `sandbox`.

## Files that reach route internals

| File | Test function | What it checks | Verdict | Requirement | Note |
|---|---|---|---|---|---|
| test_condition_as_time.py | condition / unknown-kind / no-kind label tests (3) | a non-time label axis never reaches MIOFlow as time | keep | | they fake `_run_mioflow_experiment` |
| test_condition_as_time.py | test_real_time_labels_still_reach_mioflow | a real time axis reaches MIOFlow | keep | | |
| test_condition_as_time.py | test_the_fix_is_label_typing_not_over_pruning | `cflows` legal on manifold through the invented axis | retire | Time is declared, never invented | replaced by `test_mioflow_time.py::test_cflows_needs_real_time` |
| test_display_routes.py | test_analysis_axes_and_coordinates_do_not_change_with_display | time axis and coordinates do not depend on the display colour | keep | | |
| test_display_routes.py | test_mioflow_receives_the_actual_time_tensor_for_each_display | the in-process datamodule's label tensor | keep | | |
| test_engine_extras.py | test_the_mioflow_seam_returns_its_byproducts_instead_of_dropping_them | the source of `_run_mioflow_experiment` | keep | | |
| test_manylatents_engine.py | test_latent_via_api_and_mioflow_via_run_experiment | latent dispatch, chaining into mioflow | keep | | |
| test_manylatents_engine.py | test_real_timepoint_labels_reach_mioflow | time labels reach the step | keep | | |
| test_manylatents_engine.py | test_mioflow_plot_uses_pre_mioflow_embedding_as_background | `mioflow.png` over the embedding the step received | keep | | the figure is now drawn by `apply_step` from `state["trajectories"]`, over the same frame |
| test_manylatents_engine.py | test_mioflow_first_without_embedding_is_skipped | mioflow first is skipped | keep | | |
| test_mioflow_params.py | test_a_declared_mioflow_param_reaches_the_engine | declared params reach MIOFlow | adjust | | time need: passes real time labels; drops the `n_timepoints` param |
| test_mioflow_params.py | test_a_bounded_param_reaches_the_engine_clamped | the clamp, the caveat and the engine agree | adjust | | time need: `_state` carries a real time axis |
| test_mioflow_params.py | test_an_unknown_mioflow_param_errors_the_step_instead_of_running_the_default | a typo is not silent | adjust | | time need: as above |
| test_mioflow_params.py | test_a_param_the_run_owns_is_refused_with_its_own_reason | `init_seed` is the run's | adjust | | time need: as above |
| test_mioflow_params.py | test_the_lightning_cli_path_refuses_params_it_cannot_forward | the CLI route drops no params silently | keep | | |
| test_mioflow_params.py | test_the_param_namespaces_are_disjoint_and_the_signature_is_closed | the two MIOFlow signatures are disjoint and closed | adjust | | time need: drops the one line that read `_TIME_PARAMS`, which the requirement "Time is declared, never invented" removes; the rest is unchanged |
| test_mioflow_params.py | test_the_time_axis_params_are_manyruns_own_and_are_not_forwarded | `_TIME_PARAMS` are consumed and never forwarded | retire | Time is declared, never invented | replaced by `test_mioflow_time.py::test_the_old_time_params_are_unknown` |
| test_mioflow_params.py | test_init_seed_means_weight_initialization_on_both_construction_paths, test_the_run_seed_reaches_network_construction_before_the_experiment, test_another_constructor_overlap_is_still_refused, test_manyruns_defaults_survive_a_silent_recipe, test_an_explicit_null_reaches_the_engine_as_none, test_every_manyruns_default_stays_overridable_after_the_spread, test_every_keyword_the_call_site_passes_by_hand_is_reserved | `_resolve_kwargs`, `_RESERVED` and the MIOFlow signatures | keep | | |
| test_model_retention.py | test_model_is_a_fact_a_named_step_produces | `STEP_PRODUCES["mioflow"]` | adjust | | probe slot: `("model", "trajectories")` |
| test_model_retention.py | test_a_probe_before_a_fit_is_refused_at_plan_time, test_a_filter_between_the_fit_and_the_probe_is_refused | probe ordering at plan time | adjust | | probe slot: the missing / cleared fact is `trajectories` |
| test_model_retention.py | test_a_probe_after_a_fit_is_legal | probe ordering at plan time | keep | | |
| test_model_retention.py | test_model_is_never_a_frame_fact | disjointness of the probed fact | adjust | | probe slot: adds the same two assertions for `trajectories` |
| test_model_retention.py | test_sample_trajectories_reads_the_model_and_writes_to_the_g_vector | the probe answer | adjust | | probe slot: renamed `..._reads_the_slot_...`; reads `state["trajectories"]` |
| test_model_retention.py | test_a_probe_with_no_model_declines_with_a_reason | the decline | adjust | | probe slot: "no trajectories"; a model probe still says "no trained model" |
| test_model_retention.py | test_a_probe_is_dispatchable_only_where_a_model_can_exist | where a probe is offered | adjust | | probe slot: renamed `..._where_trajectories_can_exist`; the in-process loop offers it, mock does not |
| test_named_dataset_facts.py | test_unmet_calls_a_trajectory_recipe_legal_on_a_time_course_declaration | `unmet` trusts the declaration | keep | | |

## Files that build a `mioflow` step directly

| File | Test function | What it checks | Verdict | Requirement | Note |
|---|---|---|---|---|---|
| test_artifacts.py | 6 tests (round trip, derived slots, unchanged slot, COMPLETE marker, figure ownership, no-plot step) | artifacts and figure attribution | keep | | |
| test_devices.py | test_recipe_requires_fp64_only_for_mioflow | device precision by step | keep | | |
| test_figure_pane.py | 2 tests (figure per step, dashboard) | figure attribution in the pane | keep | | |
| test_lineage.py | every chain / rewind / branch test | lineage and artifact layout | keep | | |
| test_narrowing_calculus.py | 11 tests on the `MIOFLOW` constant | a narrowing clears a produced embedding | keep (7), adjust (4) | | time need, time-course: test_a_column_narrowing_clears_the_embedding_for_the_same_reason_a_row_one_does, test_a_caller_who_says_they_hold_the_embedding_keeps_it_across_a_narrowing, test_detect_doublets_narrows_only_when_the_recipe_asks_it_to, test_a_fact_nothing_ever_produced_is_missing_but_was_never_TAKEN_away |
| test_narrowing_calculus.py | test_no_step_produced_fact_is_also_a_frame_fact | producer and frame facts are disjoint | adjust | | time need: the overlap is exactly `time` (`discretize_time`); `step_facts()` stays disjoint from the frame |
| test_narrowing_calculus.py | test_every_bundled_recipe_is_still_legal_on_the_data_its_declared_shape_describes | every recipe legal on its suit | adjust | | time need: `sandbox` (suit-less, read on `unknown`) needs `time` |
| test_offloaded_compute.py | test_retained_model_survives_another_attempt_and_parent_rollback | worker retention and rollback | adjust | | probe slot: the fake fills `state["trajectories"]` as the route does |
| test_offloaded_compute.py | test_real_mioflow_retains_its_model_for_a_later_probe | MIOFlow model then probe | keep | | |
| test_ok_and_complete.py | 3 tests (decline keeps ok, ok without complete, complete implies ok) | the `ok` / `complete` split | keep | | |
| test_order_calculus.py | tests on the `MIOFLOW` constant | `unmet` sees step order | adjust (6) | | time need: backwards / forward expect `time` off time-course; self-feeding, other-embedder, groupless and the plan test move to time-course |
| test_order_calculus.py | test_every_bundled_recipe_keeps_its_exact_legality_on_every_shape | the legality table | adjust | | time need: re-pinned |
| test_order_calculus.py | test_every_declared_need_is_providable_by_something | every need has a supplier | keep | | |
| test_recipe_convention.py | test_the_in_process_loop_announces_that_mioflow_is_a_stand_in | the stand-in note | keep | | |
| test_recipe_convention.py | 5 stand-in mechanics tests | `standins` / `via` mechanics | keep | | |
| test_recipe_convention.py | test_known_steps_carries_via_only_when_declared | `known_steps` fields | keep | | |
| test_run_panel.py | test_per_step_geometry_lands_in_the_steps_panel_the_live_panel_and_the_plain_text | panel rendering of records | keep | | |
| test_session.py | test_a_prep_step_that_never_RAN_does_not_rewrite_the_runs_provenance, other session tests naming mioflow | session flow | keep | | |
| test_step_order.py | test_the_refusal_is_at_BOTH_the_calculus_and_the_ENGINE | both levels refuse | adjust | | time need: level 1 expects `time` off time-course |
| test_step_order.py | the other 3 tests | the engines refuse an ordering before an embedding | keep | | |
| test_supplied_embedding.py | test_a_supplied_embedding_makes_a_bare_trajectory_recipe_legal, test_the_uncomposable_recipe_is_still_refused_with_an_embedding_nowhere, test_an_unmet_need_is_never_reported_as_merely_unusual | a supplied embedding meets the need | adjust | | time need, time-course |
| test_supplied_embedding.py | the other 4 tests | a supplied embedding runs | keep | | |
| test_synthetic_timecourse.py | test_cflows_is_legal_here_and_contrast_still_is_not | cflows legal on time-course | keep | | |
| test_tui_figure.py, test_tui_integration.py, test_tui_run.py | figure and feed rendering | rendering of synthetic records | keep | | |
| test_tui_state.py | test_a_refusal_caused_by_a_filter_does_not_claim_that_nothing_embeds, test_a_recipe_a_NARROWING_blocks_is_hollow_on_both_surfaces_too | MISSING vs CLEARED reasons | adjust | | time need, time-course |
| test_tui_state.py | test_a_fact_no_step_needs_can_never_reach_a_ledger_row | `time` never reaches a ledger row | adjust | | time need: inverted as its docstring said it would be, renamed `test_a_time_need_reaches_a_ledger_row` |
| test_tui_state.py | test_the_ledger_shows_the_refused_recipe_instead_of_hiding_it, test_the_capability_count_discriminates_once_a_row_reads_its_own_facts | ledger counts | adjust | | time need: `cflows` and `traced` are hollow for `time`; the shape-only count is 7 on time-course and 5 elsewhere |

## Files that read the model or the probe

| File | Test function | What it checks | Verdict | Requirement | Note |
|---|---|---|---|---|---|
| test_commands.py | alias and command-table tests | command names | keep | | |

## Files outside the planned list

| File | Test function | Verdict | Requirement | Note |
|---|---|---|---|---|
| test_recipe_catalog.py | legality on bundled data | keep | | |
| test_recipe_catalog.py | test_the_recipe_runs[cflows], test_the_recipe_runs[traced] | adjust | | time need: a recipe that needs `time` runs with three real time points |
| test_narrate.py | test_offer_recommends_the_right_question | keep | | |
| test_analysis_selection.py | time-axis selection tests | keep | | two tests added for the commercial flag |
| test_check.py | test_coverage_warns_when_a_recipe_has_no_legal_dataset, test_coverage_clears_when_a_case_control_dataset_exists | adjust | | time need: the three MIOFlow recipes are unexercisable without a time-course dataset |
| test_front_door.py, test_tui_ledger.py (other tests) | menu and ledger tests | keep | | |
| test_tui_ledger.py | test_the_mark_and_the_selectability_are_both_narrates_verdict, test_the_same_analysis_fills_in_once_the_data_carries_the_fact | adjust | | time need: `cflows` and `traced` are hollow rows |
| test_shell.py | test_plain_prompter_drives_the_shell_without_a_tty | adjust | | time need: `cflows` is not legal on swissroll, so the script drives `cluster` |
| test_extension_contract.py | demanded-fact and cleared-phrase tests | keep | | |
| test_model_seam_legality.py | test_manifest_replaces_name_table | adjust | | probe slot: uses `growth_rate`, which reads `model`; `sample_trajectories` reads `trajectories`, which both routes supply |
| test_analysis_readouts.py | test_the_bundled_recipes_keep_their_legality | adjust | | time need: re-pinned |
| test_experiment.py | test_the_bundled_set_cannot_yet_exercise_contrast | adjust | | time need: the pruning ledger gains the three MIOFlow recipes |
| test_experiment.py | test_recipes_that_fall_back_are_not_over_pruned | retire | Time is declared, never invented | replaced by `test_mioflow_time.py::test_cflows_needs_real_time` |
| test_gene_axis_fact.py | test_the_sweep_prices_the_gene_demand_and_every_pruned_cell_is_named, test_every_bundled_recipe_is_legal_on_its_declared_shape_given_data_that_carries_genes[sandbox] | adjust | | time need: re-pinned (92 pruned cells) |
| test_stubs.py | test_the_sandbox_recipe_is_legal_on_every_shape | adjust | | time need: renamed `test_the_sandbox_recipe_is_refused_only_for_time` |
| test_named_dataset_facts.py | test_load_inputs_refuses_a_named_time_course_dataset | adjust | | time need: the refusal no longer names a fabricated pseudotime; it says the step refuses |
| test_ok_and_complete.py | test_a_stub_spoils_ok_because_it_is_a_capability_that_does_not_exist | keep | | it failed on a defect in the change (a stub probe on the in-process loop read as a benign skip); fixed in `runner._run_probe_step`, test unchanged |
| test_doc_references.py | test_docs_keep_only_contract_sidecars_and_guides | keep | | fails only while this file exists; passes after task 9.3 |

## Retire verdicts for review (task 1.3)

| Test | Requirement that removes the behaviour |
|---|---|
| test_condition_as_time.py::test_the_fix_is_label_typing_not_over_pruning | Time is declared, never invented |
| test_mioflow_params.py::test_the_time_axis_params_are_manyruns_own_and_are_not_forwarded | Time is declared, never invented |
| test_experiment.py::test_recipes_that_fall_back_are_not_over_pruned | Time is declared, never invented |
