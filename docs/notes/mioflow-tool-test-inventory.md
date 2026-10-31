# MIOFlow tool route: test inventory

This file is on the branch only. It records, for each test file that depended on the in-process
MIOFlow route, the verdict of each affected test function (tasks.md, task 1). It is deleted in
task 9.3, so it stays in the branch history for review.

| Verdict | Meaning |
|---|---|
| keep | the behaviour still exists; the test changes nothing or only adds `tool: mioflow` |
| port | the behaviour still exists; the test is rewritten (against the stub MIOFlow folder, or with `dpt` where `mioflow` was only a generic step) |
| retire | the behaviour is removed; the column "Requirement" names the requirement of `spec.md` that removes it |

"Swapped to `dpt`" means that `mioflow` was only a convenient in-process step in that test (a
step that needs an embedding, sets `pseudotime` and leaves `emb`). `dpt` does the same, so the
test keeps its subject and its assertions.

## Files that reach route internals

| File | Test function | What it checks | Verdict | Requirement | Note |
|---|---|---|---|---|---|
| test_condition_as_time.py | condition / unknown-kind / no-kind label tests (3) | a non-time label axis never reaches MIOFlow as time | port | | stub tool: no `time.npy` is sent, the step declines for want of a time axis |
| test_condition_as_time.py | test_real_time_labels_still_reach_mioflow | a real time axis reaches MIOFlow | port | | `time.npy` equals the labels; numeric labels |
| test_condition_as_time.py | test_the_fix_is_label_typing_not_over_pruning | `cflows` legal on manifold through the invented axis | retire | Time is declared, never invented | replaced by `test_cflows_needs_a_declared_time_axis_on_every_shape_without_one` |
| test_display_routes.py | test_analysis_axes_and_coordinates_do_not_change_with_display | time axis and coordinates do not depend on the display colour | port | | stub tool; time read from `time.npy` |
| test_display_routes.py | test_mioflow_receives_the_actual_time_tensor_for_each_display | the in-process datamodule's label tensor | retire | The in-process route is gone | the port above covers the property |
| test_engine_extras.py | test_the_mioflow_seam_returns_its_byproducts_instead_of_dropping_them | the source of `_run_mioflow_experiment` | retire | The in-process route is gone | |
| test_manylatents_engine.py | test_latent_via_api_and_mioflow_via_run_experiment | latent dispatch, chaining into mioflow | port | | renamed `..._via_the_tool`; the tool's `input.npy` equals phate's output |
| test_manylatents_engine.py | test_real_timepoint_labels_reach_mioflow | time labels reach the step | port | | `time.npy` equals the labels |
| test_manylatents_engine.py | test_mioflow_plot_uses_pre_mioflow_embedding_as_background | `mioflow.png` over the embedding the step received | port | | stub tool |
| test_manylatents_engine.py | test_mioflow_first_without_embedding_is_skipped | mioflow first is skipped | keep | | adds `tool: mioflow` |
| test_mioflow_params.py | test_a_declared_mioflow_param_reaches_the_engine | declared params reach MIOFlow | port | | params reach the tool's `request.json` |
| test_mioflow_params.py | test_a_bounded_param_reaches_the_engine_clamped | the clamp, the caveat and the engine agree | port | | the request carries the clamped value |
| test_mioflow_params.py | test_an_unknown_mioflow_param_errors_the_step_instead_of_running_the_default | a typo is not silent | port | | the adapter declines it: `skipped` with the allow-list |
| test_mioflow_params.py | test_a_param_the_run_owns_is_refused_with_its_own_reason | `init_seed` is the run's | port | | the run seed is in the request; `init_seed` is declined |
| test_mioflow_params.py | test_the_lightning_cli_path_refuses_params_it_cannot_forward | the CLI route drops no params silently | keep | | |
| test_mioflow_params.py | test_the_param_namespaces_are_disjoint_and_the_signature_is_closed, test_init_seed_means_weight_initialization_on_both_construction_paths, test_the_run_seed_reaches_network_construction_before_the_experiment, test_another_constructor_overlap_is_still_refused, test_manyruns_defaults_survive_a_silent_recipe, test_the_time_axis_params_are_manyruns_own_and_are_not_forwarded, test_an_explicit_null_reaches_the_engine_as_none, test_every_manyruns_default_stays_overridable_after_the_spread, test_every_keyword_the_call_site_passes_by_hand_is_reserved | `_resolve_kwargs`, `_RESERVED`, `_TIME_PARAMS` and the manylatents MIOFlow signatures | retire | The in-process route is gone | |
| test_model_retention.py | test_model_is_a_fact_a_named_step_produces | `STEP_PRODUCES["mioflow"] == ("model",)` | retire | MIOFlow readouts read the trajectories slot | replaced by `test_the_mioflow_step_produces_trajectories_not_a_model` |
| test_model_retention.py | test_a_probe_before_a_fit_is_refused_at_plan_time, test_a_probe_after_a_fit_is_legal, test_a_filter_between_the_fit_and_the_probe_is_refused | probe ordering at plan time | port | | the fact is `trajectories`; the fit is the tool step |
| test_model_retention.py | test_model_is_never_a_frame_fact | disjointness of the probed fact | port | | renamed; asserts it for `trajectories` |
| test_model_retention.py | test_sample_trajectories_reads_the_model_and_writes_to_the_g_vector, test_a_probe_with_no_model_declines_with_a_reason | the probe answer and its decline | port | | reads `state["trajectories"]` |
| test_model_retention.py | test_a_probe_is_dispatchable_only_where_a_model_can_exist | where a probe is offered | port | | the in-process loop now offers it; mock does not |
| test_named_dataset_facts.py | test_unmet_calls_a_trajectory_recipe_legal_on_a_time_course_declaration | `unmet` trusts the declaration | port | | the step has `tool: mioflow`; stub on the tool search path |

## Files that build a `mioflow` step directly

| File | Test function | What it checks | Verdict | Requirement | Note |
|---|---|---|---|---|---|
| test_artifacts.py | 6 tests (round trip, derived slots, unchanged slot, COMPLETE marker, figure ownership, no-plot step) | artifacts and figure attribution | port | | swapped to `dpt` |
| test_devices.py | test_recipe_requires_fp64_only_for_mioflow | device precision by step | keep | | unchanged, passes |
| test_figure_pane.py | 2 tests (figure per step, dashboard) | figure attribution in the pane | port | | swapped to `dpt` |
| test_lineage.py | every chain / rewind / branch test | lineage and artifact layout | port | | swapped to `dpt` |
| test_narrowing_calculus.py | 11 tests on the `MIOFLOW` constant | a narrowing clears a produced embedding | port | | swapped to `dpt` |
| test_narrowing_calculus.py | test_no_step_produced_fact_is_also_a_frame_fact | producer and frame facts are disjoint | port | | the overlap is exactly `time` (`discretize_time`) |
| test_narrowing_calculus.py | bundled-recipe legality on its own suit | every recipe legal on its suit | port | | stub on the path; `sandbox` named as never legal |
| test_offloaded_compute.py | test_retained_model_survives_another_attempt_and_parent_rollback | worker retention and rollback | port | | the fake also returns trajectories |
| test_offloaded_compute.py | test_real_mioflow_retains_its_model_for_a_later_probe | MIOFlow model then probe | port | | replaced by `test_a_tool_steps_trajectories_reach_a_later_probe` (stub tool) |
| test_ok_and_complete.py | 3 tests (decline keeps ok, ok without complete, complete implies ok) | the `ok` / `complete` split | port | | swapped to `dpt` |
| test_order_calculus.py | tests on the `MIOFLOW` constant | `unmet` sees step order | port | | swapped to `dpt` |
| test_order_calculus.py | test_every_bundled_recipe_keeps_its_exact_legality_on_every_shape | the legality table | port | | re-pinned: `tool:mioflow` on a plain install |
| test_order_calculus.py | test_every_declared_need_is_providable_by_something | every need has a supplier | port | | tool output facts count; declared stubs exempt |
| test_recipe_convention.py | test_the_in_process_loop_announces_that_mioflow_is_a_stand_in | the stand-in note | retire | The in-process route is gone | replaced by `test_nothing_stands_in_for_mioflow_any_more` |
| test_recipe_convention.py | 5 stand-in mechanics tests (`via` suppresses, no `via`, caveat channel, finished record, recipe run xfail) | `standins` / `via` mechanics | port | | a stand-in declared by a fixture, on `dpt` |
| test_recipe_convention.py | test_known_steps_carries_via_only_when_declared | `known_steps` fields | port | | `tool` rides along |
| test_run_panel.py | test_per_step_geometry_lands_in_the_steps_panel_the_live_panel_and_the_plain_text | panel rendering of records | keep | | synthetic records |
| test_session.py | test_a_prep_step_that_never_RAN_does_not_rewrite_the_runs_provenance | a skipped prep step keeps provenance | port | | swapped to `dpt` |
| test_session.py | other session tests naming mioflow | session flow on mock | keep | | the mock never launches a tool |
| test_step_order.py | 4 tests | both engines refuse an ordering before an embedding | port | | swapped to `dpt` |
| test_supplied_embedding.py | 7 tests | a supplied embedding meets the need | port | | swapped to `dpt` |
| test_synthetic_timecourse.py | test_cflows_is_legal_here_and_contrast_still_is_not | cflows legal on time-course | port | | stub on the path |
| test_tui_figure.py, test_tui_integration.py, test_tui_run.py | figure and feed rendering | rendering of synthetic records | keep | | unchanged, pass |
| test_tui_state.py | test_a_refusal_caused_by_a_filter_does_not_claim_that_nothing_embeds, test_a_recipe_a_NARROWING_blocks_is_hollow_on_both_surfaces_too | MISSING vs CLEARED reasons | port | | swapped to `dpt` |
| test_tui_state.py | test_a_fact_no_step_needs_can_never_reach_a_ledger_row | `time` never reaches a ledger row | port | | inverted: the tool needs `time` now |
| test_tui_state.py | ledger counts (2 tests) | refused rows and capability count | port | | counts move by the two blocked recipes |

## Files that read the model or the probe

| File | Test function | What it checks | Verdict | Requirement | Note |
|---|---|---|---|---|---|
| test_commands.py | alias and command-table tests | command names | keep | | unchanged, pass |

## Files outside the planned list that the change also touched

| File | Test function | Verdict | Note |
|---|---|---|---|
| test_recipe_catalog.py | legality on bundled data; recipe runs [cflows, traced] | port | stub on the path, time labels; `out_dir` passed by name (it was passed as `data_ref`, so runs wrote into the working folder); `sandbox` named as blocked by design |
| test_narrate.py | test_offer_recommends_the_right_question | port | `pseudotime` heads the time-course menu on a plain install, `cflows` with the tool |
| test_analysis_selection.py | time-axis selection tests | port | the same rule for `_choose_recipe` |
| test_check.py | 2 coverage tests | port | the three MIOFlow recipes WARN with the tool phrase |
| test_front_door.py, test_tui_ledger.py, test_shell.py | menu, ledger and plain-prompter tests | port | the two blocked recipes move counts; the shell drives `cluster` |
| test_extension_contract.py | test_every_demanded_fact_can_be_supplied_by_something | port | tool output facts count; declared stubs exempt |
| test_extension_contract.py, test_tui_state.py | cleared-phrase coverage | keep | pass after `step_facts()` subtracts the frame facts |
| test_model_seam_legality.py | test_manifest_replaces_name_table | port | untooled mioflow needs `tool:mioflow` |
| test_analysis_readouts.py, test_experiment.py, test_gene_axis_fact.py | bundled legality sweeps | port | stub on the path, or re-pinned for a plain install |
| test_experiment.py | test_recipes_that_fall_back_are_not_over_pruned | retire (Time is declared, never invented) | replaced by `test_cflows_needs_a_real_time_axis_and_embed_needs_nothing` |
| test_stubs.py | test_the_sandbox_recipe_is_legal_on_every_shape | retire (MIOFlow readouts read the trajectories slot) | replaced: `sandbox` is refused only for `model` (and `time` off time-course) |
