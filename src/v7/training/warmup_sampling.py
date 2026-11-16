"""Complete nodule visits, with the original priority and background sampling."""
POLICY = 'all_retained_nodules_plus_priority_paired'


def visit_plan(metadata, rng, background_ratio=1.):
    """Visit each retained nodule once, then add one priority-selected pair."""
    ids = [n['nodule_id'] for n in metadata['nodules'] if n['retained']]
    if len(ids) != len(set(ids)):
        raise ValueError('Retained nodule identifiers must be unique within a CT')
    if background_ratio not in (.5, 1., 2.):
        raise ValueError('Background/positive crop ratio must be 0.5, 1 or 2')
    plan = []
    for index in rng.permutation(len(ids)):
        plan.extend(((ids[int(index)], True), (None, False)))
    # None + True invokes the existing nearby/label/size priority sampler.
    # Empty CTs retain the original context pair and genuine packed labels.
    plan.extend(((None, True), (None, False)))
    if background_ratio != 1.:
        positives = [entry for entry in plan if entry[1]]
        # Keep every explicit nodule visit and at least one background crop
        # per CT, including CTs with no retained nodules.
        count = max(1, int(round(len(positives)*background_ratio)))
        plan = positives + [(None, False)]*count
        plan = [plan[int(i)] for i in rng.permutation(len(plan))]
    return plan


def coverage(expected, case_visits, selections):
    """Count explicit selections, not inferred per-nodule supervision."""
    if any(len(ids) != len(set(ids)) for ids in expected.values()):
        raise ValueError('Retained nodule identifiers must be unique within a CT')
    unexpected = set(case_visits)-set(expected) | set(selections)-set(expected)
    if unexpected:
        raise ValueError('Nodule coverage contains cases outside the training cohort')
    for name, counts in selections.items():
        if set(counts)-{str(i) for i in expected[name]}:
            raise ValueError('Nodule coverage contains an unknown nodule')
    if any(n < 1 for n in case_visits.values()) or any(
            n < 1 for counts in selections.values() for n in counts.values()):
        raise ValueError('Nodule coverage counts must be positive')
    missing_cases = [name for name in expected if not case_visits.get(name)]
    missing_nodules = [f'{name}/{i}' for name, ids in expected.items()
                       for i in ids if not selections.get(name, {}).get(str(i))]
    total = sum(map(len, expected.values()))
    return dict(passed=not missing_cases and not missing_nodules,
                visited_cases=len(expected)-len(missing_cases), total_cases=len(expected),
                selected_nodules=total-len(missing_nodules), total_nodules=total,
                missing_cases=len(missing_cases), missing_nodules=len(missing_nodules))


def require_coverage(saved):
    config = saved['config']
    if config.get('nodule_sampling_policy') != POLICY:
        raise ValueError('Warmup lacks complete nodule sampling')
    expected = config.get('retained_nodule_ids')
    if not expected or set(expected) != set(config['anatomy_case_ids']):
        raise ValueError('Warmup nodule coverage cohort differs')
    if saved.get('nodule_coverage_epoch') != saved['epoch']:
        raise ValueError('Nodule coverage must describe the saved epoch')
    result = coverage(expected, saved.get('coverage_case_visits', {}),
                      saved.get('nodule_selection_counts', {}))
    if not result['passed'] or result != saved.get('nodule_coverage'):
        raise ValueError('Warmup has not visited every training CT and retained nodule')
