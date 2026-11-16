"""Reuse per-case anatomy labels while strictly selecting the new training subset."""
import hashlib
import json
from pathlib import Path
from v7.data.split import case_id, validate_split


def select_training_cases(manifest, plan):
    path = Path(manifest)
    payload = json.loads(path.read_text())
    if payload.get('tuning_split'):
        original_path = Path(payload['tuning_split']['original_manifest'])
        if hashlib.sha256(original_path.read_bytes()).hexdigest() != payload['tuning_split']['original_manifest_sha256']:
            raise ValueError('Original split manifest changed')
        validate_split(payload, json.loads(original_path.read_text()))
    elif hashlib.sha256(path.read_bytes()).hexdigest() != plan['manifest_sha256']:
        raise ValueError('Warmup manifest/labels mismatch')
    allowed = {case_id(c): c for c in plan['cases'] if c['split'] == 'training'}
    selected = []
    for row in payload['training']:
        name = case_id(row)
        if name not in allowed or allowed[name]['image'] != row['image']:
            raise ValueError(f'Case outside original anatomy training cohort: {name}')
        selected.append(allowed[name])
    if not selected or len({case_id(c) for c in selected}) != len(selected):
        raise ValueError('Empty or duplicate anatomy training selection')
    return selected
