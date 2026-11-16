# Evaluation and HTML export

```bash
python scripts/evaluate.py --checkpoint checkpoints/v7_epoch18.pt \
  --manifest data/fold_0.json --output outputs/evaluation
python scripts/plot_results.py --output outputs/evaluation
```

Evaluation rejects overlap with patients used to fit the risk bank. It does not update that bank. Official boxes, masks and ratings are not provided to the whole-CT prediction forward pass. Separate annotation-level analyses deliberately use official masks or features to evaluate individual components.

For parallel evaluation, launch independent workers with distinct `--shard` indices and the same `--shards` count. Run plotting only after every shard finishes. The report checks that the entire expected cohort is present.

Output folders:

| Folder | Contents |
|---|---|
| `annotation_lvl/` | Official reader segmentation → semantic features; official features → reader malignancy |
| `nodule_lvl/` | Segmentation overlap, semantic threshold AUC, and nodule malignancy ROC |
| `volume_lvl/` | FROC, excluded predictions, and scan malignancy ROC |
| `volume_lvl/exclusion_metrics/malignancy_vs_nodule_size/` | Size filtering and performance among small nodules |
| `_data/` | Per-case records and metric tables used to generate the plots |

At the default operating point, object probability is at least 0.5 and the hard mask uses `sigmoid(logits / temperature) ≥ 0.5`. Predictions and physical nodules are matched one-to-one, requiring mask IoU ≥ 0.1 with at least one reader. Unmatched predictions count as false positives; unmatched official nodules count as misses. FROC sweeps the object score, not the mask threshold.

Malignancy is evaluated using reader scores with three label rules: exclude mean rating 3, rating ≥ 3 positive, and rating > 3 positive. Nodule reports distinguish individual readers, closest reader and reader mean. The all-nodule ROC includes missed nodules with a zero prediction score; detected-only ROC uses matched nodules. These are different evaluation populations.

Semantic threshold AUC uses an official rating ≥ t as positive. AUC is undefined when one class is absent, including t = 1. All labels are reader-derived, not biopsy-confirmed diagnoses.

For one CT:

```bash
python scripts/infer.py --checkpoint checkpoints/v7_epoch18.pt \
  --manifest data/fold_0.json --case LIDC-IDRI-0259__scan259 \
  --output outputs/example
python scripts/export_html.py --result-dir outputs/example --output outputs/example.html
```

`infer.py` saves numerical outputs and masks, then attaches the selected case's official annotations. HTML export also needs the original CT and reader mask paths to display individual reader masks. Its first run downloads the NiiVue JavaScript bundle used by roivue; the resulting HTML is self-contained. The `--scan-step 2` default reduces only the whole-CT preview resolution. Nodule regions remain at the saved prediction resolution.

The provided HTML examples and their nodule selection are documented in [missing_malignant](../missing_malignant/README.md).
