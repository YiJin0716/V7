# Checkpoints

Large model checkpoints are not included in Git or published as release assets. The code and the nine pre-rendered HTML examples are available independently of the weights.

| Local filename | Purpose |
|---|---|
| `checkpoints/vista_public.pt` | Public pretrained VISTA model for fresh warmup |
| `checkpoints/vista_nodule.pt` | Exported VISTA weights after organ/nodule warmup |
| `outputs/vista_warmup/complete.pt` | Completed VISTA warmup, including readiness metadata |
| `outputs/component_warmup/warmup.pt` | Completed DETR/refiner/semantic warmup |
| `checkpoints/v7_epoch18.pt` | Reference full joint model for inference and evaluation |

Obtain the public VISTA weights through the [official VISTA repository](https://github.com/Project-MONAI/VISTA) and its model download instructions. The model weights have their own terms; they are not redistributed here. V7's semantic CNN is trained from random initialization and does not need MedicalNet weights.

The reference full V7 checkpoint is run **13220013**, epoch **18**, with SHA-256:

```text
63b9693aea3cf79b44becc4f7757d6a7ccf874da4f35daae207331c1051d5f5a
```

It includes the networks and the fitted risk bank. `load_joint_model` restores it strictly and does not need the original public initialization files. Training warmup artifacts additionally require their adjacent `config.json` and matching data provenance. Only load trusted PyTorch checkpoints.
