# XRF-to-Optical Field-of-View Localization

This repository contains the compact reference implementation accompanying the
manuscript *XRF-to-Optical Field-of-View Localization with Vision Language
Models*. It is provided for inspection of the methods, not as a supported
software package or a standalone reproduction artifact.

## Scope

The repository includes the image preparation, prompting, localization,
baseline, proposal-and-verify, and statistical routines described in the
manuscript. It does not include microscopy data, prepared images, frozen model
responses, model weights, credentials, generated results, or live model access.
The reported results therefore cannot be reproduced from this repository alone.

The optical/XRF data are available from the corresponding authors upon
reasonable request.

## Code Map

| Manuscript component | Reference implementation |
| --- | --- |
| Optical and phosphorus-channel XRF preparation | `data.py`, `preprocessing.py` |
| Direct VLM prompts and four coordinate formats | `prompting.py`, `direct_localization.py` |
| Geometric controls | `controls.py` |
| Gradient + NCC template matching | `baselines/classical.py` |
| DINOv2 dense-feature baseline | `baselines/dinov2.py` |
| multiGradICON adaptation | `baselines/multigradicon.py` |
| Proposal generation and image-based verification | `proposal_verify/` |
| IoU, bootstrap intervals, sign-flip tests, and leave-one-pair-out selection | `evaluation.py` |
| Paper-specific settings and policy thresholds | `settings.py` |

The VLM code constructs provider-neutral, OpenAI-compatible message payloads
and parses supplied response text. Network transport, endpoint configuration,
credentials, retries, and provider-specific integration are intentionally
excluded.

## Environment

The default Pixi environment contains the dependencies required for the core
implementation and offline tests:

```bash
pixi install
pixi run test
```

The optional Linux baseline environment adds the packages used by DINOv2 and
multiGradICON. Its smoke checks do not download model weights:

```bash
pixi install -e baselines
pixi run -e baselines test-baseline-imports
```

The source modules expose ordinary Python functions for inspection and testing,
but no stable library API or command-line interface is promised.

## Citation

Please cite the accompanying manuscript when using this reference implementation.

## License

The code is released under the BSD 3-Clause License. See `LICENSE`.
