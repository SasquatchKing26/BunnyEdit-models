# BunnyEdit models

AI model files that [BunnyEdit](https://github.com/SasquatchKing26) downloads once, the first time a tool needs
them. Nothing else lives here: no app code, no photos. The files are converted to ONNX by
`.github/workflows/models.yml` from each model's original public release, and published as a release (`models-N`).

| File | What it does | From | License |
|---|---|---|---|
| `general_x4v3_192.onnx`, `general_wdn_x4v3_192.onnx` | Sharpen, unblur and enlarge (fast) | [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) realesr-general-x4v3 | BSD-3-Clause |
| `esrgan_x4_128_fp16.onnx` | Sharpen, unblur and enlarge (best detail, slower) | Real-ESRGAN x4plus | BSD-3-Clause |
| `scunet_256_fp16.onnx` | Noise and grain removal for real photos | [SCUNet](https://github.com/cszn/SCUNet) (real, PSNR) | Apache-2.0 |
| `gfpgan14_512_fp16.onnx` | Face restoration | [GFPGAN](https://github.com/TencentARC/GFPGAN) v1.4 | Apache-2.0 (model: see GFPGAN) |
| `zerodce_256.onnx` | Low-light brightening (curve estimation) | [Zero-DCE++](https://github.com/Li-Chongyi/Zero-DCE_extension) | see the authors' repository |

`tools/archs` holds the model architectures from those projects, unchanged except for imports.
