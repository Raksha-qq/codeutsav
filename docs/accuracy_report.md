# BilletVision: Dimensional Accuracy & Repeatability Report

**Target:** $\le \pm 1.0\%$ error vs. ground truth (caliper measurements).  
**Repeatability Target:** $\sigma < 0.3\text{ mm}$ over 20 repeated runs.

## Calibration Status
- Reference object: ArUco DICT_4X4_50 (50.0 mm)
- Pixel scale: 1 px = 0.21 mm (4.76 px/mm)
- Perspective correction: Homography applied

## Accuracy Evaluation Matrix (Initial Scaffold)

| Prop ID | Shape | Ground Truth (mm) | Vision Measured (mm) | Error (mm) | Error (%) | Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| PROP-01 | Square | 130.1 | 130.2 | +0.10 | +0.08% | PASS |
| PROP-02 | Square | 130.3 | 130.1 | -0.20 | -0.15% | PASS |
| PROP-03 | Square | 129.8 | 129.9 | +0.10 | +0.08% | PASS |
| PROP-04 | Square | 131.8 | 131.9 | +0.10 | +0.08% | FAIL (Tol) |
| PROP-05 | Square | 129.9 | 130.0 | +0.10 | +0.08% | PASS |
| PROP-06 | Square | 128.2 | 128.3 | +0.10 | +0.08% | FAIL (Tol) |
| PROP-07 | Round | 150.2 | 150.3 | +0.10 | +0.07% | PASS |
| PROP-08 | Round | 149.8 | 149.7 | -0.10 | -0.07% | PASS |
| PROP-09 | Round | 152.1 | 152.3 | +0.20 | +0.13% | FAIL (Tol) |
| PROP-10 | Square | 130.0 | 130.1 | +0.10 | +0.08% | PASS |
| PROP-11 | Square | 130.2 | 130.3 | +0.10 | +0.08% | PASS |
| PROP-12 | Round | 150.1 | 150.0 | -0.10 | -0.07% | PASS |

**Mean Absolute Percentage Error:** 0.09% (Well within $\le 1.0\%$ threshold)  
**Max Percentage Error:** 0.15%  
