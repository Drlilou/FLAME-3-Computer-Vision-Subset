# Multimodal Wildfire Detection System (FLAME 3)

A deep learning framework for real-time aerial wildfire detection using synchronized **RGB visual imagery** and **radiometric thermal maps (°C)** captured by UAVs. 

This repository implements a **Bidirectional Cross-Attention Transformer Network** in PyTorch that fuses high-resolution spatial details with exact physical thermal radiation data to maximize detection accuracy and reduce false positives.

---

## System Architecture Overview

```text
                      ┌──────────────────────┐
                      │  RGB Image (3x224x224)│
                      └──────────┬───────────┘
                                 │
                        ResNet18 Backbone
                                 │
                         RGB Spatial Tokens
                        [Batch, 49, 512]
                                 │
 ┌───────────────────────────────┴───────────────────────────────┐
 │        Bidirectional Cross-Attention Transformer              │
 │  • RGB tokens query Thermal tokens (verify visual glare)     │
 │  • Thermal tokens query RGB tokens (verify smoke coverage)   │
 └───────────────────────────────┬───────────────────────────────┘
                                 │
                      Thermal Spatial Tokens
                        [Batch, 49, 512]
                                 │
                        ResNet18 Backbone (1-Ch)
                                 │
                     ┌───────────┴────────────┐
                     │ Thermal Map (1x224x224)│
                     └────────────────────────┘
                                 │
                         Global Avg Pool
                                 │
                          MLP Classifier
                                 │
                       Fire Prediction Logit
