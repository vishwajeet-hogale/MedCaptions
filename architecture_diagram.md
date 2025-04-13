# Updated Architecture Diagram

## Fig. 1: DeiT-BERT Medical Image Captioning Framework

The architecture diagram should be updated to show both decoder approaches:

```
┌─────────────┐                                     ┌───────────────────┐
│   Medical   │                                     │ Ground Truth      │
│   MRI Image │                                     │ Caption           │
└──────┬──────┘                                     └─────────┬─────────┘
       │                                                      │
       ▼                                                      ▼
┌──────────────┐                                     ┌───────────────────┐
│ DEiT-Small   │                                     │ MediCareBERT      │
│ Vision       │                                     │ Text Encoder      │
│ Transformer  │                                     │                   │
└──────┬───────┘                                     └─────────┬─────────┘
       │                                                      │
       │ Image Features                                       │ Caption Embedding
       │ (384-dim)                                            │ (768-dim)
       │                                                      │
       └─────────────┐                           ┌─────────────┘
                     │                           │
                     ▼                           ▼
          ┌──────────────────────┐      ┌─────────────────┐
          │  Decoder Selection   │◄─────┤   Training      │
          └──────────┬───────────┘      │   Objective     │
                     │                  └─────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
┌───────────────────┐    ┌───────────────────────┐
│ LSTM Decoder      │    │ Transformer Decoder   │
│ - 2 Layer LSTM    │    │ - Biomedical          │
│ - Hidden init     │    │   Transformer base    │
│   with image      │    │ - Image projection    │
│   features        │    │ - Cross-modal fusion  │
│ - Hybrid loss     │    │ - Contrastive loss    │
└────────┬──────────┘    └──────────┬────────────┘
         │                          │
         ▼                          ▼
┌───────────────────┐    ┌───────────────────────┐
│ Sequential        │    │ Embedding-based       │
│ Caption           │    │ Caption Retrieval     │
│ Generation        │    │ via Similarity        │
└───────────────────┘    └───────────────────────┘
```

## Key Differences Between Approaches

The diagram highlights the two parallel decoding paths:

1. **LSTM Path (Left Side):**
   - Takes image features to initialize hidden states
   - Processes tokens sequentially (traditional NLP approach)
   - Uses hybrid cosine-MSE loss for training
   - Outputs caption as generated token sequence

2. **Transformer Path (Right Side):**
   - Projects image features to transformer embedding space
   - Processes input in parallel (transformer approach)
   - Uses contrastive learning with InfoNCE loss for training
   - Outputs caption embedding for similarity-based retrieval

The diagram shows how both approaches use the same encoder components (DEiT for images, MediCareBERT for text) but differ in their decoding strategy and training objectives.

## Implementation Notes

The diagram should emphasize that:
1. Both approaches can be used interchangeably at inference time
2. They offer complementary strengths (generation vs. retrieval)
3. The shared encoder architecture enables fair comparison
4. The approach represents a novel combination of vision transformers, biomedical language models, and dual decoding strategies 