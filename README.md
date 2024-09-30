# Kinship Verification with Custom Sampling and Hard Contrastive Loss

This repository contains the code and resources for our paper on facial kinship verification using contrastive learning techniques. Our method combines a custom batch sampling strategy with Hard Contrastive Loss (HCL) to enhance the discriminative power of learned facial features for kinship recognition.

## Abstract

Facial Kinship Verification, a challenging task in computer vision, faces significant challenges due to subtle interclass differences and high intraclass variations. This research introduces a novel approach that leverages supervised contrastive learning techniques with a focus on strategic sample selection. We propose a method that combines a custom batch sampling strategy with Hard Contrastive Loss (HCL) to enhance the discriminative power of learned facial features for kinship recognition.

## Key Features

- Custom batch sampling strategy for diverse and informative training batches
- Integration of Hard Contrastive Loss (HCL) to focus on challenging negative examples
- Competitive performance on the Families in the Wild (FIW) dataset
- Strong performance on complex tasks such as tri-subject verification and search and retrieval

## Results

Our method achieves:
- Average accuracy of 81.99% on the kinship verification task
- State-of-the-art performance on tri-subject verification with 86.0% average accuracy
- Competitive performance on search and retrieval tasks

## Methodology

1. Custom Batch Sampler: Creates diverse and balanced batches with unique family representations
2. Hard Contrastive Loss (HCL): Focuses on the most challenging negative samples
3. Fine-tuned face recognition model (AdaFace ResNet-101)

## Experiments and Ablation Studies

We conducted extensive experiments and ablation studies to evaluate the impact of:
- Sampling methods (sequential, random, custom)
- Hard Contrastive Loss
- Temperature parameter tuning

## Reproducing Results

To reproduce the results presented in our paper:

1. **Environment Setup**: Instructions for setting up the required environment will be provided upon publication.

2. **Data Preparation**: Details on how to obtain and preprocess the FIW dataset will be included.

3. **Model Training**: 
   - Scripts for training the model with different configurations will be made available.
   - Instructions on how to use the custom batch sampler and implement HCL will be provided.

4. **Evaluation**:
   - Scripts for evaluating the model on kinship verification, tri-subject verification, and search and retrieval tasks will be included.
   - Instructions on how to reproduce the results reported in the paper will be detailed.

5. **Pretrained Models**: 
   - Links to download pretrained models will be provided to allow for quick reproduction of our results.

6. **Ablation Studies**:
   - Scripts and instructions for running the ablation studies reported in the paper will be made available.

Detailed instructions, code, and additional resources necessary for reproducing our results will be added to this repository upon publication of our paper.

## Future Work

- Explore larger batch sizes and more sophisticated negative sampling strategies
- Investigate advanced feature transformation techniques
- Develop multi-task learning frameworks
- Address dataset challenges for imbalanced data and rare kinship relations

## Citation

If you find this work useful in your research, please consider citing:

```
[Citation information will be added upon publication]
```

## Contact

For any questions or concerns, please open an issue in this repository or contact the authors directly.
