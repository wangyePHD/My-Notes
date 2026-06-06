# OmniStyle English Interview Q&A Cheat Sheet

This sheet is for the two English rounds:

- Technical Interview: answer with algorithm + engineering detail.
- Research Lead Interview: answer with ownership, collaboration, tradeoffs, and product thinking.

## Phrases to Buy Time

Use these when you need a few seconds:

- Let me break it into two parts.
- Let me first give the high-level answer, then I can go deeper.
- If I understand correctly, you are asking about ...
- I have not tested that exact setting, but my intuition is ...
- That is a good point. The tradeoff is ...
- I may need a moment to organize the answer.

## Core Answers You Should Memorize

**What is the paper about?**

OmniStyle is a style transfer framework. The main idea is to treat style transfer as a data quality problem. We built a large paired dataset, filtered it with content, style, and aesthetic signals, and trained a DiT-based feed-forward model for both instruction-guided and image-guided stylization.

**What is the main contribution?**

There are three contributions: OmniStyle-1M, a million-scale paired style transfer dataset; OmniFilter, a task-specific filtering framework; and OmniStyle, a DiT-based model that supports both text instruction and style-reference image guidance.

**Why is it important?**

Existing methods are often slow, hard to control, or weak on fine-grained styles. Our method makes style transfer more controllable and scalable by improving the training data and using a feed-forward model.

**What is your personal contribution?**

I led the project end to end, especially the problem formulation, data construction, filtering design, model training, evaluation, and analysis. I also coordinated with collaborators on experiments and paper writing.  

Adjust this if your exact contribution was different.

## Technical Q&A

### 1. Why did you focus on data instead of only architecture?

Because style transfer quality depends heavily on paired examples. Architecture can help, but if the data is noisy, the model learns unstable mappings. We wanted the model to learn from examples that already preserve content, match style, and look visually appealing.

### 2. How did you build OmniStyle-1M?

We generated content images from 20 categories using FLUX. We selected 1,000 style images from Style30K. For each content-style pair, we used six existing style transfer models to produce stylized candidates. This gave us more than one million triplets.

### 3. Why use six source models?

Each source model has different strengths and failure modes. Using six models increases diversity and avoids depending on a single generator. But it also creates noisy data, so we need OmniFilter to select the best candidate.

### 4. How does OmniFilter work?

For each candidate, OmniFilter computes three scores: content preservation, style consistency, and aesthetic appeal. The final score is a weighted sum. We keep the candidate with the highest score for each content-style pair.

### 5. Why these three filtering dimensions?

They match the real objective of style transfer. A good result should preserve the content, capture the target style, and look good to users. Optimizing only one of them can cause failure, such as content distortion or weak stylization.

### 6. Why is style consistency weighted higher?

In our setting, style fidelity was the main failure mode. Content and aesthetics are also important, but without style consistency the task is not solved. The weight is a design choice and can be tuned for different product requirements.

### 7. How did you evaluate content preservation?

We used semantic and structural signals. CLIP helps measure semantic alignment with the content description. DINOv2 is better for structural similarity because it captures visual features and layout more robustly than text-image similarity alone.

### 8. How did you evaluate style consistency?

We used contrastive learning on style images. Images from the same style category are positive pairs, and images from different styles are negative pairs. This trains an encoder where similar styles are close in feature space.

### 9. How did you evaluate aesthetic appeal?

We used visual attributes such as composition, color harmony, lighting, contrast, and saturation. InternVL extracts image and attribute features, and an MLP predicts the aesthetic score. We train with AVA and fine-tune with BAID to reduce the domain gap for artistic images.

### 10. What are the limitations of automatic metrics?

Automatic metrics are scalable but imperfect. They can miss human preference, cultural style nuances, or product-specific quality requirements. I would combine them with human evaluation, artist annotation, and online feedback in production.

### 11. What is OmniStyle-150K?

It is the high-quality filtered subset selected from OmniStyle-1M. It contains 150K triplets across 1,000 style categories and is used to train the model.

### 12. What is the model architecture?

The model is based on FLUX-dev / MM-DiT. It uses VAE features for content and style images and concatenates image tokens, text tokens, and noisy latents before the diffusion transformer. It supports both instruction-guided and image-guided stylization.

### 13. Why use DiT / MM-DiT?

DiT is scalable and handles token-based conditioning naturally. MM-DiT is a good fit because we need to combine image information, text instruction, and diffusion latents in one generative model.

### 14. Why freeze most components and fine-tune only the diffusion transformer?

Freezing reduces training instability and memory cost. The pretrained VAE and text encoder already provide strong representations. Fine-tuning the transformer adapts the generative process to style transfer without changing every component.

### 15. How would you debug weak stylization?

I would check whether the style reference is encoded correctly, whether the style score is too low in the training data, and whether the conditioning signal is being ignored. I would also inspect attention behavior, compare with stronger style prompts, and adjust training data or loss weighting.

### 16. How would you debug content distortion?

I would check content preservation metrics, visualize failure cases, and compare the original and stylized layouts. Possible fixes include stronger content conditioning, better filtering, balancing the data, and adding evaluation cases where identity or structure is important.

### 17. How would you optimize inference?

I would profile first. Likely directions include fewer denoising steps, distillation, mixed precision, attention optimization, batching, VAE caching, and model quantization if quality remains acceptable.

### 18. How would you scale training?

I would use distributed data parallel or FSDP/ZeRO depending on model size, mixed precision, gradient checkpointing, efficient dataloading, and monitoring for GPU utilization. I would also cache expensive features when possible.

### 19. How would you design this as a production system for Canva?

I would separate offline and online parts. Offline: generate data, filter data, train models, evaluate safety and quality. Online: accept user input, retrieve or encode style references, run fast inference, apply safety checks, and collect feedback. I would measure latency, quality, controllability, and user satisfaction.

### 20. What is the biggest risk for production?

The biggest risks are inconsistent quality, latency, style misuse, and user trust. The model should provide controls such as style strength, preserve important content, and avoid unsafe or copyrighted style behavior.

## Research Lead Q&A

### 1. Tell me about yourself.

I work on generative vision models, especially controllable image generation and style transfer. My recent research focuses on making generative models more useful in practice by improving data quality, evaluation, and model controllability.

### 2. Why did you choose this project?

I noticed that many style transfer methods were limited by data and evaluation. The model architecture was not the only bottleneck. So I wanted to build a more complete research loop: generate data, filter it, train a model, and evaluate both quantitatively and qualitatively.

### 3. What was the hardest decision?

The hardest decision was how to filter generated data. If we are too strict, we lose diversity. If we are too loose, the model learns bad examples. We chose a weighted score and kept the best candidate for each pair to balance quality and coverage.

### 4. What did you learn from the project?

I learned that evaluation design is part of the method. For creative generation, metrics are not just reporting tools. They shape the data, the training signal, and the final user experience.

### 5. What would you improve next?

I would add stronger human preference data, professional artist annotations, better style-intensity control, and model distillation for faster deployment. I would also evaluate more product-specific use cases.

### 6. How do you work with collaborators?

I try to make the research plan explicit: what problem we are solving, what experiments are needed, and what success means. I also prefer frequent check-ins on failure cases, because visual generation problems are easier to discuss with concrete examples.

### 7. How do you handle disagreement?

I first clarify the goal and evidence. If the disagreement is about model design, I try to turn it into an experiment. If it is about paper direction or product direction, I compare the options by impact, cost, and risk.

### 8. Why Canva?

Canva is a strong fit because it is a design platform, not just an image generation demo. The research problems I care about, such as controllability, evaluation, latency, and user trust, are directly connected to real creative workflows.

### 9. What kind of researcher are you?

I like research that connects algorithms with systems. I care about model quality, but I also care about data pipelines, evaluation, and deployment constraints. I think this is important for research that needs to become useful in products.

### 10. How will you handle English communication?

I will speak clearly and structure my answers. If I do not understand a question, I will restate it and ask for confirmation. I am comfortable discussing technical details in English, but I may speak a little slower to be precise.

## Questions You Can Ask Them

- What are the most important research problems for the Canva AI team this year?
- How do you evaluate creative generation quality in production?
- How do research scientists collaborate with product and engineering teams?
- What is the balance between publishing research and building product features?
- For this role, what would success look like in the first six months?

