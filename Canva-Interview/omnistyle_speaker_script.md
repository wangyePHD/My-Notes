
# Polished Interview Speaker Script

## Opening

Hi Hojin, nice to meet you. Thank you for taking the valuable time today.

I prepared presentation slides to introduce myself and my research work. 

May I now share my screen to walk through the slides? 

---

## Slide 1: About Me

I’ll keep this part brief.

I am Ye Wang, a Ph.D. candidate at Jilin University. My research focuses on style transfer, image editing, and controllable visual generation.

I expect to graduate in December 2026.

<!-- Today, I will mainly talk about OmniStyle, because it is a representative project that shows how I think about research: from problem definition, to data construction, to model training and evaluation. -->

---

## Slide 2: About Me

This slide summarizes the two parts of my background that are most relevant.

On the research side, I have 7 publications, including papers at CVPR 2025 and AAAI 2025. I am also currently working on ECCV and NeurIPS submissions.

On the industry side, I have had research experience at Tencent, Alibaba, Baidu, JD.com, and Kunlun.

One experience I value a lot is Tencent’s Qingyun Talent Program. It gave me the chance to work on research problems under more practical industry constraints. Tencent Qingyun Talent Program is Tencent’s highly selective track for outstanding student researchers and engineers.

So overall, my background is a mix of academic research and industry experience.

---

## Slide 3: Research Overview

This slide gives a quick overview of my research portfolio.

My projects cover style transfer, vector graphics, font generation, and multimodal post-training.

For today’s interview, I will focus on OmniStyle.


---

## Slide 4: OmniStyle Cover

Now I will move to the technical part.

OmniStyle is our CVPR 2025 work on controllable style transfer.

I will use it as a concrete example to explain the full research process: what problem we wanted to solve, how we built the dataset, how we filtered the data, and how we trained the final model.

---

## Slide 5: Motivation

The motivation comes from three limitations we observed in previous style transfer methods.

First, generalization is still limited. Many methods work well for some styles, but they do not handle truly diverse and fine-grained styles very reliably.

Second, controllability is not strong enough. During stylization, the model may change the identity, structure, or layout of the original image.

Third, efficiency is also a problem. Some methods rely on optimization or inversion, which makes them expensive and less suitable for real applications.

So the question we asked was simple:

Can we build a controllable and scalable style transfer model by learning from better paired data?

---

## Slide 6: Our Solution

Our solution is a data-centric supervised learning pipeline.

It has three main stages.

First, we build a large-scale paired dataset.

Second, we filter the generated data with task-specific quality signals.

Third, we train a unified feed-forward model on the filtered data.

The key idea is that we do not treat style transfer only as an inference-time technique.

Instead, we treat it as a data construction and learning problem.

In the next few slides, I will go through these three stages one by one.

---

## Slide 7: Step 1 — OmniStyle-1M Overview

The first part is the dataset.

We build OmniStyle-1M, which contains more than one million triplets.

Each triplet includes a content image, a style reference, and a stylized result.

The scale is important, but diversity is equally important.

The dataset covers 1,000 style categories and 20 content categories.

So the goal of this stage is to convert style transfer into a paired supervised learning problem.

---

## Slide 8: Step 1 — How We Built OmniStyle-1M

This slide shows how we constructed the triplets.

We first generated content images with FLUX across 20 content categories.

Then we collected style reference images from Style30K, covering 1,000 style categories.

For each content-style pair, we used six existing style transfer models to generate multiple candidate stylized images.

This gives us diverse results.

But it also introduces noise, because not every generated candidate is good.

Some outputs preserve the content but do not match the style. Some match the style but damage the content. Some looks low-quality.

That is why the next stage, OmniFilter, is very important.

---

## Slide 9: Step 2 — OmniFilter Overview

OmniFilter is the quality control stage.

For each content-style pair, we have six candidate outputs.

Instead of using all of them, we score each candidate and keep the best one.

The scoring considers three dimensions:

content preservation, style consistency, and aesthetic quality.

In other words, OmniFilter tries to answer three questions:

Does the result preserve the original content?

Does it match the reference style?

And does it look visually good?

The purpose is to turn noisy synthetic data into cleaner training data.

---

## Slide 10: Step 2A — Content Preservation

The first score is content preservation.

Here, we want the stylized image to keep the main subject, semantics, and spatial structure of the content image.

We use two signals.

For semantic consistency, we compute CLIP similarity between the stylized result and the content caption.

For structural consistency, we compare the content image and the stylized image using DINOv2 features.

Then we combine these two scores with equal weight.

This way, the selected sample needs to preserve both the semantic and the structure of the original image.

---

## Slide 11: Step 2B — Style Consistency

The second score is style consistency.

This part is more challenging because style is often subjective.

Simple style loss are not enough to capture style similarity.

So we use Style30K as supervision.

Images from the same style category are treated as positive pairs, and images from different style categories are treated as negative pairs.

Then we fine-tune a CLIP image encoder with contrastive learning.

After that, images with similar styles become closer in the embedding space.

So we can measure the similarity between the reference style image and the stylized output as the style consistency score.

---

## Slide 12: Step 2C — Aesthetic Appeal

The third score is aesthetic quality.

Even if an image preserves the original content and matches the target style, it may still look visually weak.

So we add a dedicated aesthetic scorer.

This scorer considers 40 visual attributes, such as composition, balance, color harmony, lighting, contrast, and saturation.

We use InternVL2 to generate attribute-level descriptions, then combine multimodal features and use an MLP to predict the final aesthetic score.

The scorer is first trained on AVA dataset and then adapted to artistic images with BAID dataset.

This helps the filter prefer results that are not only correct, but also visually pleasing.

---

## Slide 13: Step 3 — Train the Model

After filtering the data, we train the final model, OmniStyle.

OmniStyle is a DiT-based feed-forward model built on FLUX-dev and MM-DiT.

It supports both text-guided and image-guided stylization.

For the inputs, content and style images are encoded by VAE modules. Image tokens and text tokens are then combined with noisy latents and processed by the diffusion transformer.

During training, we fine-tune only the diffusion transformer, while keeping the other modules frozen.

This makes training more stable and focuses the adaptation on the generative backbone.

---

## Slide 14: Quantitative Evaluation

This slide shows the quantitative results.

The left side shows the instruction-guided setting.

The right side shows the image-guided setting.

OmniStyle achieves strong style consistency while still maintaining competitive content preservation and aesthetic quality.


<!-- 
Across both settings, the results suggest that high-quality filtered paired data can improve controllability without sacrificing overall image quality. -->

---

## Slide 15: Qualitative Evaluation

This slide shows instruction-guided qualitative results.

What I want to highlight here is not just stronger texture or more obvious stylization.

The more important point is balance.

OmniStyle can follow fine-grained style instructions while keeping the identity, layout, and main content more stable.

This is important for real creative tools.

Users usually do not want the model to completely change the original scene. They want controlled changes that still respect the input image.

---

## Slide 16: Qualitative Evaluation

This slide shows image-guided qualitative results.

In this setting, the model takes a style reference image as input.

We can see that OmniStyle transfers color, texture, and composition cues from the reference image more stable.

<!-- Again, the goal is not simply to maximize style strength.

The goal is controlled transfer: the output should reflect the style reference, but still preserve the structure of the original content image. -->
<!-- 
These examples support the same conclusion as the quantitative results: better filtered data leads to a better trade-off between style transfer and content preservation. -->

---

## Slide 17: Summary

To summarize, the main story of OmniStyle is quite simple.

We first build a large paired dataset.

Then we filter it using task-specific quality signals.

Finally, we train a unified stylization model on the filtered pairs.

In this way, OmniStyle turns style transfer into a scalable supervised learning problem.

---

## Slide 18: Backup — Limitations and Next Steps

This is a backup slide. I would use it if we discuss limitations or future work.

There are several directions I would like to improve.

First, synthetic data is useful, but it can still inherit artifacts or biases from the source generators.

Second, automatic metrics are scalable, but creative quality is ultimately related to human preference.

So in future work, I would like to add stronger human-aligned evaluation, professional artist annotation, and preference learning.

For real deployment, I would also consider safety filtering, latency profiling, style-strength control, and online feedback loops.

These are important if we want the system to be useful not only in research, but also in real creative workflows.

---

# Two-Minute Version

OmniStyle is our CVPR 2025 work on controllable style transfer.

The key idea is to treat style transfer as a data and quality problem.

We first build OmniStyle-1M, a million-scale triplet dataset. Each triplet contains a content image, a style reference, and a stylized result. The dataset covers 1,000 style categories and 20 content categories.

Because the generated triplets are noisy, we propose OmniFilter. It scores each candidate from three perspectives: content preservation, style consistency, and aesthetic quality. Then we keep the best candidate for training.

Finally, we train a DiT-based feed-forward model that supports both instruction-guided and image-guided stylization.

The main takeaway is that better data curation and filtering can improve both controllability and visual quality.

From a product perspective, I think this is important because creative tools need to be predictable, efficient, and easy for users to control.

---