Bidirectional Causal Audio-Visual Editing.
Given source video frames V, source audio A, a user instruction I, and optionally a visual mask, audio span, object click, or reference sound, generate edited video \hat V and edited audio \hat A such that:

1. the target edit is satisfied;
2. unrelated visual regions are preserved;
3. unrelated audio sources are preserved;
4. visible sound sources and audible events remain temporally and semantically aligned.

Given a video with audio, the user can edit either a visible object/part or an audible sound source, and the system updates the other modality only when needed to maintain causal audio-visual consistency.

Previous methods insert visual objects or generate Foley audio, but they do not treat the inserted character and its sound as a single reference-conditioned causal audio-visual source. Our method jointly preserves visual identity, acoustic identity, motion-event synchronization, spatial audio consistency, and original scene/audio preservation.

Reference-guided audio-visual source editing: insertion, removal, and replacement are all operations on one causal AV source layer: a visible character/object, its sound stem, its motion, its events, and its visual/audio side effects


https://openreview.net/forum?id=NCmgCSJTLm

https://arxiv.org/pdf/2603.18524

https://arxiv.org/pdf/2603.19224

你可以先看看 https://arxiv.org/pdf/2603.19224 这个没声音 你可以看看能不能配一点声音

https://javisverse.github.io/JavisDiT-page/

baseline 可能需搭一个 video audio 的
我之前跑过 qwen tts 挺不错的

怎么标出 video object 和 语音是关键


3. Audio-Visual Source Separation — the deeper version where you actually link a sound to the visual object making it. This is an active research area:

Meta's "Audio-Visual Segmentation" models can highlight the pixel region in a video frame that corresponds to each sound.
"Looking to Listen" (Google) — separates speech of individual speakers using their face as a guide.
"Sound of Pixels" (MIT) — learns to associate visual objects with their sounds and can separate audio per-object.
PixelPlayer and Music Gesture — similar research projects from MIT CSAIL.


This is a two-part problem: isolating a specific sound source and then removing it while preserving the ambient background. Here's how to approach it:
Step 1 — Separate the Target Sound
Since you've already labeled which dog is barking, you need to separate that dog's audio from everything else.
Best tools for this:

Meta's Demucs / HTDemucs — originally for music separation, but newer versions handle general audio. It can separate stems, and you can fine-tune it for your use case.
AudioSep (from Microsoft Research) — this is probably the closest to what you need. You give it a text prompt like "dog barking" or even a visual reference, and it separates that sound from the mix.
Bandit — another general-purpose sound separation model.

Step 2 — Remove It Cleanly
Once you've isolated the target dog's barking as a separate audio track, you subtract it from the original mix. But naive subtraction often leaves artifacts. Better approaches:
Option A — Spectral Subtraction

Convert both the original mix and the isolated bark to spectrograms.
Subtract the bark's spectrogram from the mix.
Reconstruct the audio using the original phase.
This preserves background texture reasonably well.

https://arxiv.org/pdf/2601.22143