---
title: "VIOLIN – A large Scale Dataset for video and language inference"
categories:
  - Multimodal-learning
tags:
  - Dataset
---

## Introduction 
Natural language inference (NLI) is the task judging entailment and contradiction relations between premise and hypothesis sentences. e.g. positive/negative between two sentences

Visual Entailment (VE) is similar to NLI, but the difference is the premise is image. Thus, the goal of this task is to judge whether the textual hypothesis can be confirmed based on the visual content in the image.

The task proposed in this paper is to judge the textual hypothesis based on the combination of video and its subtitles. Therefore, this is more advanced multimodal task than VE.

In this paper, a new large scale of dataset for this task is proposed and testing results of multimodal architecture utilizing existing models are given.

## Why our proposed task is challenging? 
Deep understanding of videos is difficult due to following reasons:
  1. videos contain the complex temporal dynamics and relationship between different visual scenes 
  2. our dataset is collected from TV shows and movie clips, which contain rich social interactions and diverse scenes. This requires a model to not only understand explicit visual cues, but also infer in-depth rationale behind the scenes 
  3. VE task which dataset is less sophisticated sentence as it only contains factual description that can be explicitly derived from the visual content in the image. On the other hand, VIOLIN mainly consists of implicit statements that cannot be solved without in-depth understanding of the video and text.

Therefore, VIOLIN is designed specifically to evaluate a model’s multimodal reasoning skills.

## Overview of Dataset
f(V,S,H) -> {0,1} where V is video clip consisting of a sequence of video frames $$ { \{ v_{i} \} }^{T}_{i=1} $$, paired with its aligned text $$ S = {\{ s_i, t^{(0)}_i, t^{(1)}_i \} }^n_{i=1} $$ ($$ s_i $$ is the subtitle within time span $$ (t^{(0)}_i \rightarrow t^{(1)}_i) $$ in the video) and a natural language statement $ H $ as the hypothesis aiming to describe the video clip.

For every (V,S,H) triplet, a system needs to perform binary classification: f(V,S,H) -> {0,1}, deciding whether the statement $ H $ is entailed (label 1) from or contradicts (label 0) the given video clip.

Diverse sources are used (4 popular TV shows of different genres and YouTube movie Clips from thousands of movies) so that the dataset has high coverage and versatility. 

The below figure shows the comparison between VIOLIN and other existing vision-and-language datasets. VIOLIN is the first dataset that provides both video and subtitles to accomplish NLI. Therefore, the solution model needs to encode all three different types of data: video, subtitles and statement, and then, it performs the binary classification.

<img src="/assets/imgs/J_Lin(2020)/1.png" alt="1.jpg">

VIOLIN is composed of various types of statement as shown in the below diagram. While visual recognition, identifying character and action recognition are more focused on explicit information, human dynamics, conversation reasoning and inferring reasons are requiring an additional implicit information. Human dynamics includes inferring human emotions/relations/intentions. Conversation reasoning is that performing inference over characters’ dialogues and other forms of interactions (body language, hand gestures, etc). inferring reason is about inferring causal relations in complex events.

Thus, the former types of statement require relatively low-level reasoning, whereas the latter types of statement require in-depth understanding and commonsense reasoning. Overall, 54% of the dataset is explicit information recognition and common-sense reasoning takes up the remaining 46%.

One of the emphasized points of VIOLIN is that it is more focused on reasoning rather than surface-level grounding; In TVQA, only 8.5% of the questions require reasoning.

<img src="/assets/imgs/J_Lin(2020)/2.png" alt="2.jpg">

## Model Architecture
The model composed of two encoders: video encoder and text encoder. Video features, statement features and subtitles features are extracted by sing these encoders, and then, fusion modules combine these cross modal features. Finally, the final bi-LSTM layer and fc layer are located. 1-dim output from the fc layer input to the sigmoid activation function so that the probability of the input statement being positive is computed.

<img src="/assets/imgs/J_Lin(2020)/3.png" alt="3.jpg">

**Preprocess of video data:**
The videos are down-sampled to 3 fps for image-level feature extraction, while C3D features are extracted for every 16 frames from the original video without down-sampling. 

The video encoder firstly extracts the visual features $$ V \in \mathbb{R}^{^{T \times d_v}} $$, where $$T$$ is # time steps, and $$d_v$$ is the dim of feature vector by 3 approaches:
  1. image feature using ResNet101 trained on ImageNet to extract the global image feature for each frame 
        * 2048-dim feature vector from the last avg pool layer.
  2. detection feature using Faster R-CNN trained on Visual Genome to extract the detected objects’ regional features for each frame
        * 2048-dim feature vector. After ROI Pooling layer, the feature vector is (512x7x7,4096), but the exact method to extract 2048-dim is not mentioned in the paper. 
  3. video feature using C3D (3D conv net) to extract spatial-temporal video feature for each small clip of video
        * 4096-dim feature vector

Then, bi-directional LSTM captures the temporal correlation among consecutive frames and extracts video representation $$ H_V \in \mathbb{R}^{T \times 2d}$$, where d is dim of LSTM encoder hidden-state which is built by concatenating the last hidden state of LSTM from both directions.

<img src="/assets/imgs/J_Lin(2020)/4.png" alt="4.jpg">

**Preprocess of text data:**
Both statement and subtitle are tokenized into a word sequence $$ {\{ w_i\} }^{n_{stmt}}_{i=1} $$ and $$ {\{ u_i\} }^{n_{subtt}}_{i=1} $$ , where $$n_{stmt}$$ and $$n_{subtt}$$ are lengths of statement and subtitle, respectively. All the lines are tokenized and concatenated into one single sequence. 

The same text encoder is used to extract the feature from statement and subtitles. LSTM encoder, which does not consider contextual info, and BERT encoder are experimented. LSTM encoder (i.e. GloVe), converts word tokens to their embeddings, and then produces text representations $$ H_{stmt} \in \mathbb{R}^{n_{stmt} \times 2d} $$ and $$ H_{subtt} \in \mathbb{R}^{n_{subtt} \times 2d} $$. BERT encoder is initially finetuned on VIOLIN. Its output at each position is 768-dim so it is projected to 2d dimensions. However, the method of projection is not stated by the author.

## Combining Multimodality Streams
Statement representations are jointly modeled with video and subtitles via a shared fusion module, which is implemented with bidirectional attention following the mechanism of query-context matching. For example, statement representations $$ H_{stmt} \in \mathbb{R}^{^{n_{stml}} \times 2d} $$ are used as context, and video representations $$ H_V \in \mathbb{R}^{T \times 2d} $$ as query since the full statement needs to be supported by evidence from either video or subtitles. Thus, each word in the statement attends to every time step in the video representations. This attention weights are mathematcially expressed as:

Let $$ a_i \in \mathbb{R}^T $$ be attention weights for the i-th word in the statement, $$ \sum^{T}_{j=1}a_{i,j}=1 \text{ for all } i=1,…,n_{stmt}, a \in \mathbb{R}^{n_{stmt} \times T} $$

Using this video-statement attention weights, a video-aware statement representation can be computed: $$ M^V_{stmt}=aH_V \in \mathbb{R}^{n_{stmt} \times 2d} $$. Likewise, subtitle-aware statement representation $$ M^{subtt}_{stmt} $$ can be copmuted. Finally, these two sets of representations are fused to result matrix $$ M^{all}_{stmt} \in \mathbb{R}^{n_{stmt} \times 10d} $$, and then this matrix is fed into the final bidirectional LSTM:

$$ M^{all}_{stmt} = [H_{stmt}; M^V_{stmt}; M^{subtt}_{stmt}; H_{stmt} \odot M^V_{stmt}; H_{stmt} \odot M^{subtt}_{stmt} ] $$

In addition, the author evaluates the pre-trained model LXMERT that jointly learns multimodal features. In this model, the visual input is input rather than video. the middle frame of the corresponding video segment was used.

<img src="/assets/imgs/J_Lin(2020)/5.png" alt="5.jpg">

## Evaluations
Testing results show that adding visual features to the model only slightly improved the accuracy, while human predicts much better with the additional video input. Also, according to Table 6, statements of visual recognition do not improve with the additional video input, and the accuracy of human dynamics statements is even decreased. Therefore, we can conclude that the fusion of visual data and text data is not well achieved for the existing models. LXMERT is only using single frame rather than the video, so this point can be the consideration of low accuracy. 

Possible future directions are suggested by authors: 
1. developing models to localize key frames
2. better utilizing the alignment between video and subtitles to improve reasoning ability.

<img src="/assets/imgs/J_Lin(2020)/6.png" alt="6.jpg">

<img src="/assets/imgs/J_Lin(2020)/7.png" alt="7.jpg">

<img src="/assets/imgs/J_Lin(2020)/8.png" alt="8.jpg">

<img src="/assets/imgs/J_Lin(2020)/9.png" alt="9.jpg">


## References
<a href="https://arxiv.org/pdf/2003.11618.pdf">[1] J. Lin, et al., “VIOLIN: A Large-Scale Dataset for Video-and-Language Inference”, arXiv:2003.11618v1 [cs.CV] </a>

<a href="https://arxiv.org/pdf/1908.07490.pdf">[2] H. Tan, M. Bansal, “LXMERT: Learning Cross-Modality Encoder Representations from Transformers”, arXiv:1908.07490v3 [cs.CL]  </a>

<a href="http://ethereon.github.io/netscope/#/gist/b21e2aae116dc1ac7b50">[3] Visualization of ResNet-101 </a>

<a href="https://github.com/mitmul/chainer-faster-rcnn">[4] Visualization of Fater R-CNN </a>
