---
title: "Segmentation: Metrics and Losses in 40 Easy Steps"
date: 2023-04-10T12:00:00+03:00
draft: false
katex: true
tags: ["segmentation", "metrics", "losses", "deep learning", "computer vision"]
showtoc: true
---

Semantic segmentation is a crucial component of visual perception, with high demand in both low-latency and precision-focused applications. I have decided to create an overview of the current state of this task.

The first part contains a brief overview of various approaches to evaluating segmentation model performance. Choosing the right set of metrics is often the first step when developing a segmentation system. Understanding the common performance measurements helps clarify the task at hand and provides insights into the challenges that arise when creating such a system.

The second part delves into loss functions, a topic closely connected to metrics. Selecting an appropriate loss function greatly impacts a model's ability to improve on a specific metric.

An overview of the architectures is not included in this post but is likely to be covered in Part 2.

## Notation

| Syntax                                                                        | Description                                                   |
|:-----------------------------                                                 |:--------------------------------------------------------------|
| $\Omega = \\{x_1,...,x_N\\}$                                                  | Image represented by a point set                              |
| $N=\|\Omega\|$                                                                | Number of pixels                                              |
| $C$                                                                           | Number of classes                                             |
| $S_g=\left\\{y_i\| y_i\in\\{0,1\\} \text{ or }[0,1]\right\\}_i^N$             | Partition of the image representing ground truth segmentation |
| $S_t=\left\\{\hat{y}_i\| \hat{y}_i\in\\{0,1\\} \text{ or }[0,1]\right\\}_i^N$ | Partition of the image representing predicted segmentation    |
| $S^1=\\{y_i\| y_i=1\\}$                                                     | Foreground of the segmentation                                |
| $S^0=\\{y_i\| y_i=0\\}$                                                     | Background of the segmentation                                |
| $\mathbf{1}_{\text{condition}}$                                               | Binary indication function                                    |
| $A \odot B$                                                                   | Hadamard (element-wise) product                               |


## Metrics

When evaluating a predicted segmentation against the ground truth, various factors need to be taken into account, depending on the objective. These factors include overall alignment and shape errors, volume errors, boundary errors, density, and the general quality of the segmentation. It is important to note that some metrics are sensitive to outliers, class imbalance, segmentation density, and other specific details.

This section is based on the survey by [[Taha et al. 2015](https://rdcu.be/c9uJP)] and offers definitions for 20 metrics categorized into six groups: overlap-based, volume-based, pair counting-based, information theory-based, probabilistic, and spatial distance-based. While insights on the behavior of each specific metric are provided, readers interested in a more comprehensive overview are encouraged to refer to the original study.

### Common definitions

Let the image be represented by a point set $\Omega = \\{x_1,...,x_N\\}$ with $|\Omega| = w \times h \times d = N$, where $w$, $h$ and $d$ are the width, height and depth of the grid on which the image is defined.

Let the ground truth segmentation be represented by the partition $S_g=\left\\{y_i|y_i\in\\{0,1\\} \text{ or }(0,1)\right\\}_i^N$ of $\Omega$.
Let the predicted segmentation be represented by the partition $S_t=\left\\{\hat{y}_i|\hat{y}_i\in\\{0,1\\} \text{ or }(0,1)\right\\}_i^N$.
Assume that $S^1=\\{y_i|y_i=1\\}$ is the foreground and $S^0=\\{y_i|y_i=0\\}$ is the background.

### Basic components: confusion matrix

For two crisp segmentations $S_g$ and $S_t$, the confusion matrix consists of the four basic components that reflect the overlap between them, namely True Positive, False Positive, False Negative and True Negative. These components are counting based, which is reflected in the case when $\hat{y_i}\in\\{0,1\\}$. However, I abuse the notation here by pointing out that their soft probabilistic approximations are defined by the same formulas when $\hat{y}_i\in[0,1]$. This distinction will be relevant when discussing loss functions, as the latter are differentiable, whereas the former are not.

$$
\begin{align}
TP&=\sum_i^N\hat{y}_i y_i\\\\
FP&=\sum_i^N\hat{y}_i(1-y_i)\\\\
FN&=\sum_i^N(1-\hat{y}_i) y_i\\\\
TN&=\sum_i^N(1-\hat{y}_i)(1-y_i) \\\\
\end{align}
$$

Note that the fuzzy definitions are available in the original study [[Taha et al. 2015](https://rdcu.be/c9uJP)]. I have omitted them for clarity, but it should be assumed that all metrics based on these components are applicable in the fuzzy case as well.

When evaluating metrics defined using the confusion matrix, it is important to consider that some segmentation algorithms may produce segmentations with lower density. These resulting segmentations may have numerous uniformly distributed holes in the foreground, which are counted as false negatives. Such segmentations will be scored lower than denser counterparts with equivalent volume and alignment. This effect may be undesirable if the objective is to prioritize boundary precision over density. In these cases, spatial distance-based metrics are a more suitable choice.

Another aspect to consider, specifically for metrics that include TN, is that they reward segmentations with bias towards background and penalize larger segments more heavily. This can be particularly disadvantageous when aiming for maximum recall.

### Overlap-based metrics

A shared characteristic of all overlap-based metrics is that they do not account for the pixel positions of false positives and false negatives. This is because these pixels are not part of the overlapping region, causing overlap-based metrics to fail in reflecting the distance-wise magnitude of the error. As a result, these metrics are not suitable for situations where low or zero overlap is likely to occur due to alignment errors, such as with class imbalance or when the segmented regions are small (in at least one dimension). As mentioned earlier, they also penalize low-density segmentations.

However, overlap-based metrics are suitable when dealing with outliers and generally low-quality segmentations, as they reflect poor alignment and low overlap. They are also appropriate when prioritizing volume.

**Dice score and Intersection over Union**

The Dice coefficient is the most commonly used metrics in segmentation.

$$
\begin{equation}
Dice = \frac{2\big|S_g^1 \cap S_p^1\big|}{\big| S_g^1 \big| + \big| S_p^1 \big|} = \frac{2TP}{2TP + FP + FN}
\end{equation}
$$

Similarly, Intersection over Union (IoU), also known as the Jaccard index, is defined as

$$
\begin{equation}
IoU= \frac{\big|S_g^1 \cap S_p^1\big|}{\big| S_g^1 \cup S_p^1 \big|}=\frac{TP}{TP+FP+FN}
\end{equation}
$$

The two metrics can be shown to be related:

$$
\begin{align}
IoU &= \frac{Dice}{2 - Dice}, & Dice &= \frac{2IoU}{1+IoU}
\end{align}
$$

Therefore there’s no additional information gain in considering both of them at the same time.

**TPR, TNR, FPR, FNR**

True Positive Rate (TPR, Recall, Sensitivity) measures the proportion of positive pixels in the ground truth that are identified as positive in the evaluated segmentation. True Negative Rate (TNR, Specificity) measures the proportion of negative (background) pixels in the ground truth that are identified as negative in the evaluated segmentation. Both of these metrics are highly sensitive to segment size, as they penalize errors in small segments more severely than in larger ones.

$$
\begin{equation}
TPR = Recall = Sensitivity = \frac{TP}{TP+FN}
\end{equation}
$$

$$
\begin{equation}
TNR = Specificity = \frac{TN}{TN+FP}
\end{equation}
$$

In some applications, missing regions are more undesirable than added regions, requiring the segmentation to include at least all true positives and thus prioritizing recall over precision.

False Positive Rate (FPR) and False Negative Rate are directly related to TPR and TNR, respectively. This implies that one should select at most one metric from each pair for evaluation.

$$
\begin{equation}
FPR = \frac{FP}{FP+TN} = 1 - TNR
\end{equation}
$$

$$
\begin{equation}
FNR = \frac{FN}{FN+TP} = 1 - TPR
\end{equation}
$$

**Precision**

Precision, also know as the Positive Predictive Value (PPV) is used to calculate F-Measure.

$$
\begin{equation}
Precision = PPV = \frac{TP}{TP+FP}
\end{equation}
$$

$**F_{\beta}$-Measure**

$F_{\beta}$-Measure represents the trade-off between precision and recall:

$$
\begin{equation}
F_{\beta} = \frac{(\beta^2 + 1)\cdot Precision \cdot Recall}{\beta^2 \cdot Precision \cdot Recall}
\end{equation}
$$

At $\beta=1$ it becomes an $F_1$-Measure, also called the harmonic mean. By substitution it can be shown to be equivalent to the Dice index.

$$
\begin{equation}
F_1 = \frac{2\cdot Precision \cdot Recall}{Precision \cdot Recall} = Dice
\end{equation}
$$

**Global Consistency Error**

Let $R(S,x)$ be a set of all the pixels that belong to the same segment as $x$ in the segmentation $S$ (foreground or background). The error between two segmentations $S_1$ and $S_2$ at the pixel $x$ is defined as follows:

$$
\begin{equation}
LRE(S_1, S_2, x) = \frac{\left| R(S_1, x) \setminus R(S_2, x) \right|}{\left| R(S_1, x) \right|}
\end{equation}
$$

Note that it’s not symmetric. Now, the Global Consistency Error (GCE) can be defined as the average $LRE$ over all pixels: 

$$
\begin{equation}
GCE(S_p, S_g) = \frac1N\min \left\\{ \sum_i^N LRE(S_p, S_g, x_i), \sum_i^N LRE(S_g, S_p, x_i) \right\\}
\end{equation}
$$

Or, in terms of the confusion matrix,

$$
\begin{equation}
\begin{split}
GCE(S_p, S_g) = \frac1N\min \bigg\\{ &\frac{FN(FN+2TP)}{TP+FN} + \frac{FP(FP+2TN)}{TN+FP}, \\\\[15pt]
&\frac{FP(FP+2TP)}{TP+FP} + \frac{FN(FN+2TN)}{TN+FN} \bigg\\}
\end{split}
\end{equation}
$$

### Note on the multiple label case

In many real-world scenarios, it's common to compare segmentations with multiple labels. In practice, a typical approach is to compare each label individually and then calculate the average across all labels. Assume $M$ is a metric function defined for the binary case,

$$
\begin{equation}
M_{ml}=\frac1C\sum_cM^c
\end{equation}
$$

where $C$ is the number of classes.

**Generalized Dice score**

The approach presented above, however, ignores class imbalance. In literature one can find a weighted multi-label variant of the Dice score called the Generalized Dice score:

$$
\begin{equation}
GD=\frac{2\sum_cw_c\big|S_g^c \cap S_p^c\big|}{\sum_cw_c\Big[\big| S_g^c \big| + \big| S_t^c \big|\Big]} = \frac{2\sum_cw_cTP_c}{\sum_cw_c[2TP_c + FP_c + FN_c]}
\end{equation}
$$

where $w_c=\left|S_g^c\right|^{-2} = \left(TP_c+FN_c\right)^{-2}$. Fuzzy definition can be found in the original study. One can obtain a generalized IoU score by applying (7).

### Volume based metrics

**Volumetric Similarity**

Volumetric Similarity (VS) is defined as $1-VD$, where $VD$ is the volumetric distance. We calculate it as the difference between the absolute volumes of the segments divided by the sum of these volumes.

$$
\begin{equation}
VS = 1- \frac{\Big| \left| S_p^1 \right| - \left| S_g^1 \right| \Big|}{\left| S_p^1 \right|+ \left| S_g^1 \right|} = 1 - \frac{\left|FN - FP\right|}{2TP+FP+FN}
\end{equation}
$$

Note that even though we expressed it through the confusion matrix, VS is not considered to be an overlap metric since it only compares the absolute volumes. In fact, it can reach 1 even at zero overlap.

It is important to keep in mind that VS only compares the volume of the segments and carries no information about their shape or alignment.

### Pixel pair groups

In order to advance to the next group of metrics, we are going to define the base pair-counting elements. Let $P$ be the set of $\frac{n(n-1)}{2}$ pairs representing all pixel pairs in $X$. We’re going to classify each pair $(x_i,x_j)\in P\text{, }i,j\in[0,N]$ into one of four categories based on which subset (foreground or background) those pixels are placed to according to each of the segmentations. To avoid the $O(n^2)$ runtime, it is shown to be possible to calculate these categories using the values in the confusion matrix.

For pairs $(x_i,x_j)$ where $x_i$ and $x_j$ are placed in the same subset in both $S_g$ and $S_t$:

$$
\begin{equation}
a = \frac{1}{2}\bigg[TP(TP-1)+FP(FP-1)+TN(TN-1)+FN(FN-1)\bigg]
\end{equation}
$$

Where $x_i$ and $x_j$ are placed in the same subset in $S_g$, but in different subsets in $S_t$:

$$
\begin{equation}
b = \frac{1}{2}\bigg[ (TP+FN)^2+(TN+FP)^2-\Big(TP^2+TN^2+FP^2+FN^2\Big) \bigg]
\end{equation}
$$

Where $x_i$ and $x_j$ are placed in different subsets in $S_g$, but in the same subset in $S_t$:

$$
\begin{equation}
c = \frac{1}{2}\bigg[ (TP+FP)^2+(TN+FN)^2-\Big(TP^2+TN^2+FP^2+FN^2\Big) \bigg]
\end{equation}
$$

Finally, where $x_i$ and $x_j$ are placed in different subsets in both $S_g$ and $S_t$:

$$
\begin{equation}
d = \frac{N(N-1)}{2}-(a+b+c)
\end{equation}
$$

### Pair counting based metrics

**Rand Index**

The Rand Index (RI) was originally proposed for measuring the similarity between clusterings and was later adapted for classification.

$$
\begin{equation}
RI(S_g, S_p) = \frac{a+b}{a+b+c+d}
\end{equation}
$$

**Adjusted Rand Index**

The Adjusted Rand Index (ARI) is a modification of RI with a correction for chance. It can be expressed by the pair-counting groups as:

$$
\begin{equation}
ARI(S_g, S_p) = \frac{2(ad-bc)}{c^2+b^2+2ad+(a+d)(c+b)}
\end{equation}
$$

Having the built-in chance adjustment makes ARI a good choice for when there is a heavy class imbalance.

### Information theory based metrics

Assuming $N=TP+FP+FN+TN$ is the total number of pixels, the probability of a randomly sampled pixel belonging to each of the classes in either segmentation can be expressed using the confusion matrix as follows:

$$
\begin{align}
p\left(S_g^1\right) &= \frac{TP+FN}{N} \\\\[15pt]
p\left(S_g^0\right) &= \frac{TN+FP}{N} \\\\[15pt]
p\left(S_t^1\right) &= \frac{TP+FP}{N} \\\\[15pt]
p\left(S_t^0\right) &= \frac{TN+FN}{N}
\end{align}
$$

The joint probabilities for some are given by

$$
\begin{align}
p\left(S_g^1,S_p^1\right) = \frac{TP}{N} \\\\[15pt]
p\left(S_g^1,S_p^0\right) = \frac{FN}{N} \\\\[15pt]
p\left(S_g^0,S_p^1\right) = \frac{FP}{N} \\\\[15pt]
p\left(S_g^0,S_p^0\right) = \frac{TN}{N}
\end{align}
$$

Having the definitions above we can express the marginal entropy between the regions

$$
\begin{equation}
H(S)=-\sum_i p\left(S^i\right)\log p\left(S^i\right)
\end{equation}
$$

and the joint entropy between the segmentations

$$
\begin{equation}
H(S_1,S_2)=-\sum_{i,j} p\left(S_1^i, S_2^j\right)\log p\left(S_2^i,S_2^j\right)
\end{equation}
$$

**Mutual Information**

The Mutual Information (MI) measures the reduction in uncertainty of one variable when the other one is known.

$$
\begin{equation}
MI(S_g,S_p)=H(S_g)+H(S_p)-H(S_g,S_p)
\end{equation}
$$

Note that MI essentially measures how much information the segmentations have in common and therefore rewards high recall.

**Variation of Information**

The Variation of Information (VoI) measures the amount of information lost when transitioning from one variable to the other.

$$
\begin{equation}
VoI(S_g,S_p)=H(S_g)+H(S_p)-2MI(S_g,S_p)
\end{equation}
$$

### Probabilistic metrics

**Intraclass Correlation Coefficient**

Intraclass Correlation Coefficient (ICC) is sometimes used as a measure of consistency between two segmentations, specifically in the medical imaging domain.

$$
\begin{equation}
ICC = \frac{\sigma_S^2}{\sigma_S^2+\sigma_{\epsilon}^2}
\end{equation}
$$

Here $\sigma_S$ denotes variance caused by differences between segmentations and $\sigma_{\epsilon}$ denotes variance cause by differences between the points within each segmentation. For segmentations $S_g$ and $S_t$, we express it via the mean squares between segmentations $MS_b$ and the mean squares within the segmentations $MS_w$:

$$
\begin{gather}
&ICC = \frac{MS_b-MS_w}{MS_b+MS_w}
\end{gather}
$$

where

$$
\begin{align}
&MS_b = \frac{2}{N-1}\sum_i \left(\frac{\hat{y}_i+y_i}{2}-\mu\right)^2 \\\\[15pt]
&MS_w = \frac1N\sum_i\left(y_i-\frac{\hat{y}_i+y_i}{2}\right)^2 + \left(\hat{y}_i-\frac{\hat{y}_i+y_i}{2}\right)^2 \\\\
\end{align}
$$

Here $\mu$ is the mean of means of the two segmentations.

**Probabilistic Distance**

The Probabilistic Distance (PBD) [[Guido et al. 2001](https://link.springer.com/chapter/10.1007/3-540-45468-3_62)] is designed as a measure of distance between two fuzzy segmentations.

$$
\begin{equation}
PBD(S_g,S_p)=\frac{\sum_i \big|y_i-\hat{y}_i\big|}{2\sum_i y_i \hat{y}_i}
\end{equation}
$$

Note that in contrast to Dice, the PBD over-penalizes false positives and false negatives, as they both reduce the denominator and increase the numerator to the point where PBD reaches infinity at zero overlap. This results in PBD strongly reflecting alignment errors, i.e., when the volume is correct and the overlap is low.

**Cohen’s Kappa Coefficient**

The Cohen’s Kappa Coefficient is a robust measure of agreement between the samples that takes into account the agreement caused by chance.

$$
\begin{equation}
\kappa=\frac{2(TP\cdot TN-FN\cdot FP)}{(TP+FP)(FP+TN)+(TP+FN)(FN+TN)}
\end{equation}
$$

Chance adjustment makes Kappa a good choice when there is high class imbalance.

**ROC AUC**

The ROC curve is a plot of TPR against FPR at every possible threshold. The area under the ROC curve reflects the probability for the classifier to rank a positive example higher than the negative one. It is possible to calculate a rough estimate of the AUC for a single measurement case.

$$
\begin{equation}
AUC=1-\frac{FPR+FNR}{2}
\end{equation}
$$

Note that calculating AUC this way is generally not recommended, since it significantly underestimates the value.

### Spatial distance based metrics

Spatial distance-based metrics possess the notable property of taking into account pixel positions outside the overlap region, as well as those within it. This enables them to offer more meaningful rankings in situations where overlap is likely to be low, such as with small object segmentations and low-density segmentations. These metrics are also applicable when prioritizing boundary or overall alignment and in cases involving low-quality segmentations.

**Hausdorff Distance**

The Hausdorff Distance (HD) is defined as the maximum distance from a point in one set to the nearest point in the other set.

$$
\begin{equation}
HD(S_g^i,S_p^i)=\max\big(h(S_g^i, S_p^i),h(S_p^i, S_g^i)\big)
\end{equation}
$$

where $h(A,B)$ is the directed Hausdorff distance given by

$$
\begin{equation}
h(A,B)=\max_{a\in A}\min_{b\in B} \|a-b\|
\end{equation}
$$

HD can be viewed as an indicator of the largest segmentation error. It is computed between boundaries of the ground truth and the predicted segmentation.

![hausdorff](/segmentation-metrics-and-losses/hausdorff.png)

Algorithms have been developed that calculate the HD in near-linear time. Note that HD is sensitive to outliers, which is why it is recommended to use the quantile version instead of applying it directly.

**Average Hausdorff Distance**

The Average Hausdorff Distance is the HD averaged over all points. It is known to be more robust than the original.

$$
\begin{equation}
AHD(S_g^i,S_p^i)=\max\big(d(S_g^i,S_p^i),d(S_p^i,S_g^i)\big)
\end{equation}
$$

where $d(A,B)$ is the directed average Hausdorff distance that is defined as

$$
\begin{equation}
d(A,B)=\frac{1}{N}\sum_{a\in A}\min_{b\in B}\|a-b\|
\end{equation}
$$

**Mahalanobis Distance**

The Mahalanobis Distance (MD) is a metric that measures the distance between a point and a distribution while considering the shape of the distribution. However, when comparing image segmentations, we need to measure the distance between two distributions. To achieve this, we calculate the distance between their means as follows:

$$
\begin{equation}
MD(S_g^i,S_p^i) = \sqrt{(\mu_{g,i}-\mu_{t,i})^T K^{-1}(\mu_{g,i}-\mu_{t,i})}
\end{equation}
$$

where $\mu_{g,i}$ and $\mu_{t,i}$ are the means for the $i$th category in the respective segmentation, and their common covariance matrix is given by

$$
\begin{equation}
K=\frac{n_{g,i} K_{g,i} + n_{t,i} K_{t,i}}{n_{g,i} + n_{t,i}}
\end{equation}
$$

Here $K_{g,i}$ and $K_{t,i}$ are respective covariance matrices and $n_{g,i},n_{t,i}$ are the numbers of pixels.

Note that MD ignores boundary details and considers only general shape and alignment, i. e. the two ellipsoids that best represent the segmentations.

![summary_table](/segmentation-metrics-and-losses/summary_table.png)

## Loss functions

After examining various popular metrics and gaining insight into the challenges involved, we can proceed to select the appropriate loss function. The choice of loss function can profoundly influence a model's capacity to learn the nuances of a task, which often includes class imbalance, disparate object sizes, and intricate boundary delineation.

This section draws in part on Jun Ma's survey [[Jun Ma 2020](https://arxiv.org/abs/2005.13449)], with some additions. It does not maintain a one-to-one correspondence with the metric section, as numerous metrics discussed earlier are non-differentiable and thus unsuitable for backpropagation. However, some have been approximated with differentiable functions, making them compatible with gradient-based optimization.

### Distribution-based loss

Distribution-based loss functions treat both the ground truth and predicted segmentation as probability distributions, with the goal of minimizing their differences. The cross-entropy serves as the foundational loss function in this category, from which others are derived.

**Cross entropy loss**

Cross-entropy (CE) is related to the Kullback-Leibler (KL) divergence, a metric that quantifies the dissimilarity between two probability distributions. In machine learning, when the data distribution is determined by the training set, minimizing KL divergence is equivalent to minimizing CE.

$$
\begin{equation}
L_{CE}=-\frac1N\sum_i\sum_c y_i^c \log \hat{y}_i^c
\end{equation}
$$

where $g_i^c$ indicates if class label $c$ is a correct classification for the pixel $i$, and $s_i^c$ is the corresponding predicted probability.

Weighted cross entropy (WCE) [[Ronneberger et al. 2015](https://arxiv.org/abs/1505.04597)] is a popular modification to the CE:

$$
\begin{equation}
L_{WCE}=-\frac1N\sum_i\sum_c w_c y_i^c \log \hat{y}_i^c
\end{equation}
$$

where $w_c$ is the corresponding class weight. It is common to set $w_c$ as inversely proportional to the class frequency to balance out majority classes.

**TopK loss**

TopK loss [[Wu et al. 2016](https://arxiv.org/abs/1605.06885)] focuses the training on the hard pixels by discarding the ones that model evaluates with enough confidence. Notably this approach automatically balances the biased data by skipping the over-learned majority class.

$$
\begin{equation}
L_{TopK}= -\frac{\sum_i\sum_c \bm{1}_{\hat{y}_i^c<t} \log \hat{y}_i^c}{\sum_i\sum_c \bm{1}\_{\hat{y}_i^c<t}}
\end{equation}
$$

where $t\in(0,1]$ is the discarding threshold and $\bm{1}_{\dots}$ is the binary indication function.

**Focal loss**

The Focal loss [[Lin et al. 2017](https://arxiv.org/abs/1708.02002v2)] directly addresses class imbalance by adding an extra multiplier to the CE. This multiplier diminishes the loss value for the well classified examples, effectively the function heavily towards the hard samples.

$$
\begin{equation}
L_{focal} = -\frac1N \sum_i\sum_c \big(1-\hat{y}_i^c\big)^{\gamma} y_i^c\log \hat{y}_i^c
\end{equation}
$$

### Overlap-based loss

Overlap-based loss functions, such as the popular Dice loss, aim to directly optimize overlap metrics. However, these metrics are count-based, making them non-differentiable and unsuitable for gradient-based optimization.

A common solution to this issue, with a few exceptions, involves using a surrogate function to create a smooth approximation of the confusion matrix elements. Although these soft versions don't precisely match the original representations, they closely resemble them, making optimization feasible.

Note that in this section equations (1) - (4) should be interpreted as defined on soft probabilistic labels $\hat{y}_i\in[0,1]$. This interpretation makes these equations differentiable with respect to both the predictions and the model parameters, enabling their use in gradient-based optimization during model training.

**Sensitivity-specificity loss**

Sensitivity-specificity (SS) loss [[Brosch et al. 2015](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_1)] combines both of these metrics to facilitate training with heavy class imbalance.

$$
\begin{equation}
L_{SS}=r\cdot \frac{\sum_i(y_i-\hat{y}_i)^2y_i}{\sum_i y_i}+(1-r)\cdot \frac{\sum_i(y_i-\hat{y}_i)^2(1-y_i)}{\sum_i (1-y_i)}
\end{equation}
$$

where $r$ denotes the weight and is set to 0.05 by default.

**Dice loss**

The Dice loss [[Milletari et al. 2016](https://arxiv.org/abs/1606.04797)] optimizes the Dice coefficient by utilizing a soft approximation. It accounts for class imbalance by definition.

$$
\begin{equation}
L_{Dice}=1-Dice
\end{equation}
$$

**IoU loss**

IoU loss [[Rahman et al. 2016](https://link.springer.com/chapter/10.1007/978-3-319-50835-1_22)] is defined similarly to the Dice loss.

$$
\begin{equation}
L_{IoU}=1-IoU
\end{equation}
$$

**Tversky loss**

Tversky loss [[Salehi et al. 2017](https://arxiv.org/abs/1706.05721)] adds additional parameters to control the trade-off between false positives and false negatives, since assigning the same weight to false positives and false negatives may result in low recall for small regions. When $\alpha=\beta=0.5$ the Tversky loss becomes equivalent to the Dice loss, and when $\alpha=\beta=1$, it becomes equivalent to the Jaccard loss.

$$
\begin{equation}
Tversky(S_g,S_p;\alpha,\beta)=\frac{TP}{TP+\alpha FP + \beta FN}
\end{equation}
$$

$$
\begin{equation}
L_{Tversky}(S_g,S_p;\alpha,\beta)=1-Tversky
\end{equation}
$$

**Generalized Dice loss**

Generalized Dice loss optimized the multiclass extension of the Dice coefficient.

$$
\begin{equation}
L_{GD}=1-GD
\end{equation}
$$

**Focal Tversky loss**

The Focal Tversky loss [[Abraham et al. 2018](https://arxiv.org/abs/1810.07842)] is an extension of the Tversky loss, specifically tailored to enhance performance when dealing with smaller objects. It uses the focusing parameted $\gamma\in[1,3]$ to put heavier emphasis on the hard misclassifier examples.

$$
\begin{equation}
L_{FTL}=\big(1-Tversky\big)^{\frac1\gamma}
\end{equation}
$$

**Asymmetric similarity loss**

Asymmetric similarity loss [[Hashemi et al. 2018](https://arxiv.org/abs/1803.11078)] is a modified version of Tversky loss that establishes more precise rules for weighting false positives (FP) and false negatives (FN), placing greater emphasis on reducing the latter. Suitable values for the hyperparameter beta can be determined based on class imbalance ratios, ensuring a more balanced approach to addressing discrepancies between classes.

$$
\begin{equation}
L_{Asym}(S_g,S_p;\beta)=1-\frac{TP}{TP+ \frac{\beta^2}{1+\beta^2}FP +  \frac{1}{1+\beta^2}FN}
\end{equation}
$$

**Penalty loss**

The Penalty loss [[Yang et al. 2019](https://openreview.net/forum?id=H1lTh8unKN)] builds on the Generalized Dice loss by incorporating a coefficient k, which facilitates adding extra weight for false positives (FP) and false negatives (FN).

$$
\begin{equation}
L_{pGD}=\frac{L_{GD}}{1+k\left(1-L_{GD}\right)}
\end{equation}
$$

**Lovász Hinge loss**

Lovász Hinge loss [[Berman et al. 2017](https://arxiv.org/abs/1705.08790)] offers an alternative differentiable surrogate function for IoU, distinct from the IoU loss itself. In this approach, the IoU loss is reformulated as a set function on the set of mispredictions. It can be shown that such a set function is submodular, therefore its tight convex hull can be computed efficiently as its Lovász extension.

Denote $\Delta: \\{0,1\\}^N\rarr\mathbb{R}$ the rewritten IoU loss. We calculate its Lovász extension as

$$
\begin{equation}
\overline{\Delta} = \sum_i^N m_ig_i(\bm{m})
\end{equation}
$$

Here vector $\bm{g}(\bm{m})$ is the derivative of $\overline{\Delta}$ with respect to $\bm{m}$.

$$
\begin{equation}
g_i(\bm{m})=\Delta(\\{\pi_1,\dots,\pi_i\\})-\Delta(\\{\pi_1,\dots,\pi_{i-1}\\})
\end{equation}
$$

$\bm{\pi}$ is a permutation ordering of the components of $\bm{m}$ in decreasing order.  $\bm{m}$ is a vector of all pixel errors, that is calculated as follows:

- Rewrite ground truth labels as $y_i^*\in\\{-1, 1\\}$,
- Denote $F_i$ the logit, such that the predicted label $\hat{y}_i^*=\textrm{sign}(F_i)$,
- $m_i=\max \big(1-F_i\cdot y_i^*,0\big)$

The obtained function is the the Lovász hinge applied to the IoU loss.

$$
\begin{equation}
L_{Lovász}=\overline{\Delta}(\bm{m}(\bm{F}))
\end{equation}
$$

$\overline{\Delta}(\bm{m})$ is a sum of errors $\bm{m}$ that weights them according to the interpolated discrete loss.

The algorithm above gets a bit more clear when put in code, provided below. Note that some technical details were omitted for clarity.

```python
def lovasz_grad(gt_sorted):
    """Compute gradient of the Lovasz extension w.r.t sorted errors
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.cumsum(0)
    union = gts + (1 - gt_sorted).cumsum(0)
    iou_loss = 1.0 - intersection / union
    return iou_loss

def lovasz_hinge(logits, labels):
    """Binary Lovasz hinge loss
    Args:
        logits: logits at each prediction (-infinity, +infinity)
        labels: binary ground truth labels {0, 1}
    """
    signs = 2.0 * labels - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, descending=True)
		gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss
```

**Exponential Logarithmic Loss**

Exponential Logarithmic Loss [[Wong et al. 2018](https://arxiv.org/abs/1809.00076)] is a combination of the exponential logarithmic Dice loss and the weighted exponential cross-entropy loss. It has been proposed to enhance segmentation accuracy for small structures in tasks with significant variability in the sizes of objects to be segmented, ensuring a more balanced performance across different scales.

$$
\begin{equation}
L_{Exp}=w_{eld}L_{eld}+w_{elce}L_{elce}
\end{equation}
$$

$$
\begin{equation}
L_{eld}=\frac1N \big(-\ln( Dice)^{\gamma_{eld}}\big)
\end{equation}
$$

$$
\begin{equation}
L_{elce}=\frac1{2N} \Big[w_1 (-\ln\hat{y})^{\gamma_{elce}}+w_2\big(-\ln(1-\hat{y})\big)^{\gamma_{elce}}\Big]
\end{equation}
$$

where $w_c=\left(\frac{\sum_k f_k}{f_c}\right)^{\frac12}$ is the weight to increase the influence of the rare labels, $f_k$ is the frequency of the label $k$.

**Matthews correlation coefficient**

Matthews correlation coefficient loss [[Abhishek et al. 2020](https://arxiv.org/abs/2010.13454)] addresses the fact that the Dice
loss does not include a penalty for misclassifying the false negative pixels, which affects the accuracy of background segmentation.

$$
\begin{equation}
MCC=\frac{TP\cdot TN-FP\cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}
\end{equation}
$$

$$
\begin{equation}
L_{MCC}=1-MCC
\end{equation}
$$

### Compound loss

Achieving a good balance of qualities in a loss function can often be accomplished by combining several of them with assigned weights. Various combinations have proven successful, and below is one such example.

**Combo loss**

The Combo loss [[Taghanaki et al. 2018](https://arxiv.org/abs/1805.02798)] aims to tackle class imbalance and offers control over the tradeoff between false positives and false negatives.

$$
\begin{equation}
\begin{split}
L_{Combo}=-\alpha\left(\frac1N\sum_i^N\beta(y_i \log \hat{y}_i)+(1-\beta)(1-y_i)\log(1-\hat{y}_i)\right)- \\\\[15pt]
-(1-\alpha)Dice
\end{split}
\end{equation}
$$

### Boundary-based loss

Losses in this category primarily rely on the distance transform. Essentially, the distance transform converts a binary segmentation map into a distance map, where each pixel is assigned a value representing its distance to the nearest foreground pixel. However, each approach utilizes the distance transform differently, combining and inverting maps and integrating them into the objective function in unique ways.

**Distance map penalized cross entropy loss (DPCE)**

Distance maps for DPCE [[Caliva et al. 2019](https://arxiv.org/abs/1908.03679)] are generated by computing the distance transform on the segmentation masks and then inverting them by pixel-wise subtracting the binary segmentation from the mask's maximum distance value. This process aims to create a distance mask where pixels close to the foreground are assigned higher weight compared to those further away. A similar procedure is conducted on the inverted version of the segmentation mask to calculate a distance map inside the foreground regions. The resulting maps are then applied to the loss with element-wise multiplication, imposing heavier penalties for errors near the boundary.

$$
\begin{equation}
L_{DPCE}=-\frac1N\sum_{i=1}^N(1+\Phi)\odot\sum_c^C y_i^c\log \hat{y}_i^c
\end{equation}
$$

where $\Phi$ is the distance penalty and $\odot$ denotes the element-wise product.

![dpce.png](/segmentation-metrics-and-losses/dpce.png)

**Boundary loss**

Distance maps for the Boundary loss [[Kervadec et al. 2018](https://arxiv.org/abs/1812.07032)] are constructed as follows: $\phi_G=-D_G(q)$ if $q\in S_g^1$ and $\phi_G=D_G(q)$ otherwise. Here, the distance transform on the ground truth is combined with a negative signed distance transform on the inverted map.

```python
def dist(seg):
    C = len(seg)
    res = np.zeros_like(seg)
    for c in range(C):
        posmask = seg[c].astype(np.bool)
        if posmask.any():
            negmask = ~posmask
            res[c] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
    return res
```

The main advantage of the Boundary loss is its ability to bypass the class imbalance issue by focusing on the boundaries rather than the regions themselves.

$$
\begin{equation}
L_{DB}=\frac1N \sum\phi_G\odot \hat{S}
\end{equation}
$$

**Hausdorff distance (HD) loss**

Minimizing HD directly can be intractable and may result in unstable training. Nonetheless, it can be approximated using the distance transforms of the ground truth and predicted segmentation [[Karimi et al. 2019](https://arxiv.org/abs/1904.10030)]. $d_p$ and $d_g$ denote regular distance transforms on their respective segmentation maps. Parameter $\alpha$ determines the emphasis placed on larger errors and is set to 2 by default. Since recomputing $d_p$ at every step can be expensive, the authors also propose a one-sided variant of this loss.

$$
\begin{equation}
L_{HD}=\frac1N\sum_{i=1}^N\Big(\left(\hat{y}_i-y_i\right)^2\odot\left(d_p^{\alpha}+d_g^{\alpha}\right)\Big)
\end{equation}
$$

### Topological loss

This class of loss functions focuses on ensuring the topological correctness of the predicted segmentation. Specifically, it compares two segmentations to confirm that they have the same number of connected components and holes, which is also referred to as the Betti number.

**Topology-Preserving loss**

Since the Betti number is discrete, it cannot be directly used for gradient-based optimization. To overcome this issue, one can employ persistent homology [[Hu et al. 2019](https://arxiv.org/abs/1906.05404)]. In this approach, instead of using a single threshold to obtain the predicted segmentation map, all thresholds are considered. As the threshold decreases, certain topological components "get born" (a separate disconnected component appears, a cycle gets bridged) and "die" (one component merges into another, a hole gets filled). We record the threshold value at which such an event occurs as birth time and death time of a component. A full set of these values for every threshold forms a persistence diagram. Each component can be visualized as a dot by plotting the birth time against the death time. It is important to note that, for the ground truth, all components land on the same spot, since they remain unchanged for every value of the threshold.

![topo.png](/segmentation-metrics-and-losses/topo.png)

Next, the optimal one-to-one correspondence between the ground truth and the predicted segmentation is determined, with any residual predicted points being matched to the diagonal line. The loss is then calculated as the combined Euclidean distance between each pair of corresponding points.

$$
\begin{equation}
L_{Topo}=\sum_{p\in Dgm(f)} \Big[birth(p)-birth\big(\gamma^{\star}(p)\big)\Big]^2 + \Big[\big(\gamma^{\star}(p)\big)\Big]^2
\end{equation}
$$

where $\gamma^{\star}$ is the optimal matching between two different point sets.

![topo2.png](/segmentation-metrics-and-losses/topo2.png)

## Citation

Cited as:

>Buzin, Andrey. (Apr 2023). Semantic Segmentation: Metrics and Losses in 40 Easy Steps. Computer Visions. [https://deepbuzin.github.io/posts/segmentation-metrics-and-losses/](https://deepbuzin.github.io/posts/segmentation-metrics-and-losses/).

Or

```
@article{buzin2023segmentationmetrics,
    title   = "Semantic Segmentation: Metrics and Losses in 40 Easy Steps",
    author  = "Buzin, Andrey",
    journal = "deepbuzin.github.io",
    year    = "2023",
    month   = "Apr",
    url     = "https://deepbuzin.github.io/posts/segmentation-metrics-and-losses/"
}
```

## References

[1] Taha, A.A., Hanbury, A. [“Metrics for evaluating 3D medical image segmentation: analysis, selection, and tool”](https://rdcu.be/c9uJP) BMC Med Imaging **(2015).

[2] Guido Gerig, et al. [“Valmet: A New Validation Tool for Assessing and Improving 3D Object Segmentation”](https://link.springer.com/chapter/10.1007/3-540-45468-3_62) MICCAI 2001.

[3] Jun Ma [“Segmentation Loss Odyssey”](https://arxiv.org/abs/2005.13449) arXiv:2005.13449 (2020).

[4] Olaf Ronneberger, et al. [“U-Net: Convolutional Networks for Biomedical Image Segmentation”](https://arxiv.org/abs/1505.04597) arXiv:1505.04597 (2015).

[5] Zifeng Wu, et al. [“Bridging Category-level and Instance-level Semantic Image Segmentation”](https://arxiv.org/abs/1605.06885) arXiv:1605.06885 (2016).

[6] Tsung-Yi Lin, et al. [“Focal Loss for Dense Object Detection”](https://arxiv.org/abs/1708.02002v2) arXiv:1708.02002 (2017).

[7] Tom Brosch, et al. [“Deep Convolutional Encoder Networks for Multiple Sclerosis Lesion Segmentation”](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_1) MICCAI 2015.

[8] Fausto Milletari, et al. [“V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation”](https://arxiv.org/abs/1606.04797) arXiv:1606.04797 (2016).

[9] Rahman, M.A., Wang, Y. [“Optimizing Intersection-Over-Union in Deep Neural Networks for Image Segmentation.”](https://link.springer.com/chapter/10.1007/978-3-319-50835-1_22)  ISVC 2016.

[10] Seyed Sadegh Mohseni Salehi, et al. [“Tversky loss function for image segmentation using 3D fully convolutional deep networks”](https://arxiv.org/abs/1706.05721) arXiv:1706.05721 (2017).

[11] Nabila Abraham, et al. [“A Novel Focal Tversky loss function with improved Attention U-Net for lesion segmentation”](https://arxiv.org/abs/1810.07842) arXiv:1810.07842 (2018).

[12] Seyed Raein Hashemi, et al. [“Asymmetric Loss Functions and Deep Densely Connected Networks for Highly Imbalanced Medical Image Segmentation: Application to Multiple Sclerosis Lesion Detection”](https://arxiv.org/abs/1803.11078) arXiv:1803.11078 (2018).

[13] Su Yang, et al. [“Major Vessel Segmentation on X-ray Coronary Angiography using Deep Networks with a Novel Penalty Loss Function”](https://openreview.net/forum?id=H1lTh8unKN) MIDL 2019.

[14] Maxim Berman, et al. [“The Lovász-Softmax loss: A tractable surrogate for the optimization of the intersection-over-union measure in neural networks”](https://arxiv.org/abs/1705.08790) arXiv:1705.08790 (2017).

[15] Ken C. L. Wong, et al. [“3D Segmentation with Exponential Logarithmic Loss for Highly Unbalanced Object Sizes”](https://arxiv.org/abs/1809.00076) arXiv:1809.00076 (2018).

[16] Kumar Abhishek, et al. [“Matthews Correlation Coefficient Loss for Deep Convolutional Networks: Application to Skin Lesion Segmentation”](https://arxiv.org/abs/2010.13454) arXiv:2010.13454 (2020).

[17] Saeid Asgari Taghanaki, et al. [“Combo Loss: Handling Input and Output Imbalance in Multi-Organ Segmentation”](https://arxiv.org/abs/1805.02798) arXiv:1805.02798 (2018).

[18] Francesco Caliva, et al. [“Distance Map Loss Penalty Term for Semantic Segmentation”](https://arxiv.org/abs/1908.03679) arXiv:1908.03679 (2019).

[19] Hoel Kervadec, et al. [“Boundary loss for highly unbalanced segmentation”](https://arxiv.org/abs/1812.07032) arXiv:1812.07032 (2018)

[20] Davood Karimi, et al. [“Reducing the Hausdorff Distance in Medical Image Segmentation with Convolutional Neural Networks”](https://arxiv.org/abs/1904.10030) arXiv:1904.10030 (2019).

[21] Xiaoling Hu, et al. [“Topology-Preserving Deep Image Segmentation”](https://arxiv.org/abs/1906.05404) arXiv:1906.05404 (2019).




