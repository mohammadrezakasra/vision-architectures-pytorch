<div align="center">

# 🧠 Vision Architectures in PyTorch
### Modular CNN Implementations, Controlled Empirical Benchmarks & Hardware-Aware Profiling

<p>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-F59E0B?style=for-the-badge&logo=python&logoColor=white&labelColor=2563EB" alt="Python"></a>
  &nbsp;&nbsp;
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.x-F97316?style=for-the-badge&logo=pytorch&logoColor=white&labelColor=DC2626" alt="PyTorch"></a>
  &nbsp;&nbsp;
  <a href="https://jupyter.org/"><img src="https://img.shields.io/badge/Notebooks-Jupyter-EC4899?style=for-the-badge&logo=jupyter&logoColor=white&labelColor=7C3AED" alt="Jupyter"></a>
</p>

<p>
  <a href="https://github.com/mohammadrezakasra/vision-architectures-pytorch/commits/main"><img src="https://img.shields.io/github/last-commit/mohammadrezakasra/vision-architectures-pytorch?style=for-the-badge&logo=github&logoColor=white&labelColor=0D9488&color=10B981" alt="Last Commit"></a>
  &nbsp;&nbsp;
  <a href="https://github.com/mohammadrezakasra/vision-architectures-pytorch"><img src="https://img.shields.io/github/repo-size/mohammadrezakasra/vision-architectures-pytorch?style=for-the-badge&logo=files&logoColor=white&labelColor=6366F1&color=06B6D4" alt="Repo Size"></a>
  &nbsp;&nbsp;
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge&logo=opensourceinitiative&logoColor=white&labelColor=0284C7" alt="License"></a>
</p>

<p>
  <strong>🌐 Language:</strong>
  <a href="#1-overview">English</a> •
  <a href="#بخش-فارسی-پیادهسازی-ماژولار-و-بنچمارک-معماریهای-بینایی-ماشین-در-pytorch">فارسی</a>
</p>

</div>

---

## 1. Overview

**`vision-architectures-pytorch`** is a modular research and engineering repository implementing classical, modern, and custom convolutional neural network (CNN) architectures from scratch in PyTorch. Every study is paired with an isolated, reproducible benchmark notebook featuring unified training pipelines (`Trainer`), mixed-precision (`AMP`) acceleration, theoretical complexity profiling (`Params`, `MACs`, `FLOPs`), and hardware latency evaluations.

* **Detailed mathematical analyses, per-class breakdowns, and training logs are documented directly at the top of each benchmark notebook.**

---

## 2. Architecture Zoo (Implemented Models)

> *To add a new model in the future, simply append a single row to this table.*

| Model Name | Family / Paradigm | Key Architectural Features | Params (CIFAR/MNIST) | FLOPs | Best Test Acc |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **`KasraNet-V1-IncepNeXt`** | Custom Hybrid | Multi-scale branches (`1×1`, `3×3`, `5×5`) + Grouped Conv (`C=16`) | `1.06M` | `315.33M` | **95.27%** *(CIFAR-10)* |
| **`ResNeXt-29`** | Residual / Grouped | Aggregated residual transformations (`depth=29, C=2`) | `1.42M` | `433.77M` | **95.15%** *(CIFAR-10)* |
| **`WideResNet-28-2`** | Wide Residual | Dense $3\times3$ convolutions with widened channels (`k=2`) | `1.47M` | `432.52M` | **94.45%** *(CIFAR-10)* |
| **`DualBranchResNet-28`** | Lightweight Multi-Path | Parallel two-branch residual topology for high throughput | `0.74M` | `222.37M` | **94.34%** *(CIFAR-10)* |
| **`VGG-16` (Light Head)** | Deep Homogeneous | Deep stacked $3\times3$ filters + funnel classifier for $32\times32$ inputs | — | — | **89.30%** *(CIFAR-10)* |
| **`VGG-10` (Large Kernels)** | Custom Shallow | Early large spatial filters ($11\times11, 7\times7, 5\times5$) | — | — | **84.79%** *(CIFAR-10)* |
| **`LeNet-5 Classic`** | Classical CNN | Explicit spatial subsampling / pooling layers | — | — | **98.84%** *(MNIST)* |
| **`LeNet-5 Modern`** | Strided CNN | Learnable strided convolutions (`stride > 1`) replacing pooling | — | — | **98.68%** *(MNIST)* |

---

## 3. Benchmark Notebooks Index

> *Each notebook contains comprehensive bilingual (English/Persian) documentation, experimental setups, and engineering insights.*

| # | Notebook / Study | Dataset | Core Research Question | Winner & Key Takeaway |
| :-: | :--- | :---: | :--- | :--- |
| **01** | **`01_mnist_lenet5_classic_vs_modern.ipynb`** | `MNIST` | **Pooling vs. Strided Conv** across `SGD` & `AdamW` (synchronized initial weights) | **`LeNet-5 Classic + SGD` (98.84%)** — Pooling provides superior translation invariance; `AdamW` excels on strided landscapes. |
| **02** | **`02_CIFAR_10_vgg10_vs_vgg16.ipynb`** | `CIFAR-10` | **Deep $3\times3$ Stacks (`VGG16`) vs. Large Kernels (`VGG10`)** ($11\times11, 7\times7, 5\times5$) | **`VGG-16` (89.30% vs. 84.79%)** — Large kernels blur fine animal textures (`cat` drops by `-19.4%`); $3\times3$ stacks boost non-linearity. |
| **03** | **`03_CIFAR_10_resnet_resnext_kasranet.ipynb`** | `CIFAR-10` | **Multi-Scale Grouped Conv (`KasraNet-V1`) vs. `WRN-28`, `ResNeXt-29`, `DualBranch-28`** | **`KasraNet-V1` (95.27% Acc, 315M FLOPs)** — Highest accuracy with 27% fewer FLOPs; `DualBranch-28` leads GPU speed (`20,645 img/s`). |

---

## 4. Datasets & Preprocessing Summary

| Dataset | Input Resolution | Splits (`Train / Val / Test`) | Augmentation & Regularization Pipeline |
| :--- | :---: | :---: | :--- |
| **MNIST** | $1 \times 28 \times 28$ | Standard Stratified Splits | Channel Normalization, Isolated Weight Cloning (`state_dict`) |
| **CIFAR-10** | $3 \times 32 \times 32$ | `45,000 / 5,000 / 10,000` (Stratified, `seed=42`) | `RandomCrop(32, pad=4)`, `RandomHorizontalFlip`, `RandomErasing`, `LabelSmoothing(0.1)` |

---

## 5. Repository Structure & Quick Start

```text
vision-architectures-pytorch/
├── models/                  # Modular PyTorch architecture definitions (LeNet, VGG, ResNet, KasraNet, etc.)
├── utils/                   # Unified Trainer class (AMP, CosineAnnealingLR, checkpointing) & dataset loaders
├── notebooks/               # Self-contained bilingual benchmark notebooks (01, 02, 03, ...)
└── README.md                # High-level project index and summary tables
```

### Quick Usage

```bash
git clone [https://github.com/mohammadrezakasra/vision-architectures-pytorch.git](https://github.com/mohammadrezakasra/vision-architectures-pytorch.git)
cd vision-architectures-pytorch
pip install torch torchvision thop pandas scikit-learn
```

---
---

<div dir="rtl" align="right" style="line-height: 1.8;">

# بخش فارسی: پیاده‌سازی ماژولار و بنچمارک معماری‌های بینایی ماشین در PyTorch

## ۱. معرفی مخزن

مخزن **`vision-architectures-pytorch`** یک چارچوب مهندسی و پژوهشی ماژولار برای پیاده‌سازی از پایه‌ی معماری‌های کلاسیک، مدرن و اختصاصی شبکه‌های عصبی کانولوشنی (CNN) در PyTorch است. هدف این پروژه، مقایسه کاملاً عادلانه و کنترل‌شده‌ی معماری‌ها تحت پایپ‌لاین آموزش یکپارچه (`Trainer`)، شتاب‌دهی سخت‌افزاری (`AMP FP16` و `channels_last`)، پروفایلینگ پیچیدگی محاسباتی (`Params`، `MACs`، `FLOPs`) و زمان‌سنجی واقعی روی GPU است.

* **توضیحات تفصیلی، تحلیل‌های ریاضی و جداول کامل هر آزمایش در ابتدای نوت‌بوک مربوطه مستندسازی شده‌اند.**

---

## ۲. بانک معماری‌ها (Architecture Zoo)

> *ساختار منعطف: برای افزودن مدل‌های جدید در آینده، کافی است تنها یک سطر جدید به این جدول اضافه کنید.*

| نام مدل | خانواده / پارادایم طراحی | ویژگی کلیدی معماری | پارامترها | FLOPs | بهترین دقت تست (Test Acc) |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **`KasraNet-V1-IncepNeXt`** | معماری ترکیبی اختصاصی | شاخه‌های چندمقیاسی (`1×1`, `3×3`, `5×5`) + کانولوشن گروهی (`C=16`) | `1.06M` | `315.33M` | **95.27%** *(CIFAR-10)* |
| **`ResNeXt-29`** | باقیمانده تجمیعی | تبدیل‌های باقیمانده گروهی (`depth=29, C=2`) | `1.42M` | `433.77M` | **95.15%** *(CIFAR-10)* |
| **`WideResNet-28-2`** | باقیمانده عریض | کانولوشن‌های متراکم $3\times3$ با دو برابر شدن عرض کانال‌ها | `1.47M` | `432.52M` | **94.45%** *(CIFAR-10)* |
| **`DualBranchResNet-28`** | چندمسیره سبک‌وزن | معماری دو شاخه‌ای موازی با تمرکز بر سرعت استنتاج بالا | `0.74M` | `222.37M` | **94.34%** *(CIFAR-10)* |
| **`VGG-16` (Light Head)** | عمیق همگن | پشته عمیق فیلترهای $3\times3$ + طبقه‌بند قیفی سبک برای ابعاد $32\times32$ | — | — | **89.30%** *(CIFAR-10)* |
| **`VGG-10` (Large Kernels)** | کم‌عمق با هسته بزرگ | استفاده از فیلترهای بزرگ ($11\times11, 7\times7, 5\times5$) در لایه‌های ابتدایی | — | — | **84.79%** *(CIFAR-10)* |
| **`LeNet-5 Classic`** | کلاسیک | کاهش ابعاد مکانی با لایه‌های صریح `Pooling` | — | — | **98.84%** *(MNIST)* |
| **`LeNet-5 Modern`** | کانولوشن گام‌دار | جایگزینی لایه‌های Pooling با کانولوشن‌های یادگیرنده گام‌دار (`Stride > 1`) | — | — | **98.68%** *(MNIST)* |

---

## ۳. فهرست نوت‌بوک‌های بنچمارک (Benchmark Index)

> *هر نوت‌بوک دارای مستندات کامل دوزبانه در سلول ابتدایی خود است.*

| شماره | نام نوت‌بوک | دیتاست | محور اصلی آزمایش | برنده و نتیجه کلیدی |
| :-: | :--- | :---: | :--- | :--- |
| **01** | **`01_mnist_lenet5_classic_vs_modern.ipynb`** | `MNIST` | تقابل **Pooling و Strided Conv** تحت بهینه‌سازهای `SGD` و `AdamW` با وزن اولیه یکسان | **`LeNet-5 Classic + SGD` (98.84%)** — برتری کمینه‌های هموار در SGD و کارایی بالاتر AdamW در فضای ناهموار مدل استرایدی. |
| **02** | **`02_CIFAR_10_vgg10_vs_vgg16.ipynb`** | `CIFAR-10` | تقابل **پشته‌های عمیق $3\times3$ (`VGG16`) و فیلترهای بزرگ (`VGG10`)** به تفکیک کلاس‌ها | **`VGG-16` (89.30% در برابر 84.79%)** — تخریب بافت‌های ظریف حیوانات توسط فیلترهای بزرگ (سقوط ۱۹.۴٪ در کلاس گربه). |
| **03** | **`03_CIFAR_10_resnet_resnext_kasranet.ipynb`** | `CIFAR-10` | بنچمارک **`KasraNet-V1`** در برابر **`WRN-28`، `ResNeXt-29` و `DualBranch-28`** (دقت، FLOPs و تأخیر GPU) | **`KasraNet-V1` (95.27% دقت با ۲۷٪ محاسبات کمتر)** — بالاترین دقت و کمترین خطا؛ پیشتازی `DualBranch` در سرعت (`20,645 img/s`). |

---

## ۴. خلاصه دیتاست‌ها و پایپ‌لاین داده

| دیتاست | ابعاد ورودی | تقسیم‌بندی (`Train / Val / Test`) | تکنیک‌های افزونه‌سازی و منظم‌سازی (Regularization) |
| :--- | :---: | :---: | :--- |
| **MNIST** | $1 \times 28 \times 28$ | تقسیم‌بندی استاندارد | نرمال‌سازی کانال‌ها، همگام‌سازی وزن‌های اولیه با `state_dict` |
| **CIFAR-10** | $3 \times 32 \times 32$ | `45,000 / 5,000 / 10,000` (لایه‌ای با `seed=42`) | `RandomCrop(32, pad=4)`, `RandomHorizontalFlip`, `RandomErasing`, `LabelSmoothing(0.1)` |

---

</div>
