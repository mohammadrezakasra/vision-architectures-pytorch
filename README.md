<div align="center">

# 👁️ Vision Architectures in PyTorch | معماری‌های بینایی ماشین در پایتورچ

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-CNN%20Architectures-blue?style=for-the-badge)

🌐 **[English Version](#-english-documentation)** | 🇮🇷 **[نسخه فارسی](#-مستندات-فارسی)**

</div>

---

<a name="-english-documentation"></a>
<h2 align="center">🌐 English Documentation</h2>

An engineering-focused repository dedicated to implementing, modifying, and benchmarking foundational and custom Convolutional Neural Network (CNN) architectures from scratch in PyTorch. 

Instead of merely importing `torchvision` models, this repository explores **why** specific architectural inductive biases work—comparing classic designs against modern adaptations through strictly controlled experiments.

### 📂 Project Structure

```text
vision-architectures-pytorch/
├── checkpoints/                          # Saved best model weights (.pth)
├── models/
│   └── cnn/
│       ├── lenet5_classic.py             # Classic Pooling-based LeNet-5
│       ├── lenet5_modern.py              # Strided-Convolution LeNet-5
│       ├── vgg_16_classic.py             # Deep 3x3 stack VGG-16 (w/ BatchNorm & Light FC)
│       ├── costom_vgg_10.py              # Hybrid AlexNet-VGG with large initial kernels
│       └── kasranet_v1_incepnext.py      # [Upcoming] Custom Inception-ResNeXt Hybrid
├── notebooks/
│   ├── 01_mnist_lenet5_classic_vs_modern.ipynb
│   └── 02_CIFAR_10_vgg10_vs_vgg16.ipynb
├── utils/
│   └── trainer.py                        # Modular training, validation & checkpoint engine
└── README.md
```

### 🧠 Implemented Architectures (`models/cnn/`)

| Model File | Class Name | Key Architectural Characteristics |
| :--- | :--- | :--- |
| `lenet5_classic.py` | `LeNet5Classic` | Traditional architecture utilizing spatial pooling layers for translation invariance. |
| `lenet5_modern.py` | `LeNet5Modern` | Replaces explicit pooling layers with learnable **Strided Convolutions** for spatial downsampling. |
| `vgg_16_classic.py` | `VGG16` | Pure 3x3 deep convolutional stack enhanced with `BatchNorm2d` and a custom **Light Funnel Classifier** (`512 -> 256 -> num_classes`) to prevent overfitting on low-resolution datasets. |
| `costom_vgg_10.py` | `VGG10` | Experimental hybrid combining **AlexNet-style large kernels** (11x11, 7x7, 5x5) in early stages with VGG-style 3x3 blocks in deeper stages. |
| `kasranet_v1_incepnext.py` | `KasraNet_V1_IncepNeXt` | *(In Progress)* First generation of the custom `KasraNet` family, merging multi-scale **Inception** branches, **ResNeXt** grouped convolutions (cardinality), and **ResNet** skip connections. |

### 📊 Experimental Benchmarks & Notebooks

#### 1️⃣ Experiment 01: Pooling vs. Strided Convolutions & Optimizer Dynamics
* **Notebook:** `01_mnist_lenet5_classic_vs_modern.ipynb`
* **Dataset:** MNIST (28x28 Grayscale)
* **Experimental Control:** Identical weight initialization (`state_dict` synchronization) across all 4 runs for 50 epochs.

| Architecture | Optimizer | Test Accuracy | Key Takeaway |
| :--- | :---: | :---: | :--- |
| **LeNet-5 Classic** | **SGD** | **98.84%** | **Winner:** Flat minima via SGD + strong inductive bias of explicit Pooling. |
| LeNet-5 Classic | AdamW | 98.81% | Fast initial convergence, slightly sharper minima than SGD. |
| LeNet-5 Modern | AdamW | 98.68% | Adaptive learning rates handle the complex loss landscape of strided convs better. |
| LeNet-5 Modern | SGD | 98.56% | Struggles to optimize strided filters without adaptive momentum. |

#### 2️⃣ Experiment 02: Small Deep Filter Stacks vs. Large Early Kernels
* **Notebook:** `02_CIFAR_10_vgg10_vs_vgg16.ipynb`
* **Dataset:** CIFAR-10 (32x32 RGB) with stratified 90/10 Train-Validation split and Data Augmentation (`RandomCrop`, `RandomHorizontalFlip`, `ColorJitter`).
* **Experimental Goal:** Empirically test the core VGG hypothesis—whether stacked 3x3 filters outperform fewer layers with large receptive fields (11x11, 7x7, 5x5).

| Metric / Class | VGG16 (3x3 Deep Stack) | VGG10 (Large Kernels) | Delta (Δ) |
| :--- | :---: | :---: | :---: |
| **Overall Test Accuracy** | **89.30%** | **84.79%** | **+4.51%** |
| **Test Loss** | **0.4219** | **0.5750** | **-0.1531** |
| **Rigid Objects** (e.g., *Car / Ship*) | 94.70% / 94.70% | 93.80% / 91.80% | High performance on both |
| **Deformable** (*Cat*) | **85.30%** | **65.90%** | **+19.40%** |
| **Deformable** (*Bird*) | **83.90%** | **75.50%** | **+8.40%** |

##### 💡 Key Engineering Insights from Experiment 02:
1. **High-Frequency Texture Destruction:** While `VGG10` performs well on rigid geometric vehicles (*plane*: 94.00%, *car*: 93.80%), it suffers a catastrophic drop on deformable animals (*cat* drops by **19.4%**). Applying an 11x11 filter on a tiny 32x32 image acts as a destructive low-pass filter, wiping out fine-grained fur and facial textures in the very first layer.
2. **Non-Linearity Density:** Three stacked 3x3 layers cover the same 7x7 receptive field as a single 7x7 layer, but pass the feature maps through **three `ReLU` activations** instead of one, vastly improving decision boundary expressiveness while requiring fewer parameters.
3. **Funnel Classifier Regularization:** Replacing the original 4096-dim VGG linear layers with a compact bottleneck (`512 -> 256 -> 10`) paired with `Dropout(0.5)` and `BatchNorm2d` curbed the generalization gap from ~15% down to ~8%.

### 🛠️ Training Pipeline & Utilities (`utils/trainer.py`)
The repository includes a custom, reusable `Trainer` engine featuring:
* Automated training and validation loops.
* Stratified validation set evaluation without data augmentation leakage.
* Automatic best-checkpoint serialization (`checkpoints/<model_name>.pth`) based on validation accuracy.
* Per-class accuracy breakdown for granular error analysis.

### 🚀 Roadmap: The `KasraNet` Custom Family
- [x] **Phase 1:** Classic & Modernized Baselines (`LeNet-5`, `VGG16`, `VGG10`)
- [ ] **Phase 2:** `KasraNet_V1_IncepNeXt` — Custom hybrid combining Inception multi-branching, ResNeXt grouped convolutions, and ResNet residual connections.
- [ ] **Phase 3:** `KasraNet_V2_SEMobile` — Lightweight architecture combining Depthwise Separable Convolutions (MobileNet) with Squeeze-and-Excitation (SENet) channel attention.

---
---

<div dir="rtl" align="right" style="line-height: 1.8;">

<a name="-مستندات-فارسی"></a>
<h2 align="center">🇮🇷 مستندات فارسی</h2>

این ریپازیتوری یک پروژه مهندسی متمرکز بر پیاده‌سازی از صفر (From Scratch)، بهینه‌سازی و بنچمارک معماری‌های پایه و سفارشی شبکه‌های عصبی پیچشی (CNN) در PyTorch است.

به‌جای استفاده آماده از مدل‌های `torchvision`، هدف اصلی این پروژه بررسی عمیق و تجربی **چرایی** عملکرد سوگیری‌های استقرایی (Inductive Biases) در معماری‌های مختلف است؛ به‌طوری که طراحی‌های کلاسیک در شرایط کاملاً کنترل‌شده و ایزوله با اقتباس‌های مدرن مقایسه می‌شوند.

### 🧠 معماری‌های پیاده‌سازی‌شده (`models/cnn/`)

| فایل مدل | نام کلاس | ویژگی‌های کلیدی معماری |
| :--- | :--- | :--- |
| `lenet5_classic.py` | `LeNet5Classic` | معماری سنتی مبتنی بر لایه‌های Pooling صریح برای ایجاد تغییرناپذیری انتقالی (Translation Invariance). |
| `lenet5_modern.py` | `LeNet5Modern` | حذف لایه‌های Pooling و جایگزینی آن‌ها با **کانولوشن‌های گام‌دار (Strided Convolutions)** قابل یادگیری جهت کاهش ابعاد مکانی. |
| `vgg_16_classic.py` | `VGG16` | پشته عمیق و خالص از فیلترهای 3x3 مجهز به `BatchNorm2d` و یک **طبقه‌بند قیفی سبک (Light Funnel Classifier)** با ساختار `512 -> 256 -> num_classes` برای جلوگیری از بیش‌برازش در رزولوشن‌های پایین. |
| `costom_vgg_10.py` | `VGG10` | مدل هیبریدی آزمایشی با ترکیب **هسته‌های بزرگ به سبک AlexNet** (ابعاد 11x11، 7x7 و 5x5) در لایه‌های ابتدایی و بلوک‌های 3x3 به سبک VGG در لایه‌های عمیق‌تر. |
| `kasranet_v1_incepnext.py` | `KasraNet_V1_IncepNeXt` | *(در دست توسعه)* نسل اول خانواده مدل‌های اختصاصی `KasraNet`؛ تلفیق شاخه‌های چندمقیاسی **Inception**، کانولوشن‌های گروهی **ResNeXt** (مفهوم Cardinality) و اتصالات میان‌بر **ResNet**. |

### 📊 بنچمارک‌ها و آزمایش‌های تجربی (Notebooks)

#### 1️⃣ آزمایش اول: تقابل Pooling و Strided Convolution در کنار دینامیک بهینه‌سازها
* **نوت‌بوک:** `01_mnist_lenet5_classic_vs_modern.ipynb`
* **دیتاست:** MNIST (تصاویر خاکستری 28x28)
* **کنترل آزمایش:** یکسان‌سازی کامل وزن‌های اولیه (`state_dict`) در هر 4 مدل و آموزش به مدت 50 اپوک.

| معماری | بهینه‌ساز | دقت تست (Test Acc) | دستاورد و تحلیل کلیدی |
| :--- | :---: | :---: | :--- |
| **LeNet-5 Classic** | **SGD** | **98.84%** | **برنده:** نشستن در کمینه‌های هموار (Flat Minima) توسط SGD + سوگیری استقرایی قوی لایه Pooling. |
| LeNet-5 Classic | AdamW | 98.81% | همگرایی اولیه سریع‌تر، اما متوقف شدن در کمینه‌های تیزتر نسبت به SGD. |
| LeNet-5 Modern | AdamW | 98.68% | نرخ یادگیری تطبیقی، فضای بهینه‌سازیِ پیچیده حاصل از کانولوشن‌های گام‌دار را بهتر مدیریت می‌کند. |
| LeNet-5 Modern | SGD | 98.56% | دشواری در بهینه‌سازی فیلترهای Strided بدون مومنتوم تطبیقی. |

#### 2️⃣ آزمایش دوم: پشته‌های عمیق با فیلترهای کوچک در برابر فیلترهای بزرگ ابتدایی
* **نوت‌بوک:** `02_CIFAR_10_vgg10_vs_vgg16.ipynb`
* **دیتاست:** CIFAR-10 (تصاویر رنگی 32x32) به همراه جداسازی اصولی و Stratified داده‌های اعتبارسنجی (90/10) و اعمال Data Augmentation (`RandomCrop`, `RandomHorizontalFlip`, `ColorJitter`).
* **هدف آزمایش:** ارزیابی تجربی فرضیه اصلی مقاله VGG — آیا پشته‌ای از فیلترهای کوچک 3x3 عملکرد بهتری نسبت به لایه‌های کمتر با میدان دید بزرگ (11x11، 7x7 و 5x5) دارد؟

| معیار / کلاس | VGG16 (پشته عمیق 3x3) | VGG10 (هسته‌های بزرگ) | اختلاف (Δ) |
| :--- | :---: | :---: | :---: |
| **دقت کل تست (Overall Acc)** | **89.30%** | **84.79%** | **+4.51%** |
| **خطای تست (Test Loss)** | **0.4219** | **0.5750** | **-0.1531** |
| **اشیای صلب** (*ماشین / کشتی*) | 94.70% / 94.70% | 93.80% / 91.80% | عملکرد بالا در هر دو مدل |
| **اشیای منعطف** (*گربه - Cat*) | **85.30%** | **65.90%** | **+19.40%** |
| **اشیای منعطف** (*پرنده - Bird*) | **83.90%** | **75.50%** | **+8.40%** |

##### 💡 استنتاج‌های مهندسی از آزمایش دوم:
1. **تخریب بافت‌های فرکانس بالا توسط فیلترهای بزرگ:** در حالی که `VGG10` روی وسایل نقلیه با هندسه صلب عملکرد خوبی دارد (هواپیما: 94.00%، ماشین: 93.80%)، در تشخیص حیوانات که دارای بافت‌های ظریف و شکل منعطف هستند دچار سقوط آزاد می‌شود (افت **19.4 درصدی** در کلاس گربه). اعمال فیلتر 11x11 روی تصویر کوچک 32x32 مانند یک فیلتر پایین‌گذر مخرب عمل کرده و جزئیات ظریف صورت و موی حیوانات را در همان لایه اول محو می‌کند.
2. **چگالی توابع غیرخطی (Non-Linearity Density):** سه لایه متوالی 3x3 دقیقاً میدان دیدی معادل یک لایه 7x7 ایجاد می‌کنند، اما نگاشت‌های ویژگی را از **سه تابع فعال‌ساز `ReLU`** عبور می‌دهند (به‌جای یک تابع در حالت 7x7)؛ این امر قدرت تفکیک‌پذیری مرزهای تصمیم‌گیری را با مصرف پارامترهای کمتر به شدت ارتقا می‌دهد.
3. **تنظیم‌کنندگی طبقه‌بند قیفی (Funnel Classifier Regularization):** جایگزینی لایه‌های سنگین 4096 تایی VGG با یک گلوگاه فشرده (`512 -> 256 -> 10`) در کنار `Dropout(0.5)` و `BatchNorm2d`، شکاف بیش‌برازش (Generalization Gap) را از حدود 15% به حدود 8% کاهش داد.

### 🛠️ پایپ‌لاین آموزش و ابزارها (`utils/trainer.py`)
این پروژه شامل یک موتور آموزش ماژولار و قابل استفاده مجدد (`Trainer`) با قابلیت‌های زیر است:
* حلقه‌های خودکار آموزش (Train) و اعتبارسنجی (Validation).
* ارزیابی ایزوله روی داده‌های Validation بدون نشت تغییرات Data Augmentation.
* ذخیره‌سازی خودکار بهترین وزن‌های مدل (`checkpoints/<model_name>.pth`) بر مبنای بالاترین دقت اعتبارسنجی.
* گزارش‌گیری تفکیک‌شده دقت برای تک‌تک کلاس‌ها جهت تحلیل دقیق خطاها.

### 🚀 نقشه راه: خانواده مدل‌های اختصاصی `KasraNet`
- [x] **فاز ۱:** پیاده‌سازی و بنچمارک مدل‌های پایه کلاسیک و مدرن (`LeNet-5`, `VGG16`, `VGG10`)
- [ ] **فاز ۲:** معماری `KasraNet_V1_IncepNeXt` — مدل هیبریدی سفارشی با ترکیب شاخه‌های موازی Inception، کانولوشن‌های گروهی ResNeXt و اتصالات میان‌بر ResNet.
- [ ] **فاز ۳:** معماری `KasraNet_V2_SEMobile` — شبکه سبک‌وزن با ترکیب کانولوشن‌های تفکیک‌پذیر عمقی (MobileNet) و مکانیزم توجه کانالی (Squeeze-and-Excitation).

</div>
