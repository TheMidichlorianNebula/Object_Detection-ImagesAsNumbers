# Teaching Computers to See: Images as Numbers & Object Detection

This project has two short Jupyter notebooks and one helper file. Together they show how a computer "sees" a picture and how a pre-trained AI model can find objects in a photo, draw boxes around them, and read out what it found.

You don't need a machine learning background to follow along. Each section below explains the idea in plain language first, then what the code does.

---

## What's in this folder

| File | What it is | In one sentence |
|---|---|---|
| `Images_as_numerical_data.ipynb` | Notebook | Shows that an image is really a big grid of numbers, and how to simplify it before feeding it to an AI model. |
| `Object_detection.ipynb` | Notebook | Uses a ready-made AI model to find objects in a photo, turns the result into a small web app, and makes the computer describe the photo out loud. |
| `helper.py` | Python file | A small toolbox of reusable functions the object detection notebook calls: drawing boxes, writing a sentence summary, and hiding noisy warnings. |

**Image files the notebooks expect** (keep them in the same folder as the notebooks):

- `seven.png`: a handwritten number 7 (used in the "images as numbers" notebook)
- `OD_test.png`: a street photo with people and bicycles (the test photo for object detection)
- `class_object.PNG`, `Object_detection.png`, `pipeline.PNG`: illustration diagrams shown inside the object detection notebook

**Suggested order:** start with `Images_as_numerical_data.ipynb`, then move on to `Object_detection.ipynb`.

---

## Part 1: Images as Numbers (`Images_as_numerical_data.ipynb`)

### The big idea

To you, a photo is a picture. To a computer, it's a spreadsheet of numbers.

Every image is made of tiny coloured squares called **pixels**. Each pixel stores a number (or a few numbers) that says how bright or what colour it is. An AI model never "looks" at the picture. It only ever reads these numbers. This notebook makes that visible.

### Step by step

**1. Load the image and check its size**

The notebook opens `seven.png` and prints its shape: `(1480, 1490, 4)`. Read that as:

- **1480**: rows of pixels (the image's height)
- **1490**: columns of pixels (the image's width)
- **4**: numbers stored for each pixel, called **channels**

> Tip: Python image libraries list size as *(height, width, channels)*, meaning rows first, then columns. It's easy to mix those up.

**2. Colour channels**

Each pixel's colour is built by mixing a few ingredients:

| Channel | Index in code | Meaning |
|---|---|---|
| Red | `0` | How much red is in the pixel |
| Green | `1` | How much green |
| Blue | `2` | How much blue |
| Alpha | `3` | How see-through the pixel is (transparency) |

A normal photo has 3 channels (RGB). A PNG with transparency has 4 (RGBA). The notebook pulls out each channel on its own and displays it, so you can see that each one is just its own grid of numbers.

**3. Peeking at the actual numbers**

`image[:, :, 0]` shows the red channel as a raw table of values between **0.0** (none) and **1.0** (full). `image[100, 200, 0]` grabs the single number at row 100, column 200.

**4. Converting to grayscale (black and white)**

`cv2.cvtColor(..., COLOR_RGB2GRAY)` combines the colour channels into one brightness value per pixel. The image goes from **3D** `(1480, 1490, 4)` to **2D** `(1480, 1490)`.

*Why bother?* For recognising a handwritten digit, colour doesn't matter. Dropping it gives the model a quarter as many numbers to process, which makes it faster and simpler. This is called **dimensionality reduction**.

**5. Pixels are the model's inputs**

The notebook builds a tiny 5×5 image by hand from 25 numbers (0 = black, 255 = white) and displays it. This is the key lesson: **every pixel value becomes one input ("feature") to the model.** A 5×5 image means 25 inputs. The original 1480×1490 image means over 2.2 million inputs.

**6. Shrinking the image (downscaling)**

`cv2.resize(gray_image, (28, 28))` shrinks the 7 down to 28×28 pixels, which is 784 numbers instead of 2.2 million. It still clearly looks like a 7.

28×28 grayscale is the standard size used by **MNIST**, the famous dataset of handwritten digits that many beginner models (like CNNs, or *convolutional neural networks*) are trained on. So this step gets the image into the shape such a model expects.

### Key takeaways

- An image is a grid of numbers. Colour images are several grids stacked together.
- Grayscale plus resizing makes images smaller and simpler without losing what matters.
- Those numbers are exactly what gets fed into a machine learning model.

---

## Part 2: Object Detection (`Object_detection.ipynb`)

### The big idea

**Object detection** means answering two questions about a photo:

1. **What** is in it? (a bicycle, a person, a backpack…)
2. **Where** is each one? (shown as a box drawn around it)

This is the technology behind self-driving cars, security cameras, robots, and medical scan analysis.

Instead of training a model from scratch (which takes huge amounts of data and computing power), this notebook uses a **pre-trained model** that someone else already trained, downloaded from **Hugging Face**, a free online library of AI models.

### The model: DETR

The notebook uses `facebook/detr-resnet-50`, released by Facebook AI in 2020 ([paper](https://arxiv.org/abs/2005.12872), [model page](https://huggingface.co/facebook/detr-resnet-50)).

- **DETR** stands for *DEtection TRansformer*. It uses the same "transformer" technology behind modern language AI.
- **ResNet-50** is the part that scans the image and picks out visual patterns (edges, shapes, textures).
- It was trained on the COCO dataset, so it recognises about 80 everyday object types: people, cars, bikes, animals, furniture, and so on.

### Step by step

**1. Build the pipeline**

```python
od_pipe = pipeline("object-detection", "facebook/detr-resnet-50")
```

A **pipeline** is Hugging Face's shortcut that bundles everything into one call: preparing the image, running the model, and cleaning up the answer. The first time you run this, it downloads the model (a few hundred MB).

**2. Run it on a photo**

The notebook opens `OD_test.png` and passes it to the pipeline. The output is a list where each item is one object found:

```python
{'score': 0.95, 'label': 'bicycle', 'box': {'xmin': 84, 'ymin': 358, 'xmax': 347, 'ymax': 646}}
```

| Field | Meaning |
|---|---|
| `label` | What the model thinks the object is |
| `score` | How confident it is (0.95 = 95% sure) |
| `box` | The corners of the rectangle around the object, in pixels: left edge (`xmin`), top (`ymin`), right (`xmax`), bottom (`ymax`) |

On the test photo, the model found 6 bicycles, 5 people, 2 backpacks and 1 handbag.

**3. Draw the boxes**

`render_results_in_image()` (from `helper.py`) draws a green box around each object with a red label like `bicycle: 95.0%`.

**4. Turn it into a web app with Gradio**

**Gradio** builds a simple web page around a Python function with no HTML or web coding needed. The notebook wraps "detect objects, then draw boxes" into one function, `get_pipeline_prediction()`, and Gradio gives it:

- an **upload box** for your own image, and
- an **output panel** showing the labelled result.

`demo.launch(share=True)` starts the app (locally at `http://127.0.0.1:7860`) and tries to create a temporary public link you can send to others. When you're done, `demo.close()` shuts it down.

**5. Audio narration: making the computer describe the photo**

This section chains two AI models together:

```
Photo → [Object detector] → list of objects → [helper: summarise] → sentence → [Text-to-speech] → spoken audio
```

- `summarize_predictions_natural_language()` turns the raw list into a sentence: *"In this image, there are six bicycles two backpacks five persons and one handbag."*
- A **text-to-speech (TTS)** model then reads that sentence aloud, and the audio plays right in the notebook.

This is the basic idea behind accessibility tools that describe surroundings to people with low vision.

The notebook contains two TTS options:

- `kakao-enterprise/vits-ljs`: small and fast, one clear English voice
- `suno/bark`: larger and more natural-sounding, but much slower and heavier to download

Because the `suno/bark` cell runs second, it replaces the first one. Run only the one you want.

---

## Part 3: The Helper File (`helper.py`)

These functions keep the notebooks short and readable by tucking the fiddly code away.

| Function | What it does |
|---|---|
| `load_image_from_url(url)` | Downloads an image from a web link and opens it, so you can test photos straight from the internet. |
| `render_results_in_image(image, results)` | Takes the photo and the model's results, draws a green rectangle and a red "label: confidence%" tag for each object, and returns the new image. |
| `summarize_predictions_natural_language(results)` | Counts each type of object and writes a plain-English sentence, using the `inflect` library to spell numbers as words ("six" instead of "6"). |
| `ignore_warnings()` | Hides harmless but noisy warning messages from the AI libraries so the output stays readable. |

---

## Setup: how to run this yourself

**1. Install Python 3.10+ and Jupyter** (Anaconda or VS Code with the Jupyter extension both work).

**2. Install the libraries:**

```bash
pip install numpy matplotlib opencv-python pillow requests inflect
pip install transformers torch timm gradio
```

| Library | Used for |
|---|---|
| `numpy` | Working with grids of numbers |
| `matplotlib` | Displaying images and drawing boxes |
| `opencv-python` (`cv2`) | Grayscale conversion and resizing |
| `pillow` (`PIL`) | Opening and handling image files |
| `transformers` | Downloading and running the Hugging Face models |
| `torch` | The engine the models run on (PyTorch) |
| `timm` | Extra image-model building blocks DETR depends on |
| `gradio` | The quick web app |
| `inflect` | Turning numbers into words |
| `requests` | Downloading images from URLs |

The notebook also mentions `phonemizer` and `espeak`. Some older text-to-speech setups needed these, but you usually don't need them for the models used here.

**3. Put all files in one folder**: both notebooks, `helper.py`, and the image files listed at the top.

**4. Open a notebook and run the cells top to bottom** (Shift + Enter on each cell).

**Heads-up:**

- **First run is slow.** Models download the first time (DETR is about 160 MB; `suno/bark` is several GB). After that they're cached on your computer.
- **Internet is needed** to download models and for Gradio's share link.
- **NumPy version clash:** installing `opencv-python` may upgrade NumPy to 2.x, which breaks older packages like `gensim`. If you see a dependency warning, a separate virtual environment for this project avoids the problem.

---

## Known quirks in the code

These don't stop anything from running, but they're worth knowing about:

- **Image resize doesn't stick.** `raw_image.resize((500, 400))` displays a resized copy but doesn't save it, so the model actually runs on the full-size image. To really resize it, write `raw_image = raw_image.resize((500, 400))`.
- **A comment mislabels the blue channel.** In the images notebook, the cell for channel `2` says "green channel". It is actually blue.
- **The summary sentence has no commas**, and it pluralises by adding "s", so you get "persons" instead of "people".
- **Transparency channel:** `OD_test.png` is RGBA (4 channels). DETR handles it here, but if a model complains, convert first with `raw_image.convert("RGB")`.

---

## Glossary

| Term | Plain-English meaning |
|---|---|
| **Pixel** | One tiny square of colour in an image |
| **Channel** | One "layer" of colour information (red, green, blue, or transparency) |
| **Grayscale** | Black-and-white image: one brightness number per pixel |
| **Feature** | One piece of input information the model uses; here, one pixel value |
| **Dimensionality reduction** | Shrinking the amount of data while keeping what matters |
| **Pre-trained model** | An AI model already trained by someone else, ready to use |
| **Pipeline** | A one-line shortcut that handles every step of running a model |
| **Bounding box** | The rectangle drawn around a detected object |
| **Confidence score** | How sure the model is about a prediction, from 0 to 1 (0% to 100%) |
| **Transformer** | A type of AI architecture, originally built for language, now used for images too |
| **CNN** | Convolutional neural network, a model designed to spot patterns in images |
| **Text-to-speech (TTS)** | AI that turns written text into spoken audio |
| **Gradio** | A Python library for turning a model into a clickable web app |
| **Hugging Face** | A website and library that hosts thousands of free, ready-to-use AI models |

---

## Try it yourself

- Run the detector on your own photos, or on web images with `load_image_from_url()`.
- Browse other object detection models on the [Hugging Face Hub](https://huggingface.co/models?pipeline_tag=object-detection&sort=trending) and swap the model name into the pipeline.
- Filter out low-confidence results (for example, keep only objects with `score > 0.9`) and see how the boxes change.
- Resize the handwritten 7 to other sizes (14×14, 64×64) and see how much detail survives.
