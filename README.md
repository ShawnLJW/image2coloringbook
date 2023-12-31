---
title: image2coloringbook
emoji: ✏️
colorFrom: red
colorTo: indigo
sdk: gradio
sdk_version: "3.46.0"
app_file: app.py
pinned: false
---

# image2coloringbook

![image2coloringbook](https://raw.githubusercontent.com/ShawnLJW/random-assets-hosting/main/image2coloringbook.png)

A simple tool that converts an image into a coloring book. It runs on a custom implementation of the k-means clustering algorithm by default but comes with the option to use scikit-learn's implementation.

This [Colab notebook](https://colab.research.google.com/drive/1S91AsP2XHUKuxtUBEaFlboWd8ScAndcz?usp=sharing) explains how the coloring books are generated.

## Usage

The application is available as a [Hugging Face space](https://shawnljw-image2coloringbook.hf.space).

To run the application locally, clone this repo and open the directory in your terminal.

Install all requirements with pip:

```shell
pip install -r requirements.txt
```

Once all requirements are installed, you can run the web ui with:

```shell
gradio app.py
```