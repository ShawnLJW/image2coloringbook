# image2coloringbook

![image2coloringbook](assets/image2coloringbook.png)

A simple tool that converts an image into a coloring book. It runs on a custom implementation of the k-means clustering algorithm by default but comes with the option to use scikit-learn's implementation.

This [Colab notebook](https://colab.research.google.com/drive/1S91AsP2XHUKuxtUBEaFlboWd8ScAndcz?usp=sharing) explains how the coloring books are generated.

## Usage

To run the application locally, cloned this repo and open the directory in your terminal.

Install all requirements with pip:

```shell
pip install -r requirements.txt
```

Once all requirements are installed, you can run the web ui with:

```shell
gradio app.py
```