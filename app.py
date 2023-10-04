import gradio as gr
import numpy as np
import cv2
from tqdm import trange
from sklearn.cluster import KMeans

class KMeansClustering():
    def __init__(self, n_clusters=8, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X):
        self.inertia_ = float('inf')

        # random init of clusters
        idx = np.random.choice(range(X.shape[0]), self.n_clusters, replace=False)
        self.cluster_centers_ = X[idx]

        print(f'Training for {self.max_iter} epochs')
        epochs = trange(self.max_iter)
        for i in epochs:
            distances = X[:, np.newaxis, :] - self.cluster_centers_[np.newaxis, :, :]
            distances = np.linalg.norm(distances, axis=2)
            self.labels_ = np.argmin(distances, axis=1)
            new_inertia = np.sum(np.min(distances, axis=1) ** 2)

            epochs.set_description(f'Epoch-{i+1}, Inertia-{new_inertia}')

            if new_inertia < self.inertia_:
                self.inertia_ = new_inertia
            else:
                epochs.close()
                print('Early Stopping. Inertia has converged.')
                break

            self.cluster_centers_ = np.empty_like(self.cluster_centers_)
            for cluster in range(self.n_clusters):
                in_cluster = (self.labels_ == cluster)
                if np.any(in_cluster):
                    self.cluster_centers_[cluster] = np.mean(X[in_cluster], axis=0)
                else:
                    # cluster is empty, pick random point as next centroid
                    self.cluster_centers_[cluster] = X[np.random.randint(0, X.shape[0])]

        return self

    def predict(self, X):
        distances = X[:, np.newaxis, :] - self.cluster_centers_[np.newaxis, :, :]
        distances = np.linalg.norm(distances, axis=2)
        labels = np.argmin(distances, axis=1)
        return labels

    def fit_predict(self, X):
        return self.fit(X).labels_

def segment_image(image, model: KMeansClustering):
    w, b, c = image.shape
    image = image.reshape(w*b, c) / 255
    
    idx = np.random.choice(range(image.shape[0]), image.shape[0]//5, replace=False)
    image_subset = image[idx]
    model.fit(image_subset) # fit model on 20% sample of image
    
    labels = model.predict(image)
    return labels.reshape(w,b), model
    
def generate_outputs(image, implementation, num_colours):
    if implementation == 'custom':
        model = KMeansClustering(n_clusters=num_colours, max_iter=10)
    elif implementation == 'sk-learn':
        model = KMeans(n_clusters=num_colours, n_init='auto')
    label_map, model = segment_image(image, model)
    
    clustered_image = model.cluster_centers_[label_map]
    clustered_image = (clustered_image * 255).astype('uint8')
    clustered_image = cv2.medianBlur(clustered_image,5)
    edges = 255 - cv2.Canny(clustered_image, 0, 1)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    
    return [(edges, 'Coloring Page'), (clustered_image, 'Filled Picture')]

with gr.Blocks() as demo:
    gr.Markdown(
    """
    # image2coloringbook
    
    (image2coloringbook)[https://github.com/ShawnLJW/image2coloringbook] is a simple tool that converts an image into a coloring book.
    """)
    with gr.Row():
        with gr.Column():
            image = gr.Image()
            submit = gr.Button('Generate')
        with gr.Column():
            num_colours = gr.Slider(
                minimum=1,
                maximum=40,
                value=24,
                step=1,
                label='Number of colours'
            )
            implementation = gr.Dropdown(
                choices=['sk-learn','custom'],
                value='sk-learn',
                label='Implementation'
            )
    with gr.Row():
        output = gr.Gallery(preview=True)
            
    submit.click(
        generate_outputs,
        inputs=[image, implementation, num_colours],
        outputs=[output]
    )

if __name__ == '__main__':
    demo.launch()