"""
Visualize one saved graph per class with clickable buttons.

Buttons:
- Prev Class
- Next Class
- Prev Sample
- Next Sample
- Toggle Image
"""

from pathlib import Path
import torch
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import cv2
import matplotlib.pyplot as plt


GRAPH_ROOT = "data/processed/graphs/train"


def load_graph(graph_file):
    return torch.load(graph_file, weights_only=False)


def collect_graphs_by_class(root_dir):
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Folder not found: {root}")

    class_to_files = {}
    for class_dir in sorted([d for d in root.iterdir() if d.is_dir()]):
        files = sorted(class_dir.glob("*.pt"))
        if files:
            class_to_files[class_dir.name] = files

    if not class_to_files:
        raise ValueError(f"No graph files found in {root}")

    return class_to_files


def draw_pose_graph(ax, graph, title="", show_node_ids=False):
    x = graph.x.cpu().numpy()
    edge_index = graph.edge_index.cpu().numpy()

    xs = x[:, 0]
    ys = x[:, 1]

    for i in range(edge_index.shape[1]):
        src = edge_index[0, i]
        dst = edge_index[1, i]
        ax.plot([xs[src], xs[dst]], [ys[src], ys[dst]], linewidth=1)

    ax.scatter(xs, ys, s=30)

    if show_node_ids:
        for i, (px, py) in enumerate(zip(xs, ys)):
            ax.text(px, py, str(i), fontsize=7)

    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.grid(True)
    ax.set_title(title)


class GraphBrowser:
    def __init__(self, root_dir, show_node_ids=False):
        self.class_to_files = collect_graphs_by_class(root_dir)
        self.class_names = list(self.class_to_files.keys())
        self.class_idx = 0
        self.sample_indices = {class_name: 0 for class_name in self.class_names}
        self.image_mode = False
        self.show_node_ids = show_node_ids

        self.fig = plt.figure(figsize=(12, 7))
        self.fig.subplots_adjust(bottom=0.2)

        self.ax_main1 = None
        self.ax_main2 = None

        # Buttons
        ax_prev_class = self.fig.add_axes([0.10, 0.05, 0.12, 0.06])
        ax_next_class = self.fig.add_axes([0.24, 0.05, 0.12, 0.06])
        ax_prev_sample = self.fig.add_axes([0.40, 0.05, 0.12, 0.06])
        ax_next_sample = self.fig.add_axes([0.54, 0.05, 0.12, 0.06])
        ax_toggle = self.fig.add_axes([0.72, 0.05, 0.16, 0.06])

        self.btn_prev_class = Button(ax_prev_class, "Prev Class")
        self.btn_next_class = Button(ax_next_class, "Next Class")
        self.btn_prev_sample = Button(ax_prev_sample, "Prev Sample")
        self.btn_next_sample = Button(ax_next_sample, "Next Sample")
        self.btn_toggle = Button(ax_toggle, "Toggle Image")

        self.btn_prev_class.on_clicked(self.prev_class)
        self.btn_next_class.on_clicked(self.next_class)
        self.btn_prev_sample.on_clicked(self.prev_sample)
        self.btn_next_sample.on_clicked(self.next_sample)
        self.btn_toggle.on_clicked(self.toggle_image)

        self.refresh()

    def get_current(self):
        class_name = self.class_names[self.class_idx]
        files = self.class_to_files[class_name]
        sample_idx = self.sample_indices[class_name] % len(files)
        graph_file = files[sample_idx]
        graph = load_graph(graph_file)
        return class_name, sample_idx, graph_file, graph

    def refresh(self):
        class_name, sample_idx, graph_file, graph = self.get_current()

        print(f"\nClass: {class_name}")
        print(f"Sample: {sample_idx + 1}/{len(self.class_to_files[class_name])}")
        print(f"File: {graph_file}")

        self.fig.clf()
        self.fig.subplots_adjust(bottom=0.2)

        if self.image_mode:
            self.ax_main1 = self.fig.add_subplot(1, 2, 1)
            self.ax_main2 = self.fig.add_subplot(1, 2, 2)

            image_path = getattr(graph, "image_path", "")
            if image_path:
                image = cv2.imread(image_path)
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    self.ax_main1.imshow(image)
                    self.ax_main1.set_title(f"Original Image\n{class_name}")
                    self.ax_main1.axis("off")
                else:
                    self.ax_main1.text(0.5, 0.5, "Could not load image", ha="center", va="center")
                    self.ax_main1.axis("off")
            else:
                self.ax_main1.text(0.5, 0.5, "No image path stored", ha="center", va="center")
                self.ax_main1.axis("off")

            draw_pose_graph(
                self.ax_main2,
                graph,
                title=f"Graph\n{class_name} | {graph.sample_name}",
                show_node_ids=self.show_node_ids
            )
        else:
            self.ax_main1 = self.fig.add_subplot(1, 1, 1)
            draw_pose_graph(
                self.ax_main1,
                graph,
                title=f"{class_name} | {graph.sample_name}",
                show_node_ids=self.show_node_ids
            )

        self.fig.suptitle(
            f"Class: {class_name} | Sample {sample_idx + 1}/{len(self.class_to_files[class_name])}",
            fontsize=12
        )

        # Recreate buttons after clf()
        ax_prev_class = self.fig.add_axes([0.10, 0.05, 0.12, 0.06])
        ax_next_class = self.fig.add_axes([0.24, 0.05, 0.12, 0.06])
        ax_prev_sample = self.fig.add_axes([0.40, 0.05, 0.12, 0.06])
        ax_next_sample = self.fig.add_axes([0.54, 0.05, 0.12, 0.06])
        ax_toggle = self.fig.add_axes([0.72, 0.05, 0.16, 0.06])

        self.btn_prev_class = Button(ax_prev_class, "Prev Class")
        self.btn_next_class = Button(ax_next_class, "Next Class")
        self.btn_prev_sample = Button(ax_prev_sample, "Prev Sample")
        self.btn_next_sample = Button(ax_next_sample, "Next Sample")
        self.btn_toggle = Button(ax_toggle, "Toggle Image")

        self.btn_prev_class.on_clicked(self.prev_class)
        self.btn_next_class.on_clicked(self.next_class)
        self.btn_prev_sample.on_clicked(self.prev_sample)
        self.btn_next_sample.on_clicked(self.next_sample)
        self.btn_toggle.on_clicked(self.toggle_image)

        plt.draw()

    def prev_class(self, event):
        self.class_idx = (self.class_idx - 1) % len(self.class_names)
        self.refresh()

    def next_class(self, event):
        self.class_idx = (self.class_idx + 1) % len(self.class_names)
        self.refresh()

    def prev_sample(self, event):
        class_name = self.class_names[self.class_idx]
        self.sample_indices[class_name] = (self.sample_indices[class_name] - 1) % len(self.class_to_files[class_name])
        self.refresh()

    def next_sample(self, event):
        class_name = self.class_names[self.class_idx]
        self.sample_indices[class_name] = (self.sample_indices[class_name] + 1) % len(self.class_to_files[class_name])
        self.refresh()

    def toggle_image(self, event):
        self.image_mode = not self.image_mode
        self.refresh()


def main():
    browser = GraphBrowser(GRAPH_ROOT, show_node_ids=False)
    plt.show()


if __name__ == "__main__":
    main()