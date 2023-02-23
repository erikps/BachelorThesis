import networkx as nx
import matplotlib.pyplot as plt
import matplotlib

import random

from core import AttackInferenceProblem
from dataset import AttackInferenceDataset

def draw_problem_instance(instance: AttackInferenceProblem):
    def remove_fake_edges(graph: nx.DiGraph):
        # remove all edges that are neither actual or predicted
        edges = list(graph.edges.data())
        for a, b, data in edges:
            actual_edge = data["actual_edge"]
            predicted_edge = data["predicted_edge"]
            if not (actual_edge or predicted_edge):
                graph.remove_edge(a, b)
            else: 
                predicted_edge = random.random() > 0.3
                color = 1 if predicted_edge else 0 
                graph.add_edge(a, b, color=color, predicted_edge=predicted_edge) 
        return graph

    graph = instance.framework.graph.copy()
    remove_fake_edges(graph)


    # setup node and edge colors based on weight and actual / predicted status respectivley
    node_colors = [weight for _, weight in graph.nodes.data("weight")]
    edge_colors = [int(color) for _, _, color in graph.edges.data("color")]
    print(edge_colors)

    # create a binary colormap; gray for actual edges, red for edges that are also predicted
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#816868", "#D93e3e"])

    position = nx.spring_layout(graph, seed=0)
    options = {
            "node_color": node_colors,
            "edge_color": edge_colors,
            "width": 1,
            "cmap": plt.cm.Reds,
            "edge_cmap": cmap,
            "with_labels": False
            }
    nx.draw(graph, **options)
    plt.draw()
    plt.show()


if __name__ == "__main__":
    instance = AttackInferenceDataset()[0]
    draw_problem_instance(instance)

