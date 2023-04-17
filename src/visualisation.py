from typing import Tuple
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib

import random

from src.core import AttackInferenceProblem

class Visualiser:
    def __init__(self, problem: AttackInferenceProblem, render_interval: float = 0.05):
        def remove_fake_edges(graph: nx.DiGraph):
            # remove all edges that are neither actual or predicted
            edges = list(graph.edges.data())
            for a, b, data in edges:
                actual_edge = data["actual_edge"]
                predicted_edge = data["predicted_edge"]
                if not (predicted_edge):
                    graph.remove_edge(a, b)
                else:
                    predicted_edge = random.random() > 0.3
                    color = 1 if predicted_edge else 0
                    graph.add_edge(a, b, color=color, predicted_edge=predicted_edge)
            return graph

        self._render_interval = render_interval
        self._graph: nx.DiGraph = problem.framework.graph.copy()
        self._position = nx.circular_layout(self._graph)
        remove_fake_edges(self._graph)

        self.closed = False

        def on_close(*args):
            self.closed = True

        plt.connect("close_event", on_close)

    def flip_edge(self, edge: Tuple[str, str]):
        a, b = edge
        if (a, b) in self._graph.edges():
            self._graph.remove_edge(a, b)
        else:
            self._graph.add_edge(a, b)

    def draw(self, error=None):
        # respect decision of user to close window
        if not self.closed:
            # create a binary colormap; gray for actual edges, red for edges that are also predicted
            plt.clf()

            if error != None:
                plt.text(
                    0, 1, f"verifier result: {error}", transform=plt.gca().transAxes
                )

            options = {"width": 1, "pos": self._position, "with_labels": False}

            nx.draw(self._graph, **options)
            plt.pause(self._render_interval)
            plt.show(block=False)





    
    



