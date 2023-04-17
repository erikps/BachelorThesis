from flask import Flask, render_template
import networkx

from src.dataset import AttackInferenceDataset

app = Flask(__name__)

# application needs to be defined in this file so it
#   can be found by gunicorn
application = app


@app.route("/")
def index():
    return render_template("index.jinja")


@app.route("/about")
def about():
    return render_template("about.jinja")


@app.route("/graph-viewer")
def graph_viewer():
    return render_template("graph-viewer.jinja")


@app.route("/model")
def model():
    return render_template("model.jinja")


@app.route("/api/graph")
def api_graph():
    dataset = AttackInferenceDataset.example_dataset()
    graph = dataset[0].framework.graph

    return networkx.json_graph.node_link.node_link_data(graph)
