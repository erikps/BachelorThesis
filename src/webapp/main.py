from fractions import Fraction
from functools import reduce
import math
import re
import secrets
import json

import subsetsum
from flask import Flask, flash, redirect, render_template, request, jsonify
from flask_wtf import FlaskForm
import networkx
import networkx as nx
import numpy as np
from wtforms import TextAreaField
from src.categoriser import CardBasedCategoriser, HCategoriser
from src.core import AttackInferenceProblem, WeightedArgumentationFramework

from src.dataset import AttackInferenceDataset
from src.solver import solve

app = Flask(__name__)
app.secret_key = secrets.token_urlsafe(16)

# application needs to be defined in this file so it
#   can be found by gunicorn
application = app


# proof of concept solver below
def set_degrees(G):
    num_nodes = G.number_of_nodes()
    current_degrees = np.array([G.nodes[i]["weight"]
                               for i in range(num_nodes)])

    error = 1

    while error != 0:
        temp_degrees = np.array([])

        for i in range(num_nodes):
            att_i = [x for x in G.predecessors(i)]
            temp_degrees = np.append(
                temp_degrees, G.nodes[i]["weight"] / (1 + sum([current_degrees[j] for j in att_i])))

        error = 0
        for i in range(num_nodes):
            error += temp_degrees[i] - current_degrees[i]

        current_degrees = temp_degrees

    nx.set_node_attributes(
        G, {i: deg for (i, deg) in enumerate(current_degrees)}, name="degree")
    return G


def verifier(G):
    result = True
    num_nodes = G.number_of_nodes()

    for i in range(num_nodes):
        if (G.nodes[i]["weight"]/G.nodes[i]["degree"] - 1) - sum([G.nodes[j]["degree"] for j in [x for x in G.predecessors(i)]]) != 0:
            result = False

    return result


def poof_of_concept_solver(G):
    num_nodes = G.number_of_nodes()

    all_weights = {Fraction(G.nodes[i]["weight"]) for i in range(num_nodes)}
    all_degrees = {Fraction(G.nodes[i]["ground_truth_degree"])
                   for i in range(num_nodes)}

    lcm = reduce(lambda x, y: x*y // math.gcd(x, y),
                 [f.denominator for f in all_weights.union(all_degrees)])

    M = [(Fraction(G.nodes[i]["ground_truth_degree"]) *
          lcm).numerator for i in range(num_nodes)]

    for i in range(num_nodes):
        w = Fraction(G.nodes[i]["weight"])
        deg = Fraction(G.nodes[i]["ground_truth_degree"])

        # find target value
        target = (((w - deg)/deg)*lcm).numerator

        if target != 0:  # This means that there are attackers
            # We add the edges

            for j in next(subsetsum.solutions(M, target)):
                G.add_edge(j, i)


def attack_inference_from_string(string: str) -> "AttackInferenceProblem":
    """ Generate an attack inference problem instance from the provided string. """
    ARGUMENT_REGEX = r"arg\(\s*(\w+)\s*\)\.\s"
    WEIGHT_REGEX = r"wgt\((\w)\s*,\s*(\d+\.\d+)\s*\)\.\s"
    DEGREE_REGEX = r"dgr\((\w)\s*,\s*(\d+\.\d+)\s*\)\.\s"

    string = string.replace(").", "). ")

    def byfirst(x):
        return x[0]

    weighted_argumentation_framework = WeightedArgumentationFramework()

    categoriser = CardBasedCategoriser() if string.find("CBCAT") != - \
        1 else HCategoriser()

    arguments = sorted(re.findall(ARGUMENT_REGEX, string))
    arguments = [int(arg) for arg in arguments]

    label_mapping = {argument: index for index,
                     argument in enumerate(arguments)}

    # intially weights are (index, float) tuples ...
    weights = sorted(
        [
            (label_mapping[int(argument)], float(weight))
            for argument, weight in re.findall(WEIGHT_REGEX, string)
        ],
        key=byfirst,
    )

    degrees = sorted(
        [
            (label_mapping[int(argument)], float(degree))
            for argument, degree in re.findall(DEGREE_REGEX, string)
        ],
        key=byfirst,
    )

    # ... then transformed to just weights, ordered by the index
    weights = [weight for _, weight in weights]
    degrees = [degree for _, degree in degrees]

    for argument, weight in zip(arguments, weights):
        weighted_argumentation_framework.graph.add_node(
            argument, weight=weight, predicted_degree=weight
        )

    for argument, degree in zip(arguments, degrees):
        weighted_argumentation_framework.graph.add_node(
            argument, ground_truth_degree=degree
        )

    return AttackInferenceProblem(weighted_argumentation_framework, categoriser, run_categoriser=False)


class GraphForm(FlaskForm):
    DEFAULT_TEXT = "arg(0).\narg(1).\nwgt(0,1.0).\nwgt(1,1.0).\ndgr(0,1.0).\ndgr(1,0.5).\n"
    graph = TextAreaField(
        "Input Your Attack Inference Problem Below", default=DEFAULT_TEXT)


@app.route("/")
def index():
    return render_template("index.jinja")


@app.route("/create-graph", methods=["GET", "POST"])
def create_graph():
    form = GraphForm()
    if form.validate_on_submit():
        # flash("Form Submitted")
        instance_string = form.data["graph"]
        instance = attack_inference_from_string(instance_string)
        json_instance = networkx.json_graph.node_link.node_link_data(
            instance.framework.graph)
        return redirect(f"/graph-viewer?graph={json.dumps(json_instance)}&instance={instance_string}")
    return render_template("create-graph.jinja", form=form)


@app.route("/graph-viewer")
def graph_viewer():
    return render_template("graph-viewer.jinja")


@app.route("/api/graph")
def api_graph():
    dataset = AttackInferenceDataset.example_dataset()
    graph = dataset[0].framework.graph

    return nx.json_graph.node_link.node_link_data(graph)


@app.route("/api/solver")
def run_solver():

    graph = attack_inference_from_string(
        request.args.get("instance")).framework.graph

    graph.remove_edges_from(list(graph.edges()))

    poof_of_concept_solver(graph)

    # solve(instance)
    return jsonify(list(graph.edges()))
