# Graph Neural Networks for Gradual Semantics in Argumentation

This code can also be found at github: https://github.com/erikps/BachelorThesis.

## Setup

- `python -m venv venv`
- Enable the venv
- `python -m pip install -r requirements\_freeze.txt`
- `python -m pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.13.0.html`

## Run Web Application
- `python -m flask --app src.webapp.main:application run`

## Testing

In the root folder, run `python -m unittest discover tests -v` to execute all test cases.

## Architecture 

The application has four main modules: the core components, the environment and the solvers and then the web application.

The core components include the files `core.py, verifier.py, categoriser.py`. These handle how attack inference problems are handled, how semantics are applied and how verification of successful completion of the attack inference problems is done. The weighted argumentation frameworks are represented as `networkx.DiGraph` objects that are two-dimensional graphs wrapped in classes that seek to only represent valid states for weighted argumentation frameworks.

The second module is the environment which resides in the `envs` directory with support from the files `visualisation.py, dataset.py`. The datasets used are a subclass of the standard pytorch dataset base class. This has the advantage of enabling interoperability with third-party code. The environment is created as a subclass of the `gymnasium.Environment` base class. It can optionally be rendered using matplotlib using the `render\_mode="human"` setting. 

The third module comprises the solver and the GNN agent. The GNN agent can be found in the `learning` subfolder. The solver is located in the `solver.py`.

The fourth and last module is the web application located in the `webapp` folder. It makes use of the three other modules. It uses Flask to render the Jinja templates located in the `templates` subfolder. For the styling of the page `tailwindcss` is used.

There are also some unit tests located in the `tests` folder which is a subfolder of the root directory.