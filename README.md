# DeepTactics Arena

<div align="center">

![GitHub Workflow Status (with event)](https://img.shields.io/github/actions/workflow/status/CogitoNTNU/deeptactics-arena/ci.yml)
![GitHub top language](https://img.shields.io/github/languages/top/CogitoNTNU/deeptactics-arena)
![GitHub language count](https://img.shields.io/github/languages/count/CogitoNTNU/deeptactics-arena)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Project Version](https://img.shields.io/badge/version-0.0.1-blue)](https://img.shields.io/badge/version-0.0.1-blue)

<img src="docs/images/project-logo.webp" width="50%" alt="Cogito Project Logo" style="display: block; margin-left: auto; margin-right: auto;">
</div>

<details> 
<summary><b>📋 Table of contents </b></summary>

- [DeepTactics Arena](#deeptactics-arena)
  - [Description](#description)
  - [🛠️ Prerequisites](#️-prerequisites)
  - [Getting started](#getting-started)
  - [Usage](#usage)
    - [📖 Generate Documentation Site](#-generate-documentation-site)
  - [Testing](#testing)
  - [Team](#team)
    - [License](#license)

</details>

## Description

<!-- TODO: Provide a brief overview of what this project does and its key features. Please add pictures or videos of the application -->

## 🛠️ Prerequisites

<!-- TODO: In this section you put what is needed for the program to run.
For example: OS version, programs, libraries, etc.  

-->

- **Git**: Ensure that git is installed on your machine. [Download Git](https://git-scm.com/downloads)
- **Python 3.12**: Required for the project. [Download Python](https://www.python.org/downloads/)
- **UV**: Used for managing Python environments. [Install UV](https://docs.astral.sh/uv/getting-started/installation/)
- **Docker** (optional): For DevContainer development. [Download Docker](https://www.docker.com/products/docker-desktop)

## Getting started

1. **Clone the repository**:

   ```sh
   git clone https://github.com/CogitoNTNU/deeptactics-arena.git
   cd deeptactics-arena
   ```

1. **Install dependencies**:

   ```sh
   uv sync
   ```

1. **Configure environment variables**:

   This project uses environment variables for configuration. Copy the example environment file to create your own:

   ```sh
   cp .env.example .env
   ```

   Then edit the `.env` file to include your specific configuration settings.

1. **Set up pre commit** (only for development):

   ```sh
   uv run pre-commit install
   ```

## Usage

Run training with the default configuration (`configs/config.yaml`):

```bash
uv run python main.py
```

To use a specific config, pass it as a positional argument or with `--config`:

```bash
uv run python main.py connect_four.yaml
uv run python main.py --config chess.yaml
```

Available configurations in `configs/`: `config.yaml`, `chess.yaml`, `connect_four.yaml`, `tic-tac-toe.yaml`. You can also create your own by using any of these as a template.

Training progress is logged to [Weights & Biases](https://wandb.ai/) under the `deeptactics-arena` entity. To disable W&B logging, uncomment the `mode="disabled"` line in [main.py](main.py).


### 📖 Generate Documentation Site

To build and preview the documentation site locally:

```bash
uv run mkdocs build
uv run mkdocs serve
```

This will build the documentation and start a local server at [http://127.0.0.1:8000/](http://127.0.0.1:8000/) where you can browse the docs and API reference. Get the documentation according to the lastes commit on main by viewing the `gh-pages` branch on GitHub: [https://cogitontnu.github.io/deeptactics-arena/](https://cogitontnu.github.io/deeptactics-arena/).

## Testing

To run the test suite, run the following command from the root directory of the project:

```bash
uv run pytest --doctest-modules --cov=src --cov-report=html
```

## Team

This project would not have been possible without the hard work and dedication of all of the contributors. Thank you for the time and effort you have put into making this project a reality.

<table align="center">
    <tr>
        <td align="center">
            <a href="https://github.com/sverrenystad">
              <img src="https://github.com/sverrenystad.png?size=100" width="100px;" alt="Sverre Nystad"/><br />
              <sub><b>Sverre Nystad</b></sub>
            </a>
        </td>
        <td align="center">
            <a href="https://github.com/knolaisen">
              <img src="https://github.com/knolaisen.png?size=100" width="100px;" alt="Kristoffer Nohr Olaisen"/><br />
              <sub><b>Kristoffer Nohr Olaisen</b></sub>
            </a>
        </td>
        <td align="center">
            <a href="https://github.com/ludvigovrevik">
              <img src="https://github.com/ludvigovrevik.png?size=100" width="100px;" alt="Ludvig Øvrevik"/><br />
              <sub><b>Ludvig Øvrevik</b></sub>
            </a>
        </td>
        <td align="center">
            <a href="https://github.com/hako2807">
              <img src="https://github.com/hako2807.png?size=100" width="100px;" alt="Håkon Støren"/><br />
              <sub><b>Håkon Støren</b></sub>
            </a>
        </td>
        <td align="center">
            <a href="https://github.com/Parleenb">
              <img src="https://github.com/Parleenb.png?size=100" width="100px;" alt="Parleen Brar"/><br />
              <sub><b>Parleen Brar</b></sub>
            </a>
        </td>
        <td align="center">
            <a href="https://github.com/Vetlets05">
              <img src="https://github.com/Vetlets05.png?size=100" width="100px;" alt="Vetle Støren"/><br />
              <sub><b>Vetle Støren</b></sub>
            </a>
        </td>
        <td align="center">
            <a href="https://github.com/jacobNTNU">
              <img src="https://github.com/jacobNTNU.png?size=100" width="100px;" alt="Jacob Gullesen Hagen"/><br />
              <sub><b>Jacob Gullesen Hagen</b></sub>
            </a>
        </td>
        <td align="center">
            <a href="https://github.com/jessicaliu03">
              <img src="https://github.com/jessicaliu03.png?size=100" width="100px;" alt="Jessica Liu"/><br />
              <sub><b>Jessica Liu</b></sub>
            </a>
        </td>
    </tr>
</table>

### License

______________________________________________________________________

Distributed under the MIT License. See `LICENSE` for more information.
