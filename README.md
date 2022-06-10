<div align="center">
  <img src="./assets/images/logo.svg" width="350"/>
</div>

<div align="center"><h1>DORA: Data-agnOstic Representation Analysis</h1>
<h5>A toolkit to explore Representation Spaces of Deep Neural Networks</h5>
<h6>PyTorch version</h6></div>
<div align="center">

[![Build status](https://github.com/lapalap/dora/workflows/build/badge.svg?branch=master&event=push)](https://github.com/lapalap/dora/actions?query=workflow%3Abuild)
[![Python Version](https://img.shields.io/pypi/pyversions/dora.svg)](https://pypi.org/project/dora/)
[![Dependencies Status](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)](https://github.com/lapalap/dora/pulls?utf8=%E2%9C%93&q=is%3Apr%20author%3Aapp%2Fdependabot)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Security: bandit](https://img.shields.io/badge/security-bandit-green.svg)](https://github.com/PyCQA/bandit)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/lapalap/dora/blob/master/.pre-commit-config.yaml)
[![Semantic Versions](https://img.shields.io/badge/%20%20%F0%9F%93%A6%F0%9F%9A%80-semantic--versions-e10079.svg)](https://github.com/lapalap/dora/releases)
[![License](https://img.shields.io/github/license/lapalap/dora)](https://github.com/lapalap/dora/blob/master/LICENSE)
![Coverage Report](assets/images/coverage.svg)

</div>

<div align="left">
<img src="./assets/images/About.svg" height="32"/>
</div>
<hr />

<div align="left">
<img src="./assets/images/Getting%20started.svg" height="32"/>
</div>
<hr />

<div align="left">
<img src="./assets/images/Installation.svg" height="32"/>
</div>
<hr />
<div align="left">
<img src="./assets/images/Contributing.svg" height="32"/>
</div>
<hr />
<div align="left">
<img src="./assets/images/Citation.svg" height="32"/>
</div>
<hr />

```bibtex
@misc{https://doi.org/10.48550/arxiv.2206.04530,
  doi = {10.48550/ARXIV.2206.04530},
  url = {https://arxiv.org/abs/2206.04530},
  author = {Bykov, Kirill and Deb, Mayukh and Grinwald, Dennis and Müller, Klaus-Robert and Höhne, Marina M. -C.},
  keywords = {Machine Learning (cs.LG), Artificial Intelligence (cs.AI), Computer Vision and Pattern Recognition (cs.CV), Machine Learning (stat.ML), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {DORA: Exploring outlier representations in Deep Neural Networks},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```

<div align="left">
<img src="./assets/images/License.svg" height="32"/>
</div>
<hr />

[![License](https://img.shields.io/github/license/lapalap/dora)](https://github.com/lapalap/dora/blob/master/LICENSE)

This project is licensed under the terms of the `GNU GPL v3.0` license. See [LICENSE](https://github.com/lapalap/dora/blob/master/LICENSE) for more details.


[comment]: <> (### Initialize your code)

[comment]: <> (1. Initialize `git` inside your repo:)

[comment]: <> (```bash)

[comment]: <> (cd dora && git init)

[comment]: <> (```)

[comment]: <> (2. If you don't have `Poetry` installed run:)

[comment]: <> (```bash)

[comment]: <> (make poetry-download)

[comment]: <> (```)

[comment]: <> (3. Initialize poetry and install `pre-commit` hooks:)

[comment]: <> (```bash)

[comment]: <> (make install)

[comment]: <> (make pre-commit-install)

[comment]: <> (```)

[comment]: <> (4. Run the codestyle:)

[comment]: <> (```bash)

[comment]: <> (make codestyle)

[comment]: <> (```)

[comment]: <> (5. Upload initial code to GitHub:)

[comment]: <> (```bash)

[comment]: <> (git add .)

[comment]: <> (git commit -m ":tada: Initial commit")

[comment]: <> (git branch -M main)

[comment]: <> (git remote add origin https://github.com/lapalap/dora.git)

[comment]: <> (git push -u origin main)

[comment]: <> (```)

[comment]: <> (### Set up bots)

[comment]: <> (- Set up [Dependabot]&#40;https://docs.github.com/en/github/administering-a-repository/enabling-and-disabling-version-updates#enabling-github-dependabot-version-updates&#41; to ensure you have the latest dependencies.)

[comment]: <> (- Set up [Stale bot]&#40;https://github.com/apps/stale&#41; for automatic issue closing.)

[comment]: <> (### Poetry)

[comment]: <> (Want to know more about Poetry? Check [its documentation]&#40;https://python-poetry.org/docs/&#41;.)

[comment]: <> (<details>)

[comment]: <> (<summary>Details about Poetry</summary>)

[comment]: <> (<p>)

[comment]: <> (Poetry's [commands]&#40;https://python-poetry.org/docs/cli/#commands&#41; are very intuitive and easy to learn, like:)

[comment]: <> (- `poetry add numpy@latest`)

[comment]: <> (- `poetry run pytest`)

[comment]: <> (- `poetry publish --build`)

[comment]: <> (etc)

[comment]: <> (</p>)

[comment]: <> (</details>)

[comment]: <> (### Building and releasing your package)

[comment]: <> (Building a new version of the application contains steps:)

[comment]: <> (- Bump the version of your package `poetry version <version>`. You can pass the new version explicitly, or a rule such as `major`, `minor`, or `patch`. For more details, refer to the [Semantic Versions]&#40;https://semver.org/&#41; standard.)

[comment]: <> (- Make a commit to `GitHub`.)

[comment]: <> (- Create a `GitHub release`.)

[comment]: <> (- And... publish 🙂 `poetry publish --build`)

[comment]: <> (## 🎯 What's next)

[comment]: <> (Well, that's up to you 💪🏻. I can only recommend the packages and articles that helped me.)

[comment]: <> (- [`Typer`]&#40;https://github.com/tiangolo/typer&#41; is great for creating CLI applications.)

[comment]: <> (- [`Rich`]&#40;https://github.com/willmcgugan/rich&#41; makes it easy to add beautiful formatting in the terminal.)

[comment]: <> (- [`Pydantic`]&#40;https://github.com/samuelcolvin/pydantic/&#41; – data validation and settings management using Python type hinting.)

[comment]: <> (- [`Loguru`]&#40;https://github.com/Delgan/loguru&#41; makes logging &#40;stupidly&#41; simple.)

[comment]: <> (- [`tqdm`]&#40;https://github.com/tqdm/tqdm&#41; – fast, extensible progress bar for Python and CLI.)

[comment]: <> (- [`IceCream`]&#40;https://github.com/gruns/icecream&#41; is a little library for sweet and creamy debugging.)

[comment]: <> (- [`orjson`]&#40;https://github.com/ijl/orjson&#41; – ultra fast JSON parsing library.)

[comment]: <> (- [`Returns`]&#40;https://github.com/dry-python/returns&#41; makes you function's output meaningful, typed, and safe!)

[comment]: <> (- [`Hydra`]&#40;https://github.com/facebookresearch/hydra&#41; is a framework for elegantly configuring complex applications.)

[comment]: <> (- [`FastAPI`]&#40;https://github.com/tiangolo/fastapi&#41; is a type-driven asynchronous web framework.)

[comment]: <> (Articles:)

[comment]: <> (- [Open Source Guides]&#40;https://opensource.guide/&#41;.)

[comment]: <> (- [A handy guide to financial support for open source]&#40;https://github.com/nayafia/lemonade-stand&#41;)

[comment]: <> (- [GitHub Actions Documentation]&#40;https://help.github.com/en/actions&#41;.)

[comment]: <> (- Maybe you would like to add [gitmoji]&#40;https://gitmoji.carloscuesta.me/&#41; to commit names. This is really funny. 😄)

[comment]: <> (## 🚀 Features)

[comment]: <> (### Development features)

[comment]: <> (- Supports for `Python 3.7` and higher.)

[comment]: <> (- [`Poetry`]&#40;https://python-poetry.org/&#41; as the dependencies manager. See configuration in [`pyproject.toml`]&#40;https://github.com/lapalap/dora/blob/master/pyproject.toml&#41; and [`setup.cfg`]&#40;https://github.com/lapalap/dora/blob/master/setup.cfg&#41;.)

[comment]: <> (- Automatic codestyle with [`black`]&#40;https://github.com/psf/black&#41;, [`isort`]&#40;https://github.com/timothycrosley/isort&#41; and [`pyupgrade`]&#40;https://github.com/asottile/pyupgrade&#41;.)

[comment]: <> (- Ready-to-use [`pre-commit`]&#40;https://pre-commit.com/&#41; hooks with code-formatting.)

[comment]: <> (- Type checks with [`mypy`]&#40;https://mypy.readthedocs.io&#41;; docstring checks with [`darglint`]&#40;https://github.com/terrencepreilly/darglint&#41;; security checks with [`safety`]&#40;https://github.com/pyupio/safety&#41; and [`bandit`]&#40;https://github.com/PyCQA/bandit&#41;)

[comment]: <> (- Testing with [`pytest`]&#40;https://docs.pytest.org/en/latest/&#41;.)

[comment]: <> (- Ready-to-use [`.editorconfig`]&#40;https://github.com/lapalap/dora/blob/master/.editorconfig&#41;, [`.dockerignore`]&#40;https://github.com/lapalap/dora/blob/master/.dockerignore&#41;, and [`.gitignore`]&#40;https://github.com/lapalap/dora/blob/master/.gitignore&#41;. You don't have to worry about those things.)

[comment]: <> (### Deployment features)

[comment]: <> (- `GitHub` integration: issue and pr templates.)

[comment]: <> (- `Github Actions` with predefined [build workflow]&#40;https://github.com/lapalap/dora/blob/master/.github/workflows/build.yml&#41; as the default CI/CD.)

[comment]: <> (- Everything is already set up for security checks, codestyle checks, code formatting, testing, linting, docker builds, etc with [`Makefile`]&#40;https://github.com/lapalap/dora/blob/master/Makefile#L89&#41;. More details in [makefile-usage]&#40;#makefile-usage&#41;.)

[comment]: <> (- [Dockerfile]&#40;https://github.com/lapalap/dora/blob/master/docker/Dockerfile&#41; for your package.)

[comment]: <> (- Always up-to-date dependencies with [`@dependabot`]&#40;https://dependabot.com/&#41;. You will only [enable it]&#40;https://docs.github.com/en/github/administering-a-repository/enabling-and-disabling-version-updates#enabling-github-dependabot-version-updates&#41;.)

[comment]: <> (- Automatic drafts of new releases with [`Release Drafter`]&#40;https://github.com/marketplace/actions/release-drafter&#41;. You may see the list of labels in [`release-drafter.yml`]&#40;https://github.com/lapalap/dora/blob/master/.github/release-drafter.yml&#41;. Works perfectly with [Semantic Versions]&#40;https://semver.org/&#41; specification.)

[comment]: <> (### Open source community features)

[comment]: <> (- Ready-to-use [Pull Requests templates]&#40;https://github.com/lapalap/dora/blob/master/.github/PULL_REQUEST_TEMPLATE.md&#41; and several [Issue templates]&#40;https://github.com/lapalap/dora/tree/master/.github/ISSUE_TEMPLATE&#41;.)

[comment]: <> (- Files such as: `LICENSE`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, and `SECURITY.md` are generated automatically.)

[comment]: <> (- [`Stale bot`]&#40;https://github.com/apps/stale&#41; that closes abandoned issues after a period of inactivity. &#40;You will only [need to setup free plan]&#40;https://github.com/marketplace/stale&#41;&#41;. Configuration is [here]&#40;https://github.com/lapalap/dora/blob/master/.github/.stale.yml&#41;.)

[comment]: <> (- [Semantic Versions]&#40;https://semver.org/&#41; specification with [`Release Drafter`]&#40;https://github.com/marketplace/actions/release-drafter&#41;.)

[comment]: <> (## Installation)

[comment]: <> (```bash)

[comment]: <> (pip install -U dora)

[comment]: <> (```)

[comment]: <> (or install with `Poetry`)

[comment]: <> (```bash)

[comment]: <> (poetry add dora)

[comment]: <> (```)



[comment]: <> (### Makefile usage)

[comment]: <> ([`Makefile`]&#40;https://github.com/lapalap/dora/blob/master/Makefile&#41; contains a lot of functions for faster development.)

[comment]: <> (<details>)

[comment]: <> (<summary>1. Download and remove Poetry</summary>)

[comment]: <> (<p>)

[comment]: <> (To download and install Poetry run:)

[comment]: <> (```bash)

[comment]: <> (make poetry-download)

[comment]: <> (```)

[comment]: <> (To uninstall)

[comment]: <> (```bash)

[comment]: <> (make poetry-remove)

[comment]: <> (```)

[comment]: <> (</p>)

[comment]: <> (</details>)

[comment]: <> (<details>)

[comment]: <> (<summary>2. Install all dependencies and pre-commit hooks</summary>)

[comment]: <> (<p>)

[comment]: <> (Install requirements:)

[comment]: <> (```bash)

[comment]: <> (make install)

[comment]: <> (```)

[comment]: <> (Pre-commit hooks coulb be installed after `git init` via)

[comment]: <> (```bash)

[comment]: <> (make pre-commit-install)

[comment]: <> (```)

[comment]: <> (</p>)

[comment]: <> (</details>)

[comment]: <> (<details>)

[comment]: <> (<summary>3. Codestyle</summary>)

[comment]: <> (<p>)

[comment]: <> (Automatic formatting uses `pyupgrade`, `isort` and `black`.)

[comment]: <> (```bash)

[comment]: <> (make codestyle)

[comment]: <> (# or use synonym)

[comment]: <> (make formatting)

[comment]: <> (```)

[comment]: <> (Codestyle checks only, without rewriting files:)

[comment]: <> (```bash)

[comment]: <> (make check-codestyle)

[comment]: <> (```)

[comment]: <> (> Note: `check-codestyle` uses `isort`, `black` and `darglint` library)

[comment]: <> (Update all dev libraries to the latest version using one comand)

[comment]: <> (```bash)

[comment]: <> (make update-dev-deps)

[comment]: <> (```)

[comment]: <> (<details>)

[comment]: <> (<summary>4. Code security</summary>)

[comment]: <> (<p>)

[comment]: <> (```bash)

[comment]: <> (make check-safety)

[comment]: <> (```)

[comment]: <> (This command launches `Poetry` integrity checks as well as identifies security issues with `Safety` and `Bandit`.)

[comment]: <> (```bash)

[comment]: <> (make check-safety)

[comment]: <> (```)

[comment]: <> (</p>)

[comment]: <> (</details>)

[comment]: <> (</p>)

[comment]: <> (</details>)

[comment]: <> (<details>)

[comment]: <> (<summary>5. Type checks</summary>)

[comment]: <> (<p>)

[comment]: <> (Run `mypy` static type checker)

[comment]: <> (```bash)

[comment]: <> (make mypy)

[comment]: <> (```)

[comment]: <> (</p>)

[comment]: <> (</details>)

[comment]: <> (<details>)

[comment]: <> (<summary>6. Tests with coverage badges</summary>)

[comment]: <> (<p>)

[comment]: <> (Run `pytest`)

[comment]: <> (```bash)

[comment]: <> (make test)

[comment]: <> (```)

[comment]: <> (</p>)

[comment]: <> (</details>)

[comment]: <> (<details>)

[comment]: <> (<summary>7. All linters</summary>)

[comment]: <> (<p>)

[comment]: <> (Of course there is a command to ~~rule~~ run all linters in one:)

[comment]: <> (```bash)

[comment]: <> (make lint)

[comment]: <> (```)

[comment]: <> (the same as:)

[comment]: <> (```bash)

[comment]: <> (make test && make check-codestyle && make mypy && make check-safety)

[comment]: <> (```)

[comment]: <> (</p>)

[comment]: <> (</details>)

[comment]: <> (<details>)

[comment]: <> (<summary>8. Docker</summary>)

[comment]: <> (<p>)

[comment]: <> (```bash)

[comment]: <> (make docker-build)

[comment]: <> (```)

[comment]: <> (which is equivalent to:)

[comment]: <> (```bash)

[comment]: <> (make docker-build VERSION=latest)

[comment]: <> (```)

[comment]: <> (Remove docker image with)

[comment]: <> (```bash)

[comment]: <> (make docker-remove)

[comment]: <> (```)

[comment]: <> (More information [about docker]&#40;https://github.com/lapalap/dora/tree/master/docker&#41;.)

[comment]: <> (</p>)

[comment]: <> (</details>)

[comment]: <> (<details>)

[comment]: <> (<summary>9. Cleanup</summary>)

[comment]: <> (<p>)

[comment]: <> (Delete pycache files)

[comment]: <> (```bash)

[comment]: <> (make pycache-remove)

[comment]: <> (```)

[comment]: <> (Remove package build)

[comment]: <> (```bash)

[comment]: <> (make build-remove)

[comment]: <> (```)

[comment]: <> (Delete .DS_STORE files)

[comment]: <> (```bash)

[comment]: <> (make dsstore-remove)

[comment]: <> (```)

[comment]: <> (Remove .mypycache)

[comment]: <> (```bash)

[comment]: <> (make mypycache-remove)

[comment]: <> (```)

[comment]: <> (Or to remove all above run:)

[comment]: <> (```bash)

[comment]: <> (make cleanup)

[comment]: <> (```)

[comment]: <> (</p>)

[comment]: <> (</details>)

[comment]: <> (## 📈 Releases)

[comment]: <> (You can see the list of available releases on the [GitHub Releases]&#40;https://github.com/lapalap/dora/releases&#41; page.)

[comment]: <> (We follow [Semantic Versions]&#40;https://semver.org/&#41; specification.)

[comment]: <> (We use [`Release Drafter`]&#40;https://github.com/marketplace/actions/release-drafter&#41;. As pull requests are merged, a draft release is kept up-to-date listing the changes, ready to publish when you’re ready. With the categories option, you can categorize pull requests in release notes using labels.)

[comment]: <> (### List of labels and corresponding titles)

[comment]: <> (|               **Label**               |  **Title in Releases**  |)

[comment]: <> (| :-----------------------------------: | :---------------------: |)

[comment]: <> (|       `enhancement`, `feature`        |       🚀 Features       |)

[comment]: <> (| `bug`, `refactoring`, `bugfix`, `fix` | 🔧 Fixes & Refactoring  |)

[comment]: <> (|       `build`, `ci`, `testing`        | 📦 Build System & CI/CD |)

[comment]: <> (|              `breaking`               |   💥 Breaking Changes   |)

[comment]: <> (|            `documentation`            |    📝 Documentation     |)

[comment]: <> (|            `dependencies`             | ⬆️ Dependencies updates |)

[comment]: <> (You can update it in [`release-drafter.yml`]&#40;https://github.com/lapalap/dora/blob/master/.github/release-drafter.yml&#41;.)

[comment]: <> (GitHub creates the `bug`, `enhancement`, and `documentation` labels for you. Dependabot creates the `dependencies` label. Create the remaining labels on the Issues tab of your GitHub repository, when you need them.)

[comment]: <> (## Credits [![🚀 Your next Python package needs a bleeding-edge project structure.]&#40;https://img.shields.io/badge/python--package--template-%F0%9F%9A%80-brightgreen&#41;]&#40;https://github.com/TezRomacH/python-package-template&#41;)

[comment]: <> (This project was generated with [`python-package-template`]&#40;https://github.com/TezRomacH/python-package-template&#41;)
