# Contributing to PyMC4

As a scientific community-driven software project, PyMC4 welcomes contributions
from users. This document describes how users can contribute to the PyMC4
project, and what workflow to follow to contribute as quickly and seamlessly as
possible.

There are four main ways of contributing to PyMC4 (in descending order of
difficulty or scope):

1. **Adding new or improved functionality** to the codebase: these contributions
   directly extend PyMC4's functionality.
2. **Fixing outstanding issues or bugs** with the codebase: these range from
   low-level software bugs to high-level design problems.
3. **Contributing to the documentation or examples**: improving the
   documentation is _just as important_ as improving the codebase itself.
4. **Submitting bug reports or feature requests** via the GitHub issue tracker:
   even something as simple as leaving a "thumbs up" reaction to issues that are
   relevant to you!

The first three types of contributions involve [opening a pull
request](#opening-a-pull-request), whereas the fourth involves [creating an
issue](#creating-an-issue).

Finally, it also helps us if you spread the word: reference the project from
your blog and articles, link to it from your website, or simply star it in
GitHub to say "I use it"!

## Creating an Issue

> Creating your first GitHub issue? Check out [the official GitHub
> documentation](https://help.github.com/articles/creating-an-issue/) on how to
> do that!

We appreciate being notified of problems with the existing PyMC4 codebase. We
prefer that issues be filed the on [GitHub issue
tracker](https://github.com/pymc-devs/pymc4/issues), rather than on social media
or by direct email to the developers.

Please check that your issue is not being currently addressed by other issues or
pull requests by using the GitHub search tool.

## Opening a Pull Request

While reporting issues is valuable, we welcome and encourage users to submit
patches for new or existing issues via pull requests (a.k.a. "PRs"). This is
especially the case for simple fixes, such as fixing typos or tweaking
documentation, which do not require a heavy investment of time and attention.

The preferred workflow for contributing to PyMC4 is to fork the [GitHub
repository](https://github.com/pymc-devs/pymc4/), clone it to your local
machine, and develop on a feature branch.

### Step-by-step instructions

1. Fork the [project repository](https://github.com/pymc-devs/pymc4/) by
   clicking on the `Fork` button near the top right of the main repository page.
   This creates a copy of the code under your GitHub user account.

2. Clone your fork of the PyMC4 repo from your GitHub account to your local
   computer, and add the base repository as an upstream remote.

   ```bash
   $ git clone git@github.com:<your GitHub handle>/pymc4.git
   $ cd pymc4
   $ git remote add upstream git@github.com:pymc-devs/pymc4.git
   ```

3. Check out a `feature` branch to contain your edits.

   ```bash
   $ git checkout -b my-feature
   ```

   Always create a new `feature` branch. It's best practice to never work on the
   `master` branch of any repository.

4. To set up your development environment, you can use the `make` command line
   utility. Depending on whether you want to develop using a [Python virtual
   environment](https://docs.python.org/3/library/venv.html), [conda
   environment](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
   or [Docker
   image](https://docs.docker.com/develop/develop-images/image_management/), you
   can run one of `make venv`, `make conda` or `make docker`, respectively, and
   follow the resulting instructions (in blue) after the setup is finished.

5. Develop the feature on your feature branch. This is the fun part!

6. Once you are done developing, run `make black` and `make check` from the root
   `pymc4/` directory to blackify, lint and test the codebase. If you like, you
   can run `make lint` and `make test` to lint and test separately. Work through
   and fix any lint errors or failing tests. Don't hesitate to reach out to us
   through the [GitHub issue tracker](https://github.com/pymc-devs/pymc4/issues)
   if you run into problems!

7. Add changed files using `git add` and then `git commit`:

   ```bash
   $ git add you_modified_file.py
   $ git commit
   ```

   to record your changes locally.

   Then push the changes to your GitHub account with:

   ```bash
   $ git push -u origin my-feature
   ```

8. Go to the GitHub web page of your fork of the PyMC4 repo. Click the `Pull
   request` button to send your changes to the project's maintainers for review.
   This will notify the PyMC4 developers.

## Code of Conduct

The PyMC4 project abides by the [Contributor
Covenant](https://www.contributor-covenant.org/). You can find our code of
conduct
[here](https://github.com/pymc-devs/pymc4/blob/master/CODE_OF_CONDUCT.md).
