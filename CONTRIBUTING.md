# SecMLT: Contribution Guide for Adversarial Machine Learning and Robustness Evaluation

SecMLT is an open-source Python library for Adversarial Machine Learning and robustness evaluation. We welcome contributions from the research community to expand its capabilities, improve its functionality, or add new features. In this guide, we will discuss how to contribute to SecMLT through forks, pull requests, and code formatting using Black.

## Prerequisites

Before contributing to SecMLT:

1. Familiarize yourself with the library by reviewing the [official documentation](https://secml-torch.readthedocs.io/en/latest/) and exploring the existing codebase.
2. Install the required dependencies (refer to [the installation guide](https://secml-torch.readthedocs.io/en/latest/installation.html)).

## Setting up your development environment

To contribute to SecMLT, follow these steps:

1. **Fork the repository**: Go to the [SecMLT GitHub page](https://github.com/pralab/secml-torch) and click "Fork" in the upper-right corner. This will create a
copy of the SecMLT repository under your GitHub account.

1. **Clone your forked repository**: Clone your forked repository to your local machine using `git clone` command:
   ```bash
   git clone <your_forked_repo_URL> secmlt
   ```
2. **Set up remote repositories**: Add the original SecMLT repository as an upstream remote and set the tracking branch to be `master`:
   ```bash
   cd secmlt
   git remote add upstream <original_repo_URL>
   git fetch upstream
   git checkout master --track upstream/master
   ```

## Making changes

1. Create a new branch for your feature, bug fix, or documentation update:
   ```bash
   git checkout -c <new_branch_name>
   ```
2. Make the necessary changes to the codebase (add features, fix bugs, improve documentation, etc.). Be sure to write clear and descriptive commit messages.
3. Test your changes locally using appropriate testing frameworks and tools.

## Formatting your code

SecMLT uses Black for ensuring high-quality code formatting. Before submitting a pull request, make sure your code adheres to the SecMLT style guide by running the following command in the root directory:
   ```bash
   black .
   ```
4. If there are no issues with your code, commit the changes using the `git add` command and push them to your forked repository:
   ```bash
   git add .
   git commit -m "Your commit message"
   git push origin <new_branch_name>
   ```

## Submitting a pull request

1. Go to your forked repository on GitHub and click the "New pull request" button.
2. Choose the branch you've created as the source branch, and select `master` as the target branch.
3. Review the changes you're submitting and write a clear and descriptive pull request title and description.
4. Submit your pull request by clicking "Create pull request".
5. The SecMLT maintainers will review your pull request, provide feedback, or merge it into the main repository as appropriate.

We appreciate your contributions to SecMLT! If you have any questions or need assistance during the process, please don't hesitate to reach out to us on GitHub or other communication channels.