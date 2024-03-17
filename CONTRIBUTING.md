# SecMLT: Contribution Guide

SecMLT is an open-source Python library for Adversarial Machine Learning and robustness evaluation. We welcome contributions from the research community to expand its capabilities, improve its functionality, or add new features. In this guide, we will discuss how to contribute to SecMLT through forks, pull requests, and code formatting using Ruff.

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

In our project, we leverage [Ruff](https://docs.astral.sh/ruff/) and [pre-commit](https://pre-commit.com) to enhance code quality and streamline the development process.
Ruff is a static code linter, while Pre-commit is a framework for defining pre-commit hooks.

### Using Ruff

Ruff is integrated into our project to perform code linting.
It helps ensure adherence to coding standards, identifies potential bugs, and enhances overall code quality. Here's how to use Ruff:

1. **Installation**: Make sure you have Ruff installed in your development environment. You can install it via pip:
    ```
    pip install ruff
    ```

2. **Running Ruff**: To analyze your codebase using Ruff, navigate to the project directory and run the following command:
    ```
    ruff check
    ```
    Ruff will analyze the codebase and provide feedback on potential issues and areas for improvement.

### Using Pre-commit

Pre-commit is employed to automate various tasks such as code formatting, linting, and ensuring code consistency across different environments. We use it to enforce Ruff formatting *before* commit.
Here's how to utilize Pre-commit:

1. **Installation**: Ensure that Pre-commit is installed in your environment. You can install it using pip:
    ```
    pip install pre-commit
    ```

2. **Configuration**: The project includes a `.pre-commit-config.yaml` file that specifies the hooks to be executed by Pre-commit. These hooks can include tasks such as code formatting, static analysis, and more.

3. **Installation of Hooks**: Run the following command in the project directory to install the Pre-commit hooks:
    ```
    pre-commit install
    ```
    This command will set up the hooks specified in the configuration file to run automatically before each commit.

4. **Running Pre-commit**: Whenever you make changes and attempt to commit them, Pre-commit will automatically execute the configured hooks. If any issues are found, Pre-commit will prevent the commit from proceeding and provide feedback on the detected issues.

### Contributing with your Code

When contributing code to the project, follow these guidelines to ensure a smooth and efficient contribution process:

1. **Run Ruff and Pre-commit Locally**: Before making a pull request, run Ruff and Pre-commit locally to identify and fix potential issues in your code.

2. **Address Ruff and Pre-commit Warnings**: If Ruff or Pre-commit identifies any issues, address them before submitting your code for review. This ensures that the codebase maintains high standards of quality and consistency.

3. **Document Changes**: Clearly document any changes you make, including the rationale behind the changes and any potential impact on existing functionality.

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
