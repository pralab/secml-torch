# SecML 2
New version of SecML

### Implement new functionalities
1. Create a card inside the project, by specifying what will be implemented (not how), and convert it to an issue. *Please be clear and concise*.
2. Create a new branch from main that will be used to implement the new functionality. Please provide a meaningful name while creating the new branch.
Do it from the GitHub web interface to avoid creating non-updated branches.
**If you create the branch locally, remember to pull the latest commit before branching!**
3. Checkout the new branch on your local machine, and implement unit tests and the intended functionality. Check out the “code style” section before pushing any commits.
4. Once the implementation of both unit testing and new functionality is completed, open a pull request to merge the newly created branch and main.
Please, write inside the description of the pull request a detailed changelog of the implemented functionalities and refactoring. Move the card you created at step 1 inside the “Review” column.
5. Now that the pull request has been open, wait until the maintainers review and approve your changes. DO NOT click on “Merge Pull Request” until somebody reviews your changes.
**Please remember that modified file outside the scope of the pull request will not be approved.**
6. Once this is done, you can move the card to “Done”. 
7. Congrats! You have included a new functionality in SecML2!

### Code style
We leverage “Black” (https://github.com/psf/black) as default format for SecML2.
Before pushing to any branch of SecML2, execute:

```
black .
```

from the source of the repository.
If you work with an IDE, follow this guide to configure “Black”: https://black.readthedocs.io/en/stable/integrations/editors.html 









