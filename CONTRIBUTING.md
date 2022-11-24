
## Contributing Guidelines

You are **Awesome!** Thank you for your Interest in Contributing to this Project 🤗
For Contributions we strictly follow [Github Flow](https://guides.github.com/introduction/flow/).

## Contents
- [Setting Up the Project](#user-content-setting-up-the-project)
- [Contributing](#user-content-contributing)


### Setting Up the Project
- The Project works seamlessly on Python version >= `3.8`
- `git clone https://github.com/Praful932/Tf-Rec.git` - Clone the Repo Directly
- `cd Tf-Rec/`
- Setup conda/miniconda if you don't have already
- Install conda-lock - `conda install --channel=conda-forge conda-lock`
- Install an environment called `tf-rec` which has all the requirements for - `conda-lock install --name tf-rec conda-lock.yml`
- You're good to Go!
- Run tests to make sure everything's working as expected `python -m unittest discover`

### Contributing
- Please go through [Github Flow](https://guides.github.com/introduction/flow/), if not already. :)
- Take up an [Issue](https://github.com/Praful932/Tf-Rec/issues) or [Raise](https://github.com/Praful932/Tf-Rec/issues/new) one.
- Discuss your proposed changes.
- If your changes are approved, do the changes in branch `[branch_name]`.
- Run tests
- `pre-commit run --all-files`, `python -m unittest discover`
- Fix if any test fails.
- Still in branch `[branch_name].`
- **Stage and Commit only the required files.**
- `git push --set-upstream origin [branch_name]`
- You'll get a link to Create a Pull Request.
- Fill the PR Details.
- That's it!
