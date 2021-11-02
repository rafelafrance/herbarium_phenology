# Setting up a development environment
This is a slew of personal preferences, but I'll tell you how I do it. As you become comfortable with things I fully expect you to change it to suit your needs.

## OS
I code exclusively on Linux (Pop!_OS), so that will taint my selections. If you're a Mac or Windows person I can help some but not a lot.

## Editor
I don't have a strong preference here, I typically bounce between 3 editors:
- Neovim: It's the old standby, I use it for quick edits.
- Visual Studio Code: I use it when I'm not coding Python, and I'll often use it for Python too. It's a very good free IDE.
- PyCharm Community Edition: It's also a great free IDE geared towards Python and R. I use it roughly half the time when I'm coding Python. The one thing that has kept me from switching to VS Code for everything are the automated refactorings, PyCharm has excellent refactoring tools.

## Project setup
All the details are contained in the `dev_env.bash` script.
- I use virtual environments and not Conda. Personal preference. I suppose we could containerize the setup, but I don't want to lock people into my setup just yet.
  - If you use a virtual environment then you need to activate it every time you run anything in the project. So you need to `source ./.venv/bin/activate` at the start of every session (just once), and `deactivate` when you are done.
- I've started using the pre-commit hooks for auditing and modifying code.
- I use GitHub actions for continuous integration (CI). Please don't break the build.
- The other development libraries are a solid selection, use them or not, your call.

## Collaboration
- We'll work via pull requests at first. Once we get organized you'll get full access to the repository.
- We'll also assign tasks using GitHub issues.
- We'll use the branch and merge workflow. With the small team I don't think we need to get fancy.
- I've been pretty lax about my git commit messages and creating issues. I'll need to up my game here.

You can often get in touch with me via g-chat.
