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
- I've started using the `black` code linter/formatter. It's OK but if you have a strong preference like `autopep8` I'd be willing to switch.
- I use GitHub actions for continuous integration (CI). It runs all tests and does some light code quality analysis via `flake8`. I really should change the linter once we've decided on which one fits our needs best. GitHub actions is going to stay... Please don't break the build.
- The other development libraries are a solid selection. Use them or not. Your call.
- Pytorch is the machine learning framework that I've decided to use. I don't like TensorFlow, and we're not switching to it any time soon. Then again, having used Flux.jl, I could see us using something else in the future.
- I hope you have a reasonably sized NVIDIA GPU, because it will make training models so much easier. I have a fairly puny 8 GB card, and it still makes all the difference.
  - If you have an appropriate GPU card then you'll also need to install the CUDA and cuDNN libraries.
  - If you don't have a GPU, then you can still use a CPU it'll just be orders of magnitude slower for some things like model training.

## Collaboration
- We'll work via pull requests at first. Once we get organized you'll get full access to the repository.
- We'll also assign tasks using GitHub issues.
- We'll use the branch and merge workflow. With the small team I don't think we need to get fancy.
- I've been pretty lax about my oen git commit messages and creating issues. I'll need to up my game here.

You can often get in touch with me via g-chat.
