# How to install a virtual environment
- install anaconda
    - make sure you include in path
- create anaconda virtual env with:
    - run this command: `conda create --name myenv python=3.9` in cmd
    - activate your environment `conda activate <path_to_env>`
- run this command in cmd: `conda env update -f environment.yml`

# How to be able to run tests
- apart from installing pytest by installing the env as described above you also need to
- add a file named `custom.pth` inside the path:
    - `<path to your env>/lib/<your python>/site-packages/custom.pth`
    - inside that file you should indicate the path to your project 
        - i.e. the `/Nepheline_syenite` folder

# How to run
- just run main.py