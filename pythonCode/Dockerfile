# Start with nvidia-enabled pytorch image
FROM nvcr.io/nvidia/pytorch:21.10-py3

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install tmux
RUN apt-get update
RUN apt-get install -y tmux

# Install python packages
RUN python -m pip install torch==2.2.*
RUN python -m pip install sentence-transformers==2.5.*
RUN python -m pip install wandb
RUN python -m pip install ipdb
RUN python -m pip install scikit-learn
RUN python -m pip install jupyter
RUN python -m pip install nltk
RUN python -m pip install pandas
RUN python -m pip install openpyxl
RUN python -m pip install pytorch_lightning

