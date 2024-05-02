FROM continuumio/miniconda3:24.3.0-0

# see https://stackoverflow.com/questions/55123637/activate-conda-environment-in-docker
# and https://pythonspeed.com/articles/activate-conda-dockerfile/#working

RUN conda config --set auto_activate_base false

# make docker use bash instead of sh
SHELL ["/bin/bash", "--login", "-c"]


# set up flusion files
WORKDIR /flusion

COPY code ./code
COPY data-raw ./data-raw
COPY environment.yml .
COPY docker_entrypoint.sh /usr/local/bin/

# make entrypoint script executable
RUN chmod u+x /usr/local/bin/docker_entrypoint.sh

# create conda environment
RUN conda env create -f environment.yml

# update .bashrc to activate the flusion conda environment
# see https://stackoverflow.com/questions/76215066/how-to-make-docker-container-automatically-activate-a-conda-environment
RUN echo "conda activate flusion" >> ~/.bashrc

# set entrypoint
ENTRYPOINT ["/usr/local/bin/docker_entrypoint.sh"]
