FROM bethgelab/deeplearning:cuda10.0-cudnn7

RUN pip install --upgrade pip

RUN pip install jupyterlab && \
    jupyter serverextension enable --py jupyterlab --sys-prefix

RUN pip3 install h5py

RUN pip3 install torch torchvision datajoint --upgrade

RUN pip3 install --upgrade seaborn

RUN chmod 777 /root

CMD ["/usr/local/bin/run_jupyterlab.sh"]
