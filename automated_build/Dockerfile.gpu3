FROM floydhub/dl-docker:gpu_temp2

MAINTAINER Sai Soundararaj <saip@outlook.com>

ARG THEANO_VERSION=rel-0.8.2
ARG TENSORFLOW_VERSION=0.8.0
ARG TENSORFLOW_ARCH=gpu
ARG KERAS_VERSION=1.0.3
ARG LASAGNE_VERSION=v0.1
ARG TORCH_VERSION=latest
ARG CAFFE_VERSION=master


# Install the latest versions of nn, cutorch, cunn, cuDNN bindings and iTorch
RUN luarocks install nn > /dev/null && \
	luarocks install cutorch > /dev/null && \
	luarocks install cunn > /dev/null&& \
	\
	cd /root && git clone https://github.com/soumith/cudnn.torch.git && cd cudnn.torch && \
	git checkout R4 && \
	luarocks make > /dev/null && \
	\
	cd /root && git clone https://github.com/facebook/iTorch.git && \
	cd iTorch && \
	luarocks make > /dev/null


# Set up notebook config
COPY jupyter_notebook_config.py /root/.jupyter/

# Jupyter has issues with being run directly: https://github.com/ipython/ipython/issues/7062
COPY run_jupyter.sh /root/

# Expose Ports for TensorBoard (6006), Ipython (8888)
EXPOSE 6006 8888

WORKDIR "/root"
CMD ["/bin/bash"]
