# Base image tag is a build arg so it can be bumped without editing this file.
ARG BASE_TAG=26.11-py3
FROM nvcr.io/nvidia/pytorch:${BASE_TAG}

# The NGC image sets PIP_CONSTRAINT to pin its own packages. requirements.txt pins
# conflicting versions (e.g. numpy==1.26.4), so lift the constraint for this one step.
# torch/torchvision are commented out in requirements.txt and come from the base image.
COPY requirements.txt /tmp/requirements.txt
RUN PIP_CONSTRAINT= pip install --no-cache-dir -r /tmp/requirements.txt tensorboard

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

WORKDIR /workspace/minimind
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
