FROM python:3.9

# Creates a non-root user in the development image
# This user should correspond to the host user to avoid file permission errors
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID default \
    && useradd --uid $USER_UID --gid $USER_GID --create-home --shell /bin/bash default

USER default

WORKDIR /workspace/SCIP

COPY . .

# RUN pip install -e .[dev]
