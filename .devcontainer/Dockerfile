FROM mcr.microsoft.com/devcontainers/python:3.9
WORKDIR /workspaces
RUN \
    # devcontainer dependencies
    apt-get update && \
    apt-get install -y git graphviz && \
    # MICrONS dependencies
    pip3 install git+https://github.com/AllenInstitute/em_coregistration.git@phase3 && \
    pip3 install seaborn ipykernel && \
    pip3 install git+https://github.com/datajoint/microns_phase3_nda.git &&\
    
    #con-con-models dependencies
    pip3 install standard-transform && \
    pip3 install scikit-learn
    
ENV DJ_HOST db.datajoint.com
ENV DJ_USER microns
ENV DJ_PASS microns2023
ENV DATABASE_PREFIX microns_