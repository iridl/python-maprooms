FROM centos:7.9.2009 AS common

COPY docker/CentOS-Vault.repo /etc/yum.repos.d/
RUN rm /etc/yum.repos.d/CentOS-Base.repo

# Dependencies that are used in both build stage and final stage

# mesa-libGL is a dependency of opencv that is not packaged for conda (see
# https://conda-forge.org/docs/maintainer/knowledge_base.html#core-dependency-tree-packages-cdts)
# This step is reused from cache in second stage.
RUN yum install -y httpd mesa-libGL

RUN rm -r /etc/httpd/conf.d /etc/httpd/conf.modules.d


FROM common AS build

# httpd-devel and gcc are to support building mod_wsgi from source.
RUN yum install -y httpd-devel gcc

# miniconda
RUN curl -L \
    https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh \
    -o /miniconda-installer.sh && \
    bash /miniconda-installer.sh -b -p /conda
RUN eval "$('/conda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)" && \
    conda config --set auto_update_conda False && \
    conda install -c conda-forge conda==22.9.0 conda-lock==1.2.1


# build conda environment
COPY conda-lock.yml /build/conda-lock.yml
RUN eval "$('/conda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)" && \
    conda-lock install --no-dev -n app /build/conda-lock.yml && \
    conda clean -afy

# mod_wsgi: use pip to compile mod_wsgi from source for the particular versions
# of apache and python that we're using.
RUN eval "$('/conda/bin/conda' 'shell.bash' 'hook' 2> /dev/null)" && \
    conda activate app && \
    pip install mod_wsgi==4.7.1



FROM common

COPY --from=build /conda /conda

# httpd config
COPY docker/httpd.conf /etc/httpd/conf/httpd.conf

# The following is bad security practice if running httpd as
# root, but we will run it as apache.
RUN chmod g+rwx /run/httpd

# install application
COPY . /app/

USER apache:apache
WORKDIR /app
ENTRYPOINT ["/app/docker/entrypoint"]
CMD ["/app/docker/service"]
