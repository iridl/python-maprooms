FROM centos:7.9.2009 as common

ARG PIXI_VERSION="v0.41.4"

COPY docker/CentOS-Vault.repo /etc/yum.repos.d/
RUN rm /etc/yum.repos.d/CentOS-Base.repo



FROM common as build

# install pixi
RUN curl --silent --location \
    https://github.com/prefix-dev/pixi/releases/download/${PIXI_VERSION}/pixi-x86_64-unknown-linux-musl.tar.gz \
    | tar zxf - --to-stdout \
    > /usr/local/bin/pixi \
    && chmod +x /usr/local/bin/pixi

# build pixi environment
WORKDIR /app
COPY pixi.lock pixi.toml .
RUN pixi install --locked --environment prod
RUN pixi shell-hook --locked --shell bash --environment prod > activate



FROM common

COPY --from=build /app /app
COPY . /app

USER nobody
WORKDIR /app
ENTRYPOINT ["/app/docker/entrypoint"]

# Use environment variable WEB_CONCURRENCY to set number of worker
# processes.
CMD ["gunicorn", "--bind=0.0.0.0", "fbfmaproom:SERVER"]
