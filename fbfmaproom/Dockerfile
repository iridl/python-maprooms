ARG PIXI_VERSION=0.46.0@sha256:c12bcbe8ba5dfd71867495d3471b95a6993b79cc7de7eafec016f8f59e4e4961
ARG CENTOS_VERSION=stream9-minimal@sha256:5b62dfd81b3357910f498a65dc40e42a16249531c7b509272419283a937ce98d

FROM ghcr.io/prefix-dev/pixi:${PIXI_VERSION} as build

# build pixi environment
WORKDIR /app
COPY pixi.lock pixi.toml .
RUN pixi install --locked --environment prod
RUN pixi shell-hook --locked --shell bash --environment prod > activate



FROM quay.io/centos/centos:${CENTOS_VERSION}

COPY --from=build /app /app
COPY . /app

USER nobody
WORKDIR /app
ENTRYPOINT ["/app/docker/entrypoint"]

# Use environment variable WEB_CONCURRENCY to set number of worker
# processes.
CMD ["gunicorn", "--bind=0.0.0.0", "fbfmaproom:SERVER"]
