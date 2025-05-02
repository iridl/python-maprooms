ARG PIXI_VERSION=0.46.0@sha256:c12bcbe8ba5dfd71867495d3471b95a6993b79cc7de7eafec016f8f59e4e4961
ARG CENTOS_VERSION=stream9-minimal@sha256:6fab90c6b8d1348bcf63d486b4c75e51a18d94e1bd4ba7a4e855b33f824e45f7

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

# OPENBLAS multithreading is incompatible with gunicorn's use of fork().
# Also its default is to use one thread per CPU, which isn't appropriate
# in a multi-application, multi-user environment.
ENV OPENBLAS_NUM_THREADS=1

# Use environment variable WEB_CONCURRENCY to set number of worker
# processes.

CMD ["gunicorn", "--bind=0.0.0.0", "fbfmaproom:SERVER"]
