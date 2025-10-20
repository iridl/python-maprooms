ARG PIXI_VERSION=0.55.0@sha256:423daf63e4e876ab42b1be9ddee3fc867ba8b9fa511e0634de3b9df12ff1a90d
ARG CENTOS_VERSION=7.9.2009

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
