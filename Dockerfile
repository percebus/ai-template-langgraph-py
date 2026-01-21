FROM python:3.13 AS base
WORKDIR /usr/app
COPY . .

# SRC: https://docs.astral.sh/uv/guides/integration/docker/
FROM base AS uv
RUN curl -LsSf https://astral.sh/uv/0.8.9/install.sh | sh
ENV PATH="/root/.local/bin/:$PATH"
RUN uv --version
RUN uv venv
ENV PATH="/usr/app/.venv/bin:$PATH"


FROM uv AS dev
RUN bash scripts/uv/tool/install.ba.sh
RUN uvx --from poethepoet poe sync
RUN uvx --from poethepoet poe build
CMD [ "uvx", "--from", "poethepoet", "poe", "ci" ]

FROM dev AS tested
RUN uvx --from poethepoet poe ci

FROM uv AS built
RUN uvx --from poethepoet poe sync_release
RUN uvx --from poethepoet poe build

# TODO use light image. alpine?
FROM built AS release
# CMD [ "python", "-m", "FIXME" ]
