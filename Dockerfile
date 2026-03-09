####################################################################################################
## Build
####################################################################################################
FROM rust:alpine AS chef
RUN apk add --no-cache musl-dev pkgconf git lld
RUN cargo install cargo-chef --locked

WORKDIR /app

# Plan dependencies (only changes when Cargo.toml/Cargo.lock change)
FROM chef AS planner
COPY Cargo.toml Cargo.lock* ./
COPY src/ src/
RUN cargo chef prepare --recipe-path recipe.json

# Build dependencies (cached unless Cargo.toml/Cargo.lock change)
FROM chef AS builder
COPY --from=planner /app/recipe.json recipe.json
RUN cargo chef cook --release --recipe-path recipe.json

# Build real source (only recompiles our code)
COPY . .
RUN cargo build --release

####################################################################################################
## Final image
####################################################################################################
FROM alpine:latest

RUN apk update && apk upgrade --no-cache && \
    apk add --no-cache ca-certificates tzdata && \
    update-ca-certificates && \
    adduser \
      --disabled-password \
      --gecos "" \
      --home "/nonexistent" \
      --shell "/sbin/nologin" \
      --no-create-home \
      --uid 10001 \
      bot

COPY --from=builder /app/target/release/polymarket-bot /bin/polymarket-bot

# Model files are optional — sidecar serves the full ensemble at runtime.
# Local XGBoost is kept as fallback; files come from the shared Docker volume.
RUN mkdir -p /app/model

USER bot:bot
WORKDIR /app
ENTRYPOINT ["/bin/polymarket-bot"]
