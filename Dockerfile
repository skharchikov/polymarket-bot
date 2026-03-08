####################################################################################################
## Build
####################################################################################################
FROM rust:alpine AS builder

RUN apk update && apk upgrade --no-cache && \
    apk add --no-cache musl-dev pkgconf git lld upx

WORKDIR /app

# Cache dependencies
COPY Cargo.toml Cargo.lock* ./
RUN mkdir src && echo "fn main() {}" > src/main.rs && \
    cargo build --release 2>/dev/null || true

# Build real source
COPY . .
RUN touch src/main.rs && cargo build --release

# Compress binary with UPX (~60% smaller)
RUN upx --best --lzma /app/target/release/polymarket-bot

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

USER bot:bot
WORKDIR /app
ENTRYPOINT ["/bin/polymarket-bot"]
