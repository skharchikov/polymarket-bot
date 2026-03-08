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
## Minimal CA certs + timezone data
####################################################################################################
FROM alpine:latest AS files

RUN apk update && apk upgrade --no-cache && \
    apk add --no-cache ca-certificates tzdata

RUN update-ca-certificates

ENV USER=bot
ENV UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    "${USER}"

####################################################################################################
## Final scratch image
####################################################################################################
FROM scratch

COPY --from=files --chmod=444 /etc/passwd /etc/group /etc/nsswitch.conf /etc/
COPY --from=files --chmod=444 /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/
COPY --from=files --chmod=444 /usr/share/zoneinfo /usr/share/zoneinfo

COPY --from=builder /app/target/release/polymarket-bot /bin/polymarket-bot

USER bot:bot
WORKDIR /app
ENTRYPOINT ["/bin/polymarket-bot"]
