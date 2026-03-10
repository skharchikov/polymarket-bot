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

COPY target/x86_64-unknown-linux-musl/release/polymarket-bot /bin/polymarket-bot

RUN mkdir -p /app/model

USER bot:bot
WORKDIR /app
ENTRYPOINT ["/bin/polymarket-bot"]
