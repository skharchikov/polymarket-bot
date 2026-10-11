FROM alpine:latest

RUN apk update && apk upgrade --no-cache && \
    apk add --no-cache ca-certificates tzdata && \
    update-ca-certificates && \
    adduser --disabled-password --gecos "" --home "/nonexistent" \
      --shell "/sbin/nologin" --no-create-home --uid 10001 admin

COPY target/x86_64-unknown-linux-musl/release/admin /bin/admin

USER admin:admin
ENV ADMIN_BIND=0.0.0.0:8080
EXPOSE 8080
ENTRYPOINT ["/bin/admin"]
