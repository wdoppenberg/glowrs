ARG PACKAGE="glowrs-server"
ARG FEATURES=""

# Build container
FROM lukemathwalker/cargo-chef:latest-rust-1 AS chef
WORKDIR /build

FROM chef AS planner
COPY . .
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder
COPY --from=planner /build/recipe.json recipe.json
# Build dependencies - this is the caching Docker layer!
RUN cargo chef cook --release --recipe-path recipe.json
# Build application
ARG PACKAGE

COPY . .

RUN cargo build --release --bin $PACKAGE --features "$FEATURES"

# Final image
FROM debian:stable-slim

# Installing nativetls dependencies
RUN apt-get update && \
	apt-get install -y openssl libssl-dev ca-certificates && \
	rm -rf /var/lib/apt/lists/*

WORKDIR /app
ARG PACKAGE
COPY --from=builder /build/target/release/$PACKAGE  /app/$PACKAGE

HEALTHCHECK --interval=30s --start-period=10s --retries=3 CMD bash -c ':> /dev/tcp/0.0.0.0/3000' || exit 1

ENV RUST_LOG="glowrs=debug,server=debug,tower_http=debug,axum::rejection=trace"

# Running
EXPOSE 3000
ENTRYPOINT ["./glowrs-server", "--host", "0.0.0.0"]