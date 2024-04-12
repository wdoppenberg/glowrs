LOCAL_VERSION=$(grep -e '^version\s*=\s*"' Cargo.toml | head -1 | cut -d '"' -f2)

DOCKER_BUILDKIT=0 docker build -f Dockerfile.cuda -t "ghcr.io/wdoppenberg/glowrs-server:${LOCAL_VERSION}-cuda" . || exit 1
docker tag "ghcr.io/wdoppenberg/glowrs-server:${LOCAL_VERSION}-cuda" "ghcr.io/wdoppenberg/glowrs-server:latest-cuda"

docker push "ghcr.io/wdoppenberg/glowrs-server:${LOCAL_VERSION}-cuda"
docker push "ghcr.io/wdoppenberg/glowrs-server:latest-cuda"
