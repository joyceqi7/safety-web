#!/usr/bin/env bash
# Read-only deployment audit. Does not print environment variables or secrets.
set -u
printf '\nSystem\n'
if [ -r /etc/os-release ]; then cat /etc/os-release; fi
uname -m
printf '\nResources\n'
getconf _NPROCESSORS_ONLN
free -m
df -h / /opt 2>/dev/null || df -h /
printf '\nRuntime availability\n'
for executable in docker python3 nginx caddy unzip curl; do command -v "$executable" || true; done
if command -v docker >/dev/null 2>&1; then
  docker --version
  docker compose version || true
  docker ps --format 'table {{.Names}}\t{{.Image}}\t{{.Ports}}\t{{.Status}}' || true
fi
printf '\nListening TCP ports\n'
ss -lnt 2>/dev/null || true
printf '\nExisting deployment directory\n'
if [ -d /opt/safety-web ]; then ls -ld /opt/safety-web; else printf 'Not present\n'; fi
