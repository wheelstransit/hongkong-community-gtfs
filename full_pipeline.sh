#!/bin/bash

set -e

REPO_ROOT="$(pwd)"


docker compose up --build -d

sleep 10
docker compose exec -T app python main.py
    
docker compose down

bash clean_gtfs.sh

echo "Completed:"
echo "  - $REPO_ROOT/hk.gtfs.zip"
echo "  - $REPO_ROOT/includefares-hk.gtfs.zip"

