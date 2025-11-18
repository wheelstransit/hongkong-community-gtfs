#!/bin/bash

set -e

REPO_ROOT="$(pwd)"


docker compose up --build -d

sleep 10
docker compose exec -T app python main.py
    
docker compose down

bash clean_gtfs.sh

echo "Completed: $REPO_ROOT/hk.gtfs.zip"

