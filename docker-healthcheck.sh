#!/bin/bash
# Health check script for Docker container
# Run with: bash docker-healthcheck.sh  (or ./docker-healthcheck.sh on Unix)

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo "Checking API health..."

if ! docker ps | grep -q car-price-api; then
    echo -e "${RED}Container is not running${NC}"
    exit 1
fi

HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health 2>/dev/null || echo "000")

if [ "$HTTP_CODE" = "200" ]; then
    echo -e "${GREEN}API is healthy${NC}"
    curl -s http://localhost:8000/health
    exit 0
else
    echo -e "${RED}API health check failed (HTTP $HTTP_CODE)${NC}"
    exit 1
fi
