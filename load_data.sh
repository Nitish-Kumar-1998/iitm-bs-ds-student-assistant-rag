#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}================================${NC}"
echo -e "${YELLOW}IITM BS RAG - Data Pipeline${NC}"
echo -e "${YELLOW}================================${NC}\n"

# Check if backend container is running
if ! docker-compose ps backend | grep -q "Up"; then
    echo -e "${RED}❌ Backend container is not running!${NC}"
    echo "Run: docker-compose up -d"
    exit 1
fi

echo -e "${GREEN}✅ Backend container is running${NC}\n"

# Step 1: Scrape
echo -e "${YELLOW}Step 1/4: Scraping Google Docs...${NC}"
docker-compose exec -T backend python scraper.py
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Scraper failed!${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Scraping complete\n${NC}"

# Step 2: Chunk
echo -e "${YELLOW}Step 2/4: Chunking documents...${NC}"
docker-compose exec -T backend python chunker.py
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Chunker failed!${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Chunking complete\n${NC}"

# Step 3: Embed
echo -e "${YELLOW}Step 3/4: Generating embeddings...${NC}"
docker-compose exec -T backend python embedder.py
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Embedder failed!${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Embeddings complete\n${NC}"

# Step 4: Upload
echo -e "${YELLOW}Step 4/4: Uploading to Qdrant...${NC}"
docker-compose exec -T backend python uploader.py
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Uploader failed!${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Upload complete\n${NC}"

echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}🎉 Data pipeline complete!${NC}"
echo -e "${GREEN}================================${NC}"
