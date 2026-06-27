@echo off
REM Colors and labels for Windows
setlocal enabledelayedexpansion

echo.
echo ================================
echo IITM BS RAG - Data Pipeline
echo ================================
echo.

REM Check if backend container is running
docker-compose ps backend | findstr "Up" >nul
if errorlevel 1 (
    echo [ERROR] Backend container is not running!
    echo Run: docker-compose up -d
    exit /b 1
)

echo [OK] Backend container is running
echo.

REM Step 1: Scrape
echo [INFO] Step 1/4: Scraping Google Docs...
docker-compose exec -T backend python scraper.py
if errorlevel 1 (
    echo [ERROR] Scraper failed!
    exit /b 1
)
echo [OK] Scraping complete
echo.

REM Step 2: Chunk
echo [INFO] Step 2/4: Chunking documents...
docker-compose exec -T backend python chunker.py
if errorlevel 1 (
    echo [ERROR] Chunker failed!
    exit /b 1
)
echo [OK] Chunking complete
echo.

REM Step 3: Embed
echo [INFO] Step 3/4: Generating embeddings...
docker-compose exec -T backend python embedder.py
if errorlevel 1 (
    echo [ERROR] Embedder failed!
    exit /b 1
)
echo [OK] Embeddings complete
echo.

REM Step 4: Upload
echo [INFO] Step 4/4: Uploading to Qdrant...
docker-compose exec -T backend python uploader.py
if errorlevel 1 (
    echo [ERROR] Uploader failed!
    exit /b 1
)
echo [OK] Upload complete
echo.

echo ================================
echo [SUCCESS] Data pipeline complete!
echo ================================
