"""
PostgreSQL Ingestion Service — FastAPI + SQLAlchemy + APScheduler.

Fetches records from AWS RDS and saves them as CSV files in secret_data/
to synchronize the ML pipeline with the live production database.
"""

import os
import time
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, BackgroundTasks, HTTPException
from sqlalchemy import create_engine, text
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from config import DATABASE_URL, SECRET_DATA_DIR

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("db_ingestor")

app = FastAPI(title="Maya Database Ingestor")

# Tables to sync: mapping { table_name_in_rds: output_csv_filename }
TABLES_TO_SYNC = {
    "users": "users.csv",
    "sessions": "sessions.csv",
    "feedbacks": "feedbacks.csv",
    "whatsapp_messages": "whatsapp_messages.csv",
}

# Shared state for sync status
sync_status = {
    "last_run": None,
    "last_result": "Never run",
    "is_running": False,
    "details": {}
}

def sync_data():
    """Core logic to fetch data from RDS and save to CSV."""
    if sync_status["is_running"]:
        logger.warning("Sync already in progress, skipping...")
        return

    sync_status["is_running"] = True
    sync_status["last_run"] = datetime.now().isoformat()
    
    try:
        if not DATABASE_URL:
            raise ValueError("DATABASE_URL not configured. Check your .env file.")

        engine = create_engine(DATABASE_URL)
        SECRET_DATA_DIR.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting database sync from RDS... {len(TABLES_TO_SYNC)} tables found.")
        
        counts = {}
        for table, filename in TABLES_TO_SYNC.items():
            try:
                logger.info(f"  → Fetching table: {table}")
                with engine.connect() as conn:
                    df = pd.read_sql(text(f"SELECT * FROM {table}"), conn)
                
                # Standardize user ID columns for the ML pipeline
                if table == "whatsapp_messages" and "sender_user_id" in df.columns:
                    df = df.rename(columns={"sender_user_id": "user_id"})
                if table == "users" and "id" in df.columns:
                    df = df.rename(columns={"id": "user_id"})
                
                output_path = SECRET_DATA_DIR / filename
                df.to_csv(output_path, index=False)
                counts[table] = len(df)
                logger.info(f"    ✓ Saved {len(df):,} rows to {output_path}")
            except Exception as e:
                logger.error(f"    ✗ Failed to fetch {table}: {str(e)}")
                counts[table] = f"Error: {str(e)}"

        sync_status["last_result"] = "Partial Success" if any("Error" in str(v) for v in counts.values()) else "Success"
        sync_status["details"] = counts
        logger.info("✅ Database sync complete.")

    except Exception as e:
        logger.error(f"❌ Sync failed: {str(e)}")
        sync_status["last_result"] = f"Failed: {str(e)}"
    finally:
        sync_status["is_running"] = False

# --- Scheduler Setup ---
scheduler = BackgroundScheduler()
# Default: Weekly on Monday at 12:00 AM
scheduler.add_job(
    sync_data,
    CronTrigger(day_of_week="mon", hour=0, minute=0),
    id="weekly_sync",
    replace_existing=True
)
scheduler.start()

@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI Ingestor started. Scheduler active.")

@app.get("/")
async def root():
    return {"message": "Maya Database Ingestor API", "status": sync_status}

@app.get("/status")
async def get_status():
    return sync_status

@app.post("/sync")
async def trigger_sync(background_tasks: BackgroundTasks):
    """Manually trigger a data synchronization."""
    if sync_status["is_running"]:
        return {"message": "Sync already in progress"}
    
    background_tasks.add_task(sync_data)
    return {"message": "Sync started in background"}

if __name__ == "__main__":
    import uvicorn
    # Allow port override via env
    port = int(os.getenv("INGESTOR_PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
