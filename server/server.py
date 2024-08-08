from fastapi import FastAPI, HTTPException
from services import BlogService
import asyncio

import random

app = FastAPI()

blog_service = BlogService()


@app.get("/")
async def root():
    return {"Hello": "World"}


@app.get("/first")
async def blog():
    try:
        return await blog_service.generate_blog_post("Local Business Automation")
    except asyncio.CancelledError:
        # Handle task cancellation
        raise HTTPException(
            status_code=503, detail="Service unavailable due to task cancellation.")
    except KeyboardInterrupt:
        # Handle manual interruption
        raise HTTPException(
            status_code=500, detail="Service interrupted manually.")
    except Exception as e:
        # Handle any other exceptions
        raise HTTPException(status_code=500, detail=str(e))

# Graceful shutdown handling


@app.get("/test/{limit}")
async def test(limit: int):
    return {"test": limit}


@app.on_event("shutdown")
async def shutdown_event():
    tasks = [task for task in asyncio.all_tasks(
    ) if task is not asyncio.current_task()]
    [task.cancel() for task in tasks]
    await asyncio.gather(*tasks, return_exceptions=True)
