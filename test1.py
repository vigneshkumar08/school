from fastapi import FastAPI, HTTPException, Query
import psycopg2
import requests
import cv2
import numpy as np
from pydantic import BaseModel
from datetime import datetime
import logging
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)

# Database connection settings
DB_CONFIG = {
    "host": "192.168.51.236",
    "database": "postgres",
    "user": "admin",
    "password": "godspeed123"
}

# Threshold for contrast
CONTRAST_THRESHOLD = 25


class ImageInfo(BaseModel):
    camera_id: int
    minio_url: str
    timestamp: datetime
    status: int


def get_image_data(from_timestamp: datetime, to_timestamp: datetime):
    """
    Fetch image data from the database within a timestamp range.
    """
    try:
        with psycopg2.connect(
            host=DB_CONFIG["host"],
            dbname=DB_CONFIG["database"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"]
        ) as conn:
            with conn.cursor() as cursor:
                query = """
                    SELECT camera_id, minio_url, timestamp, status
                    FROM camera_images
                    WHERE timestamp BETWEEN %s AND %s
                """
                cursor.execute(query, (from_timestamp, to_timestamp))
                rows = cursor.fetchall()

                # Convert rows to a list of dictionaries
                return [
                    ImageInfo(
                        camera_id=row[0],
                        minio_url=row[1],
                        timestamp=row[2],
                        status=row[3]
                    )
                    for row in rows
                ]
    except Exception as e:
        logging.error(f"Error fetching image data: {e}")
        return []


def calculate_contrast(image_url: str):
    """
    Calculate contrast for an image given its URL.
    """
    try:
        response = requests.get(image_url, timeout=10)
        if response.status_code == 200:
            image = np.array(bytearray(response.content), dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)

            if image is None:
                logging.error(f"Error: Unable to decode image from {image_url}")
                return None

            _, stddev = cv2.meanStdDev(image)
            contrast = stddev[0][0]
            return contrast
        else:
            logging.error(f"Failed to retrieve image from {image_url}. Status: {response.status_code}")
            return None
    except Exception as e:
        logging.error(f"Error calculating contrast for {image_url}: {e}")
        return None


def update_image_status(camera_id: int, timestamp: datetime, status: int):
    """
    Update the status of an image in the database.
    """
    try:
        with psycopg2.connect(
            host=DB_CONFIG["host"],
            dbname=DB_CONFIG["database"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"]
        ) as conn:
            with conn.cursor() as cursor:
                query = """
                    UPDATE camera_images
                    SET status = %s
                    WHERE camera_id = %s AND timestamp = %s
                """
                cursor.execute(query, (status, camera_id, timestamp))
                conn.commit()  
                logging.info(f"Updated status for camera_id {camera_id} at {timestamp} to {status}")
    except Exception as e:
        logging.error(f"Error updating status for camera_id {camera_id} at {timestamp}: {e}")
        raise


@app.get("/")
def filter_images(
    from_timestamp: datetime = Query(..., description="Start of the timestamp range"),
    to_timestamp: datetime = Query(..., description="End of the timestamp range")
):
    """
    API endpoint to process images within a timestamp range and return those with contrast below the threshold.
    """
    try:
        logging.info(f"Filtering images from {from_timestamp} to {to_timestamp}")

        # Fetch image data within the timestamp range
        image_data_list = get_image_data(from_timestamp, to_timestamp)
        if not image_data_list:
            raise HTTPException(status_code=404, detail="No image data found for the specified time range.")

        results = []
        for image_info in image_data_list:
            contrast = calculate_contrast(image_info.minio_url)
            if contrast is not None and contrast < CONTRAST_THRESHOLD:
                
                update_image_status(image_info.camera_id, image_info.timestamp, 1)

                results.append({
                    "camera_id": image_info.camera_id,
                    "contrast": contrast,
                })

        if not results:
            raise HTTPException(status_code=404, detail="No images with contrast below threshold found.")

        return {"results": results}

    except HTTPException as e:
        logging.error(f"HTTP error: {e.detail}")
        raise e
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while processing images.")



# Add the entry point to run the app with uvicorn
if __name__ == "__main__":
    
    uvicorn.run("test1:app", host="127.0.0.1", port=8000, reload=True)
