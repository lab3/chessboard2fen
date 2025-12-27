"""FastAPI service to convert chessboard images to FEN strings.

This module exposes a small HTTP API that accepts an uploaded image of a
chessboard, runs the detection/classification pipeline, and returns the
resulting FEN string (plus a few optional evaluation details).
"""
from typing import Any, Dict

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

from boardDetection import ChessboardDetector
from chessboard import Chessboard

app = FastAPI(title="Chessboard2FEN", description="Convert chessboard images to FEN strings")


class DetectorService:
    """Encapsulates the heavy detector instances so they are created once."""

    def __init__(self) -> None:
        self.detector = ChessboardDetector("models/detection", "models/classification.h5")
        self.board = Chessboard()

    def image_from_upload(self, upload: UploadFile) -> np.ndarray:
        data = upload.file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        image_array = np.frombuffer(data, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Unable to decode image. Ensure it is a valid image file.")
        return image

    def build_response(self, predictions: np.ndarray) -> Dict[str, Any]:
        rotated_predictions = self.board.rotate_predictions(predictions)
        fen = self.board.predictions_to_fen(rotated_predictions)
        board_changed = self.board.predictions_to_move(predictions)

        response: Dict[str, Any] = {"fen": fen, "board_updated": board_changed}
        if board_changed:
            response.update(
                {
                    "last_engine_move": str(self.board.mv),
                    "suggested_best_move": str(getattr(self.board, "bestMv", "")),
                    "score": self.board.score,
                }
            )
        return response


detector_service = DetectorService()


@app.get("/health")
def health() -> Dict[str, str]:
    """Simple readiness check."""
    return {"status": "ok"}


@app.post("/fen")
def image_to_fen(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Return the FEN string for an uploaded chessboard image."""
    image = detector_service.image_from_upload(file)
    corners = detector_service.detector.predict_board_corners(image)
    if len(corners) != 4:
        raise HTTPException(status_code=422, detail="Could not detect chessboard corners in the image")

    predictions = detector_service.detector.predictBoard(image, corners)
    return detector_service.build_response(predictions)


@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "Chessboard2FEN API. POST /fen with an image file to receive a FEN string."}
