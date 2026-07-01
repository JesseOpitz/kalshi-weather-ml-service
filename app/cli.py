from __future__ import annotations

import argparse
import json
import secrets
from pathlib import Path

from app.db.base import Base
from app.db.models import ModelVersion, Station
from app.db.session import SessionLocal, engine


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    print("Database initialized.")


def seed_stations(path: str) -> None:
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    with SessionLocal() as db:
        for payload in records:
            station = db.query(Station).filter(Station.station_code == payload["station_code"]).one_or_none()
            if station is None:
                station = Station(
                    station_code=payload["station_code"],
                    station_name=payload["station_name"],
                    latitude=payload["latitude"],
                    longitude=payload["longitude"],
                )
                db.add(station)
            station.station_name = payload["station_name"]
            station.latitude = payload["latitude"]
            station.longitude = payload["longitude"]
            station.elevation_m = payload.get("elevation_m")
            station.timezone = payload.get("timezone", "UTC")
            station.official_source = payload.get("official_source")
        db.commit()
    print(f"Seeded {len(records)} stations.")


def model_status() -> None:
    with SessionLocal() as db:
        champion = db.query(ModelVersion).filter(ModelVersion.status == "champion").first()
        if champion is None:
            print(json.dumps({"status": "unavailable", "message": "No champion model is promoted."}))
            return
        print(json.dumps({"status": champion.status, "version": champion.version, "metrics": champion.metrics}, default=str))


def main() -> None:
    parser = argparse.ArgumentParser(prog="weather-ml")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init-db")
    seed = commands.add_parser("seed-stations")
    seed.add_argument("path", nargs="?", default="config/stations.example.json")
    commands.add_parser("model-status")
    commands.add_parser("generate-key")
    args = parser.parse_args()
    if args.command == "init-db":
        init_db()
    elif args.command == "seed-stations":
        seed_stations(args.path)
    elif args.command == "model-status":
        model_status()
    elif args.command == "generate-key":
        print(secrets.token_urlsafe(48))


if __name__ == "__main__":
    main()
