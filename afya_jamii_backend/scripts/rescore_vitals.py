"""Re-score vitals recorded before the risk-label fix.

The previous release read ``predict_proba(x)[0, 1]`` from a three-class model
and compared it to 0.5. Column 1 is *low risk*, so stored labels are wrong for
every row written before the fix, and ``ml_probability`` holds the probability
of low risk rather than confidence in the stored label.

The original vitals are intact, so the rows can simply be scored again.

Usage::

    # Show what would change, without writing anything
    python scripts/rescore_vitals.py --dry-run

    # Re-score every record
    python scripts/rescore_vitals.py

    # Re-score only records written before the upgrade
    python scripts/rescore_vitals.py --before 2026-09-11

Take a database backup first. The script rewrites ml_risk_label,
ml_probability and ml_feature_importances in place; it does not touch the
vitals themselves, so it is safe to run more than once.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

# Allow running this file directly from the backend root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlmodel import Session, select  # noqa: E402

from app.database import engine  # noqa: E402
from app.ml_model import InvalidFeaturesError, initialize_model, risk_model  # noqa: E402
from app.models import VitalsRecord  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("rescore")


def to_celsius(value: float, unit: str) -> float:
    """The model was trained in Celsius; stored rows may be Fahrenheit."""
    if (unit or "celsius").lower().startswith("f"):
        return round((value - 32.0) * 5.0 / 9.0, 2)
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would change without writing to the database",
    )
    parser.add_argument(
        "--before",
        metavar="YYYY-MM-DD",
        help="only re-score records created before this date",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=500,
        help="rows to commit at a time (default: 500)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    cutoff = None
    if args.before:
        try:
            cutoff = datetime.strptime(args.before, "%Y-%m-%d")
        except ValueError:
            logger.error("--before must be a date in YYYY-MM-DD form")
            return 2

    if not initialize_model():
        logger.error("The risk model could not be loaded; nothing was changed.")
        return 1

    logger.info("Model classes: %s", ", ".join(risk_model.classes))
    if args.dry_run:
        logger.info("Dry run — no changes will be written.\n")

    changed = unchanged = failed = 0

    with Session(engine) as session:
        statement = select(VitalsRecord).order_by(VitalsRecord.id)
        if cutoff is not None:
            statement = statement.where(VitalsRecord.created_at < cutoff)

        records = session.exec(statement).all()
        logger.info("Examining %d record(s).\n", len(records))

        for index, record in enumerate(records, start=1):
            features = {
                "Age": record.age,
                "SystolicBP": record.systolic_bp,
                "DiastolicBP": record.diastolic_bp,
                "BS": record.bs,
                "BodyTemp": to_celsius(record.body_temp, record.body_temp_unit),
                "HeartRate": record.heart_rate,
            }

            try:
                prediction = risk_model.predict(features)
            except InvalidFeaturesError as exc:
                # Rows captured before the range checks may fall outside them.
                logger.warning("  #%s skipped: %s", record.id, exc)
                failed += 1
                continue

            if prediction.label == record.ml_risk_label:
                unchanged += 1
                continue

            logger.info(
                "  #%-5s %s -> %s   (%s/%s mmHg, BS %s)",
                record.id,
                record.ml_risk_label,
                prediction.label,
                record.systolic_bp,
                record.diastolic_bp,
                record.bs,
            )
            changed += 1

            if not args.dry_run:
                record.ml_risk_label = prediction.label
                record.ml_probability = prediction.probability
                record.ml_feature_importances = json.dumps(prediction.feature_importances)
                session.add(record)

                if index % args.batch == 0:
                    session.commit()

        if not args.dry_run:
            session.commit()

    logger.info(
        "\n%d relabelled, %d already correct, %d skipped.",
        changed,
        unchanged,
        failed,
    )
    if args.dry_run and changed:
        logger.info("Re-run without --dry-run to apply these changes.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
