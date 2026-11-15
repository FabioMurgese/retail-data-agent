"""PII masker: catches contact data in free text and frames, without damaging analysis output."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from retail_agent.policy import load_data_policy
from retail_agent.safety.pii_masker import PiiMasker

MASKER = PiiMasker.from_policy(load_data_policy(Path(__file__).resolve().parents[1] / "config"))


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("Contact jane.doe@example.org for details", "email"),
        ("Her number is +1 415 555 0100.", "phone"),
        ("Call +39 06 1234 5678", "phone"),
        ("Office: (415) 555-0100", "phone"),
        ("Phone 415-555-0100", "phone"),
        # formats from the dataset's street_address column
        ("Ships to 403 Jefferson Walks, Baltimore", "address"),
        ("000 Marshall Falls Suite 353", "address"),
        ("Lives at 7707 Holland Harbors Apt. 519.", "address"),
        ("Location 39.2663, -76.62082", "coordinates"),
        ("Card 4111 1111 1111 1111 was used", "card_number"),
    ],
)
def test_detects_contact_data(text, kind):
    masked, counts = MASKER.mask_text(text)
    assert counts == {kind: 1}
    assert f"[{kind} redacted]" in masked


@pytest.mark.parametrize(
    "text",
    [
        "Revenue grew 12.5% to $1,234,567.89 in Q1 2024 (2023: $1,100,000).",
        "Between 2023 2024 the order count went from 125467 to 130000.",
        "Orders placed 2024-01-15 10:30:00 and shipped 2024-01-17.",
        "Top traffic source: Email (4,983 users), then Search and Organic.",
        "We sold 12 Columbia Men's Steens Mountain jackets and 30 Levi's 501 Original Fit Jeans.",
        "| avg_price | margin |\n| 45.1234 | 67.8912 |",
        "Order 1234567890123 had 3 items",  # long id that fails the Luhn check
        "Growth +15.3 points vs last month",
        "Customer 42 (age 31, Baltimore, United States) spent $1,250.",
    ],
)
def test_leaves_analysis_output_alone(text):
    masked, counts = MASKER.mask_text(text)
    assert masked == text and not counts


def test_frame_blanks_pii_columns_and_scans_strings():
    df = pd.DataFrame(
        {
            "user_id": [1, 2],
            "Email": ["a@example.com", None],  # matched case-insensitively against the policy
            "note": ["reach me at b@example.net", "nothing here"],
            "latitude": [39.26, -8.89],
            "revenue": [10.5, 20.0],
        }
    )
    masked, counts = MASKER.mask_frame(df)

    assert masked["Email"].tolist()[0] == "[redacted]" and pd.isna(masked["Email"].tolist()[1])
    assert masked["latitude"].tolist() == ["[redacted]", "[redacted]"]
    assert masked["note"].tolist() == ["reach me at [email redacted]", "nothing here"]
    assert masked["revenue"].tolist() == [10.5, 20.0] and masked["user_id"].tolist() == [1, 2]
    assert counts == {"pii_column": 3, "email": 1}
    assert df["Email"][0] == "a@example.com"  # input not modified
