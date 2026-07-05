import unittest

import pandas as pd

from xquik_export import combine_export_text, normalize_xquik_export


class XquikExportTests(unittest.TestCase):
    def test_combines_export_text_rows(self):
        frame = pd.DataFrame(
            {
                "tweet": [" Earnings guidance improved ", " ", "Revenue risk faded"],
                "created_at": ["2026-07-05", "2026-07-06", "2026-07-07"],
                "tweet_id": ["301", "302", "303"],
            }
        )

        result = combine_export_text(frame)

        self.assertEqual(result, "Earnings guidance improved\n\nRevenue risk faded")

    def test_normalizes_metadata_and_unknown_schema(self):
        normalized = normalize_xquik_export(
            pd.DataFrame({"headline": [" Market rallies "], "link": ["https://example.com"]})
        )
        empty = normalize_xquik_export(pd.DataFrame({"score": [1]}))

        self.assertEqual(
            normalized.to_dict("records"),
            [{"text": "Market rallies", "published": "", "source_id": "https://example.com"}],
        )
        self.assertEqual(len(empty), 0)


if __name__ == "__main__":
    unittest.main()
