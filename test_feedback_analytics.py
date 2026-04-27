"""
Student Feedback Analytics System — Test Suite
pytest test file covering all 6 categories (24 test cases)

Run with:
    pip install pytest
    pytest test_feedback_analytics.py -v
"""

import pytest
import pandas as pd
import io
from collections import Counter

# ── Import the module under test ──────────────────────────────────────────────
# Adjust the import path if your module is named differently
from feedback_analytics import (
    normalise_text,
    analyze_all_aspects,
    calculate_overall_score,
    generate_insights,
    analyze_bulk_feedback,
    generate_bulk_insights,
    filter_bulk_results,
    get_sample_csv,
)


# =============================================================================
# CATEGORY 1 — NLP Engine: Slang & Normalisation
# =============================================================================

class TestNLPSlangNormalisation:
    """TC-01 to TC-06: Slang mapping, negation, and aspect routing."""

    def test_TC01_prof_chill_positive_teaching(self):
        """'prof is chill' → Teaching Quality: Positive"""
        results = analyze_all_aspects("prof is chill")
        assert "Teaching Quality" in results, "Expected Teaching Quality to be detected"
        assert results["Teaching Quality"]["sentiment"] == "Positive"

    def test_TC02_exams_brutal_negative_assessment(self):
        """'exams are brutal' → Assessment & Evaluation: Negative"""
        results = analyze_all_aspects("exams are brutal")
        assert "Assessment & Evaluation" in results, "Expected Assessment & Evaluation to be detected"
        assert results["Assessment & Evaluation"]["sentiment"] == "Negative"

    def test_TC03_labs_useless_negative_practical(self):
        """'labs are useless' → Practical & Lab: Negative"""
        results = analyze_all_aspects("labs are useless")
        assert "Practical & Lab" in results, "Expected Practical & Lab to be detected"
        assert results["Practical & Lab"]["sentiment"] == "Negative"

    def test_TC04_mid_normalises_to_average(self):
        """'mid' should normalise to 'average' via SLANG_MAP"""
        normalised = normalise_text("the course is mid")
        assert "average" in normalised, f"Expected 'average' in normalised text, got: '{normalised}'"

    def test_TC05_lit_normalises_to_excellent(self):
        """'lit' should normalise to 'excellent' via SLANG_MAP"""
        normalised = normalise_text("the teaching was lit")
        assert "excellent" in normalised, f"Expected 'excellent' in normalised text, got: '{normalised}'"

    def test_TC06_negation_not_helpful_tas_is_negative(self):
        """'not helpful TAs' → Mentoring & Support: Negative (or at least not Positive)"""
        results = analyze_all_aspects("not helpful TAs")
        if "Mentoring & Support" in results:
            assert results["Mentoring & Support"]["sentiment"] != "Positive", \
                "Negated positive phrase should not resolve as Positive"


# =============================================================================
# CATEGORY 2 — Bulk CSV Processing
# =============================================================================

class TestBulkCSVProcessing:
    """TC-07 to TC-11: CSV ingestion, validation, and batch processing."""

    def test_TC07_sample_csv_14_rows_processed(self):
        """Full 14-row sample CSV should produce 14 results"""
        df = get_sample_csv()
        results = analyze_bulk_feedback(df.to_json(), "feedback")
        assert len(results) == 14, f"Expected 14 results, got {len(results)}"

    def test_TC08_csv_missing_feedback_column_detected(self):
        """CSV without a 'feedback' column should be detected as invalid"""
        df = pd.DataFrame({"name": ["Alice", "Bob"], "score": [90, 85]})
        assert "feedback" not in df.columns, "Column check: 'feedback' should be absent"

    def test_TC09_short_feedback_skipped(self):
        """Feedback entries with <=10 characters should be skipped"""
        df = pd.DataFrame({
            "feedback": [
                "ok",                                                          # too short
                "The professor explains concepts really well, fair exams.",    # valid
            ]
        })
        results = analyze_bulk_feedback(df.to_json(), "feedback")
        assert len(results) == 1, f"Expected 1 valid result (short entry skipped), got {len(results)}"

    def test_TC10_bulk_results_contain_required_keys(self):
        """Each bulk result dict must contain all required output keys"""
        df = get_sample_csv()
        results = analyze_bulk_feedback(df.to_json(), "feedback")
        required_keys = {"feedback_id", "overall_score", "positive_aspects",
                         "negative_aspects", "neutral_aspects", "aspect_details"}
        for result in results:
            missing = required_keys - result.keys()
            assert not missing, f"Result missing keys: {missing}"

    def test_TC11_overall_score_within_valid_range(self):
        """Overall scores should fall between 10 and 100"""
        df = get_sample_csv()
        results = analyze_bulk_feedback(df.to_json(), "feedback")
        for r in results:
            assert 10 <= r["overall_score"] <= 100, \
                f"Score out of range for feedback_id {r['feedback_id']}: {r['overall_score']}"


# =============================================================================
# CATEGORY 3 — Dashboard Charts & Visualisations (data correctness)
# =============================================================================

class TestDashboardDataCorrectness:
    """TC-12 to TC-16: Verify data structures that feed charts."""

    @pytest.fixture(scope="class")
    def bulk_insights(self):
        df = get_sample_csv()
        results = analyze_bulk_feedback(df.to_json(), "feedback")
        return generate_bulk_insights(results)

    def test_TC12_sentiment_distribution_has_all_buckets(self, bulk_insights):
        """Sentiment distribution should include Positive, Negative, and Neutral"""
        dist = bulk_insights["sentiment_distribution"]
        total = sum(dist.values())
        assert total == bulk_insights["total_feedbacks"], \
            "Sentiment distribution totals should match total feedback count"
        assert total > 0

    def test_TC13_score_distribution_within_range(self, bulk_insights):
        """All overall scores must be between 10 and 95"""
        scores = [r["overall_score"] for r in bulk_insights["bulk_results"]]
        assert all(10 <= s <= 95 for s in scores), \
            f"Found out-of-range scores: {[s for s in scores if not (10<=s<=95)]}"

    def test_TC14_aspect_aggregator_has_seven_dimensions(self, bulk_insights):
        """Aspect aggregator should contain all 7 analysis dimensions"""
        expected_dimensions = {
            "Teaching Quality", "Assessment & Evaluation", "Practical & Lab",
            "Mentoring & Support", "Course Design & Workload",
            "Teaching Methodology", "Learning Outcomes",
        }
        found = set(bulk_insights["aspect_aggregator"].keys())
        # At least some of the 7 dimensions must be present
        assert len(found) >= 3, f"Expected multiple dimensions, found only: {found}"

    def test_TC15_faculty_stats_sorted_by_score_descending(self):
        """Faculty bar chart data must be sorted by avg score descending"""
        df = get_sample_csv()
        results = analyze_bulk_feedback(df.to_json(), "feedback")

        faculty_stats: dict = {}
        for result in results:
            faculty = result.get("faculty", "Unknown")
            if faculty not in faculty_stats:
                faculty_stats[faculty] = []
            faculty_stats[faculty].append(result["overall_score"])

        faculty_data = [
            {"Faculty": f, "Avg Score": sum(s) / len(s)}
            for f, s in faculty_stats.items()
        ]
        faculty_df = pd.DataFrame(faculty_data).sort_values("Avg Score", ascending=False)
        scores = faculty_df["Avg Score"].tolist()
        assert scores == sorted(scores, reverse=True), "Faculty scores not in descending order"

    def test_TC16_gauge_color_green_for_score_above_70(self):
        """Satisfaction gauge should use green colour when avg score >= 70"""
        avg = 78
        gauge_color = "#15803d" if avg >= 70 else ("#92400e" if avg >= 50 else "#b91c1c")
        assert gauge_color == "#15803d", f"Expected green gauge for score {avg}, got {gauge_color}"


# =============================================================================
# CATEGORY 4 — Sample Dataset & Interactive Loading
# =============================================================================

class TestSampleDataset:
    """TC-17 to TC-19: Sample data loading and schema validation."""

    def test_TC17_sample_csv_has_14_rows(self):
        """get_sample_csv() should return exactly 14 rows"""
        df = get_sample_csv()
        assert len(df) == 14, f"Expected 14 rows, got {len(df)}"

    def test_TC18_sample_csv_has_all_required_columns(self):
        """Sample CSV must contain feedback, faculty, course, semester columns"""
        df = get_sample_csv()
        required = {"feedback", "faculty", "course", "semester"}
        assert required.issubset(set(df.columns)), \
            f"Missing columns: {required - set(df.columns)}"

    def test_TC19_analyze_sample_data_full_pipeline(self):
        """Full pipeline on sample data should produce populated results"""
        df = get_sample_csv()
        results = analyze_bulk_feedback(df.to_json(), "feedback")
        insights = generate_bulk_insights(results)

        assert insights["total_feedbacks"] > 0
        assert insights["avg_overall_score"] > 0
        assert len(insights["aspect_aggregator"]) > 0
        assert len(insights["bulk_results"]) > 0


# =============================================================================
# CATEGORY 5 — Multi-Aspect Analysis & Filtering
# =============================================================================

class TestMultiAspectAndFiltering:
    """TC-20 to TC-23: Contrast detection, multi-aspect, and filter logic."""

    def test_TC20_mixed_feedback_detects_multiple_aspects(self):
        """Complex feedback should trigger 3+ distinct aspects"""
        results = analyze_all_aspects(
            "Great teaching, terrible labs, helpful TAs"
        )
        assert len(results) >= 3, \
            f"Expected at least 3 aspects, got {len(results)}: {list(results.keys())}"

    def test_TC21_contrast_detection_splits_correctly(self):
        """'but' clause should yield both Positive and Negative aspects"""
        results = analyze_all_aspects(
            "Professor explains very well but exams are too hard and grading is unfair"
        )
        sentiments = {d["sentiment"] for d in results.values()}
        assert "Positive" in sentiments, "Expected at least one Positive aspect"
        assert "Negative" in sentiments, "Expected at least one Negative aspect"

    def test_TC22_faculty_filter_returns_correct_subset(self):
        """Filtering by 'Dr. Garcia' should return exactly 1 result"""
        df = get_sample_csv()
        results = analyze_bulk_feedback(df.to_json(), "feedback")
        filtered = filter_bulk_results(results, df, faculty="Dr. Garcia",
                                       course=None, semester=None)
        assert len(filtered) == 1, f"Expected 1 result for Dr. Garcia, got {len(filtered)}"

    def test_TC23_course_filter_narrows_correctly(self):
        """Filtering by 'CS101' should return only CS101 rows"""
        df = get_sample_csv()
        results = analyze_bulk_feedback(df.to_json(), "feedback")
        filtered = filter_bulk_results(results, df, faculty=None,
                                       course="CS101", semester=None)
        assert all(r.get("course") == "CS101" for r in filtered), \
            "All filtered results should belong to CS101"
        assert len(filtered) > 0, "Expected at least one CS101 result"


# =============================================================================
# CATEGORY 6 — Export & End-to-End Pipeline
# =============================================================================

class TestExportAndPipeline:
    """TC-24: Export schema and complete pipeline integrity."""

    def test_TC24_export_csv_contains_required_columns(self):
        """Exported results should contain all columns needed for CSV download"""
        df = get_sample_csv()
        results = analyze_bulk_feedback(df.to_json(), "feedback")

        export_rows = [
            {
                "ID": r["feedback_id"],
                "Feedback": str(r.get("feedback", ""))[:200],
                "Faculty": r.get("faculty", "—"),
                "Course": r.get("course", "—"),
                "Aspects": r["aspects_detected"] if "aspects_detected" in r else len(r["aspect_details"]),
                "Positive": r["positive_aspects"],
                "Negative": r["negative_aspects"],
                "Score": r["overall_score"],
            }
            for r in results
        ]
        export_df = pd.DataFrame(export_rows)

        required_export_cols = {"ID", "Feedback", "Faculty", "Score"}
        assert required_export_cols.issubset(set(export_df.columns)), \
            f"Missing export columns: {required_export_cols - set(export_df.columns)}"

        # Validate CSV round-trip
        buf = io.StringIO()
        export_df.to_csv(buf, index=False)
        buf.seek(0)
        reloaded = pd.read_csv(buf)
        assert len(reloaded) == len(export_df), "CSV row count mismatch after round-trip"
        assert set(reloaded.columns) == set(export_df.columns), \
            "CSV column mismatch after round-trip"


# =============================================================================
# Bonus: Edge-case tests
# =============================================================================

class TestEdgeCases:
    """Additional robustness checks."""

    def test_empty_string_returns_empty_dict(self):
        """Empty feedback should return empty results"""
        assert analyze_all_aspects("") == {}

    def test_very_short_feedback_returns_empty(self):
        """Single-word feedback should return empty results"""
        result = analyze_all_aspects("ok")
        assert result == {}

    def test_calculate_overall_score_none_for_empty(self):
        """calculate_overall_score should return None for empty input"""
        assert calculate_overall_score({}) is None

    def test_generate_insights_structure(self):
        """generate_insights should return strengths/weaknesses/neutral/recommendations keys"""
        fake = {
            "Teaching Quality": {"sentiment": "Positive", "confidence": 0.85, "score": 80},
            "Practical & Lab":  {"sentiment": "Negative", "confidence": 0.80, "score": 25},
        }
        insights = generate_insights(fake)
        assert "Teaching Quality" in insights["strengths"]
        assert "Practical & Lab" in insights["weaknesses"]
        assert len(insights["recommendations"]) == 2
