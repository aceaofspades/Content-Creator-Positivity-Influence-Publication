#%%
"""
6b_human_human_agreement.py
============================
PLACEHOLDER — implementation pending second-annotator data.

Manages the human-human inter-rater agreement lifecycle, providing an
interpretive ceiling for the LLM-human kappas from 6_data_validation.py.

Phases (stateful, detected automatically):
  1. No test_set_human2.csv → sample ~50 items from completed test_set.csv
     (stratified by content type), write a clean labeling sheet with empty
     human2__* columns, print instructions for the second annotator.
  2. test_set_human2.csv exists but not fully labeled → print progress.
  3. All rows have human2_validated=True → compute human1 vs. human2
     kappa (categorical) and Pearson+MAE (numeric), write
     human_human_report.csv, print side-by-side vs. LLM-human values.

Prerequisites:
  - test_set.csv must be fully completed (all human_validated=True)
  - A second annotator must label test_set_human2.csv independently,
    blind to both the original human labels and the LLM outputs

Fields to re-label (same as 6_data_validation.py):
  - sentiment_score      (numeric, -2 to +2)
  - optimism             (numeric, -2 to +2; submissions + creators only)
  - emotional_tone       (categorical, collapsed to positive/neutral/negative)
  - still_playing        (categorical; creator rows only)
  - burned_out           (categorical; creator rows only)
  - would_recommend      (categorical; creator rows only)
"""

raise NotImplementedError(
    "6b_human_human_agreement.py is a placeholder. "
    "Implement once second-annotator labeling is ready. "
    "See module docstring for the full implementation plan."
)
