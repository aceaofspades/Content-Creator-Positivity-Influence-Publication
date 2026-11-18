# Creators Rarely Lead: Creator and Community Sentiment in Five Live-Service Games

Replication materials for the paper by Acea Sands and Levi Sands. The paper tests whether YouTube creators' sentiment about a live-service game leads the sentiment of the game's subreddit and its player counts, in Destiny 2, Warframe, Counter-Strike, Dota 2, and Dead by Daylight.

The paper is in `paper/paper.md`. Its figures are linked from `analysis/*/outputs/figures/`; run `python paper/embed_figures.py paper/paper.md` to produce a single file with the figures embedded.

## What is here

| Path | Contents |
|------|----------|
| `paper/` | The paper. |
| `analysis_plan/` | The written analysis plan the paper refers to (§3, Appendix A). |
| `1_*.py` to `6*.py`, `4_filter_reddit_files/` | Collection, filtering, LLM scoring, master-table assembly, and validation scripts. |
| `7_analysis/` | The numbered analysis scripts. `00_run_all.py` runs them for one game. |
| `analysis/shared/` | Shared modules: game configuration, parsers, loaders, aggregation, and the calendar-correct lag estimator (`lagcorr.py`). |
| `analysis/{game}/outputs/tables/` | Every output table the paper's numbers come from. |
| `analysis/{game}/outputs/figures/` | Every figure the scripts produce. |
| `analysis/cross_game/` | The cross-game multiple-testing families. |
| `other_data/{game}/` | The three LLM prompts for each game and its release calendar. |
| `data/{game}/` | Per-item scores (see below) and the release calendar as parquet. |
| `validation/` | The graded Destiny 2 validation items (identifiers, model labels, human labels) and the agreement report. |

Game folder names: `destiny2`, `warframe`, `csgo` (Counter-Strike), `dota2`, `deadbydaylight`.

## Per-item scores

`data/{game}/{game}_content_scores.parquet` has one row per Reddit submission or YouTube video; `{game}_comment_scores.parquet` has one row per Reddit comment. Each row carries the item's public identifier (`id` for Reddit, `video_id` for YouTube), its date, engagement counts, and the model's scores. Sentiment runs from −2 to +2. A missing topic score means the topic was not detected, not that it was neutral.

These files contain no post text, transcripts, usernames, or model-written summaries. Reddit's and YouTube's terms restrict redistributing that content, so the raw text has to be re-collected from the identifiers.

## What is not here

- Raw Reddit text and YouTube transcripts.
- The master tables the analysis scripts read (`analysis/{game}/data/*_master.parquet`), because they contain that text. The scripts therefore do not run from this repository alone: rebuild the masters with steps 1–5b, or adapt `analysis/shared/loaders.py` to read the score files. Script 16 (author concentration) also needs usernames, which are not released.
- Daily player-count series. They come from SteamDB (all five games) and popularity.report (Destiny 2, all platforms) and should be obtained from those sources.

## Environment

Python dependencies are pinned in `pyproject.toml` and `uv.lock` (`uv sync`). The LLM scoring step expects a vLLM server; see the header of `5_vllm_text_analysis.py`.
