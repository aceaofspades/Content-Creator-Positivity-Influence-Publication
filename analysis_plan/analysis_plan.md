# Game Community Sentiment & Player Count Analysis — Project Todo List

This document organizes all planned work into execution buckets. Each item includes notes on nuances, dependencies, and considerations important for handing off to other agents or collaborators.

---

## Project Overview

This project investigates the relationship between community sentiment and player behavior in live-service video games, using **Destiny 2** as the primary case and **Warframe, CS2, Dota 2, and Dead by Daylight** as secondary comparisons. The central research question is whether and how sentiment expressed by YouTube content creators and Reddit community members correlates with, leads, or lags changes in player counts — and whether creators function as opinion leaders whose sentiment shapes (rather than merely reflects) broader community sentiment.

### Data

Three aligned data streams, all joinable by date:

- **Reddit submissions and comments** (Pushshift archive, ~2017–2025), filtered for quality before LLM processing
- **YouTube transcripts** from hand-curated lists of opinion-forward creators per game
- **Daily Steam player counts** per game (PC only — a known platform limitation)

All text content is processed through a two-pass LLM pipeline that extracts ~100 structured sentiment fields per item: core fields (overall sentiment, fun, content, monetization, trajectory, etc.) run on everything, with topic-specific fields (PvP, sandbox balance, monetization, technical, developer communication, narrative, etc.) extracted only when relevant topics are detected. Numeric fields use a -2 to +2 scale with "unclear" treated as NULL. Creator engagement fields capture whether creators are still playing, burned out, or recommending the game.

### Core Theoretical Framing

The primary analytical lens is **elite influence theory** (Zaller; Katz & Lazarsfeld), which predicts that in domains with costly information, a small number of high-visibility opinion leaders shape broader community perception. Applied here: major content creators function as elites whose sentiment should measurably lead community (Reddit) sentiment, with effects scaling by reach and concentrating on topics requiring interpretive expertise.

### What Success Looks Like

The project produces (1) a published academic paper in a quantitative games or CSCW-adjacent venue, and (2) an accessible blog writeup with interactive visualizations on the author's personal website. Success requires the analyses to be methodologically defensible — findings should survive scrutiny from a data scientist at a game studio, which means disciplined causal language, explicit null/baseline comparisons, pipeline validation, and cross-game replication of headline results.

### How to Use This Document

Tasks are organized into execution buckets (validation, analysis, robustness, paper framing, deliverables). The **cross-cutting notes below** apply across multiple tasks and should be read before executing any individual item — they cover data conventions (NULL handling, sparsity, weighting schemes), seasonality handling, and causal-language discipline that shape how every analysis should be built and reported.

If assigned a specific task, read: (1) this overview, (2) the cross-cutting notes, (3) the task itself, and (4) any tasks explicitly referenced as dependencies.

---

## Notes for Handoff to Other Agents

If passing any of this work to other agents/collaborators, the following context is essential:

1. **The -2 to +2 scale treats "unclear" as NULL.** This affects every aggregation. Make sure any agent knows not to impute "unclear" as 0
2. **Topic fields are conditionally populated.** Missing ≠ neutral. Distinguishing "topic not discussed" from "topic discussed neutrally" is analytically important for the topic-level predictiveness work
3. **Test for seasonality before applying decomposition.** Live-service games *may* show seasonality at multiple scales:
   - **Weekly:** day-of-week reset effects (e.g., Destiny 2 Tuesday spikes)
   - **Quarterly:** seasonal release cycles (roughly every 3-4 months)
   - **Annual:** major annual expansion cycles
   But which are actually present in any given series (player counts, Reddit sentiment, creator sentiment, per game) is an empirical question. Before deseasonalizing, test for the presence and magnitude of each cycle — autocorrelation function (ACF) plots, periodogram/spectral analysis, or seasonal decomposition diagnostics. Only remove cycles that are actually present and meaningful in that specific series
   
   Note: quarterly and annual cycles are partially captured by `release_phase` and `days_since_release` features, so models including those features handle some seasonality implicitly. Pure cross-correlation analyses that don't use those features need explicit decomposition where warranted
   
   Document which cycles were detected and removed for each series — this is a methods-section item, not a silent preprocessing step
4. **Game-relevance and confidence filtering are sensitivity parameters.** Main analyses should be reported with sensible filters, plus sensitivity tests varying them
5. **Three weightings for creators, two for Reddit.** Non-negotiable — these measure different things
6. **Lead/lag ≠ causation.** Keep causal language out of discussion of findings. This is both a methodological and a credibility issue
7. **The LLM pipeline uses a two-pass system** (topic detection, then targeted extraction). Understanding this matters for interpreting sparse topic fields — they're sparse by design. If a topic field is missing fields, it should be assumed that the topic was not detected and was therefore not extracted, so assume it was not discussed

---

## 1. Data Validation

*Establishing that the LLM pipeline produces meaningful, trustworthy scores. This work should happen early — downstream analyses depend on it, and discovering pipeline issues late is expensive.*

- [ ] **Hand-label a random sample of 50–100 submissions and transcripts for human-LLM agreement**
  - Use stratified sampling across games and time periods so validation isn't concentrated in one era or game
  - Compute Cohen's kappa for categorical fields (e.g., `are_they_still_playing`, `emotional_tone`, `would_they_recommend`)
  - Compute correlation and mean absolute error (MAE) for numeric -2 to +2 fields
  - Report per-field agreement — some fields will be much more reliable than others, and that matters for interpretation downstream
  - Treat `"unclear"` consistently with how analysis treats it (as NULL)

- [ ] **Second-human subset for human-human agreement ceiling**
  - A subset (~50 items) labeled by a second person establishes the ceiling for agreement. If two humans only agree at kappa = 0.6, expecting LLM agreement above that is unrealistic
  - Without this, "LLM-human kappa = 0.55" is hard to interpret. With it, "LLM-human = 0.55, human-human = 0.60" is a strong validation

- [ ] **Prompt sensitivity / robustness check**
  - Rerun a subsample (100–200 items) through Pass 2 with a meaning-identical but surface-different prompt — reworded sentences, reordered sections, synonym swaps for instruction verbs ("assess" → "evaluate" → "rate"), different example formatting. The instruction content must stay the same; only the phrasing changes
  - Compute stability of scores across prompt versions
  - If scores shift by >0.3 on a 4-point scale for purely cosmetic phrasing changes, results are sensitive to arbitrary wording choices and need addressing before publication
  - If stable, report it — preempts a real methodological concern

- [ ] **Engagement-weighted vs. volume-weighted aggregation comparison for Reddit**
  - Not a pipeline validation per se, but a sensitivity check on aggregation choice
  - Show the main time series under both weighting schemes and note where they diverge
  - Where they disagree, that's often where the silent-majority-vs-vocal-minority distinction matters most

---

## 2. Data Analysis

*The actual analytical work that produces findings. Order roughly reflects logical build order — release-window analysis first establishes context and sanity-checks the pipeline, later analyses depend on decisions made earlier.*

**Important cross-cutting notes for all analyses:**
- **Weekly seasonality:** player counts have strong day-of-week effects (reset days drive spikes, especially in Destiny 2). Deseasonalize player counts before any correlation or lead/lag analysis, or work with 7-day rolling means as the analytical unit. This is required preprocessing, not optional
- **Weighting schemes:** creator analyses should report results under three schemes — per-video unweighted, view-weighted (reach), and creator-size-weighted (source credibility/authority). These measure different constructs and should be interpreted accordingly. Reddit analyses should report under two schemes — raw volume and engagement-weighted
- **Daily granularity:** analytical unit is daily; rolling windows are for presentation, except where smoothing is computationally necessary (cross-correlation on noisy daily data), in which case smoothing is part of methodology
- **Game-relevance filtering:** respect the `meta.game_relevance` field — `not_relevant` content should generally be excluded from sentiment aggregations; `partially_relevant` probably should be too, depending on analysis
- **Confidence-score filtering:** consider sensitivity analyses with and without low-confidence items (`meta.confidence_score` ≤ 0) to check whether findings are driven by uncertain LLM outputs
- **Steam Proxy Fit:** an analysis has been done and determined that when release weeks (T-7 days to T+7 days from release day) are excluded, Steam player counts are a near perfect proxy for all-player counts for Destiny 2 numbers only (Steam and non-Steam numbers are not available for other games). When including release weeks, Steam player counts are only a moderate proxy, good for directionality, but not magnitude

- [ ] **Release-window overlay plot**
  - T-6 weeks to T+6 weeks around each release, one line per release
  - Normalize player counts as % change from T-6 week baseline (raw counts aren't comparable across eras)
  - Color/facet by manually-assigned "success/failure" label — this labeling should happen *before* running the analysis to avoid hindsight bias
  - Build parallel versions with sentiment on the y-axis instead of player counts; side-by-side comparison tells a story
  - Serves as foundation / sanity check — if this doesn't look sensible, downstream analyses have problems

- [ ] **Release categorization: revival, retention, and floor metrics**
  - Compute three continuous per-release metrics, all defined relative to the median of other releases for the same game (so each game is evaluated against its own baseline):
    - **Revival:** peak concurrent players during launch week ÷ pre-release baseline average (pre-release baseline = mean of T-6 to T-1 weeks). Measures how much hype the release generated
    - **Retention:** % of launch-week peak still playing at week N post-launch (N=6 and apply consistently). Measures how well the release held players
    - **Floor:** stabilized player count after initial decay (e.g., mean of weeks 8–12 post-launch) ÷ pre-release baseline average. Measures long-term lift above baseline
  - Keep all three metrics continuous in the underlying analysis — do not discretize them for correlation work or regression inputs
  - For narrative/visualization purposes only, derive a 3-bucket categorization from revival and retention (floor is reported separately):
    - **Successful:** both revival and retention above their game-specific medians
    - **Nuanced:** one above median, one below
    - **Failure:** both below their game-specific medians
  - Document these thresholds explicitly as analytical conveniences, not substantive judgments — the underlying metrics are continuous and the buckets are a presentational shorthand
  - Floor is reported as an independent descriptor per release, not folded into the bucket categorization
  - All three metric definitions must be locked in before computing them on the data, to avoid threshold-tuning toward desired results

- [ ] **Within-season sentiment decay curves**
  - Use `release_phase` and `days_since_release` to map sentiment arcs within each season
  - Test whether late-season sentiment predicts next season's launch performance
  - This is a planning-relevant finding for studios, not just an academic one

- [ ] **Topic-level churn predictiveness**
  - Regularized regression (Lasso or elastic net) predicting forward player-count change from full sentiment field set
  - Run at multiple horizons (2, 4, 6 weeks) — different topics likely matter on different timescales
  - Aggregate to weekly means with volume weighting before fitting
  - Handle NULL values (sparsely populated topic fields) carefully — imputation choice matters. Consider using mean imputation with a "was_discussed" indicator variable per topic so the model can distinguish "topic was neutral" from "topic wasn't discussed"
  - Interpret non-zero coefficients as predictive, not causal

- [ ] **Creator engagement fields as leading indicators**
  - Track % of creators in `taking_a_break`, `burned_out`, etc. over time
  - Test whether these lead player-count declines
  - This is a behavioral self-report embedded in content and may be your strongest leading indicator

- [ ] **Creator → Reddit sentiment lead/lag analysis (PRIMARY elite-influence test)**
  - Cross-correlation at lags 0–28 days
  - Must deseasonalize both series first
  - Granger causality tests on residuals after removing release-window effects
  - This is the cleanest test of the elite-influence claim — Reddit sentiment is the actual opinion variable theory predicts creators should move (player counts are behavioral and noisier)
  - Report effect sizes and confidence intervals, not just significance

- [ ] **Creator heterogeneity analysis (reach + source credibility)**
  - Split videos by view-count tier (reach) AND by creator-size tier (source credibility) — these are distinct constructs
  - Under elite-influence theory, effects should be stronger for high-reach videos and high-authority creators
  - If either splits shows scaling, that supports the theory. If neither does, that's a null result worth reporting honestly
  - Note: total view counts reflect accumulated views, not contemporary reach — flag this as a limitation

- [ ] **Topic-split influence analysis (expertise-heavy vs. directly-experienced)**
  - Requires the open-question categorization to be resolved (see bottom)
  - Theory predicts creator influence concentrates on expertise-heavy topics (sandbox balance, endgame design, reading developer communication) more than directly-experienced ones (server stability, pricing)
  - If the topic-split pattern matches the theoretical prediction, this is a strong finding

- [ ] **Residualized analysis (low priority)**
  - Build a simple observable-state model predicting Reddit sentiment and player counts from release-phase dummies, days-since-release, patch indicators, major announcement dummies
  - Take residuals
  - Test whether lagged creator sentiment predicts those residuals
  - This is the most defensible "influence" evidence observational data can provide
  - Getting the observable-state model right matters — too simple and the residuals still contain game-state variance; too complex and you've absorbed the signal you're trying to find
  - Requires a more filled-out calendar to complete that, as of time of writing, has not been made. Either make a more filled-out calendar or skip this analysis
  - This should only be done for Destiny 2 if at all, not any other games

- [ ] **Reddit/creator sentiment divergence analysis**
  - Plot `creator_sentiment - reddit_sentiment` over time, overlaid with player count
  - Divergence periods (hype cycles, sponsored periods, contrarian creator takes) may be leading indicators
  - Novel angle; less crowded than straight correlation

- [ ] **Cross-game replication**
  - Repeat primary findings on Warframe, CS2, Dota 2, DBD
  - If findings replicate, the claim is stronger. If they don't, you learn something about what's game-specific
  - Reviewers will ask regardless, so bake it in from the start
  - Cross-game comparison requires harmonizing the schema — some topic fields are Destiny-specific (Trials of Osiris, Eververse) and have no analog in other games. Flag which fields are comparable vs. which aren't
  - As of writing, data for other games does not exist. The code for these analyses, however, should not change based on source as the structure of the data for the other games will remain the same. Only labels/data source/etc. should change, nothing else

- [ ] **Anomaly table**
  - Largest residuals from a seasonal baseline model of player counts
  - For each anomaly, pull surrounding-window top sentiment topics and top creator videos
  - Should rediscover major controversies (Sunsetting, Lightfall reception, etc.) without being told about them
  - Great sanity check + narrative glue for writeup

- [ ] **Clustered correlation matrix (EDA / appendix)**
  - Hierarchical clustering on correlation matrix reveals topic structure
  - Use FDR correction or treat as purely descriptive — don't report raw p-values with ~100 fields
  - Appendix material, not headline result

---

## 3. Methodological Robustness

*Defending the analyses themselves against "is this finding real or an artifact?" critiques. Run alongside or after the main analyses.*

- [ ] **Null model / permutation comparisons**
  - For lead/lag findings, shuffle dates on creator series 1000+ times and re-run analysis
  - Reports where your observed statistic falls in the null distribution
  - Answers "is correlation of 0.4 impressive?" — it depends on the null
  - This is standard for any time-series finding and reviewers will expect it

- [ ] **Autoregressive baseline comparisons**
  - Test whether creator sentiment predicts Reddit sentiment *beyond* what Reddit's own recent history predicts
  - Without this test, you can declare spurious lead relationships driven by shared autocorrelation
  - Standard approach: fit AR(n) model on Reddit sentiment alone, then test whether adding lagged creator sentiment improves fit

- [ ] **Robustness to creator subset choice**
  - Drop most-negative creators, drop largest, drop smallest — do findings survive?
  - Partially addresses the selection bias concern without requiring rigorous selection methodology

- [ ] **Deseasonalization of player counts**
  - Required preprocessing before correlation analyses
  - Standard approaches: subtract 7-day moving average, or fit day-of-week dummies and use residuals
  - Document which method used

- [ ] **Explicit documentation of smoothing choices**
  - Distinguish smoothing for computation vs. smoothing for presentation
  - Report sensitivity of findings to smoothing window choice (try 3-day, 7-day, 14-day)

- [ ] **Sensitivity to confidence-score filtering**
  - Rerun main analyses excluding low-confidence LLM outputs
  - If findings hold, robust. If findings disappear, they're driven by noisy outputs and need reframing

---

## 4. Paper Discussion / Framing

*Content for the paper itself — lit review, methods justifications, limitations. Most of this is writing, not analysis, but needs to be planned in advance so the analyses are designed to support the claims.*

- [ ] **Elite influence theory grounding (lit review)**
  - Zaller, *The Nature and Origins of Mass Opinion* (1992)
  - Katz & Lazarsfeld, *Personal Influence* (1955) — the two-step flow model
  - More recent opinion leadership literature, especially anything on online opinion leaders / influencers
  - Position your work as applying an established framework to a new domain (game communities) with novel data

- [ ] **Methods justification: LLM over off-the-shelf sentiment tools**
  - Topic-conditional extraction not possible with VADER etc.
  - Domain-specific vernacular ("broken," "meta," "grind") has inverted or specialized meanings in gaming
  - Sarcasm prevalence in gaming discourse
  - Cite or reference LLM validation work

- [ ] **Methods justification: why these games**
  - Reliable public player-count data (Steam)
  - Mature content-creator ecosystems (sufficient creator volume for analysis)
  - Phrase as a principled selection, not an arbitrary one

- [ ] **Methods justification: weighting schemes**
  - Creator analyses use three weightings: per-video (output), view-weighted (reach), creator-size-weighted (source credibility/authority)
  - Explain what each measures and why all three are relevant
  - Reddit: raw volume (discussion breadth) and engagement-weighted (discussion salience)

- [ ] **Selection bias acknowledgment (limitations)**
  - Phrase professionally — something like: "Creator selection prioritizes channels with sustained opinion-forward content and moderate-to-large followings within each game's ecosystem. This introduces selection effects we do not attempt to correct for; findings should be interpreted as describing the influential-creator segment of each community rather than creator ecosystems as a whole."
  - Do NOT use "sue me" phrasing to reviewers. Seriously

- [ ] **Specific limitations section**
  - Steam PC only — no console data; likely under-represents console-heavy games and different platform demographics
  - Silent majority — Reddit and YouTube comments reflect vocal users; lurkers invisible
  - Non-English communities not captured
  - Temporal coverage limits (start dates vary by game)
  - View counts are accumulated, not contemporary — recency effects attenuate for recent videos
  - LLM validation has a ceiling (human-human agreement); report it honestly

- [ ] **Causal language discipline**
  - Audit writeup for causal language (cause, drive, produce, lead to)
  - Replace with observational language (precede, predict, correlate, lead in time) unless you have a genuine identification strategy
  - The more disciplined the language, the more credible the claims

- [ ] **Pre-registration-style labeling**
  - In the paper, distinguish confirmatory analyses (planned before looking at results) from exploratory ones (emerged during analysis)
  - Exploratory findings are fine and often the most interesting, but must be labeled as such
  - Being honest about this reads as sophisticated, not weak

- [ ] **Ethical considerations**
  - All data is public
  - Analysis is aggregate, not individual
  - No identification of private individuals
  - Brief but necessary for academic venues

- [ ] **View-count measurement limitation**
  - Total views ≠ contemporary views at time of influence
  - Acknowledge as attenuating effect for recent videos

---

## 5. Deliverables / Presentation

*How the work reaches readers. These shape the scope and polish requirements of everything upstream.*

- [ ] **Primary paper writeup**
  - Target: quantitative games or CSCW-adjacent venue (decision deferred — write first, choose later)
  - Caveat: glance at 2-3 candidate venues' length limits early so you're drafting in a reasonable ballpark (e.g., CHI PLAY ~10 pages, *Games and Culture* 6-8k words, CSCW longer)

- [ ] **Blog post + landing page on personal website**
  - Adapt from paper — paper-to-blog is easier than blog-to-paper
  - Focus on narrative and strong visualizations; hiring managers more likely to read this than the paper

- [ ] **Interactive component for personal site**
  - Start with release-window explorer (toggle releases) and/or topic-vs-player-count time series (topic dropdown)
  - **Primary option: Tableau Public.** Embeds cleanly, handles the standard dropdowns/toggles easily, low-overhead
  - **Alternative: Observable notebooks.** Also embeds cleanly and gives more control for custom interactions beyond Tableau's built-in filters. Worth using if Tableau's interaction model proves too limiting
  - Don't overscope — one or two well-executed interactive views beats a sprawling dashboard

- [ ] **Qualitative narrative paragraphs for 3–5 key moments**
  - Pair with anomaly table from analysis
  - Context for what was happening in the game at each moment, why it showed up in the data
  - Demonstrates ability to communicate findings to non-technical stakeholders — important for hiring

---

## 6. Open Questions / Decisions Remaining

- [ ] **Topic-expertise categorization**
  - Required for the topic-split influence analysis
  - Need to classify your 11 categories (and possibly sub-fields within them) into "expertise-heavy" (where creator interpretation adds value — sandbox balance, developer communication, buildcrafting, narrative analysis) vs. "directly-experienced" (where players have their own information — server stability, pricing, load times, matchmaking queue times)
  - Some categories will be mixed (PvP matchmaking is experienced, but PvP meta analysis is expertise-heavy)
  - Worth doing this categorization collaboratively and documenting reasoning — reviewers will question any fuzzy calls

---

