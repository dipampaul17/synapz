<div align="center">
  <h1 style="font-size: 2.8em; margin-bottom: 0.4em;">synapz ⚡️</h1>
  <h3 style="margin-top: 0.4em; margin-bottom: 1em; font-weight: normal;">adaptive learning for neurodiverse students</h3>
  <p style="margin-top: 0.8em;"><i>if learning isn't adapting to how you think, it isn't really teaching you</i></p>
  <p style="margin-top: 0.8em; margin-bottom: 0.8em;">
    <img src="https://img.shields.io/badge/status-research_prototype-blue?style=flat-square" alt="status: research prototype">
    <img src="https://img.shields.io/badge/budget-$50_max-green?style=flat-square" alt="budget: $50 max">
    <img src="https://img.shields.io/badge/evidence-emerging-yellow?style=flat-square" alt="evidence: emerging">
    <img src="https://img.shields.io/badge/license-mit-lightgrey?style=flat-square" alt="license: mit">
  </p>
</div>

## the core question: does adaptive teaching actually work better for neurodiverse students?

synapz is a weekend project trying to answer this. we're specifically looking at adhd, dyslexic, and visual learners. the idea is simple: teaching that molds to your cognitive style should be more effective than a one-size-fits-all approach. but "should be" isn't good enough; we need to measure it.

this project is a sprint, built under tight constraints: a $50 openai api budget and 48 hours on an m4 macbook. these limits forced a lean, focused approach to generating evidence.

## our approach: controlled experiments and careful measurement

to get real answers, we do paired experiments:
*   **adaptive session**: the llm teacher tries to tailor its explanation to a specific learner profile (e.g., using more visuals for a visual learner, or structuring text differently for a dyslexic learner).
*   **control session**: the same llm teacher explains the same concept to the same (simulated) learner profile, but using a generic, non-adapted style.

we use a `teacheragent` to generate explanations and a `studentsimulator` (backed by an llm and heuristics) to provide feedback on clarity and engagement. a `budgettracker` keeps us honest on api costs, and a `metricscalculator` crunches the numbers. for instance, our `batch_run_20250518_121146` processed 33 experiment pairs across different profiles and concepts. everything gets logged to a sqlite database (in wal mode, because we like our data safe).

prompts are externalized in the `prompts/` directory – no magic strings in the code.

## how to run it & see what happens

the main engine is `synapz/evaluate.py`.

**to run a new batch of experiments:**
```bash
python -m synapz.evaluate --min-pairs-per-profile 10 --turns 5 --budget 2.0
```
adjust `--min-pairs-per-profile`, `--turns`, and `--budget` as needed.
results (raw csv, compiled json, logs, and charts) land in `results/batch_run_<timestamp>/`.

**to regenerate visualizations from an existing report:**
```bash
python -m synapz.evaluate --create-visuals-for results/your_batch_run_id/compiled_batch_results.json
```
this will create/update a `visualizations` folder next to your report file.

## 📊 key visualizations & insights from `batch_run_20250518_121146`

the hard numbers (p-values, specific averages, etc.) for our latest comprehensive run (`batch_run_20250518_121146`, n=33 pairs) are in `results/batch_run_20250518_121146/compiled_batch_results.json` and the detailed `experiment_pair_details.csv`. we encourage you to dig into these files to see the raw and compiled outputs.

the visualizations in `results/batch_run_20250518_121146/visualizations/` help paint the picture. here's a snapshot of what this run suggests:

<div align="center">
  <p style="margin-bottom: 0.5em;"><strong>overall effectiveness & evidence (`evidence_summary.png`, `readability_metrics.png`):</strong></p>
  <p>
    <img src="./results/batch_run_20250518_121146/visualizations/evidence_summary.png" width="48%" alt="evidence summary chart">
    <img src="./results/batch_run_20250518_121146/visualizations/readability_metrics.png" width="48%" alt="readability & effectiveness metrics chart">
  </p>
  <p style="font-size: 0.9em; margin-top: 0.2em;">
    <em>in this run of 33 pairs, adaptive teaching achieved a higher final clarity score in 39.4% of cases, while the control (non-adaptive) method never outperformed adaptive teaching (0% control wins). a significant number of pairs (60.6%) resulted in ties for final clarity. overall, the adaptive approach showed a statistically significant improvement in final clarity (p < 0.001). for specific profiles, dyslexic (54.55% win rate, p ≈ 0.026) and visual (54.55% win rate, p ≈ 0.026) learners showed significant benefits, while the results for adhd learners (9.09% win rate, p = 1.0) were not statistically significant in this batch.</em>
  </p>
  <hr style="border: none; height: 1px; background-color: #dddddd; margin: 20px 0;">
  <p style="margin-bottom: 0.5em;"><strong>clarity progression over turns (`clarity_progression_adhd.png`, etc.):</strong></p>
  <table role="presentation" style="border-collapse: collapse; width: 100%; margin: 0 auto;">
    <tr>
      <td style="text-align: center; padding: 5px;">
        <img src="./results/batch_run_20250518_121146/visualizations/clarity_progression_adhd.png" width="300" alt="clarity progression for adhd learners">
        <p style="font-size: 0.85em; margin-top: 0;">adhd learners</p>
      </td>
      <td style="text-align: center; padding: 5px;">
        <img src="./results/batch_run_20250518_121146/visualizations/clarity_progression_dyslexic.png" width="300" alt="clarity progression for dyslexic learners">
        <p style="font-size: 0.85em; margin-top: 0;">dyslexic learners</p>
      </td>
      <td style="text-align: center; padding: 5px;">
        <img src="./results/batch_run_20250518_121146/visualizations/clarity_progression_visual.png" width="300" alt="clarity progression for visual learners">
        <p style="font-size: 0.85em; margin-top: 0;">visual learners</p>
      </td>
    </tr>
  </table>
   <p style="font-size: 0.9em; margin-top: 0.2em;">
    <em>these charts track average clarity turn-by-turn. they don't just show *if* learners get it, but *how quickly* and *how consistently*. look for diverging paths between adaptive and control lines – sometimes the journey to understanding is more revealing than the destination.</em>
  </p>
   <hr style="border: none; height: 1px; background-color: #dddddd; margin: 20px 0;">
  <p style="margin-bottom: 0.5em;"><strong>magnitude of differences (`effect_sizes.png`):</strong></p>
  <p>
     <img src="./results/batch_run_20250518_121146/visualizations/effect_sizes.png" width="500" alt="effect sizes chart">
  </p>
  <p style="font-size: 0.9em; margin-top: 0.2em;">
    <em>this chart shows cohen's d effect sizes for readability metrics. it tells us *how much* the adaptive explanations differ from control in terms of standard textual properties. significant structural differences (see text/tag diffs in the raw data) might not always translate to large shifts in these standard readability scores, hinting that 'neurodiverse-friendly' might be more complex than just 'simpler text'.</em>
  </p>
</div>

**emerging (and often messy) insights:**

these points are general observations, and `batch_run_20250518_121146` provides fresh data to consider:

1.  **adaptation's impact varies**: while adaptive sessions achieved higher final clarity in 39.4% of pairs overall in the latest run, and never lost to control, the effect wasn't universal. the dyslexic and visual profiles showed ~55% win rates for adaptive, but adhd only ~9%. this highlights that the *effectiveness of current adaptive strategies differs significantly across profiles and potentially concepts.* the quality and specificity of adaptation in `prompts/` remain paramount.
2.  **profiles show diverse responses**: the adhd profile's low win rate (9.09%) and non-significant p-value in `batch_run_20250518_121146` starkly contrasts with dyslexic and visual profiles (~55% win rates, significant p-values). this underscores that a single "adaptive" approach won't be a silver bullet. digging into `experiment_pair_details.csv` for this run can show which adaptive tactics succeeded or failed for specific adhd learner/concept pairs.
3.  **simulator fidelity is crucial**: the outcomes (like the 60.6% tie rate in final clarity) are based on our simulator. if its sensitivity to subtle learning differences isn't high enough, or if it doesn't fully capture a neurodiverse student's experience, it might mask or misrepresent true effects. the current heuristics (abstractness, etc.) are a starting point.
4.  **"clarity" metrics tell different stories**: the `compiled_batch_results.json` for `batch_run_20250518_121146` shows final clarity improvements. simultaneously, the `effect_sizes.png` and `readability_metrics.png` (derived from `textstat` scores on the explanations) might show that adaptive explanations have different textual properties (e.g., sentence length, complexity). comparing these helps understand if perceived clarity aligns with standard readability, or if our adaptive strategies are achieving clarity through other means (e.g., structure, examples more than just simpler text).
5.  **statistical significance and sample size**: `batch_run_20250518_121146` (n=33 total, 11 per profile) yielded significant p-values for dyslexic and visual profiles. however, for adhd, the result wasn't significant. this scale of data is where we start to see trends, but larger runs per profile would increase confidence. the `effect_sizes.png` helps gauge if significant differences are also practically meaningful.

the journey is iterative, and each batch run (like `batch_run_20250518_121146`) is a cycle:
1.  **analyze results**: we dig into the data (`experiment_pair_details.csv`, `compiled_batch_results.json`) and visualizations from the latest run.
2.  **refine hypotheses**: based on what worked (or didn't), we update the teaching strategies in `prompts/` and consider tweaks to the `studentsimulator`'s logic or heuristic metrics.
3.  **test again**: new experiments are run with `evaluate.py` to measure the impact of our refinements.
this learn-adjust-retest loop is how synapz evolves and deepens its understanding of "effective adaptation."

## the tricky bits & what's next

this is research, so it's full of challenges, many of which were apparent during the execution and analysis of `batch_run_20250518_121146`:
*   **getting enough data**: small sample sizes (even with ~10-11 pairs per profile as in our latest run) make it hard to draw strong statistical conclusions for specific profiles vs. specific concepts. we need more runs to confirm trends.
*   **defining "good" adaptation**: what specific instructional strategies truly work best for an adhd profile when explaining algebra vs. history? our prompts are hypotheses themselves, and their performance in `batch_run_20250518_121146` gives clues for refinement.
*   **the cost constraint**: the $50 budget (though extended for the larger run) means we can't just run thousands of experiments with the most powerful models. every llm call counts, a factor carefully managed in all runs.

**immediate focus:**
1.  **more data, wisely**: run more experiment pairs, perhaps focusing on profiles/concepts where results from `batch_run_20250518_121146` were ambiguous or counterintuitive. use the `--min-pairs-per-profile` flag in `evaluate.py`.
2.  **prompt refinement**: based on the detailed data in `experiment_pair_details.csv` from `batch_run_20250518_121146`, identify adaptive sessions that underperformed and analyze *why*. update the instruction sets in `prompts/`.
3.  **simulator enhancements**: can we add more sophisticated heuristics to `studentsimulator` that better correlate with neurodiverse processing, informed by patterns seen in the latest run data? for example, tracking cognitive load indicators beyond simple readability.

## 🏗️ project structure

```
synapz/
├── core/               # teacheragent, studentsimulator, budget, llmclient, db models
├── data/               # concept .json files, profile .json files, metrics.py, visualization.py
├── prompts/            # .txt files for system & instruction prompts
├── results/            # timestamped output from batch runs
├── tests/              # (aspiring to have more of these)
├── evaluate.py         # main batch evaluation script
└── cli.py              # (currently minimal, for potential interactive testing)
```

## 📦 installation

```bash
# clone
git clone https://github.com/dipampaul17/synapz.git
cd synapz

# env
python3 -m venv .venv
source .venv/bin/activate  # on windows: .venv\\scripts\\activate

# install
pip install -r requirements.txt

# api key
export openai_api_key='your-api-key' # or pass via --api-key in evaluate.py
```

## 🏷️ tags

`adaptive-learning` `cognitive-diversity` `llm-education` `neurodiversity` `adhd` `dyslexia` `visual-learner` `personalized-learning` `prompt-engineering` `educational-technology` `learning-science` `experiment-design` `python` `openai-api` `evidence-based-education`

## 📑 license

mit

---

<div align="center">
  <p>this is a research sprint. the goal is learning, iterating, and (hopefully) finding some truth.</p>
  <p><a href="https://github.com/dipampaul17/synapz">github.com/dipampaul17/synapz</a></p>
</div> 