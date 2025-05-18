<div align="center">
  <h1 style="font-size: 2.8em; margin-bottom: 0.4em;">synapz ⚡️</h1>
  <h3 style="margin-top: 0.4em; margin-bottom: 1em; font-weight: normal;">adaptive learning for neurodiverse students</h3>
  <p style="margin-top: 0.8em;"><i>if learning isn't adapting to how you think, it isn't really teaching you</i></p>
  <p style="margin-top: 0.8em; margin-bottom: 1.2em;">
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
  <hr style="border: none; height: 1px; background-color: #dddddd; margin: 15px 0;">
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
</div>

**emerging (and often messy) insights:**

the data from `batch_run_20250518_121146` reinforces several core observations from this ongoing research:

1.  **adaptation's impact is profile-dependent**: adaptive strategies significantly benefited 'dyslexic' and 'visual' profiles in the latest run (~55% win rates, p ≈ 0.026), but not 'adhd' (9% win rate, p=1.0). this isn't just about *if* adaptation works, but *which specific strategies work for whom*. the `prompts/` are critical hypotheses here.
2.  **"neurodiverse" is not a monolith**: the varied success across profiles (e.g., adhd vs. dyslexic in `batch_run_20250518_121146`) highlights that effective support requires nuanced, profile-aware strategies rather than a single "adaptive" mode. deeper analysis of `experiment_pair_details.csv` is key to uncovering concept-specific interaction effects.
3.  **simulator fidelity limits real-world claims**: our current student simulator drives the results (like the 60.6% tie rate in final clarity). while it uses llms and heuristics, its alignment with true neurodiverse student experiences is an approximation. improving this is vital for stronger conclusions.
4.  **clarity is more than readability scores**: while adaptive methods improved perceived clarity (especially for dyslexic/visual profiles), these changes don't always correlate with simpler text based on standard readability metrics (see `readability_metrics.png`). effective adaptation likely involves structure, pacing, and modality, not just text simplification.
5.  **statistical significance guides, not dictates**: with n=11 pairs per profile in the latest run, we see significant p-values for dyslexic/visual learners. this signals a real effect. for adhd, the lack of significance means we can't yet claim an effect with this data. more data will refine these views, but effect size also matters – is a statistically significant win practically meaningful?

this project evolves through a continuous cycle of discovery and refinement:

> **the synapz iteration loop:**
> 1.  **analyze**: scrutinize quantitative results (win rates, p-values, clarity scores) and qualitative data (explanation content, simulator feedback) from the latest batch run.
> 2.  **hypothesize & refine**: based on analysis, form new hypotheses. did a prompt strategy fail for adhd? was the simulator too lenient? update `prompts/`, adjust `studentsimulator` logic, or tweak evaluation metrics.
> 3.  **experiment & evaluate**: run a new batch of experiments with `evaluate.py` to test the refinements and generate fresh data.

this learn-adjust-retest loop is fundamental to making progress.

## the tricky bits & what's next

this research journey is defined by its challenges as much as its discoveries. `batch_run_20250518_121146` highlighted several:
*   **data scale vs. insight depth**: ~10-11 pairs per profile give initial signals but aren't enough for definitive conclusions on specific concept/profile interactions. balancing the need for more data with budget and time constraints is key.
*   **the art of "good" adaptation**: designing truly effective adaptive prompts is an ongoing experiment. what makes an explanation "adhd-friendly" for algebra vs. history? the current prompts are evolving hypotheses.
*   **budget realities**: the (nominal) $50 budget forces efficiency. every llm call must be justified, pushing us towards smarter, more targeted experimentation rather than brute-force data collection.

**immediate forward vectors:**
1.  **targeted data acquisition**: future runs will focus on areas where `batch_run_20250518_121146` showed ambiguity (e.g., adhd profile) or where specific prompt strategies warrant deeper testing.
2.  **prompt evolution**: rigorously analyze underperforming adaptive sessions from `experiment_pair_details.csv` (for `batch_run_20250518_121146`) to iteratively improve instruction sets in `prompts/`.
3.  **smarter simulation**: explore enhancements to `studentsimulator` – can we incorporate more nuanced cognitive load indicators or profile-specific feedback patterns observed in the data, moving beyond broad clarity scores?

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
source .venv/bin/activate  # on windows: .venv\scripts\activate

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