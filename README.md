# Synapz: Adaptive Learning for Neurodiverse Students

Synapz is a focused project demonstrating how adaptive learning techniques can be tailored for neurodiverse students. The system generates educational content adapted to different cognitive profiles (ADHD, dyslexic, visual learners) and scientifically compares their effectiveness against non-adapted (control) approaches, all within a constrained development timeframe and budget.

## Core Thesis

The central hypothesis of Synapz is that **adaptive teaching methodologies, tailored to specific neurodiverse cognitive profiles, can produce irrefutably superior learning outcomes compared to static, one-size-fits-all approaches, even within significant constraints (budget, time, local compute).** The project aims to generate objective evidence to support or refute this.

## Key Features

-   **Targeted Adaptation**: Content generation specifically designed for ADHD, dyslexic, and visual learning profiles.
-   **Scientific Control**: Rigorous comparison against a non-adapted control group to isolate the effect of adaptation.
-   **Quantitative Evaluation**: A suite of text analysis and statistical metrics to measure adaptation effectiveness and learning outcomes.
-   **Budget-Conscious Design**: Strict OpenAI API budget ($50) enforcement with pre-call cost projection and continuous monitoring.
-   **Reliable Data Management**: WAL-enabled SQLite database for robust storage of all experimental data, including session histories and detailed metrics.
-   **Automated Batch Evaluation**: A script (`synapz/evaluate.py`) to run multiple experiment pairs, collect data, and compile comprehensive results.

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/synapz.git # Replace with your actual repo URL
cd synapz

# Create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate # On Windows use `.venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt

# Set OpenAI API key (ensure this is done securely and not hardcoded)
export OPENAI_API_KEY='your-api-key'
```

## Usage

The primary way to generate evidence and analyze results is through the batch evaluation script.

### Running Batch Evaluations & Generating Reports

```bash
# Run a batch evaluation (e.g., 5 experiment pairs, 5 turns per session)
python synapz/evaluate.py --size 5 --turns 5 --budget 10.0

# Customize further:
# --db-path: Specify a custom path for the evaluation database.
# --no-llm-simulator: Use a simpler heuristic student simulator (faster, no API cost for simulation).
# --teacher-model: Specify the OpenAI model for the TeacherAgent (e.g., gpt-4o, gpt-4o-mini).
# --simulator-model: Specify the OpenAI model for the LLM-based StudentSimulator.
# --learner-id & --concept-id: Run for a single specific learner/concept pair.

# The results, including a 'compiled_batch_results.json' and visualizations,
# will be saved in a timestamped subdirectory within the 'results/' folder
# (e.g., results/batch_run_YYYYMMDD_HHMMSS/).
```

The `compiled_batch_results.json` file is crucial as it contains:
-   Detailed metrics for each experiment pair (adaptive vs. control).
-   Aggregated results per learner profile, showing performance trends.
-   Overall aggregate results for the entire batch.
-   Statistical significance analysis (p-values, effect sizes).
-   An evidence summary concluding the strength of findings based on the run.

Visualizations (charts and plots comparing clarity, costs, etc.) are also automatically generated in a `visualizations` subfolder within the batch run's results directory, aiding in the interpretation of the data.

## Project Structure

-   `synapz/`
    -   `core/`: Core logic for the `LLMClient`, `BudgetTracker`, `TeacherAgent`, `StudentSimulator`, `CognitiveProfile` management, and `Database` interactions.
    -   `data/`:
        -   `concepts/`: JSON files defining educational concepts.
        -   `profiles/`: JSON files defining learner cognitive profiles.
        -   `metrics.py`: Functions for calculating all analytical metrics.
        -   `visualization.py`: Script to generate plots from batch results.
        -   `*.db`: SQLite database files (e.g., `synapz_eval.db`).
    -   `prompts/`: Text files containing the system prompts for the LLM agents.
    -   `tests/`: Unit tests for various components.
    -   `evaluate.py`: The main script for running batch evaluations and experiments.
    -   `cli.py`: An alternative CLI for running single interactive or simulated sessions (more for qualitative exploration).
-   `results/`: Output directory for batch evaluations, with timestamped subfolders for each run.
-   `README.md`: This file.
-   `requirements.txt`: Python package dependencies.

## Project Development & Insights: The Journey So Far

This section outlines the thought process behind Synapz, key development stages, insights from experimentation, and future directions.

### Initial Approach & Iterations

1.  **Defining Profiles & Concepts (The Ground Truth)**: The foundation was laid by creating detailed JSON representations for learner cognitive profiles (ADHD, Dyslexic, Visual) and a diverse set of educational concepts (e.g., algebra, binary search). These files in `synapz/data/profiles/` and `synapz/data/concepts/` serve as the structured knowledge base, ensuring consistency and providing rich context to the LLM agents.
2.  **LLM-Powered Teacher & Simulator (The Actors)**: An `LLMClient` (`synapz/core/llm_client.py`) was developed as the gateway to OpenAI models. Crucially, it incorporates robust budget tracking (`BudgetTracker`) from the outset to adhere to the $50 constraint, along with exponential backoff and retry mechanisms for API stability. This client powers both the `TeacherAgent` (which generates adaptive or control explanations) and the `StudentSimulator` (which provides AI-generated feedback on those explanations, mimicking a student with a specific cognitive profile). The choice of an LLM for the simulator, despite its cost, was to aim for more nuanced and realistic feedback than purely heuristic models could offer.
3.  **Adaptive vs. Control Logic (The Experiment Design)**: The `TeacherAgent` (`synapz/core/teacher.py`) is designed to operate in 'adaptive' mode (using profile-specific system prompts) or 'control' mode (using a standard, non-adaptive prompt). This A/B testing setup is fundamental to the project's goal of gathering comparative evidence.
4.  **Metrics & Evaluation Rigor (The Measurement System)**: A significant portion of development was dedicated to creating a comprehensive evaluation framework, primarily within `synapz/evaluate.py` and `synapz/data/metrics.py`. The philosophy was not just to run simulations but to gather quantifiable, actionable evidence. Key metrics include student-rated clarity improvement (primary success indicator), final clarity, API costs, and measures of pedagogical difference (textual and tag-based). Statistical significance testing (p-values via t-tests, effect sizes via Cohen's d) is applied to compare adaptive and control groups rigorously.
5.  **Database for Persistence (The Record Keeper)**: SQLite, via `synapz/core/models.py`, is used to store all session interactions, detailed metrics, and experiment results. The choice of SQLite with WAL (Write-Ahead Logging) mode enabled ensures reliable data storage and allows for more robust concurrent access if the system were to be expanded, preventing common database locking issues.

### Iterative Refinement & Debugging (The "Real Work")

The path to the current system was highly iterative, involving numerous cycles of coding, testing, and analysis:
*   **Prompt Engineering**: This was, and continues to be, a critical and ongoing task. Refining the system prompts (in `synapz/prompts/`) for the `TeacherAgent` (to elicit genuinely different and effective teaching strategies for each profile) and the `StudentSimulator` (to ensure its feedback is authentic to the specified profile and sensitive to the nuances of the explanation) is paramount. Early prompts sometimes led to generic outputs or insufficient differentiation.
*   **Metrics Calculation Logic**: Ensuring the accurate and robust calculation of all defined metrics. This involved:
    *   Debugging data aggregation in `_compile_batch_results` within `evaluate.py` to correctly summarize individual experiment pair data up to profile-level and overall batch statistics.
    *   Handling `None` values or unexpected data types gracefully, especially when calculating averages or differences where data for a particular session might be incomplete.
    *   Verifying the correct application of statistical tests from `scipy.stats` and ensuring that the data met their assumptions (or acknowledging when it might not, e.g., due to small sample sizes).
*   **Error Handling & System Stability**: Making the `evaluate.py` script resilient to API errors (beyond what `LLMClient` handles), unexpected file issues, or data inconsistencies. This included adding more detailed logging throughout the pipeline and ensuring graceful fallbacks where possible to prevent entire batch runs from failing due to isolated issues.
*   **Output & Visualization**: Developing clear, informative console outputs during batch runs was important for monitoring. The `synapz/data/visualization.py` script was created to automatically generate charts from the `compiled_batch_results.json`, but this also required iteration. Initial plots were sometimes empty or hard to interpret, often due to data parsing issues in the plotting functions or, more frequently, subtle errors in how the data was structured in the upstream results compilation. Ensuring legibility and direct relevance to the core thesis for each chart was a key refinement goal.

### Insights from Recent Batch Run (`batch_run_20250518_091436`)

The latest batch run (5 experiment pairs, 5 turns per session) yielded the following high-level observations:

*   **Overall Performance**: The adaptive approach showed an overall win rate (clarity improvement) of **40.00%** compared to the control approach. The evidence strength was categorized as **WEAK**.
    *   *Initial Hypothesis Check*: While not yet "irrefutable," this result provides a very tentative indication and highlights the need for more data. The current evidence does not strongly support the core thesis.
*   **Profile-Specific Performance**:
    *   **Dyslexic Learner Profile**: Showed an adaptive win rate of **33.33%** on clarity improvement. The average final clarity for adaptive sessions was **4.00** vs. **2.33** for control. The p-value for final clarity difference was **0.038**, suggesting a statistically significant improvement in final clarity for the adaptive approach with this profile in this limited sample.
    *   **Visual Learner Profile**: Showed an adaptive win rate of **50.00%**. Average final clarity: Adaptive **2.50**, Control **3.00**. The p-value for final clarity difference was **0.500**, suggesting no statistically significant difference in final clarity. In fact, the control group had a slightly higher average final clarity.
    *   *(ADHD profile data was not part of this specific run; more diverse runs are needed.)*
*   **Cost Analysis**:
    *   Average adaptive session cost: **$0.1065**
    *   Average control session cost: **$0.1025**
    *   The overall batch run cost **$1.05**, staying well within typical experimental budgets.
    *   *Observation*: Adaptive sessions cost slightly more than control sessions in this run, by about $0.004 per session. This small difference is currently acceptable but should be monitored.
*   **Pedagogical Differentiation**:
    *   The average text difference (Levenshtein-based) between adaptive and control explanations was **77.9%**.
    *   The average pedagogical tag difference (Jaccard-based) was **100.0%** (meaning entirely different tags were used, or one set was empty).
    *   *Observation*: This indicates that the adaptive system **is** generating significantly different textual content and employing different pedagogical strategies compared to the control, which is a necessary (but not sufficient) condition for effective adaptation. The challenge is ensuring this differentiation translates to improved outcomes.

*(Self-correction/Honesty): It's crucial to underscore that with only 5 experiment pairs, the statistical power of this run is very low. The p-value of 0.038 for the Dyslexic profile's final clarity is promising but must be interpreted with extreme caution. More extensive runs with larger sample sizes per profile are essential before drawing any firm conclusions. The "WEAK" overall evidence strength reflects this directly.*

### Current Status & Next Steps

**What Works Well:**
*   **Core Experimental Pipeline**: The system can reliably run batches of adaptive vs. control teaching sessions.
*   **Student Simulation**: The `StudentSimulator` provides differentiated, LLM-generated feedback that reflects cognitive profiles, crucial for automated testing.
*   **Metrics & Reporting**: Comprehensive metrics are collected, aggregated, and saved in structured JSON reports. Statistical calculations for p-values and effect sizes are implemented.
*   **Data Visualization**: Basic charts are automatically generated, providing an initial visual summary of results.
*   **Budget Management**: The `BudgetTracker` effectively monitors and enforces API cost limits.
*   **Modularity**: The codebase is organized into distinct components (core, data, prompts, tests), facilitating maintenance and extension.

**Key Challenges & Areas for Improvement:**
1.  **Statistical Power & Experiment Scale**: The most significant limitation is the small number of experiment pairs run so far. This prevents drawing statistically robust conclusions.
2.  **Effectiveness of Adaptation**: While the system *differentiates* its teaching (as shown by text/tag differences), this differentiation needs to translate more consistently into *significantly better learning outcomes* (clarity improvement) across all profiles. The current "WEAK" evidence highlights this.
3.  **Student Simulator Fidelity**: While the LLM-based simulator is an advancement, continuously improving its alignment with the actual learning experiences and challenges of neurodiverse students is vital. Its current feedback, while structured, might not capture all subtleties.
4.  **Depth of Analysis**:
    *   The current analysis focuses on aggregate session outcomes. A deeper dive into turn-by-turn data could reveal *how and when* adaptive strategies become effective (or fail to).
    *   Correlating specific pedagogical tags with clarity improvements for each profile could yield insights into *which* adaptive techniques are most promising.
5.  **Prompt Engineering Nuance**: The effectiveness of the `TeacherAgent` hinges on its system prompts. These need ongoing refinement to encourage not just different, but *more effective and targeted*, adaptive strategies for each specific cognitive profile and concept combination.

**Immediate Next Steps (Directionally Promising & Logical):**
1.  **Increase Experiment Scale (Top Priority)**:
    *   **Action**: Modify `evaluate.py` parameters or run multiple batches to achieve at least 20-30 experiment pairs *per cognitive profile*.
    *   **Rationale**: This is essential to increase statistical power and obtain more reliable p-values and effect sizes, moving beyond "weak" or "suggestive" evidence.
2.  **Targeted Prompt Refinement for `TeacherAgent`**:
    *   **Action**: Analyze the `experiment_pair_details.csv` and individual session data from `synapz_eval.db` for experiments where adaptive did NOT outperform control (or performed worse, like with the Visual Learner in the last run).
    *   Identify the adaptive strategies (pedagogy tags, explanation structure) used in those less successful sessions.
    *   Iteratively refine the system prompts in `synapz/prompts/adaptive_system.txt` (and potentially profile-specific instruction prompts if those are separated) to guide the LLM towards more demonstrably effective techniques for those specific learner_id-concept_id pairings where it underperformed. For example, for the Visual learner, if text-heavy explanations were still produced, the prompt needs to more strongly enforce visual-centric outputs.
    *   **Rationale**: Generic adaptation isn't enough; the adaptation must be *effective*. This requires a data-driven approach to prompt tuning.
3.  **Enhance Visualization & Reporting for Deeper Insights**:
    *   **Action**: Extend `synapz/data/visualization.py` to generate more insightful plots:
        *   Side-by-side box plots or violin plots of clarity scores (adaptive vs. control) for each profile.
        *   Effect size plots (e.g., forest plots) with confidence intervals for key metrics.
        *   Turn-by-turn clarity progression plots.
    *   **Action**: Refine the automated `evidence_summary` in `compiled_batch_results.json` to provide a more nuanced interpretation, considering both p-values and effect sizes (e.g., "small but significant effect," "large but non-significant trend").
    *   **Rationale**: Better visualizations and summaries will make it easier to understand the results and communicate findings.
4.  **Review and Refine `StudentSimulator` Prompts**:
    *   **Action**: Critically evaluate the `synapz/prompts/student_sim.txt`. Ensure its guidelines for each cognitive profile are distinct and encourage feedback that truly reflects the profile's challenges and preferences when faced with different teaching styles.
    *   **Rationale**: The quality of the entire evaluation hinges on the simulator providing authentic and differentiated feedback. If the simulator is too lenient or its feedback isn't well-aligned with profile characteristics, the "clarity" scores may not be meaningful.

**Long-Term Vision:**
*   **Achieve "Irrefutable Evidence"**: Through scaled experiments and refined adaptive strategies, aim to consistently demonstrate statistically significant and practically meaningful improvements in learning outcomes for multiple neurodiverse profiles.
*   **Explore Advanced Adaptation**: Move beyond rule-based prompt changes to enable the `TeacherAgent` to learn and dynamically adjust its teaching strategies based on a richer model of the student's understanding and cognitive state over time.
*   **Human-in-the-Loop Evaluation**: Eventually, validate the simulator's findings and the system's effectiveness with real neurodiverse students.
*   **Content Domain Expansion**: Test the system across a wider range of subjects and concept complexities.

This iterative cycle of hypothesis, development, rigorous testing, honest analysis of results (even when they don't immediately confirm the thesis), and targeted refinement is the core of the Synapz project. 