"""Experiment runner for testing chain-of-thought reasoning for ADHD learners."""

import uuid
import json
import yaml
import random
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from rich.panel import Panel
from rich.markdown import Markdown
import logging
from rich.console import Console
from rich.table import Table
from rich import box

from synapz.core.models import Database
from synapz.models.learner_profiles import load_profile
from synapz.models.concept_loader import load_concept
from synapz.core.llm_client import LLMClient
from synapz.core.teacher import TeacherAgent
from synapz.core.budget import BudgetTracker, BudgetExceededError
from synapz.core.simulation import simulate_student_clarity

CONFIG_PATH = Path(__file__).parent / "reasoning_config.yaml"
PROMPT_DIR_PLACEHOLDER = "synapz/prompts/"

# Reconfigure logging to play nicer with Rich Console
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', handlers=[logging.FileHandler("reasoning_experiment.log")]) # Log to file
    # If you still want console output for logging, but Rich will handle most prints:
    # console_handler = logging.StreamHandler()
    # console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
    # logger.addHandler(console_handler)

# Force debug level for specific troublesome modules
logging.getLogger('synapz.models.concepts').setLevel(logging.DEBUG)
logging.getLogger('synapz.models.learner_profiles').setLevel(logging.DEBUG)

# --- START HARDCODED CONCEPT DATA ---
HARDCODED_CONCEPTS = {
    "variables": {
      "id": "variables",
      "title": "Variables and Constants",
      "difficulty": 1,
      "description": "Variables are symbols (typically letters like x, y, or a) that represent quantities that can change or take on different values within a mathematical expression or equation. Constants are specific, fixed numerical values that do not change throughout a problem (e.g., 5, -2, \\u03c0). Understanding the distinction is fundamental to algebra, as variables allow us to generalize relationships and solve for unknowns, while constants provide the fixed parameters of those relationships.",
      "key_ideas": [
        "Symbolic Representation: Variables use letters or symbols to stand in for numbers.",
        "Unknown Values: Variables often represent quantities we need to find or solve for.",
        "Changing Values: The value of a variable can vary depending on the context or equation.",
        "Fixed Values: Constants maintain their value throughout a specific problem or context.",
        "Generalization: Variables enable the creation of general formulas and rules applicable to many situations.",
        "Placeholders: Variables can be seen as placeholders for numbers until their specific value is determined or assigned."
      ],
      "examples": [
        {"type": "Simple Identification", "problem": "Identify the variables and constants in the expression: 3a - 7 + b", "solution": "Variables: a, b. Constants: 3, -7.", "explanation": "Letters 'a' and 'b' can represent different numbers, so they are variables. '3' (coefficient of 'a') and '-7' are fixed numbers, hence constants."},
        {"type": "Equation Context", "problem": "In the equation for a line, y = mx + c, identify the variables and parameters often treated as constants for a specific line.", "solution": "'x' and 'y' are variables representing coordinates on the line. 'm' (slope) and 'c' (y-intercept) are parameters that are constant for a specific line but can change to define different lines.", "explanation": "As you move along a line, 'x' and 'y' change. However, for any single given line, its slope 'm' and y-intercept 'c' are fixed."},
        {"type": "Formula Application", "problem": "The perimeter P of a rectangle is given by P = 2l + 2w. If the length l is 5 units and the width w is 3 units, what is the perimeter? Identify variables and constants in this specific calculation.", "solution": "P = 2(5) + 2(3) = 10 + 6 = 16 units. In the formula P = 2l + 2w, P, l, and w are variables. The number 2 is a constant. In this specific calculation, l=5 and w=3 are treated as specific values of the variables.", "explanation": "The formula itself uses l and w as variables. When we assign them values, we are evaluating the expression for a specific case."}
      ],
      "common_misconceptions": [
        "Confusing variables with specific unknown numbers: While variables often represent unknowns we solve for, their nature is that they *can* vary. The solution to an equation is a specific value a variable takes in that context.",
        "Thinking coefficients are variables: In an term like '3x', 'x' is the variable and '3' is a constant coefficient modifying the variable.",
        "Assuming a letter always means the same variable: 'x' in one problem might be different from 'x' in another unless specified.",
        "Treating all letters as variables: In physics formulas like E=mc\\u00b2, 'E' and 'm' are variables (energy and mass), but 'c' (speed of light) is a universal physical constant."
      ],
      "real_world_applications": [
        "Programming: Variables store data that can change during program execution (e.g., user input, scores).",
        "Finance: Formulas for interest calculation use variables for principal, rate, and time.",
        "Science: Experiments use variables to represent different factors being tested and measured (e.g., temperature, pressure).",
        "Engineering: Design formulas involve variables for dimensions, material properties, and loads.",
        "Cooking: Recipes can be scaled using variables to represent ingredient quantities for different serving sizes."
      ],
      "tags": ["algebra", "foundational", "symbolism", "equations"]
    },
    "expressions": {
      "id": "expressions",
      "title": "Algebraic Expressions",
      "difficulty": 1,
      "description": "An algebraic expression is a mathematical phrase that combines numbers (constants), variables (letters representing unknown values), and operational symbols (+, -, ×, ÷, exponents). Unlike an equation, an algebraic expression does not contain an equals sign and therefore cannot be 'solved' for a specific value of the variable (though it can be evaluated if the variable's value is known, or simplified). Expressions are the building blocks of equations and inequalities.",
      "key_ideas": [
        "Variable: A symbol (usually a letter like x, y, or a) that represents an unknown quantity or a quantity that can change.",
        "Constant: A fixed numerical value (e.g., 5, -2, π).",
        "Term: A single number, a variable, or numbers and variables multiplied together. Terms are separated by + or - signs (e.g., in 3x + 2y - 5, the terms are 3x, 2y, and -5).",
        "Coefficient: The numerical factor of a term that contains a variable (e.g., in 3x, the coefficient is 3).",
        "Like Terms: Terms that have the exact same variable(s) raised to the exact same power(s) (e.g., 3x² and -5x² are like terms, but 3x and 3x² are not).",
        "Simplifying Expressions: Combining like terms to make the expression shorter and easier to understand.",
        "Evaluating Expressions: Substituting a specific value for each variable and calculating the result."
      ],
      "types_of_expressions": [
        "Monomial: An expression with only one term (e.g., 5x, 7, -2ab²).",
        "Binomial: An expression with two terms (e.g., 3x + 2, a² - b²).",
        "Trinomial: An expression with three terms (e.g., x² + 5x + 6, 2a - 3b + c).",
        "Polynomial: An expression with one or more terms, where exponents on variables are non-negative integers."
      ],
      "examples": [
        {"type": "Identifying parts of an expression", "expression_string": "4x² - 7y + 2x² + 5", "parts": {"terms": ["4x²", "-7y", "2x²", "5"], "variables": ["x", "y"], "constants": ["5"], "coefficients": {"x²": ["4", "2"], "y": "-7"}}},
        {"type": "Evaluating an expression", "expression_string": "3a + 2b - 1", "given_values": {"a": 4, "b": -2}, "solution_steps": ["Substitute the given values: 3(4) + 2(-2) - 1", "Perform multiplication: 12 - 4 - 1", "Perform addition/subtraction from left to right: 8 - 1 = 7"], "result": 7},
        {"type": "Simplifying an expression (combining like terms)", "expression_string": "5x + 3y - 2x + 7y - 4", "solution_steps": ["Identify like terms: (5x and -2x), (3y and 7y), (-4).", "Combine like terms: (5x - 2x) + (3y + 7y) - 4", "Result: 3x + 10y - 4"], "simplified_expression": "3x + 10y - 4"},
        {"type": "Using the distributive property", "expression_string": "3(2a - 5b)", "solution_steps": ["Multiply the term outside the parentheses by each term inside: 3 * 2a - 3 * 5b", "Result: 6a - 15b"], "simplified_expression": "6a - 15b"}
      ],
      "common_misconceptions": [
        "Confusing expressions with equations (trying to 'solve' an expression).",
        "Incorrectly combining unlike terms (e.g., adding 3x and 2y to get 5xy).",
        "Errors with signs, especially when subtracting terms or distributing a negative sign.",
        "Mistakes in the order of operations when evaluating."
      ],
      "real_world_applications": [
        "Representing unknown quantities in word problems.",
        "Formulas in science, engineering, and finance (e.g., Area = length × width is built from expressions).",
        "Calculating costs based on variable inputs (e.g., cost = $0.50 × number_of_items + $5.00 fixed_fee)."
      ],
      "prerequisites": ["Understanding of basic arithmetic operations (addition, subtraction, multiplication, division)", "Familiarity with the concept of a variable (as a placeholder)"]
    }
}
# --- END HARDCODED CONCEPT DATA ---

def load_config() -> Dict[str, Any]:
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)

def save_reasoning_data(
    db: Database,
    interaction_id: str,
    condition: str,
    reasoning_data: Dict[str, Any],
    clarity_ratings: Dict[str, float]
) -> None:
    """
    Saves detailed reasoning data to the database.

    Args:
        db: Database instance.
        interaction_id: The ID of the interaction this reasoning data belongs to.
        condition: The experimental condition ("baseline", "visible_reasoning", "hidden_reasoning").
        reasoning_data: A dictionary containing components of the reasoning response:
            - "reasoning_process": The full reasoning text.
            - "explanation": The teaching explanation provided to the student.
            - "metacognitive_supports": List of metacognitive supports.
            - "clarity_check": The clarity check text.
            # Token counts for components can be added if available
            # "tokens_reasoning_process": INTEGER
            # "tokens_explanation": INTEGER
            # "tokens_metacognitive_supports": INTEGER
            # "tokens_clarity_check": INTEGER
        clarity_ratings: Dict with "initial", "final", "improvement".
    """
    # Note: clarity_rating_initial, clarity_rating_final, clarity_improvement
    # are typically associated with a session or a series of interactions,
    # not a single reasoning step. This might need adjustment based on how
    # student simulation provides these ratings. For now, assume they are passed.

    db.log_reasoning_detail(
        interaction_id=interaction_id,
        condition=condition,
        reasoning_process_text=reasoning_data.get("reasoning_process"),
        metacognitive_supports=reasoning_data.get("metacognitive_supports", []),
        clarity_check_text=reasoning_data.get("clarity_check"),
        clarity_ratings=clarity_ratings
    )
    print(f"Saved reasoning detail for interaction {interaction_id}, condition {condition} via db.log_reasoning_detail")


def run_reasoning_experiment() -> Dict[str, Any]:
    """
    Run the reasoning experiment with three conditions based on reasoning_config.yaml.

    Returns:
        Dict containing experimental results and analysis pathway.
    """
    config = load_config()
    console = Console()  # Initialize Rich Console

    console.print() # Add spacing
    # Determine project root: synapz/experiments -> synapz -> WORKSPACE_ROOT
    # Assuming this script (reasoning_experiment.py) is in WORKSPACE_ROOT/synapz/experiments/
    project_root = Path(__file__).resolve().parent.parent.parent
    
    db_path_str = config["output_settings"]["results_db_path"] # e.g., "synapz/results/reasoning_experiment.db"
    db_path = (project_root / db_path_str).resolve()

    db_path.parent.mkdir(parents=True, exist_ok=True)
    db = Database(str(db_path))

    learner_profile_id = config["default_learner_profile_id"]
    concepts_to_test = config["concepts_to_test"]
    num_pairs = config["num_pairs_per_condition"]
    conditions_to_run = config.get("conditions_to_run", ["baseline", "visible_reasoning", "hidden_reasoning"])
    budget_limit = float(config["budget_usd"])
    api_config = config["api_settings"]
    student_sim_config = config["student_simulation"]

    # Initialize BudgetTracker instance
    # The BudgetTracker's db_path should ideally be the same as the experiment's db
    # or a dedicated budget DB. For simplicity here, let's assume it can be the same.
    # If different, TeacherAgent's db and BudgetTracker's db might diverge in logging.
    # For this experiment, budget is tracked primarily for OpenAI calls, so LLMClient's budget_tracker matters most.
    # The reasoning_experiment.py script manages its own budget display and halting.
    # Let's create a specific budget_tracker for this experiment's overall run.
    experiment_budget_tracker = BudgetTracker(db_path=str(db_path.parent / "experiment_overall_budget.db"), max_budget=budget_limit)

    prompt_dir_config = config["paths"]["prompt_dir"] # e.g., "synapz/prompts/"
    prompt_base_path = (project_root / prompt_dir_config).resolve()
    concept_path_config = config["paths"]["concept_dir"] # e.g., "synapz/data/concepts/"
    concept_base_path = (project_root / concept_path_config).resolve()
    profile_path_config = config["paths"]["learner_profile_dir"] # e.g., "synapz/data/profiles/"
    profile_base_path = (project_root / profile_path_config).resolve()

    llm_client = LLMClient() 
    teacher_agent = TeacherAgent(
        llm_client=llm_client,
        db=db, # Teacher agent uses the main experiment DB for its session/interaction logging
        teacher_model_name=api_config["model"] 
    )

    console.print(Panel(
        f"[bold]Starting Experiment: {config['experiment_name']}[/bold]\\n"
        f"Main DB: {db_path}\\n"
        f"Budget DB: {experiment_budget_tracker.db_path}\\n"
        f"Experiment Budget: ${budget_limit:.2f}",
        title="[bold green]Synapz Reasoning Experiment[/bold green]",
        border_style="green",
        expand=False
    ))
    console.print() # Add spacing

    results_summary = {
        "experiment_name": config["experiment_name"], 
        "conditions_run": [], 
        "total_cost_usd": 0.0, # This will be from the experiment_budget_tracker
        "errors": [], 
        "budget_remaining_usd": budget_limit # From experiment_budget_tracker
    }

    try:
        # profile_base_path is already resolved
        profile_full_path = profile_base_path / f"{learner_profile_id}.json"
        logger.debug(f"Attempting to load profile from: {str(profile_full_path)}")
        profile_data_dict = load_profile(str(profile_full_path))
        profile_data_json = json.dumps(profile_data_dict) 
    except FileNotFoundError:
        error_msg = f"Profile {profile_full_path} not found."
        console.print(f"[bold red]ERROR:[/bold red] {error_msg}")
        results_summary["errors"].append(error_msg); return results_summary
    except Exception as e:
        error_msg = f"Error loading profile {learner_profile_id}: {e}"
        console.print(f"[bold red]ERROR:[/bold red] {error_msg}")
        results_summary["errors"].append(error_msg); return results_summary

    total_pairs_processed_across_conditions = 0

    for condition in conditions_to_run:
        # console.print(f"\nRunning Condition: {condition}")
        console.print(Panel(f"Running Condition: [bold cyan]{condition}[/bold cyan]", title="[magenta]Condition Start[/magenta]", border_style="magenta", expand=False, padding=(1,2), style="on black"))
        console.print() # Add spacing
        condition_total_cost_usd = 0.0
        num_successful_pairs_in_condition = 0
        prompt_filename_map = {
            "baseline": "baseline_adhd_system.txt",
            "visible_reasoning": "visible_adhd_system.txt",
            "hidden_reasoning": "hidden_adhd_system.txt"
        }
        prompt_file_path = (prompt_base_path / prompt_filename_map[condition]).resolve()

        try:
            with open(prompt_file_path, "r") as f: prompt_template_content = f.read()
        except FileNotFoundError:
            error_msg = f"Prompt {prompt_file_path} not found."
            console.print(f"[bold red]ERROR:[/bold red] {error_msg}")
            results_summary["errors"].append(error_msg); continue 

        for i in range(num_pairs):
            # Check overall experiment budget before starting a new pair
            if experiment_budget_tracker.is_exceeded():
                console.print("[bold orange_red1]Overall experiment budget depleted. Stopping experiment.[/bold orange_red1]")
                break 

            if i >= len(concepts_to_test):
                console.print(f"[yellow]Warn:[/yellow] Not enough concepts for {num_pairs} pairs. Stopping for {condition}."); break
            
            raw_concept_id = concepts_to_test[i]
            concept_id = raw_concept_id.strip() # Ensure no leading/trailing whitespace
            if raw_concept_id != concept_id:
                logger.debug(f"    DEBUG: Stripped whitespace from concept_id: '{raw_concept_id}' -> '{concept_id}'")

            pair_identifier = f"L:{learner_profile_id}, C:{concept_id}, Cond:{condition} (Pair {i+1}/{num_pairs})"
            session_id = db.create_session(learner_id=learner_profile_id, concept_id=concept_id, experiment_type=f"reasoning_{condition}")

            console.print(Panel(
                f"[bold]Processing Pair {i+1}/{num_pairs}[/bold]\\n"
                f"Learner: {learner_profile_id}\\n"
                f"Concept ID: {concept_id}\\n"
                f"Condition: {condition}\\n"
                f"Session ID: {session_id}",
                title="[dim blue]Current Task[/dim blue]",
                border_style="blue",
                expand=False,
                padding=(1,1),
                style="on black"
            ))
            console.print() # Add spacing

            concept_data_dict: Optional[Dict[str, Any]] = None
            concept_data_json: Optional[str] = None

            if concept_id in HARDCODED_CONCEPTS:
                logging.warning(f"Using hardcoded data for concept: {concept_id}")
                concept_data_dict = HARDCODED_CONCEPTS[concept_id]
                concept_data_json = json.dumps(concept_data_dict)
            else:
                try:
                    concept_full_path = concept_base_path / f"{concept_id}.json"
                    logger.debug(f"    Attempting to load concept from: {str(concept_full_path)} (Cleaned ID: '{concept_id}')") 
                    
                    if not concept_full_path.exists():
                        logger.error(f"    Direct check: Concept file {concept_full_path} does NOT exist or is not accessible.")
                        raise FileNotFoundError(f"Direct check failed for {concept_full_path}")
                    else:
                        logger.info(f"    Direct check: Concept file {concept_full_path} exists.")

                    # Standard loading for all concepts
                    logger.critical(f"CRITICAL_LOG (reasoning_experiment): About to call load_concept for concept_id: {repr(concept_id)}, with path string: {repr(str(concept_full_path))}")
                    concept_data_dict = load_concept(str(concept_full_path))
                    
                    concept_data_json = json.dumps(concept_data_dict)
                except FileNotFoundError:
                    error_msg = f"Error loading concept {concept_id}: Concept not found: {concept_full_path}. Skip."
                    console.print(f"[bold red]ERROR:[/bold red] {error_msg}")
                    results_summary["errors"].append(f"{pair_identifier}: {error_msg}"); 
                    continue 
                except Exception as e:
                    error_msg = f"Error loading concept {concept_id}: {e}. Skip."
                    console.print(f"[bold red]ERROR:[/bold red] {error_msg}")
                    results_summary["errors"].append(f"{pair_identifier}: {error_msg}"); 
                    continue

            with open(prompt_file_path, 'r') as f_prompt:
                system_prompt_template = f_prompt.read()

            try:
                formatted_prompt = system_prompt_template.format(
                    learner_profile=profile_data_json, 
                    concept=concept_data_json
                )
            except KeyError as ke:
                logging.error(f"KeyError during prompt formatting for {prompt_file_path}. Error: {ke}")
                logging.error(f"Problematic key: {repr(ke.args[0])}")
                logging.error(f"System prompt template was:\n---{{system_prompt_template[:1000]}}...---") # Corrected escape
                results_summary["errors"].append(f"PromptFormat KeyError for {prompt_file_path}: {ke}")
                continue # Skip this pair
            
            projected_tokens_in = teacher_agent.count_tokens(formatted_prompt, api_config["model"]) 
            projected_tokens_out = api_config.get("max_tokens_completion", 700) 
            
            # Use LLMClient for cost projection as it has the most direct pricing info
            projected_call_cost = llm_client.calculate_cost(projected_tokens_in, projected_tokens_out, api_config["model"])

            try:
                # Check budget with experiment_budget_tracker
                if not experiment_budget_tracker.check_budget(projected_call_cost):
                    raise BudgetExceededError(f"Projected cost ${projected_call_cost:.4f} exceeds remaining run budget of ${experiment_budget_tracker.get_remaining_run_budget():.4f}")

                status_message = (
                    f"[cyan]LLM call for '{concept_id}' ({condition})... "
                    f"Budget Left: ${experiment_budget_tracker.get_remaining_run_budget():.4f} | "
                    f"Projected Cost: ${projected_call_cost:.4f}[/cyan]"
                )
                with console.status(status_message, spinner="dots12", spinner_style="cyan"):
                    response_data, actual_tokens_in, actual_tokens_out, current_call_cost_usd = teacher_agent.get_reasoning_response(
                        prompt_text=formatted_prompt, 
                        max_tokens=api_config.get("max_tokens_completion", 1500),
                        model_name=api_config["model"], 
                        temperature=api_config.get("temperature", 0.5)
                    )
                
                # Log usage with experiment_budget_tracker using the cost from LLMClient/TeacherAgent
                experiment_budget_tracker.log_usage(
                    model=api_config["model"],
                    tokens_in=actual_tokens_in,
                    tokens_out=actual_tokens_out,
                    cost_usd=current_call_cost_usd # Pass pre-calculated cost
                )
                
                # Update overall summary from the experiment_budget_tracker
                results_summary["total_cost_usd"] = experiment_budget_tracker.get_current_run_spend()
                results_summary["budget_remaining_usd"] = experiment_budget_tracker.get_remaining_run_budget()
                condition_total_cost_usd += current_call_cost_usd # Still track condition cost separately for condition summary

                if not isinstance(response_data, dict):
                    if condition == "baseline" and isinstance(response_data, str): 
                        response_data = {"explanation": response_data} 
                    else:
                        # If it's a reasoning condition and not a dict, or baseline and not string/dict, something's wrong
                         error_detail = response_data if isinstance(response_data, str) else json.dumps(response_data)[:200]
                         raise ValueError(f"API response was not a dict as expected for condition '{condition}'. Got: {type(response_data)} Content: {error_detail}")


                if condition == "visible_reasoning":
                    raw_reasoning_text_list = response_data.get("reasoning_process", [])
                    if not isinstance(raw_reasoning_text_list, list):
                        logger.warning(f"Visible reasoning process was not a list as expected. Got type: {type(raw_reasoning_text_list)}. Treating as single item list.")
                        raw_reasoning_text_list = [str(raw_reasoning_text_list)]
                    # Join list items with newlines for Rich Markdown rendering
                    display_reasoning_text = '\n'.join(raw_reasoning_text_list)
                    console.print(Panel(Markdown(display_reasoning_text), title="[bold blue]📝 LLM Reasoning Process (Visible)[/bold blue]", border_style="blue", expand=False))

                explanation_content = response_data.get("explanation", "Error: No explanation in response")

                initial_clarity, final_clarity, clarity_improvement = simulate_student_clarity(explanation_content, student_sim_config)
                
                interaction_id = db.log_interaction(
                    session_id, 1, explanation_content, final_clarity, 
                    f"{condition}_reasoning", 
                    response_data.get("metacognitive_supports", []), 
                    actual_tokens_in, actual_tokens_out, current_call_cost_usd # Log cost from LLM call here
                )
                
                clarity_ratings_for_log = {"initial": initial_clarity, "final": final_clarity, "improvement": clarity_improvement}
                data_to_save_for_reasoning = response_data.copy()
                
                # Prepare reasoning_process for saving
                reasoning_process_data = data_to_save_for_reasoning.get("reasoning_process")
                if isinstance(reasoning_process_data, list):
                    data_to_save_for_reasoning["reasoning_process"] = json.dumps(reasoning_process_data)
                elif reasoning_process_data is not None: # It's not a list but not None (e.g. a string from an old format or error)
                    logger.warning(f"Reasoning process for saving was not a list. Type: {type(reasoning_process_data)}. Converting to JSON string of a list.")
                    data_to_save_for_reasoning["reasoning_process"] = json.dumps([str(reasoning_process_data)])
                # If None, it will be handled by .get() in save_reasoning_data or db layer

                if condition == "baseline": 
                    # For baseline, ensure reasoning_process is explicitly None if not already,
                    # as it's not expected to produce one.
                    # The explanation_content is already extracted.
                    data_to_save_for_reasoning = {
                        "explanation": explanation_content,
                        "reasoning_process": None, # Explicitly None for baseline
                        "metacognitive_supports": response_data.get("metacognitive_supports", []), 
                        "clarity_check": response_data.get("clarity_check") 
                    }
                
                save_reasoning_data(db, interaction_id, condition, data_to_save_for_reasoning, clarity_ratings_for_log)
                num_successful_pairs_in_condition += 1
                total_pairs_processed_across_conditions +=1
                console.print(
                    f"    [green]Pair {i+1} done.[/green] Call Cost: [bold]${current_call_cost_usd:.6f}[/bold]. "
                    f"Exp. Spend: [bold]${results_summary['total_cost_usd']:.4f}[/bold]. "
                    f"Budget Left: [bold]${results_summary['budget_remaining_usd']:.4f}[/bold]"
                )
                console.print() # Add spacing

            except BudgetExceededError as be:
                error_msg = f"BUDGET EXCEEDED: {be} during {pair_identifier}. HALT."
                console.print(f"[bold red]FATAL:[/bold red] {error_msg}")
                results_summary["errors"].append(error_msg)
                results_summary["conditions_run"].append({"condition": condition, "num_pairs": num_successful_pairs_in_condition, "cost_usd": condition_total_cost_usd, "status": "Budget Exceeded"})
                console.print(Panel(f"Experiment HALTED due to budget. Total Spend: ${results_summary['total_cost_usd']:.4f}", title="[bold orange_red1]BUDGET EXCEEDED[/bold orange_red1]", border_style="red"));
                console.print() # Add spacing
                return results_summary
            except Exception as e:
                error_msg = f"Error processing {pair_identifier}: {type(e).__name__} - {e}"
                console.print(f"[bold red]ERROR:[/bold red] {error_msg}", style="on black") # Keep on black for visibility if console bg changes
                import traceback; console.print_exception(show_locals=True)
                results_summary["errors"].append(error_msg)
                if experiment_budget_tracker.is_exceeded():
                    console.print(f"[bold orange_red1]Budget ${budget_limit:.2f} reached after error. HALT.[/bold orange_red1]")
                    results_summary["conditions_run"].append({"condition": condition, "num_pairs": num_successful_pairs_in_condition, "cost_usd": condition_total_cost_usd, "status": "Halted post-error (Budget)"})
                    return results_summary
                continue 
            
            if experiment_budget_tracker.is_exceeded():
                console.print(f"[bold orange_red1]Budget ${budget_limit:.2f} reached. HALT Exp.[/bold orange_red1]")
                break # Break from pairs loop
        
        results_summary["conditions_run"].append({"condition": condition, "num_pairs_completed": num_successful_pairs_in_condition, "cost_usd": condition_total_cost_usd, "status": "Completed" if num_successful_pairs_in_condition == num_pairs else "Partially Completed"})
        if experiment_budget_tracker.is_exceeded():
            console.print("[bold orange_red1]Overall experiment budget depleted. Stop all conditions.[/bold orange_red1]")
            console.print() # Add spacing
            break # Break from conditions loop

    console.print(Panel("[bold green] Experiment Finished [/bold green]", border_style="green", expand=False, style="on black"))
    console.print() # Add spacing

    summary_table = Table(title="[bold]Overall Experiment Summary[/bold]", box=box.ROUNDED, show_header=True, header_style="bold magenta")
    summary_table.add_column("Metric", style="dim", overflow="fold")
    summary_table.add_column("Value", overflow="fold")
    summary_table.add_row("Experiment Name", results_summary["experiment_name"])
    summary_table.add_row("Total Pairs Processed", str(total_pairs_processed_across_conditions))
    summary_table.add_row("Total Cost (USD)", f"${results_summary['total_cost_usd']:.4f}")
    summary_table.add_row("Budget Allowance (USD)", f"${budget_limit:.2f}")
    summary_table.add_row("Budget Remaining (USD)", f"${results_summary['budget_remaining_usd']:.4f}")
    console.print(summary_table)
    console.print() # Add spacing

    if results_summary.get("conditions_run"):
        conditions_table = Table(title="[bold]Condition Summaries[/bold]", box=box.ROUNDED, show_header=True, header_style="bold blue")
        conditions_table.add_column("Condition")
        conditions_table.add_column("Pairs Completed")
        conditions_table.add_column("Cost (USD)")
        conditions_table.add_column("Status")

        for cond_run in results_summary["conditions_run"]:
            conditions_table.add_row(
                cond_run.get("condition"),
                str(cond_run.get("num_pairs_completed")),
                f"${cond_run.get('cost_usd', 0.0):.4f}",
                cond_run.get("status")
            )
        console.print(conditions_table)
        console.print() # Add spacing

    if results_summary["errors"]:
        console.print("\n[bold red]Errors Encountered During Experiment:[/bold red]")
        console.print() # Add spacing
        for err_idx, err in enumerate(results_summary["errors"]):
            console.print(Panel(str(err), title=f"[yellow]Error {err_idx+1}[/yellow]", border_style="red", expand=True))
            console.print() # Add spacing
    
    summary_file_path = db_path.parent / f"{config['experiment_name']}_summary.json"
    with open(summary_file_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    console.print(f"\n[cyan]Experiment summary JSON saved to:[/cyan] [link=file://{summary_file_path.resolve()}]{summary_file_path}[/link]")

    return results_summary

if __name__ == '__main__':
    console = Console() # For main block
    console.print("[yellow]Running reasoning_experiment.py example directly...[/yellow]")
    console.print() # Add spacing
    # The main script now relies on actual (or more deeply mocked) 
    # synapz.models.*, synapz.core.teacher, synapz.core.budget, synapz.core.simulation
    # Ensure these are available or provide robust mocks if running this main block directly.

    # For a true standalone test of this script's logic, if other modules are incomplete:
    # You might need to temporarily re-mock load_profile, load_concept, TeachingAgent, 
    # budget_tracker, BudgetExceededError, and simulate_student_clarity here.
    # Example of re-mocking for standalone run:
    # from unittest.mock import MagicMock
    # synapz.models.learner_profiles.load_profile = MagicMock(return_value={"id": "adhd_learner", ... })
    # synapz.core.teaching_agent.TeachingAgent = MagicMock() # etc.

    try:
        results = run_reasoning_experiment()
        # console.print("\nExperiment Results Summary (from __main__):")
        # console.print(json.dumps(results, indent=2)) # The function now prints a Rich summary
    except Exception as e:
        console.print(f"[bold red]Error running main experiment block from __main__:[/bold red] {type(e).__name__} {e}")
        console.print() # Add spacing
        console.print_exception(show_locals=True) 