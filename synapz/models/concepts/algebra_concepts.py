"""Core algebra concepts for teaching experiments."""

import json
import os
from typing import Dict, List, Any
from pathlib import Path

from synapz import PACKAGE_ROOT

# Define concepts directory
CONCEPTS_DIR = PACKAGE_ROOT / "models" / "concepts" / "data"

# Ensure concepts directory exists
os.makedirs(CONCEPTS_DIR, exist_ok=True)

ALGEBRA_CONCEPTS: List[Dict[str, Any]] = [
    {
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
            {
                "type": "Identifying parts of an expression",
                "expression_string": "4x² - 7y + 2x² + 5",
                "parts": {
                    "terms": ["4x²", "-7y", "2x²", "5"],
                    "variables": ["x", "y"],
                    "constants": ["5"],
                    "coefficients": {"x²": ["4", "2"], "y": "-7"}
                }
            },
            {
                "type": "Evaluating an expression",
                "expression_string": "3a + 2b - 1",
                "given_values": {"a": 4, "b": -2},
                "solution_steps": [
                    "Substitute the given values: 3(4) + 2(-2) - 1",
                    "Perform multiplication: 12 - 4 - 1",
                    "Perform addition/subtraction from left to right: 8 - 1 = 7"
                ],
                "result": 7
            },
            {
                "type": "Simplifying an expression (combining like terms)",
                "expression_string": "5x + 3y - 2x + 7y - 4",
                "solution_steps": [
                    "Identify like terms: (5x and -2x), (3y and 7y), (-4).",
                    "Combine like terms: (5x - 2x) + (3y + 7y) - 4",
                    "Result: 3x + 10y - 4"
                ],
                "simplified_expression": "3x + 10y - 4"
            },
            {
                "type": "Using the distributive property",
                "expression_string": "3(2a - 5b)",
                "solution_steps": [
                    "Multiply the term outside the parentheses by each term inside: 3 * 2a - 3 * 5b",
                    "Result: 6a - 15b"
                ],
                "simplified_expression": "6a - 15b"
            }
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
    },
    {
        "id": "variables",
        "title": "Variables and Constants",
        "difficulty": 1,
        "description": "Variables are symbols (typically letters like x, y, or a) that represent quantities that can change or take on different values within a mathematical expression or equation. Constants are specific, fixed numerical values that do not change throughout a problem (e.g., 5, -2, π). Understanding the distinction is fundamental to algebra, as variables allow us to generalize relationships and solve for unknowns, while constants provide the fixed parameters of those relationships.",
        "key_ideas": [
            "Symbolic Representation: Variables use letters or symbols to stand in for numbers.",
            "Unknown Values: Variables often represent quantities we need to find or solve for.",
            "Changing Values: The value of a variable can vary depending on the context or equation.",
            "Fixed Values: Constants maintain their value throughout a specific problem or context.",
            "Generalization: Variables enable the creation of general formulas and rules applicable to many situations.",
            "Placeholders: Variables can be seen as placeholders for numbers until their specific value is determined or assigned."
        ],
        "examples": [
            {
                "type": "Simple Identification",
                "problem": "Identify the variables and constants in the expression: 3a - 7 + b",
                "solution": "Variables: a, b. Constants: 3, -7.",
                "explanation": "Letters 'a' and 'b' can represent different numbers, so they are variables. '3' (coefficient of 'a') and '-7' are fixed numbers, hence constants."
            },
            {
                "type": "Equation Context",
                "problem": "In the equation for a line, y = mx + c, identify the variables and parameters often treated as constants for a specific line.",
                "solution": "'x' and 'y' are variables representing coordinates on the line. 'm' (slope) and 'c' (y-intercept) are parameters that are constant for a specific line but can change to define different lines.",
                "explanation": "As you move along a line, 'x' and 'y' change. However, for any single given line, its slope 'm' and y-intercept 'c' are fixed."
            },
            {
                "type": "Formula Application",
                "problem": "The perimeter P of a rectangle is given by P = 2l + 2w. If the length l is 5 units and the width w is 3 units, what is the perimeter? Identify variables and constants in this specific calculation.",
                "solution": "P = 2(5) + 2(3) = 10 + 6 = 16 units. In the formula P = 2l + 2w, P, l, and w are variables. The number 2 is a constant. In this specific calculation, l=5 and w=3 are treated as specific values of the variables.",
                "explanation": "The formula itself uses l and w as variables. When we assign them values, we are evaluating the expression for a specific case."
            }
        ],
        "common_misconceptions": [
            "Confusing variables with specific unknown numbers: While variables often represent unknowns we solve for, their nature is that they *can* vary. The solution to an equation is a specific value a variable takes in that context.",
            "Thinking coefficients are variables: In an term like '3x', 'x' is the variable and '3' is a constant coefficient modifying the variable.",
            "Assuming a letter always means the same variable: 'x' in one problem might be different from 'x' in another unless specified.",
            "Treating all letters as variables: In physics formulas like E=mc², 'E' and 'm' are variables (energy and mass), but 'c' (speed of light) is a universal physical constant."
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
    {
        "id": "equations",
        "title": "Linear Equations",
        "difficulty": 2,
        "description": "A linear equation is a fundamental concept in algebra representing a statement that two linear expressions are equal. A linear expression involves variables to the first power (e.g., x, not x²). Solving a linear equation means finding the specific value(s) for the variable(s) that make the statement true. These equations typically graph as straight lines when plotted on a coordinate plane (e.g., y = mx + c).",
        "key_ideas": [
            "Equality: The core principle is that both sides of the equation must always remain equal.",
            "Inverse Operations: To isolate the variable, apply inverse operations (addition/subtraction, multiplication/division) to both sides of the equation.",
            "Order of Operations (PEMDAS/BODMAS): Applied in reverse when solving (SADMEP/SAMDOB).",
            "One Variable vs. Two Variables: Linear equations can involve one variable (e.g., 3x + 1 = 7) or two variables (e.g., y = 2x - 1), the latter often representing lines."
        ],
        "examples": [
            {
                "type": "One-step equation (addition/subtraction)",
                "problem": "x + 5 = 12",
                "solution_steps": [
                    "To isolate x, subtract 5 from both sides.",
                    "x + 5 - 5 = 12 - 5",
                    "x = 7"
                ],
                "explanation": "The value that makes the statement true is 7."
            },
            {
                "type": "One-step equation (multiplication/division)",
                "problem": "4a = 20",
                "solution_steps": [
                    "To isolate a, divide both sides by 4.",
                    "4a / 4 = 20 / 4",
                    "a = 5"
                ],
                "explanation": "The value that makes the statement true is 5."
            },
            {
                "type": "Two-step equation",
                "problem": "3x + 1 = 7",
                "solution_steps": [
                    "First, subtract 1 from both sides (inverse of addition).",
                    "3x + 1 - 1 = 7 - 1",
                    "3x = 6",
                    "Next, divide both sides by 3 (inverse of multiplication).",
                    "3x / 3 = 6 / 3",
                    "x = 2"
                ],
                "explanation": "The solution requires two steps to isolate x."
            },
            {
                "type": "Variable on both sides",
                "problem": "5y - 8 = 2y + 7",
                "solution_steps": [
                    "Collect variable terms on one side. Subtract 2y from both sides:",
                    "5y - 2y - 8 = 2y - 2y + 7",
                    "3y - 8 = 7",
                    "Collect constant terms on the other side. Add 8 to both sides:",
                    "3y - 8 + 8 = 7 + 8",
                    "3y = 15",
                    "Isolate y. Divide both sides by 3:",
                    "3y / 3 = 15 / 3",
                    "y = 5"
                ],
                "explanation": "Systematically isolate the variable by moving terms."
            },
            {
                "type": "Equation with fractions",
                "problem": "x/2 + 1/3 = 5/6",
                "solution_steps": [
                    "Find the least common multiple (LCM) of the denominators (2, 3, 6), which is 6.",
                    "Multiply every term by the LCM to eliminate fractions: 6 * (x/2) + 6 * (1/3) = 6 * (5/6)",
                    "3x + 2 = 5",
                    "Solve the resulting linear equation: Subtract 2 from both sides.",
                    "3x = 3",
                    "Divide by 3.",
                    "x = 1"
                ],
                "explanation": "Eliminating fractions can simplify solving the equation."
            }
        ],
        "common_misconceptions": [
            "Forgetting to perform the same operation on both sides of the equation.",
            "Sign errors when moving terms across the equals sign (e.g., not changing addition to subtraction).",
            "Incorrectly applying the distributive property when parentheses are involved.",
            "Errors in arithmetic, especially with negative numbers or fractions."
        ],
        "real_world_applications": [
            "Calculating costs, profits, and break-even points in business.",
            "Converting temperatures between Celsius and Fahrenheit.",
            "Solving for unknown quantities in physics (e.g., distance, speed, time).",
            "Budgeting and financial planning."
        ],
        "prerequisites": ["Understanding of basic arithmetic operations", "Concept of variables and constants", "Familiarity with order of operations"]
    },
    {
        "id": "inequalities",
        "title": "Inequalities",
        "difficulty": 2,
        "description": "An inequality is a mathematical statement that compares two expressions using an inequality symbol. Unlike equations which state equality, inequalities express that one quantity is greater than, less than, greater than or equal to, or less than or equal to another. Solving an inequality means finding all values of the variable that make the inequality true. The solution set is often represented as an interval on a number line.",
        "key_ideas": [
            "Inequality Symbols: < (less than), > (greater than), ≤ (less than or equal to), ≥ (greater than or equal to), ≠ (not equal to).",
            "Solution Set: The set of all numbers that satisfy the inequality. This is often an interval of numbers rather than a single value.",
            "Properties of Inequalities: Similar to equations, you can add or subtract the same quantity to both sides. You can multiply or divide both sides by the same positive quantity.",
            "Flipping the Sign: When multiplying or dividing both sides of an inequality by a negative number, the direction of the inequality symbol must be reversed.",
            "Graphing Inequalities: Solutions can be graphed on a number line. Open circles (o) are used for < and >; closed circles (●) are used for ≤ and ≥.",
            "Compound Inequalities: Two or more inequalities joined by 'and' or 'or' (e.g., -2 < x ≤ 5, or x < 0 or x > 3)."
        ],
        "examples": [
            {
                "type": "Simple inequality (addition/subtraction)",
                "problem": "x + 3 > 7",
                "solution_steps": [
                    "Subtract 3 from both sides: x + 3 - 3 > 7 - 3",
                    "Result: x > 4"
                ],
                "solution_set_description": "All numbers greater than 4.",
                "interval_notation": "(4, ∞)"
            },
            {
                "type": "Simple inequality (multiplication/division with positive number)",
                "problem": "2y ≤ 10",
                "solution_steps": [
                    "Divide both sides by 2: 2y / 2 ≤ 10 / 2",
                    "Result: y ≤ 5"
                ],
                "solution_set_description": "All numbers less than or equal to 5.",
                "interval_notation": "(-∞, 5]"
            },
            {
                "type": "Inequality with sign flip (multiplication/division with negative number)",
                "problem": "-3a < 12",
                "solution_steps": [
                    "Divide both sides by -3 and flip the inequality sign: -3a / -3 > 12 / -3",
                    "Result: a > -4"
                ],
                "solution_set_description": "All numbers greater than -4.",
                "interval_notation": "(-4, ∞)"
            },
            {
                "type": "Two-step inequality",
                "problem": "5x - 4 ≥ 11",
                "solution_steps": [
                    "Add 4 to both sides: 5x - 4 + 4 ≥ 11 + 4",
                    "5x ≥ 15",
                    "Divide both sides by 5: 5x / 5 ≥ 15 / 5",
                    "Result: x ≥ 3"
                ],
                "solution_set_description": "All numbers greater than or equal to 3.",
                "interval_notation": "[3, ∞)"
            },
            {
                "type": "Compound inequality ('and')",
                "problem": "-1 ≤ 2x + 3 < 7",
                "solution_steps": [
                    "Solve as two separate inequalities or operate on all three parts at once.",
                    "Subtract 3 from all parts: -1 - 3 ≤ 2x + 3 - 3 < 7 - 3",
                    "-4 ≤ 2x < 4",
                    "Divide all parts by 2: -4 / 2 ≤ 2x / 2 < 4 / 2",
                    "Result: -2 ≤ x < 2"
                ],
                "solution_set_description": "All numbers greater than or equal to -2 AND less than 2.",
                "interval_notation": "[-2, 2)"
            }
        ],
        "common_misconceptions": [
            "Forgetting to flip the inequality sign when multiplying or dividing by a negative number.",
            "Incorrectly graphing the solution set on a number line (e.g., using open/closed circles incorrectly).",
            "Errors when solving compound inequalities, especially with 'or' conditions.",
            "Treating the inequality symbol like an equals sign in all situations."
        ],
        "real_world_applications": [
            "Determining minimum or maximum values (e.g., 'You must be at least 18 years old').",
            "Setting constraints or limits (e.g., 'The budget must be less than or equal to $100').",
            "Range of acceptable measurements in engineering or science.",
            "Optimization problems where certain conditions must be met."
        ],
        "prerequisites": ["Understanding of solving linear equations", "Basic arithmetic operations", "Concept of a number line"]
    },
    {
        "id": "binary_search",
        "title": "Binary Search Algorithm",
        "difficulty": 3,
        "description": "Binary search is an efficient algorithm for finding an item from a sorted list of items. It works by repeatedly dividing in half the portion of the list that could contain the item, until you've narrowed down the possible locations to just one. It is a classic example of a 'divide and conquer' strategy.",
        "key_ideas": [
            "Sorted Input: Binary search exclusively works on sorted collections (arrays, lists).",
            "Divide and Conquer: The search interval is halved in each step.",
            "Midpoint Comparison: The target value is compared with the middle element of the current interval.",
            "Interval Adjustment: If the target is less than the midpoint, search the left half; if greater, search the right half.",
            "Logarithmic Time Complexity: O(log n) in the worst-case, making it very fast for large datasets.",
            "Iterative or Recursive: Can be implemented either iteratively (using a loop) or recursively."
        ],
        "examples": [
            {
                "type": "Simple Search (Iterative)",
                "problem": "Find the index of the number 23 in the sorted array: [4, 7, 10, 13, 16, 19, 22, 23, 25, 28, 30].",
                "solution_steps": [
                    "1. Initial interval: low = 0 (value 4), high = 10 (value 30).",
                    "2. Mid = (0+10)//2 = 5. Array[5] = 19. Since 23 > 19, new low = mid + 1 = 6.",
                    "3. Interval: low = 6 (value 22), high = 10 (value 30).",
                    "4. Mid = (6+10)//2 = 8. Array[8] = 25. Since 23 < 25, new high = mid - 1 = 7.",
                    "5. Interval: low = 6 (value 22), high = 7 (value 23).",
                    "6. Mid = (6+7)//2 = 6. Array[6] = 22. Since 23 > 22, new low = mid + 1 = 7.",
                    "7. Interval: low = 7 (value 23), high = 7 (value 23).",
                    "8. Mid = (7+7)//2 = 7. Array[7] = 23. Target found at index 7."
                ],
                "result": "Index 7",
                "explanation": "The algorithm repeatedly narrows the search range until the target is found or the range becomes empty."
            },
            {
                "type": "Target Not Found",
                "problem": "Find the index of the number 11 in the sorted array: [2, 5, 8, 12, 16, 23, 38, 56, 72, 91].",
                "solution_steps": [
                    "1. low = 0 (2), high = 9 (91). mid = 4 (16). 11 < 16, so high = 3 (12).",
                    "2. low = 0 (2), high = 3 (12). mid = 1 (5). 11 > 5, so low = 2 (8).",
                    "3. low = 2 (8), high = 3 (12). mid = 2 (8). 11 > 8, so low = 3 (12).",
                    "4. low = 3 (12), high = 3 (12). mid = 3 (12). 11 < 12, so high = 2.",
                    "5. Now low = 3, high = 2. Since low > high, the target is not in the array."
                ],
                "result": "Not found (or -1, depending on implementation convention).",
                "explanation": "When the 'low' index crosses the 'high' index, it means the element is not present in the array."
            }
        ],
        "common_misconceptions": [
            "Applying to Unsorted Arrays: Binary search will produce incorrect results or fail if the array is not sorted beforehand.",
            "Off-by-One Errors: Incorrectly updating 'low' or 'high' indices (e.g., mid instead of mid+1 or mid-1) is a common bug.",
            "Infinite Loops: Errors in loop conditions or index updates can lead to infinite loops, especially if the interval doesn't shrink.",
            "Handling Duplicates: Standard binary search finds *an* occurrence of the target. Finding the first or last occurrence requires modification.",
            "Assuming it's always the best: For very small arrays, a linear search might be faster due to lower overhead."
        ],
        "real_world_applications": [
            "Searching in Databases: Efficiently finding records in indexed database tables.",
            "File System Searches: Locating files or data within sorted directory structures.",
            "Dictionary/Lookup: Finding definitions or entries in sorted dictionaries or phone books.",
            "Version Control Systems: Finding specific commits or changes in a sorted history (e.g., git bisect).",
            "Debugging: Identifying the point where a regression was introduced by searching through a sorted list of code changes (e.g., bisecting commits)."
        ],
        "prerequisites": [
            "Understanding of arrays/lists.",
            "Concept of sorted data.",
            "Basic loop structures (while/for).",
            "Familiarity with algorithmic thinking."
        ],
        "tags": ["algorithm", "search", "divide and conquer", "sorted array", "logarithmic", "efficiency"]
    },
    {
        "id": "factoring",
        "title": "Factoring Polynomials",
        "difficulty": 3,
        "description": "Factoring a polynomial means expressing it as a product of simpler polynomials (its factors). It is the reverse process of expanding or multiplying polynomials. Factoring is a crucial skill for solving polynomial equations, simplifying rational expressions, and understanding the behavior of polynomial functions (e.g., finding roots/x-intercepts).",
        "key_ideas": [
            "Factor: A polynomial that divides another polynomial evenly (with no remainder).",
            "Prime Polynomial: A polynomial that cannot be factored into polynomials of lower degree with integer coefficients (analogous to prime numbers).",
            "Greatest Common Factor (GCF): The largest monomial that is a factor of each term of the polynomial. Always try to factor out the GCF first.",
            "Zero Product Property: If the product of two or more factors is zero, then at least one of the factors must be zero. This is used when solving factored polynomial equations (e.g., if (x-a)(x-b)=0, then x-a=0 or x-b=0)."
        ],
        "common_factoring_techniques": [
            {
                "name": "Greatest Common Factor (GCF)",
                "description": "Find the GCF of all terms and factor it out.",
                "example_problem": "Factor 6x³y + 9x²y²",
                "solution": "GCF is 3x²y. Factored form: 3x²y(2x + 3y)."
            },
            {
                "name": "Factoring Trinomials (ax² + bx + c, where a=1)",
                "description": "Find two numbers that multiply to 'c' and add up to 'b'. If x² + bx + c, factors are (x+p)(x+q) where p*q=c and p+q=b.",
                "example_problem": "Factor x² + 7x + 12",
                "solution": "Numbers that multiply to 12 and add to 7 are 3 and 4. Factored form: (x + 3)(x + 4)."
            },
            {
                "name": "Factoring Trinomials (ax² + bx + c, where a≠1)",
                "description": "Methods include the 'ac method' (or grouping) or trial and error. For ac method: find two numbers that multiply to a*c and add to b. Rewrite the middle term using these numbers and factor by grouping.",
                "example_problem": "Factor 2x² + 7x + 3",
                "solution": "a*c = 2*3 = 6. Numbers that multiply to 6 and add to 7 are 1 and 6. Rewrite: 2x² + x + 6x + 3. Group: x(2x + 1) + 3(2x + 1). Factored form: (2x + 1)(x + 3)."
            },
            {
                "name": "Difference of Squares",
                "description": "Formula: a² - b² = (a - b)(a + b).",
                "example_problem": "Factor 9x² - 25",
                "solution": "9x² = (3x)², 25 = (5)². Factored form: (3x - 5)(3x + 5)."
            },
            {
                "name": "Sum of Cubes",
                "description": "Formula: a³ + b³ = (a + b)(a² - ab + b²).",
                "example_problem": "Factor x³ + 8",
                "solution": "x³ = (x)³, 8 = (2)³. Factored form: (x + 2)(x² - 2x + 4)."
            },
            {
                "name": "Difference of Cubes",
                "description": "Formula: a³ - b³ = (a - b)(a² + ab + b²).",
                "example_problem": "Factor 27y³ - 1",
                "solution": "27y³ = (3y)³, 1 = (1)³. Factored form: (3y - 1)(9y² + 3y + 1)."
            },
            {
                "name": "Factoring by Grouping (for four-term polynomials)",
                "description": "Group terms in pairs, factor out GCF from each pair, then factor out the common binomial factor.",
                "example_problem": "Factor x³ - 2x² + 5x - 10",
                "solution": "Group: (x³ - 2x²) + (5x - 10). Factor GCFs: x²(x - 2) + 5(x - 2). Factored form: (x - 2)(x² + 5)."
            }
        ],
        "examples": [
            {
                "technique": "GCF and Trinomial (a=1)",
                "problem": "Factor 2x³ + 10x² + 12x",
                "solution_steps": [
                    "First, factor out the GCF: 2x(x² + 5x + 6).",
                    "Then, factor the trinomial x² + 5x + 6. Find two numbers that multiply to 6 and add to 5 (which are 2 and 3). So, x² + 5x + 6 = (x + 2)(x + 3).",
                    "Combine: 2x(x + 2)(x + 3)."
                ],
                "factored_form": "2x(x + 2)(x + 3)"
            },
            {
                "technique": "Difference of Squares",
                "problem": "Factor 4y⁴ - 81z²",
                "solution_steps": [
                    "Recognize terms as squares: 4y⁴ = (2y²)² and 81z² = (9z)².",
                    "Apply a² - b² = (a - b)(a + b) with a = 2y² and b = 9z.",
                    "Result: (2y² - 9z)(2y² + 9z)."
                ],
                "factored_form": "(2y² - 9z)(2y² + 9z)"
            }
        ],
        "common_misconceptions": [
            "Forgetting to factor out the GCF first.",
            "Sign errors, especially when factoring trinomials or using sum/difference of cubes.",
            "Stopping too early (not factoring completely, e.g., leaving a difference of squares unfactored).",
            "Incorrectly applying the formulas for sum/difference of cubes or squares.",
            "Assuming all trinomials can be factored over integers (some are prime)."
        ],
        "real_world_applications": [
            "Solving quadratic and higher-degree polynomial equations, which model trajectories, optimization problems, etc.",
            "Simplifying expressions in engineering and physics.",
            "Finding dimensions of shapes given area or volume expressions."
        ],
        "prerequisites": ["Understanding of polynomial expressions and terms", "Multiplication of polynomials (expanding)", "Rules of exponents"]
    },
    {
        "id": "quadratics",
        "title": "Quadratic Equations",
        "difficulty": 3,
        "description": "A quadratic equation is a second-degree polynomial equation in a single variable x, with the standard form ax² + bx + c = 0, where 'a', 'b', and 'c' are coefficients (real numbers) and 'a' is not equal to zero. The solutions to a quadratic equation are called its roots or zeros, which are the x-values where the corresponding quadratic function y = ax² + bx + c intersects the x-axis. Quadratic equations graph as parabolas.",
        "key_ideas": [
            "Standard Form: ax² + bx + c = 0 (where a ≠ 0).",
            "Coefficients: 'a' (leading coefficient), 'b' (linear coefficient), 'c' (constant term).",
            "Roots/Zeros: The values of x that satisfy the equation. A quadratic equation can have two distinct real roots, one repeated real root, or two complex conjugate roots.",
            "Parabola: The U-shaped graph of a quadratic function. The parabola opens upwards if a > 0 and downwards if a < 0.",
            "Vertex: The highest or lowest point of the parabola.",
            "Axis of Symmetry: A vertical line that divides the parabola into two mirror images, passing through the vertex. Equation: x = -b/(2a).",
            "Discriminant (Δ): The part of the quadratic formula under the square root, Δ = b² - 4ac. It determines the nature of the roots:",
            "  - If Δ > 0: Two distinct real roots.",
            "  - If Δ = 0: One repeated real root (or two equal real roots).",
            "  - If Δ < 0: Two complex conjugate roots (no real roots)."
        ],
        "methods_for_solving": [
            {
                "name": "Factoring",
                "description": "Rewrite the quadratic expression as a product of two linear factors and use the zero product property. Applicable when the quadratic is easily factorable.",
                "example_problem": "Solve x² - 5x + 6 = 0",
                "solution": "Factor: (x - 2)(x - 3) = 0. So, x - 2 = 0 or x - 3 = 0. Roots: x = 2, x = 3."
            },
            {
                "name": "Quadratic Formula",
                "description": "A general formula that solves for x in any quadratic equation: x = [-b ± √(b² - 4ac)] / 2a. Always works.",
                "example_problem": "Solve 2x² + 3x - 5 = 0",
                "solution": "Here a=2, b=3, c=-5. Plug into formula: x = [-3 ± √(3² - 4*2*(-5))] / (2*2) = [-3 ± √(9 + 40)] / 4 = [-3 ± √49] / 4 = [-3 ± 7] / 4. Roots: x₁ = (-3+7)/4 = 4/4 = 1, and x₂ = (-3-7)/4 = -10/4 = -2.5."
            },
            {
                "name": "Completing the Square",
                "description": "Manipulate the equation to form a perfect square trinomial on one side. Useful for deriving the quadratic formula and for converting quadratics to vertex form.",
                "example_problem": "Solve x² + 6x - 7 = 0 by completing the square.",
                "solution_steps": [
                    "Move constant term: x² + 6x = 7.",
                    "Take half of the coefficient of x and square it: (6/2)² = 3² = 9.",
                    "Add this to both sides: x² + 6x + 9 = 7 + 9.",
                    "Factor the perfect square trinomial: (x + 3)² = 16.",
                    "Take the square root of both sides: x + 3 = ±√16 = ±4.",
                    "Solve for x: x = -3 ± 4. Roots: x₁ = -3 + 4 = 1, x₂ = -3 - 4 = -7."
                ]
            },
            {
                "name": "Square Root Property",
                "description": "If x² = k, then x = ±√k. Useful for equations of the form ax² + c = 0 (where b=0).",
                "example_problem": "Solve 3x² - 27 = 0",
                "solution": "3x² = 27 => x² = 9 => x = ±√9. Roots: x = 3, x = -3."
            }
        ],
        "examples": [
            {
                "method": "Using the Discriminant",
                "problem": "Determine the nature of roots for x² - 4x + 4 = 0",
                "solution_steps": [
                    "Identify a=1, b=-4, c=4.",
                    "Calculate Discriminant Δ = b² - 4ac = (-4)² - 4(1)(4) = 16 - 16 = 0.",
                    "Since Δ = 0, there is one repeated real root."
                ],
                "nature_of_roots": "One repeated real root."
            }
        ],
        "common_misconceptions": [
            "Errors in applying the quadratic formula, especially with signs or order of operations.",
            "Forgetting the '±' when taking the square root, leading to only one solution.",
            "Mistakes in factoring, particularly when the leading coefficient 'a' is not 1.",
            "Incorrectly interpreting the discriminant (e.g., confusing no real roots with no solutions at all)."
        ],
        "real_world_applications": [
            "Physics: Calculating projectile motion (height, range, time of flight).",
            "Engineering: Designing bridges, arches, and other structures.",
            "Optimization: Finding maximum or minimum values (e.g., maximum profit, minimum cost).",
            "Geometry: Problems involving areas of shapes that lead to quadratic equations."
        ],
        "prerequisites": ["Understanding of linear equations", "Factoring basic polynomials (trinomials)", "Operations with square roots", "Algebraic manipulation"]
    },
    {
        "id": "systems",
        "title": "Systems of Equations",
        "difficulty": 3,
        "description": "A system of equations is a set of two or more equations that share the same variables. Solving a system of equations means finding a set of values for the variables that simultaneously satisfy all equations in the system. Geometrically, for a system of two linear equations with two variables, the solution represents the point(s) of intersection of their graphs.",
        "key_ideas": [
            "Simultaneous Equations: All equations in the system must be true for the same set of variable values.",
            "Solution: An ordered set of values (e.g., an ordered pair (x,y) for two variables, an ordered triple (x,y,z) for three) that satisfies every equation in the system.",
            "Consistent System: A system that has at least one solution.",
            "Inconsistent System: A system that has no solution (e.g., parallel lines that never intersect).",
            "Dependent System: A system that has infinitely many solutions (e.g., two equations representing the same line).",
            "Independent System: A consistent system with exactly one solution."
        ],
        "methods_for_solving_linear_systems": [
            {
                "name": "Graphing Method",
                "description": "Graph each equation on the same coordinate plane. The point(s) of intersection represent the solution(s). Best for visualizing and for simple systems with integer solutions.",
                "example_scenario": "System: y = x + 1 and y = -x + 3. Graphing these lines shows they intersect at (1, 2), which is the solution."
            },
            {
                "name": "Substitution Method",
                "description": "Solve one equation for one variable in terms of the other(s). Substitute this expression into the other equation(s). Solve the resulting equation, then back-substitute to find the other variable(s).",
                "example_problem": "Solve: {x + y = 5, y = x + 1}",
                "solution_steps": [
                    "Substitute y = x + 1 into the first equation: x + (x + 1) = 5",
                    "Solve for x: 2x + 1 = 5  =>  2x = 4  =>  x = 2.",
                    "Back-substitute x = 2 into y = x + 1: y = 2 + 1 = 3.",
                    "Solution: (2, 3)."
                ]
            },
            {
                "name": "Elimination Method (or Addition/Linear Combination Method)",
                "description": "Multiply one or both equations by constants so that the coefficients of one variable are opposites. Add the equations together to eliminate that variable. Solve for the remaining variable, then back-substitute.",
                "example_problem": "Solve: {2x + 3y = 7, x - y = 1}",
                "solution_steps": [
                    "Multiply the second equation by 3: 3(x - y) = 3(1)  =>  3x - 3y = 3.",
                    "Add the modified second equation to the first equation: (2x + 3y) + (3x - 3y) = 7 + 3",
                    "5x = 10  =>  x = 2.",
                    "Back-substitute x = 2 into x - y = 1: 2 - y = 1  =>  -y = -1  =>  y = 1.",
                    "Solution: (2, 1)."
                ]
            },
            {
                "name": "Matrix Method (using Augmented Matrices and Row Operations, or Inverse Matrices)",
                "description": "Represent the system as an augmented matrix [A|B] and use Gaussian elimination or Gauss-Jordan elimination to solve. Alternatively, if it's a system AX=B, find X = A⁻¹B if A is invertible. More advanced, typically for systems with more variables.",
                "example_scenario": "Used for larger systems, e.g., 3 equations with 3 variables. Involves operations like swapping rows, multiplying a row by a scalar, adding a multiple of one row to another."
            }
        ],
        "examples": [
            {
                "type": "Consistent and Independent System (One Solution)",
                "system": "{x + y = 5, 2x - y = 4}",
                "solution": "x=3, y=2. The lines intersect at a single point (3,2)."
            },
            {
                "type": "Inconsistent System (No Solution)",
                "system": "{x + y = 3, x + y = 1}",
                "solution": "No solution. If you try to solve, you might get a contradiction like 0 = 2. The lines are parallel and distinct."
            },
            {
                "type": "Dependent System (Infinite Solutions)",
                "system": "{x + y = 2, 2x + 2y = 4}",
                "solution": "Infinite solutions. The second equation is just twice the first; they represent the same line. The solution can be expressed as y = 2 - x, where x can be any real number."
            }
        ],
        "common_misconceptions": [
            "Making arithmetic errors during substitution or elimination.",
            "Incorrectly multiplying an entire equation when trying to eliminate a variable.",
            "Stopping after finding the value of only one variable; a solution requires values for all variables.",
            "Misinterpreting the meaning of an inconsistent (0 = non-zero number) or dependent (0 = 0) system outcome."
        ],
        "real_world_applications": [
            "Resource allocation: Determining how to allocate limited resources to meet certain demands.",
            "Mixture problems: Finding the amounts of different substances to create a desired mixture (e.g., in chemistry or cooking).",
            "Break-even analysis: Finding the point where cost equals revenue in business.",
            "Navigation and GPS: Systems of equations are used in trilateration to determine location.",
            "Circuit analysis: Solving for currents and voltages in electrical circuits."
        ],
        "prerequisites": ["Understanding of linear equations and their graphs", "Algebraic manipulation (solving for a variable, substitution)", "Basic arithmetic"]
    },
    {
        "id": "functions",
        "title": "Functions and Relations",
        "difficulty": 4,
        "description": "A relation is a set of ordered pairs (x, y), showing a relationship between two sets of values. A function is a special type of relation where every input value (x, from the domain) is associated with exactly one output value (y, from the range). Functions are fundamental to mathematics and are used to model real-world phenomena where one quantity depends on another.",
        "key_ideas": [
            "Relation: Any set of ordered pairs. Can be represented by a set of points, a table, a graph, or a mapping diagram.",
            "Function: A relation where each input (x-value) has only one unique output (y-value). Passes the Vertical Line Test (a vertical line drawn on the graph intersects the function at most once).",
            "Domain: The set of all possible input values (x-values) for which the function is defined.",
            "Range: The set of all possible output values (y-values) that the function can produce.",
            "Function Notation: f(x), read as \"f of x\", represents the output of function 'f' for a given input 'x'. 'x' is the independent variable, and f(x) (or y) is the dependent variable.",
            "Independent Variable: The input to a function, usually denoted by x.",
            "Dependent Variable: The output of a function, usually denoted by y or f(x), as its value depends on the input."
        ],
        "representations_of_functions": [
            "Set of Ordered Pairs: e.g., {(1, 2), (2, 4), (3, 6)}",
            "Table of Values: Columns for x and y (or f(x)).",
            "Equation: e.g., f(x) = 2x + 1 or y = x².",
            "Graph: A visual plot of the ordered pairs on a coordinate plane.",
            "Mapping Diagram: Shows arrows from domain elements to range elements."
        ],
        "types_of_functions_overview": [
            "Linear Functions: f(x) = mx + c (graph as a straight line).",
            "Quadratic Functions: f(x) = ax² + bx + c (graph as a parabola).",
            "Polynomial Functions: General form f(x) = aₙxⁿ + ... + a₁x + a₀.",
            "Exponential Functions: f(x) = abˣ (model growth/decay).",
            "Logarithmic Functions: f(x) = logₘ(x) (inverse of exponential).",
            "Trigonometric Functions: e.g., f(x) = sin(x), f(x) = cos(x) (model periodic phenomena)."
        ],
        "examples": [
            {
                "type": "Identifying a function from ordered pairs",
                "relation1": {"pairs": "{(1,a), (2,b), (3,c)}", "is_function": True, "reason": "Each input has exactly one output."},
                "relation2": {"pairs": "{(1,a), (1,b), (2,c)}", "is_function": False, "reason": "The input 1 has two different outputs (a and b)."}
            },
            {
                "type": "Evaluating a function",
                "function_equation": "f(x) = 3x² - 2x + 5",
                "problem": "Find f(-2)",
                "solution_steps": [
                    "Substitute x = -2 into the function: f(-2) = 3(-2)² - 2(-2) + 5",
                    "Calculate powers: f(-2) = 3(4) - 2(-2) + 5",
                    "Perform multiplications: f(-2) = 12 + 4 + 5",
                    "Perform additions: f(-2) = 21"
                ],
                "result": "f(-2) = 21"
            },
            {
                "type": "Determining Domain and Range from a simple function",
                "function_equation": "y = √(x - 3)",
                "domain_reasoning": "The expression under the square root (radicand) must be non-negative. So, x - 3 ≥ 0, which means x ≥ 3.",
                "domain": "[3, ∞)",
                "range_reasoning": "The square root function √ produces non-negative values. So, y ≥ 0.",
                "range": "[0, ∞)"
            },
            {
                "type": "Vertical Line Test",
                "description": "A graph represents a function if and only if no vertical line intersects the graph at more than one point.",
                "example_graph_is_function": "A parabola opening upwards or downwards (e.g., y=x²).",
                "example_graph_not_function": "A circle (e.g., x² + y² = 9) or a sideways parabola (e.g., x=y²)."
            }
        ],
        "common_misconceptions": [
            "Confusing functions with general relations.",
            "Thinking all equations represent functions (e.g., x² + y² = r² is a circle, not a function of x).",
            "Incorrectly identifying domain and range, especially with square roots or denominators.",
            "Errors in function notation, e.g., treating f(x) as f multiplied by x.",
            "Mistakes in applying the vertical line test."
        ],
        "real_world_applications": [
            "Modeling physical phenomena: trajectory of a projectile, growth of populations, decay of radioactive substances.",
            "Economics: supply and demand curves, cost functions, revenue functions.",
            "Computer Science: algorithms often represent functions transforming input to output.",
            "Engineering: designing structures, analyzing circuits, signal processing."
        ],
        "prerequisites": ["Understanding of ordered pairs and coordinate graphing", "Algebraic expressions and solving equations", "Concept of variables"]
    },
    {
        "id": "exponents",
        "title": "Exponents and Logarithms",
        "difficulty": 4,
        "description": "Exponents indicate how many times a base number is multiplied by itself. For example, in bⁿ, 'b' is the base and 'n' is the exponent. Logarithms are the inverse operation of exponentiation. The logarithm of a number 'x' to a base 'b' (written as logₘ(x)) is the exponent to which 'b' must be raised to produce 'x'. They are crucial for solving exponential equations and analyzing phenomena with exponential growth or decay.",
        "key_ideas": {
            "exponents": [
                "Base: The number being multiplied.",
                "Exponent (or Power/Index): The number of times the base is multiplied by itself.",
                "Exponential Form: bⁿ (e.g., 2³).",
                "Expanded Form: b * b * ... * b (n times) (e.g., 2 * 2 * 2)."
            ],
            "logarithms": [
                "Logarithmic Form: y = logₘ(x) is equivalent to Exponential Form: x = bʸ.",
                "Base of Logarithm: The 'b' in logₘ(x). Common bases are 10 (common logarithm, log) and 'e' (natural logarithm, ln).",
                "Argument of Logarithm: The 'x' in logₘ(x), which must be positive."
            ]
        },
        "properties_of_exponents": [
            "Product Rule: bᵐ * bⁿ = bᵐ⁺ⁿ (e.g., 2³ * 2² = 2⁵)",
            "Quotient Rule: bᵐ / bⁿ = bᵐ⁻ⁿ (e.g., 2⁵ / 2² = 2³)",
            "Power Rule: (bᵐ)ⁿ = bᵐⁿ (e.g., (2²)³ = 2⁶)",
            "Power of a Product Rule: (ab)ⁿ = aⁿbⁿ (e.g., (2*3)² = 2² * 3²)",
            "Power of a Quotient Rule: (a/b)ⁿ = aⁿ/bⁿ (e.g., (2/3)² = 2²/3²)",
            "Zero Exponent: b⁰ = 1 (for b ≠ 0) (e.g., 5⁰ = 1)",
            "Negative Exponent: b⁻ⁿ = 1/bⁿ (e.g., 2⁻³ = 1/2³ = 1/8)",
            "Fractional Exponent (Roots): b^(m/n) = ⁿ√(bᵐ) (e.g., 8^(2/3) = ³√(8²) = ³√64 = 4)"
        ],
        "properties_of_logarithms": [
            "Product Rule: logₘ(xy) = logₘ(x) + logₘ(y) (e.g., log₂(4*8) = log₂(4) + log₂(8) = 2 + 3 = 5)",
            "Quotient Rule: logₘ(x/y) = logₘ(x) - logₘ(y) (e.g., log₂(16/2) = log₂(16) - log₂(2) = 4 - 1 = 3)",
            "Power Rule: logₘ(xⁿ) = n * logₘ(x) (e.g., log₂(4³) = 3 * log₂(4) = 3 * 2 = 6)",
            "Change of Base Formula: logₘ(x) = logₜ(x) / logₜ(b) (e.g., log₂(100) = log₁₀(100) / log₁₀(2) = 2 / 0.30103 ≈ 6.64)",
            "logₘ(b) = 1 (e.g., log₂(2) = 1)",
            "logₘ(1) = 0 (e.g., log₂(1) = 0)"
        ],
        "examples": [
            {
                "type": "Basic Exponentiation",
                "problem": "Calculate 3⁴",
                "solution": "3⁴ = 3 * 3 * 3 * 3 = 81",
                "explanation": "The base 3 is multiplied by itself 4 times."
            },
            {
                "type": "Basic Logarithm",
                "problem": "Find log₂(8)",
                "solution": "We need to find the power to which 2 must be raised to get 8. Since 2³ = 8, log₂(8) = 3.",
                "explanation": "Logarithms ask 'what exponent gives me this number?'"
            },
            {
                "type": "Solving Exponential Equation with Logarithms",
                "problem": "Solve for x: 2ˣ = 16",
                "solution_steps": [
                    "Take the logarithm of both sides (base 2 is convenient here): log₂(2ˣ) = log₂(16)",
                    "Using the power rule for logarithms (logₘ(xⁿ) = n * logₘ(x)): x * log₂(2) = log₂(16)",
                    "Since log₂(2) = 1 and log₂(16) = 4 (because 2⁴ = 16): x * 1 = 4",
                    "x = 4"
                ],
                "explanation": "Logarithms are used to bring the variable down from the exponent."
            },
            {
                "type": "Using Exponent Properties (Product Rule)",
                "problem": "Simplify: (5x³y²)(2x⁴y)",
                "solution_steps": [
                    "Group coefficients and like variables: (5*2) * (x³*x⁴) * (y²*y¹)",
                    "Multiply coefficients: 10",
                    "Apply product rule for exponents (add exponents): x^(3+4) = x⁷ and y^(2+1) = y³",
                    "Result: 10x⁷y³"
                ],
                "explanation": "When multiplying terms with the same base, add their exponents."
            }
        ],
        "common_misconceptions": [
            "Confusing xⁿ with n*x (e.g., 2³ is 8, not 6).",
            "Incorrectly applying exponent rules, especially with negative or fractional exponents.",
            "Thinking log(x+y) = log(x) + log(y) (this is incorrect; the product rule is log(xy) = log(x) + log(y)).",
            "The logarithm of a negative number or zero is undefined in real numbers.",
            "Forgetting the base of a logarithm (common log is base 10, natural log is base e)."
        ],
        "real_world_applications": [
            "Compound interest calculations (exponential growth).",
            "Radioactive decay modeling (exponential decay).",
            "Measuring earthquake intensity (Richter scale - logarithmic).",
            "Measuring sound intensity (decibels - logarithmic).",
            "Population growth models.",
            "Computer algorithm complexity (e.g., O(log n), O(n²))."
        ],
        "prerequisites": ["Understanding of multiplication and division", "Basic algebra (variables, expressions)", "Understanding of inverse operations"]
    },
    {
        "id": "sequences",
        "title": "Sequences and Series",
        "difficulty": 5,
        "description": "A sequence is an ordered list of numbers, called terms, that follow a specific rule or pattern. A series is the sum of the terms of a sequence. Understanding sequences and series is important for analyzing patterns, predicting future values, and in areas like calculus (infinite series) and finance (annuities).",
        "key_ideas": {
            "sequence": [
                "Term: Each number in the sequence (e.g., a₁, a₂, a₃, ..., aₙ).",
                "Finite Sequence: A sequence with a limited number of terms.",
                "Infinite Sequence: A sequence that continues indefinitely.",
                "General Term (or nth term): A formula, denoted aₙ, that defines any term of the sequence based on its position 'n'."
            ],
            "series": [
                "Partial Sum (Sₙ): The sum of the first 'n' terms of a sequence.",
                "Infinite Series: The sum of all terms of an infinite sequence. Can converge (sum to a finite value) or diverge (sum to infinity or oscillate).",
                "Summation Notation (Sigma Notation): Uses the Greek letter Σ to represent the sum of terms (e.g., Σ_{i=1}^{n} aᵢ = a₁ + a₂ + ... + aₙ)."
            ]
        },
        "types_of_sequences_and_series": [
            {
                "name": "Arithmetic Sequence",
                "description": "Each term after the first is found by adding a constant, called the common difference (d), to the previous term.",
                "formula_nth_term": "aₙ = a₁ + (n-1)d (where a₁ is the first term)",
                "formula_sum_first_n_terms": "Sₙ = n/2 * [2a₁ + (n-1)d]  OR  Sₙ = n/2 * (a₁ + aₙ)",
                "example_sequence": "2, 5, 8, 11, ... (a₁=2, d=3)",
                "example_series_sum": "Sum of first 4 terms: S₄ = 4/2 * (2 + 11) = 2 * 13 = 26"
            },
            {
                "name": "Geometric Sequence",
                "description": "Each term after the first is found by multiplying the previous term by a constant, called the common ratio (r).",
                "formula_nth_term": "aₙ = a₁ * r^(n-1) (where a₁ is the first term)",
                "formula_sum_first_n_terms": "Sₙ = a₁ * (1 - rⁿ) / (1 - r)  (for r ≠ 1)",
                "formula_sum_infinite_geometric_series": "S_∞ = a₁ / (1 - r) (converges only if |r| < 1)",
                "example_sequence": "3, 6, 12, 24, ... (a₁=3, r=2)",
                "example_series_sum": "Sum of first 3 terms: S₃ = 3 * (1 - 2³) / (1 - 2) = 3 * (-7) / (-1) = 21"
            },
            {
                "name": "Fibonacci Sequence",
                "description": "Each term is the sum of the two preceding ones, usually starting with 0 and 1.",
                "definition": "F₀=0, F₁=1, Fₙ = Fₙ₋₁ + Fₙ₋₂ for n > 1",
                "example_sequence": "0, 1, 1, 2, 3, 5, 8, 13, ..."
            }
        ],
        "examples": [
            {
                "type": "Finding the nth term of an arithmetic sequence",
                "problem": "Find the 10th term of the arithmetic sequence: 4, 7, 10, 13, ...",
                "solution_steps": [
                    "Identify the first term a₁ = 4.",
                    "Calculate the common difference d = 7 - 4 = 3.",
                    "Use the formula aₙ = a₁ + (n-1)d with n=10: a₁₀ = 4 + (10-1)*3 = 4 + 9*3 = 4 + 27 = 31."
                ],
                "result": "The 10th term is 31."
            },
            {
                "type": "Finding the sum of a finite geometric series",
                "problem": "Find the sum of the first 5 terms of the geometric sequence where a₁=2 and r=3.",
                "solution_steps": [
                    "Use the formula Sₙ = a₁ * (1 - rⁿ) / (1 - r) with n=5, a₁=2, r=3.",
                    "S₅ = 2 * (1 - 3⁵) / (1 - 3) = 2 * (1 - 243) / (-2) = 2 * (-242) / (-2) = -484 / -2 = 242."
                ],
                "result": "The sum of the first 5 terms is 242."
            },
            {
                "type": "Sum of an infinite geometric series",
                "problem": "Find the sum of the infinite series 1 + 1/2 + 1/4 + 1/8 + ...",
                "solution_steps": [
                    "Identify a₁ = 1 and common ratio r = (1/2)/1 = 1/2.",
                    "Since |r| = 1/2 < 1, the series converges.",
                    "Use the formula S_∞ = a₁ / (1 - r): S_∞ = 1 / (1 - 1/2) = 1 / (1/2) = 2."
                ],
                "result": "The sum of the infinite series is 2."
            }
        ],
        "common_misconceptions": [
            "Confusing arithmetic and geometric sequences/series and their formulas.",
            "Incorrectly calculating the common difference (d) or common ratio (r).",
            "Errors in using summation notation (Σ).",
            "Assuming an infinite geometric series converges when |r| ≥ 1.",
            "Mistakes with negative signs or fractions in calculations."
        ],
        "real_world_applications": [
            "Finance: Calculating compound interest, loan payments, annuities.",
            "Physics: Modeling motion with constant acceleration (arithmetic), radioactive decay (geometric).",
            "Computer Science: Analyzing algorithm complexity, data structures like linked lists.",
            "Biology: Population growth models, spread of diseases.",
            "Fractals and geometric patterns."
        ],
        "prerequisites": ["Understanding of basic arithmetic and algebra", "Exponents", "Pattern recognition"]
    }
]

def write_concepts_to_disk() -> None:
    """Write all concepts to disk as JSON files."""
    for concept in ALGEBRA_CONCEPTS:
        concept_path = CONCEPTS_DIR / f"{concept['id']}.json"
        with open(concept_path, "w") as f:
            json.dump(concept, f, indent=2)

def load_concept(concept_id: str) -> Dict[str, Any]:
    """
    Load a specific concept by ID.
    
    Args:
        concept_id: ID of the concept to load
        
    Returns:
        Concept data dictionary
    """
    # Check if we have it in memory first
    for concept in ALGEBRA_CONCEPTS:
        if concept["id"] == concept_id:
            return concept
    
    # Otherwise try to load from disk
    concept_path = CONCEPTS_DIR / f"{concept_id}.json"
    if not concept_path.exists():
        raise ValueError(f"Concept not found: {concept_id}")
    
    with open(concept_path, "r") as f:
        return json.load(f)

def get_all_concepts() -> List[Dict[str, Any]]:
    """
    Return all available concepts.
    
    Returns:
        List of concept dictionaries
    """
    return ALGEBRA_CONCEPTS

def get_concepts_by_difficulty(difficulty_level: int) -> List[Dict[str, Any]]:
    """
    Get concepts at a specific difficulty level.
    
    Args:
        difficulty_level: Difficulty level (1-5)
        
    Returns:
        List of concepts at the specified difficulty level
    """
    return [c for c in ALGEBRA_CONCEPTS if c["difficulty"] == difficulty_level]

def get_concept_sequence(start_difficulty: int = 1, max_difficulty: int = 5) -> List[Dict[str, Any]]:
    """
    Get a sequence of concepts ordered by difficulty.
    
    Args:
        start_difficulty: Minimum difficulty level to include
        max_difficulty: Maximum difficulty level to include
        
    Returns:
        List of concepts ordered by difficulty
    """
    return [c for c in ALGEBRA_CONCEPTS 
            if start_difficulty <= c["difficulty"] <= max_difficulty]

# Write concepts on module import
write_concepts_to_disk() 