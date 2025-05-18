"""Detailed cognitive profiles for different learner types."""

import json
import os
from typing import Dict, List, Any
from pathlib import Path

from synapz import PACKAGE_ROOT
from synapz.core.profiles import CognitiveProfile

# Define profiles directory
PROFILES_DIR = PACKAGE_ROOT / "models" / "profile_data"

# Ensure profiles directory exists
os.makedirs(PROFILES_DIR, exist_ok=True)

ADHD_PROFILE = {
    "id": CognitiveProfile.ADHD.value,
    "name": "Sam (ADHD Profile)",
    "cognitive_style": "adhd",
    "description": "Sam is a learner with ADHD characteristics, including a dynamic attention span, a preference for engaging and varied content, and strengths in creative thinking.",
    "attention_profile": {
        "typical_span": "Varies; can hyperfocus on engaging topics, otherwise short to moderate for less stimulating content.",
        "triggers_for_focus": ["Novelty", "High interest", "Interactive elements", "Clear goals", "Sense of urgency"],
        "distraction_factors": ["Monotony", "Lengthy text blocks", "Lack of clear structure", "Passive learning"]
    },
    "working_memory": "May experience challenges with holding and manipulating multiple pieces of information simultaneously, especially if presented verbally or in dense text.",
    "executive_functions": {
        "planning_organization": "Benefits from structured outlines, checklists, and clear, sequential steps. May struggle with self-organizing complex tasks.",
        "task_initiation": "May delay starting tasks that seem overwhelming or uninteresting. Clear starting points and immediate engagement are helpful.",
        "emotional_regulation": "May be sensitive to frustration; positive reinforcement and celebrating small wins are beneficial."
    },
    "processing_style": "Often processes information in a non-linear, divergent manner. Strong in pattern recognition and making novel connections. May find rigid, purely sequential presentations challenging.",
    "learning_preferences": [
        "Interactive exercises and simulations",
        "Visually stimulating materials (color, diagrams, varied formatting)",
        "Chunked information (short paragraphs, bullet points)",
        "Real-world examples and practical applications",
        "Gamified elements or challenges",
        "Clear and immediate feedback",
        "Opportunities for movement or physical interaction (if applicable)"
    ],
    "communication_needs": {
        "instruction_delivery": "Prefers clear, concise language. Benefits from information presented in multiple modalities (visual + auditory cues). Direct and explicit instructions are better than subtle or implied ones.",
        "feedback_reception": "Responds well to specific, constructive feedback that is delivered in a supportive tone. Appreciates acknowledgment of effort."
    },
    "strengths": [
        "High creativity and innovative thinking",
        "Strong problem-solving skills, especially with novel challenges",
        "Ability to hyperfocus on areas of interest",
        "Good at seeing the 'big picture' or making connections others might miss",
        "Energetic and enthusiastic when engaged"
    ],
    "challenges": [
        "Sustained attention on tasks perceived as low-interest or repetitive",
        "Organization and time management",
        "Following multi-step directions without visual aids or reminders",
        "Impulse control (may lead to jumping to conclusions or interrupting)",
        "Working memory overload with too much information at once"
    ],
    "motivation_triggers": [
        "Novelty and variety in content and presentation",
        "Topics aligned with personal interests",
        "Clear relevance of the material to real-life situations",
        "Sense of accomplishment and progress (frequent, small wins)",
        "Positive and enthusiastic teaching style"
    ],
    "effective_teaching_strategies": [
        "Use of visual aids, color-coding, and varied formatting.",
        "Breaking down complex information into smaller, manageable chunks.",
        "Incorporating interactive elements and frequent opportunities for engagement.",
        "Providing concrete examples and analogies, especially those related to the learner's interests.",
        "Offering choices in how to learn or demonstrate understanding when possible.",
        "Frequent check-ins for comprehension and engagement.",
        "Using timers or visual schedules to help with task management.",
        "Positive reinforcement and encouragement."
    ],
    "assessment_needs": [
        "Prefers varied assessment methods over long, traditional tests.",
        "May perform better with project-based assessments or practical demonstrations.",
        "Benefits from clear rubrics and expectations for assessments.",
        "May need accommodations like extended time or a quiet environment for formal assessments."
    ]
}

DYSLEXIC_PROFILE = {
    "id": CognitiveProfile.DYSLEXIC.value,
    "name": "Taylor (Dyslexic Profile)",
    "cognitive_style": "dyslexic",
    "description": "Taylor is a learner with dyslexia, characterized by challenges in phonological processing and working memory for text, but strong in conceptual understanding and visual-spatial reasoning.",
    "phonological_processing": "Experiences difficulty with decoding individual words, sound-symbol correspondence, and reading fluency. Spelling can also be a challenge.",
    "working_memory": "May have limitations in holding and processing text-based information in working memory, but conceptual working memory can be strong.",
    "reading_profile": {
        "reading_speed": "Typically slower than peers, requires more effort for decoding.",
        "comprehension": "Often strong when barriers to decoding are removed (e.g., via audio support, accessible text formats). May struggle if cognitive load from decoding is too high.",
        "preferred_formats": ["Text with good spacing (increased line height, paragraph spacing)", "Sans-serif fonts (e.g., Arial, Verdana)", "Larger font sizes (12-14pt or higher)", "Option for text-to-speech"]
    },
    "processing_style": "Excels at big-picture thinking, making connections between concepts, and understanding complex systems. Strong visual-spatial reasoning and narrative reasoning.",
    "learning_preferences": [
        "Auditory learning (podcasts, audio explanations, text-to-speech)",
        "Visual aids (diagrams, mind maps, flowcharts, videos)",
        "Experiential and hands-on learning",
        "Structured and well-organized content with clear headings and summaries",
        "Multi-sensory approaches (combining visual, auditory, kinesthetic)",
        "Spaced repetition for memorization of facts or vocabulary"
    ],
    "communication_needs": {
        "instruction_delivery": "Benefits from clear, direct language. Avoid jargon or provide immediate definitions. Instructions given verbally and visually are helpful.",
        "written_expression": "May find it challenging to express complex ideas in writing due to spelling or sentence construction difficulties. May benefit from using speech-to-text tools or alternative formats for showing understanding."
    },
    "strengths": [
        "Strong conceptual understanding and holistic thinking",
        "Excellent problem-solving and reasoning skills",
        "Creativity and out-of-the-box thinking",
        "Strong verbal communication skills (when not focused on decoding written text)",
        "Visual-spatial abilities and pattern recognition",
        "Empathy and strong interpersonal skills"
    ],
    "challenges": [
        "Decoding text fluently and accurately",
        "Spelling and written expression",
        "Rapid naming and phonological awareness",
        "Working memory for text-based information",
        "Organization of written work without explicit structure",
        "Time management for reading-intensive tasks"
    ],
    "motivation_triggers": [
        "Understanding the 'why' behind what is being learned (relevance)",
        "Content that leverages visual or conceptual strengths",
        "Feeling understood and supported in their learning process",
        "Opportunities to demonstrate understanding in varied ways (not just written tests)",
        "Achievable goals and recognition of effort and progress"
    ],
    "effective_teaching_strategies": [
        "Providing audio versions of text materials.",
        "Using dyslexia-friendly fonts and formatting (sans-serif, good spacing, clear headings).",
        "Breaking down information into smaller, manageable chunks with visual cues.",
        "Explicitly teaching vocabulary and concepts, pre-teaching if necessary.",
        "Using mind maps, graphic organizers, and diagrams to show relationships.",
        "Allowing for alternative methods of demonstrating knowledge (oral reports, projects).",
        "Providing ample time for reading and written tasks.",
        "Focusing on understanding and ideas over perfect spelling/grammar in drafts."
    ],
    "assessment_needs": [
        "Benefits from oral assessments or presentations.",
        "Requires clear, uncluttered test formats with dyslexia-friendly fonts.",
        "Extended time for reading and writing components of tests.",
        "Use of assistive technology (text-to-speech, speech-to-text) where appropriate.",
        "Assessment of conceptual understanding rather than solely on mechanics of writing/spelling."
    ]
}

VISUAL_LEARNER_PROFILE = {
    "id": CognitiveProfile.VISUAL.value,
    "name": "Alex (Visual Profile)",
    "cognitive_style": "visual",
    "description": "Alex is a visual learner who excels when information is presented spatially and through imagery, diagrams, and visual metaphors. They think in pictures and build strong mental models from visual input.",
    "visual_processing_strengths": {
        "spatial_reasoning": "Excellent at understanding and manipulating spatial relationships, interpreting maps, charts, and diagrams.",
        "visual_memory": "Strong memory for information that is presented visually. Can recall details from images, layouts, and visual patterns.",
        "pattern_recognition": "Quickly identifies visual patterns, trends, and anomalies in data or images."
    },
    "modality_reliance": {
        "primary_modality": "Visual. Learns best by seeing.",
        "secondary_modality_support": "Kinesthetic (learning by doing, interacting with visual models) can be supportive. Auditory information is best when paired with visual aids.",
        "challenges_with_non_visual": "May struggle to retain or deeply process purely auditory or text-heavy information without visual support. Long lectures or dense readings without illustrations can be difficult."
    },
    "information_organization": "Prefers information organized spatially. Benefits from hierarchies, flowcharts, mind maps, and content laid out with clear visual structure. Responds well to color-coding and visual cues for importance or categorization.",
    "learning_preferences": [
        "Diagrams, charts, graphs, and infographics",
        "Mind maps and concept maps for organizing ideas",
        "Videos, animations, and visual simulations",
        "Color-coding and highlighting key information",
        "Content with strong visual layout and design (clear headings, use of white space, illustrative images)",
        "Opportunities to create their own visual representations of concepts (drawing, diagramming)",
        "Visual metaphors and analogies"
    ],
    "communication_needs": {
        "instruction_delivery": "Prefers instructions that include visual components (e.g., diagrams of steps, screenshots). Written instructions should be well-formatted with visual breaks.",
        "expressing_understanding": "May prefer to demonstrate understanding through visual means (e.g., creating a presentation, drawing a diagram, building a model) over purely written or oral responses."
    },
    "strengths": [
        "Excellent spatial awareness and reasoning",
        "Strong ability to visualize concepts and processes mentally",
        "High capacity for recalling visual details",
        "Good at understanding complex systems when presented visually",
        "Often artistic or skilled in design-related tasks",
        "Integrates information effectively when it's visually structured"
    ],
    "challenges": [
        "Processing and retaining information from purely auditory lectures or discussions without visual aids",
        "Learning from dense, text-heavy materials lacking illustrations or diagrams",
        "Following complex verbal instructions without visual reference points",
        "May overlook details in text if not visually emphasized",
        "Abstract concepts that are difficult to visualize without concrete imagery or metaphors"
    ],
    "motivation_triggers": [
        "Visually appealing and well-designed learning materials",
        "Opportunities to engage with content visually (e.g., interactive diagrams, building models)",
        "Clear visual pathways through complex information",
        "Recognition for visual thinking or creative visual solutions",
        "Seeing the 'big picture' visually before diving into details"
    ],
    "effective_teaching_strategies": [
        "Using whiteboards or digital drawing tools extensively to illustrate concepts.",
        "Incorporating charts, graphs, diagrams, and photos into presentations and materials.",
        "Encouraging the learner to create their own visual notes (sketchnotes, mind maps).",
        "Using video and animation to explain processes or dynamic concepts.",
        "Employing color-coding to highlight patterns, categories, or important information.",
        "Structuring information spatially on a page or screen (e.g., using columns, grids).",
        "Providing visual outlines or roadmaps for complex topics.",
        "Using visual metaphors and analogies to explain abstract ideas."
    ],
    "assessment_needs": [
        "Excels with tasks that involve interpreting visual data (charts, graphs).",
        "Benefits from assessments that allow for visual responses (e.g., drawing diagrams, creating presentations).",
        "Multiple-choice questions with visual options or diagrams can be effective.",
        "Essay questions may be more accessible if the learner can first create a visual outline or mind map."
    ]
}

CONTROL_PROFILE = {
    "id": CognitiveProfile.CONTROL.value,
    "name": "Standard Learner (Control)",
    "cognitive_traits": {
        "attention_span": "typical",
        "processing_style": "balanced",
        "working_memory": "average capacity",
        "strengths": ["general comprehension", "standard information processing"],
        "challenges": []
    },
    "modality_preferences": {
        "primary": ["mixed"],
        "secondary": [],
        "avoid": []
    },
    "pedagogical_needs": {
        "chunk_size": "standard",
        "organization": "typical academic structure",
        "engagement": "standard educational approaches",
        "optimal_explanation_length": "comprehensive",
        "example_types": ["typical examples"]
    }
}

# All profiles dictionary
LEARNER_PROFILES = {
    CognitiveProfile.ADHD.value: ADHD_PROFILE,
    CognitiveProfile.DYSLEXIC.value: DYSLEXIC_PROFILE,
    CognitiveProfile.VISUAL.value: VISUAL_LEARNER_PROFILE,
    CognitiveProfile.CONTROL.value: CONTROL_PROFILE
}

def write_profiles_to_disk() -> None:
    """Write all profiles to disk as JSON files."""
    for profile_id, profile in LEARNER_PROFILES.items():
        profile_path = PROFILES_DIR / f"{profile_id}.json"
        with open(profile_path, "w") as f:
            json.dump(profile, f, indent=2)

def load_profile(profile_id: str) -> Dict[str, Any]:
    """
    Load a specific cognitive profile by ID.
    
    Args:
        profile_id: ID of the profile to load
        
    Returns:
        Profile data dictionary
    """
    # Check if we have it in memory first
    if profile_id in LEARNER_PROFILES:
        return LEARNER_PROFILES[profile_id]
    
    # Otherwise try to load from disk
    profile_path = PROFILES_DIR / f"{profile_id}.json"
    if not profile_path.exists():
        raise ValueError(f"Profile not found: {profile_id}")
    
    with open(profile_path, "r") as f:
        return json.load(f)

def get_all_profiles() -> List[Dict[str, Any]]:
    """
    Return all available cognitive profiles.
    
    Returns:
        List of profile dictionaries
    """
    return list(LEARNER_PROFILES.values())

def get_profile_for_adaptation(profile_id: str) -> Dict[str, Any]:
    """
    Get profile with adaptation-specific information.
    
    This provides details needed for the content adaptation process.
    
    Args:
        profile_id: ID of the profile to load
        
    Returns:
        Profile with adaptation parameters
    """
    profile_data = load_profile(profile_id)
    
    # Create a deep copy to avoid modifying the original in LEARNER_PROFILES
    profile = json.loads(json.dumps(profile_data))

    # Add adaptation-specific parameters. These are high-level summaries or specific settings
    # derived from the detailed profile, intended for direct use by an adaptation engine.
    
    if profile_id == CognitiveProfile.ADHD.value:
        profile["adaptation"] = {
            "chunk_size_preference": "small, frequent", # Derived from description
            "engagement_style": "interactive, novel, gamified", # From learning_preferences
            "feedback_needed": "immediate, positive", # From communication_needs & motivation_triggers
            "visual_complexity": "moderate with clear focal points", # Implied
            "example_type": "real-world, relatable, varied" # From learning_preferences
        }
    elif profile_id == CognitiveProfile.DYSLEXIC.value:
        profile["adaptation"] = {
            "text_format_preference": "dyslexia-friendly (sans-serif, good spacing, larger font)", # From reading_profile
            "modality_preference": "multi-sensory (audio + visual strong)", # From learning_preferences
            "language_complexity": "simple, direct, explicit vocabulary", # From communication_needs
            "structure_needs": "highly structured, clear headings, summaries", # From learning_preferences
            "pacing": "slower, allow more time for processing" # Implied from reading_profile
        }
    elif profile_id == CognitiveProfile.VISUAL.value:
        profile["adaptation"] = {
            "primary_input_channel": "visual (diagrams, charts, mind maps, video)", # From learning_preferences
            "text_to_visual_ratio": "visual-dominant", # Implied
            "spatial_organization_importance": "critical", # From information_organization
            "color_usage": "meaningful (coding, highlighting)", # From learning_preferences
            "interaction_style": "manipulating visual elements, creating visual summaries" # From learning_preferences
        }
    else:  # Control profile
        profile["adaptation"] = {
            "chunk_size_preference": "standard",
            "engagement_style": "traditional",
            "feedback_needed": "standard",
            "visual_complexity": "standard",
            "example_type": "conventional"
        }
    
    return profile

# Write profiles on module import
write_profiles_to_disk() 