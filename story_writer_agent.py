#!/usr/bin/env python3
"""
AI Short Story Writer Agent
Automates the short story writing workflow using Google Gemini AI
"""

import os
import json
import asyncio
import argparse
from typing import Dict, List, Optional, Any
from dataclasses import asdict
from data_models import (StoryConcept, StoryStructure, CharacterProfile, StorySetting, StoryProject)
import google.generativeai as genai
from datetime import datetime
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from localization import get_language_instruction, get_language_specific_prompts, get_cultural_context, get_supported_languages
from gemini_service import GeminiService





class StoryWriterAgent:
    """Main AI agent for automated short story writing"""

    def __init__(self, api_key: Optional[str] = None, config_path: str = "config.json"):
        """Initialize the agent with Gemini API key and configuration"""
        # Load environment variables from .env file
        load_dotenv()

        # Get API key from parameter or environment variable
        if api_key is None:
            api_key = os.getenv('GEMINI_API_KEY')

        if not api_key:
            raise ValueError("Gemini API key not provided. Set GEMINI_API_KEY environment variable or pass api_key parameter.")

        self.console = Console()
        self.config = self.load_config(config_path)
        self.gemini_service = GeminiService(api_key, self.config)
        self.project = StoryProject()
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        default_config = {
            "api": {
                "model": "gemini-2.5-flash",
                "max_tokens": 4000,
                "temperature": 0.7,
                "timeout": 300
            },
            "story": {
                "target_word_count": 3000,
                "min_word_count": 2000,
                "max_word_count": 5000,
                "auto_save": True,
                "save_frequency": "after_each_stage"
            },
            "ui": {
                "show_progress": True,
                "interactive_mode": True,
                "color_output": True,
                "verbose_logging": False,
                "language": "en"
            },
            "workflow": {
                "stages": [
                    "concept_generation",
                    "structure_planning",
                    "character_development",
                    "setting_creation",
                    "draft_generation",
                    "revision_enhancement",
                    "final_polish"
                ],
                "allow_stage_skipping": False,
                "require_user_approval": True
            }
        }

        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                # Merge user config with defaults
                for key, value in user_config.items():
                    if key in default_config:
                        if isinstance(value, dict) and isinstance(default_config[key], dict):
                            default_config[key].update(value)
                        else:
                            default_config[key] = value
                self.console.print(f"✅ Loaded configuration from {config_path}")
            except json.JSONDecodeError as e:
                self.console.print(f"Warning: Could not parse {config_path}: {e}")
                self.console.print("Using default configuration.")
            except Exception as e:
                self.console.print(f"Warning: Error loading {config_path}: {e}")
                self.console.print("Using default configuration.")
        else:
            self.console.print(f"Config file {config_path} not found, using defaults.")

        return default_config

    def set_language(self, language_code: str):
        """Set the language for story generation"""
        supported_languages = get_supported_languages()

        if language_code not in supported_languages:
            self.console.print(f"❌ Unsupported language: {language_code}")
            self.console.print("Supported languages:")
            for code, name in supported_languages.items():
                self.console.print(f"  {code}: {name}")
            return False

        self.config["ui"]["language"] = language_code
        self.console.print(f"✅ Language set to {supported_languages[language_code]} ({language_code})")
        return True

    

    def select_language_interactive(self):
        """Interactive language selection menu"""
        supported_languages = get_supported_languages()

        self.console.print("\n🌐 LANGUAGE SELECTION")
        self.console.print("=" * 40)
        self.console.print("Choose the language for story generation:")
        self.console.print()

        # Display languages in a numbered list
        languages_list = list(supported_languages.items())
        for i, (code, name) in enumerate(languages_list, 1):
            current_marker = " ← Current" if code == self.config.get("ui", {}).get("language", "en") else ""
            self.console.print(f"{i:2d}. {name} ({code}){current_marker}")

        self.console.print()
        self.console.print("0. Keep current language")

        while True:
            try:
                choice = input("\nEnter your choice (0-13): ").strip()

                if choice == "0":
                    current_lang = self.config.get("ui", {}).get("language", "en")
                    self.console.print(f"✅ Keeping current language: {supported_languages.get(current_lang, 'English')}")
                    return True

                choice_num = int(choice)
                if 1 <= choice_num <= len(languages_list):
                    selected_code = languages_list[choice_num - 1][0]
                    selected_name = languages_list[choice_num - 1][1]
                    return self.set_language(selected_code)
                else:
                    self.console.print(f"Please enter a number between 0 and {len(languages_list)}")

            except ValueError:
                self.console.print("Please enter a valid number")
            except KeyboardInterrupt:
                self.console.print("\nLanguage selection cancelled.")
                return False

    

    

    async def _generate_content(self, prompt: str, max_tokens: int = 4000) -> str:
        """Internal method to generate content with language instructions"""
        language = self.config.get("ui", {}).get("language", "en")
        language_instruction = get_language_instruction(language)
        full_prompt = f"{language_instruction}\n\n{prompt}"
        return await self.gemini_service.generate_content(full_prompt, max_tokens)

    def save_project(self, filename: str = None):
        """Save current project state with validation"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"story_project_{timestamp}.json"

        # Validate filename for security
        filename = self._validate_filename(filename)
        if not filename.endswith('.json'):
            filename += '.json'

        try:
            # Convert dataclasses to dictionaries for JSON serialization
            project_dict = {
                'concept': asdict(self.project.concept) if self.project.concept else None,
                'structure': asdict(self.project.structure) if self.project.structure else None,
                'characters': [asdict(char) for char in self.project.characters],
                'setting': asdict(self.project.setting) if self.project.setting else None,
                'draft': self.project.draft,
                'revisions': self.project.revisions,
                'final_story': self.project.final_story
            }

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(project_dict, f, indent=2, ensure_ascii=False)

            self.console.print(f"✅ Project saved to {filename}")
            return True

        except Exception as e:
            self.console.print(f"❌ Error saving project: {e}")
            return False

    def load_project(self, filename: str):
        """Load project from file"""
        if not os.path.exists(filename):
            self.console.print(f"❌ Error: File '{filename}' not found.")

            # Look for existing project files
            project_files = [f for f in os.listdir('.') if f.endswith('.json') and f != 'config.json']
            if project_files:
                self.console.print("\n📁 Available project files:")
                for i, f in enumerate(project_files, 1):
                    self.console.print(f"  {i}. {f}")
                self.console.print("\nPlease enter one of the above filenames or create a new project.")
            else:
                self.console.print("No project files found. You can create a new project by not loading an existing one.")

            return False

        try:
            with open(filename, 'r') as f:
                project_dict = json.load(f)

            # Reconstruct dataclasses from dictionaries
            if project_dict.get('concept'):
                self.project.concept = StoryConcept(**project_dict['concept'])
            if project_dict.get('structure'):
                self.project.structure = StoryStructure(**project_dict['structure'])
            if project_dict.get('setting'):
                self.project.setting = StorySetting(**project_dict['setting'])

            self.project.characters = [
                CharacterProfile(**char_dict) for char_dict in project_dict.get('characters', [])
            ]
            self.project.draft = project_dict.get('draft')
            self.project.revisions = project_dict.get('revisions', [])
            self.project.final_story = project_dict.get('final_story')

            self.console.print(f"✅ Project loaded from {filename}")
            return True

        except json.JSONDecodeError as e:
            self.console.print(f"❌ Error: Invalid JSON format in '{filename}': {e}")
            return False
        except Exception as e:
            self.console.print(f"❌ Error loading project from '{filename}': {e}")
            return False

    def display_concept_options(self, concepts: List[StoryConcept]):
        """Display concept options for user selection"""
        self.console.print("\n=== STORY CONCEPT OPTIONS ===")
        for i, concept in enumerate(concepts, 1):
            content = f"[bold]Genre:[/] {concept.genre}\n[bold]Premise:[/] {concept.premise}\n[bold]Conflict:[/] {concept.central_conflict}"
            self.console.print(Panel(content, title=f"{i}. {concept.title}", expand=False))

    def get_user_choice(self, options: List, prompt: str) -> int:
        """Get user choice from options"""
        while True:
            try:
                choice = int(input(f"\n{prompt} (1-{len(options)}): "))
                if 1 <= choice <= len(options):
                    return choice - 1
                else:
                    self.console.print(f"Please enter a number between 1 and {len(options)}")
            except ValueError:
                self.console.print("Please enter a valid number")

    def _validate_input(self, input_str: str, max_length: int = 1000) -> str:
        """Validate and sanitize user input"""
        if not input_str:
            return ""

        # Remove any potential harmful characters and limit length
        cleaned = input_str.strip()
        if len(cleaned) > max_length:
            cleaned = cleaned[:max_length] + "..."

        # Basic sanitization - remove potential prompt injection attempts
        dangerous_patterns = ["```", "system:", "assistant:", "user:", "<script", "javascript:"]
        for pattern in dangerous_patterns:
            cleaned = cleaned.replace(pattern, "")

        return cleaned

    def _validate_filename(self, filename: str) -> str:
        """Validate filename for security"""
        import re
        # Remove any path traversal attempts and dangerous characters
        filename = re.sub(r'[^\w\-_\.]', '', filename)
        filename = re.sub(r'[\.]{2,}', '.', filename)  # Remove multiple dots
        return filename[:255]  # Limit filename length
    async def stage1_concept_generation(self):
        """Stage 1: Concept Generation & Ideation"""
        self.console.print("\n📝 Stage 1: Concept Generation & Ideation")

        # Get language-specific prompts
        language = self.config.get("ui", {}).get("language", "en")
        language_prompts = get_language_specific_prompts(language)
        cultural_context = get_cultural_context(language)

        # Generate initial story concepts
        concept_prompt = f"{language_prompts['concept_generation']}"

        if cultural_context:
            concept_prompt += f"\n\nCultural Context: {cultural_context}"

        self.console.print("Generating story concepts...")
        concepts_response = await self._generate_content(concept_prompt)

        # Parse the concepts
        concepts = []
        lines = concepts_response.strip().split('\n')
        for line in lines:
            if '|' in line and len(line.split('|')) >= 4:
                parts = [part.strip() for part in line.split('|')]
                if len(parts) >= 4:
                    concept = StoryConcept(
                        title=parts[0],
                        genre=parts[1],
                        premise=parts[2],
                        central_conflict=parts[3]
                    )
                    concepts.append(concept)

        if not concepts:
            self.console.print("Error: Could not generate concepts. Please try again.")
            return

        # Display concepts and get user selection
        self.display_concept_options(concepts)
        selected_idx = self.get_user_choice(concepts, "Select a concept to develop")
        selected_concept = concepts[selected_idx]

        # Refine the selected concept
        refinement_prompt = f"""
        Take this concept and expand it:

        Title: {selected_concept.title}
        Genre: {selected_concept.genre}
        Premise: {selected_concept.premise}
        Conflict: {selected_concept.central_conflict}

        Provide:
        1. 3 potential protagonists with different backgrounds and motivations
        2. 3 possible settings that enhance the theme
        3. 3 different ending approaches (happy, bittersweet, tragic)
        4. A clear thematic direction

        Format your response clearly with headings for each section.
        """

        self.console.print(f"\nRefining concept: {selected_concept.title}")
        refinement_response = await self._generate_content(refinement_prompt)

        self.console.print("\n=== CONCEPT REFINEMENT ===")
        self.console.print(refinement_response)

        # Store the selected concept
        self.project.concept = selected_concept
        self.console.print(f"\n✅ Selected concept: {selected_concept.title}")

    async def stage2_structure_planning(self):
        """Stage 2: Story Structure Planning"""
        self.console.print("\n📋 Stage 2: Story Structure Planning")

        if not self.project.concept:
            self.console.print("Error: No concept selected. Please run Stage 1 first.")
            return

        structure_prompt = f"""
        Using this concept, create a three-act structure:

        Title: {self.project.concept.title}
        Genre: {self.project.concept.genre}
        Premise: {self.project.concept.premise}
        Conflict: {self.project.concept.central_conflict}

        ACT I (25% - Setup):
        - Opening scene that hooks the reader
        - Character introduction and motivation
        - Inciting incident

        ACT II (50% - Confrontation):
        - Rising action and obstacles
        - Character development moments
        - Midpoint twist or revelation

        ACT III (25% - Resolution):
        - Climax scene
        - Resolution of conflict
        - Character arc completion

        Provide specific scene descriptions for each act.
        """

        self.console.print("Creating story structure...")
        structure_response = await self._generate_content(structure_prompt)

        # Parse and store structure
        structure_data = {
            'act1_setup': {'description': structure_response},  # Simplified for now
            'act2_confrontation': {'description': ''},
            'act3_resolution': {'description': ''},
            'character_arc': {'description': ''}
        }

        self.project.structure = StoryStructure(**structure_data)
        self.console.print("✅ Story structure created")

    async def stage3_character_development(self):
        """Stage 3: Character Development Deep-Dive"""
        self.console.print("\n👤 Stage 3: Character Development Deep-Dive")

        if not self.project.concept:
            self.console.print("Error: No concept selected.")
            return

        # Get protagonist name from user or generate one
        protagonist_name = input("Enter protagonist name (or press Enter to generate): ").strip()
        if not protagonist_name:
            name_prompt = f"Generate a fitting name for the protagonist of this story: {self.project.concept.title}"
            protagonist_name = (await self._generate_content(name_prompt)).strip()

        # Get language-specific prompts and cultural context
        language = self.config.get("ui", {}).get("language", "en")
        language_prompts = get_language_specific_prompts(language)
        cultural_context = get_cultural_context(language)

        character_prompt = f"""
        Develop this character for the story "{self.project.concept.title}":

        Character: {protagonist_name}

        {language_prompts['character_background']}

        {language_prompts['personality']}

        RELATIONSHIPS:
        - Key relationships that shaped them
        - How they interact with others
        - Trust issues or social patterns

        INTERNAL CONFLICT:
        - What they want vs. what they need
        - Self-deception or blind spots
        - Character flaw that drives plot

        Format clearly with headings for each section.
        """

        if cultural_context:
            character_prompt += f"\n\nCultural Context: {cultural_context}"

        self.console.print(f"Developing character: {protagonist_name}")
        character_response = await self._generate_content(character_prompt)

        # Create character profile
        character = CharacterProfile(
            name=protagonist_name,
            background={'description': character_response},
            personality={'description': ''},
            relationships={'description': ''},
            internal_conflict={'description': ''},
            dialogue_samples=[]
        )

        self.project.characters.append(character)
        self.console.print(f"✅ Character profile created for {protagonist_name}")

    async def stage4_setting_creation(self):
        """Stage 4: Setting & Atmosphere Creation"""
        self.console.print("\n🌍 Stage 4: Setting & Atmosphere Creation")

        if not self.project.concept:
            self.console.print("Error: No concept selected.")
            return

        setting_prompt = f"""
        Create a vivid setting for this story:

        Title: {self.project.concept.title}
        Genre: {self.project.concept.genre}
        Premise: {self.project.concept.premise}

        PHYSICAL SPACE:
        - Key locations with sensory details
        - How the environment reflects character emotions
        - Symbolic elements that reinforce themes

        ATMOSPHERE:
        - Mood and tone descriptors
        - Time of day/season that enhances story
        - Weather or environmental factors as plot elements

        WORLD-BUILDING:
        - Rules or constraints of this world
        - Historical or cultural context
        - How setting influences character behavior

        SENSORY DETAILS:
        - 5 visual details that create mood
        - 3 sounds that enhance atmosphere
        - 2 smells/tastes that trigger memory
        - 3 tactile sensations that ground the reader

        Format clearly with headings for each section.
        """

        self.console.print("Creating immersive setting...")
        setting_response = await self._generate_content(setting_prompt)

        # Create setting object
        setting_data = {
            'physical_space': {'description': setting_response},
            'atmosphere': {'description': ''},
            'world_building': {'description': ''},
            'sensory_details': {'description': ''}
        }

        self.project.setting = StorySetting(**setting_data)
        self.console.print("✅ Setting and atmosphere created")

    async def stage5_draft_generation(self):
        """Stage 5: First Draft Generation"""
        self.console.print("\n✍️ Stage 5: First Draft Generation")

        if not all([self.project.concept, self.project.structure, self.project.characters, self.project.setting]):
            self.console.print("Error: Complete previous stages first.")
            return

        draft_prompt = f"""
        Write the first draft of this short story:

        Title: {self.project.concept.title}
        Genre: {self.project.concept.genre}
        Protagonist: {self.project.characters[0].name if self.project.characters else 'Unknown'}

        Structure: {self.project.structure.act1_setup.get('description', '')}

        Setting: {self.project.setting.physical_space.get('description', '')}

        Write a complete short story (2000-5000 words) that follows the three-act structure.
        Include vivid descriptions, natural dialogue, and compelling character development.
        Show, don't tell. Build tension and deliver emotional impact.
        """

        self.console.print("Writing first draft...")
        self.project.draft = await self._generate_content(draft_prompt, max_tokens=8000)
        self.console.print("✅ First draft completed")

    async def stage6_revision_enhancement(self):
        """Stage 6: Revision and Enhancement"""
        self.console.print("\n🔄 Stage 6: Revision and Enhancement")

        if not self.project.draft:
            self.console.print("Error: No draft to revise.")
            return

        revision_prompt = f"""
        Analyze this short story draft for improvement:

        {self.project.draft[:2000]}... [truncated for brevity]

        PACING:
        - Are there slow sections that drag?
        - Does tension build effectively?
        - Is the climax properly positioned?

        CHARACTER CONSISTENCY:
        - Do characters act according to their established traits?
        - Are character motivations clear throughout?
        - Does dialogue sound authentic?

        PLOT COHERENCE:
        - Are there any plot holes or logical inconsistencies?
        - Does every scene serve the overall story?
        - Is the ending satisfying and earned?

        Provide specific suggestions for improvement.
        """

        self.console.print("Analyzing draft for revisions...")
        analysis = await self._generate_content(revision_prompt)
        self.project.revisions.append(analysis)

        # Apply improvements
        improvement_prompt = f"""
        Based on this analysis:
        {analysis}

        Revise the story draft to address these issues:
        {self.project.draft}

        Focus on:
        - Improving pacing and tension
        - Strengthening character consistency
        - Fixing plot coherence issues
        - Enhancing dialogue and prose
        """

        self.console.print("Applying revisions...")
        revised_draft = await self._generate_content(improvement_prompt, max_tokens=8000)
        self.project.draft = revised_draft
        self.console.print("✅ Revisions completed")

    async def stage7_final_polish(self):
        """Stage 7: Final Polish and Proofing"""
        self.console.print("\n✨ Stage 7: Final Polish and Proofing")

        if not self.project.draft:
            self.console.print("Error: No draft to polish.")
            return

        polish_prompt = f"""
        Perform final polish and proofing on this story:

        {self.project.draft}

        Focus on:
        - Grammar and punctuation errors
        - Spelling mistakes
        - Sentence fragments or run-ons
        - Consistency in tense and POV
        - Dialogue formatting
        - Word choice and clarity
        - Flow and readability

        Provide the polished version.
        """

        self.console.print("Polishing final draft...")
        self.project.final_story = await self._generate_content(polish_prompt, max_tokens=8000)
        self.console.print("✅ Final polish completed")

    async def run_workflow(self):
        """Run the complete short story writing workflow"""
        self.console.print("🚀 Starting AI Short Story Writing Workflow")

        # Stage 1: Concept Generation & Ideation
        await self.stage1_concept_generation()

        # Stage 2: Story Structure Planning
        await self.stage2_structure_planning()

        # Stage 3: Character Development
        await self.stage3_character_development()

        # Stage 4: Setting & Atmosphere
        await self.stage4_setting_creation()

        # Stage 5: First Draft Generation
        await self.stage5_draft_generation()

        # Stage 6: Revision and Enhancement
        await self.stage6_revision_enhancement()

        # Stage 7: Final Polish and Proofing
        await self.stage7_final_polish()

        self.console.print("✅ Workflow completed! Your story is ready.")
        self.save_project()


def main():
    """Main entry point"""
    console = Console()
    parser = argparse.ArgumentParser(description="AI Short Story Writer Agent")
    parser.add_argument("--config", "-c", default="config.json", help="Path to configuration file")
    parser.add_argument("--load", "-l", help="Load existing project file")
    parser.add_argument("--api-key", help="Google Gemini API key (or set GEMINI_API_KEY env var)")
    parser.add_argument("--language", help="Language to use for the story")
    args = parser.parse_args()

    # Get API key from environment variable (loaded from .env file)
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        api_key = input("Enter your Google Gemini API key: ")

    # Initialize agent with configuration
    try:
        agent = StoryWriterAgent(api_key, args.config)
    except ValueError as e:
        console.print(f"Error initializing agent: {e}")
        return

    # Set language
    if args.language:
        if not agent.set_language(args.language):
            return
    else:
        if not agent.select_language_interactive():
            return

    # Load existing project if specified
    if args.load:
        try:
            agent.load_project(args.load)
        except FileNotFoundError:
            agent.console.print(f"Project file {args.load} not found.")
            return
        except Exception as e:
            agent.console.print(f"Error loading project: {e}")
            return
    elif os.path.exists("auto_save_project.json"):
        load_auto = input("Found auto-saved project. Load it? (y/n): ").lower().strip()
        if load_auto == 'y':
            agent.load_project("auto_save_project.json")

    # Run the workflow
    try:
        asyncio.run(agent.run_workflow())
    except KeyboardInterrupt:
        agent.console.print("\nWorkflow interrupted by user.")
        # Auto-save current progress
        agent.save_project("auto_save_project.json")
        agent.console.print("Progress auto-saved.")
    except Exception as e:
        agent.console.print(f"Error during workflow execution: {e}")
        # Auto-save current progress
        agent.save_project("auto_save_project.json")
        agent.console.print("Progress auto-saved due to error.")




if __name__ == "__main__":
    main()