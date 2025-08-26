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
from dataclasses import dataclass, asdict
import google.generativeai as genai
from datetime import datetime
from dotenv import load_dotenv


@dataclass
class StoryConcept:
    """Represents a story concept with all its components"""
    title: str
    genre: str
    premise: str
    central_conflict: str
    protagonist: Optional[Dict] = None
    setting: Optional[Dict] = None
    theme: Optional[str] = None


@dataclass
class StoryStructure:
    """Three-act story structure"""
    act1_setup: Dict[str, Any]
    act2_confrontation: Dict[str, Any]
    act3_resolution: Dict[str, Any]
    character_arc: Dict[str, Any]


@dataclass
class CharacterProfile:
    """Complete character profile"""
    name: str
    background: Dict[str, Any]
    personality: Dict[str, Any]
    relationships: Dict[str, Any]
    internal_conflict: Dict[str, Any]
    dialogue_samples: List[str]


@dataclass
class StorySetting:
    """Story setting and atmosphere"""
    physical_space: Dict[str, Any]
    atmosphere: Dict[str, Any]
    world_building: Dict[str, Any]
    sensory_details: Dict[str, Any]


@dataclass
class StoryProject:
    """Complete story project with all components"""
    concept: Optional[StoryConcept] = None
    structure: Optional[StoryStructure] = None
    characters: List[CharacterProfile] = None
    setting: Optional[StorySetting] = None
    draft: Optional[str] = None
    revisions: List[str] = None
    final_story: Optional[str] = None

    def __post_init__(self):
        if self.characters is None:
            self.characters = []
        if self.revisions is None:
            self.revisions = []


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

        self.config = self.load_config(config_path)

        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(self.config["api"]["model"])
        except Exception as e:
            raise ValueError(f"Failed to initialize Gemini API: {e}")

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
                print(f"✅ Loaded configuration from {config_path}")
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse {config_path}: {e}")
                print("Using default configuration.")
            except Exception as e:
                print(f"Warning: Error loading {config_path}: {e}")
                print("Using default configuration.")
        else:
            print(f"Config file {config_path} not found, using defaults.")

        return default_config

    def set_language(self, language_code: str):
        """Set the language for story generation"""
        supported_languages = {
            "en": "English",
            "zh": "Chinese (Simplified)",
            "zh-tw": "Chinese (Traditional)",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "ja": "Japanese",
            "ko": "Korean",
            "it": "Italian",
            "pt": "Portuguese",
            "ru": "Russian",
            "ar": "Arabic",
            "hi": "Hindi"
        }

        if language_code not in supported_languages:
            print(f"❌ Unsupported language: {language_code}")
            print("Supported languages:")
            for code, name in supported_languages.items():
                print(f"  {code}: {name}")
            return False

        self.config["ui"]["language"] = language_code
        print(f"✅ Language set to {supported_languages[language_code]} ({language_code})")
        return True

    def get_language_instruction(self, language_code: str) -> str:
        """Get detailed language instruction for the AI model"""
        language_instructions = {
            "en": "Please respond in English.",
            "zh": "Please respond in simplified Chinese (中文简体). Use natural, fluent Chinese that follows modern writing conventions.",
            "zh-tw": "Please respond in traditional Chinese (中文繁體). Use natural, fluent traditional Chinese characters.",
            "es": "Please respond in Spanish (español). Use natural, fluent Spanish with appropriate regional expressions.",
            "fr": "Please respond in French (français). Use natural, fluent French with proper grammar and expressions.",
            "de": "Please respond in German (Deutsch). Use natural, fluent German with proper grammar.",
            "ja": "Please respond in Japanese (日本語). Use natural, fluent Japanese with appropriate honorifics and expressions.",
            "ko": "Please respond in Korean (한국어). Use natural, fluent Korean with proper grammar and particles.",
            "it": "Please respond in Italian (italiano). Use natural, fluent Italian with proper grammar.",
            "pt": "Please respond in Portuguese (português). Use natural, fluent Portuguese with Brazilian expressions where appropriate.",
            "ru": "Please respond in Russian (русский). Use natural, fluent Russian with proper grammar and expressions.",
            "ar": "Please respond in Arabic (العربية). Use modern standard Arabic with natural expressions.",
            "hi": "Please respond in Hindi (हिन्दी). Use natural, fluent Hindi with proper grammar and expressions."
        }

        return language_instructions.get(language_code, "Please respond in English.")

    def select_language_interactive(self):
        """Interactive language selection menu"""
        supported_languages = {
            "en": "English",
            "zh": "Chinese (Simplified)",
            "zh-tw": "Chinese (Traditional)",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "ja": "Japanese",
            "ko": "Korean",
            "it": "Italian",
            "pt": "Portuguese",
            "ru": "Russian",
            "ar": "Arabic",
            "hi": "Hindi"
        }

        print("\n🌐 LANGUAGE SELECTION")
        print("=" * 40)
        print("Choose the language for story generation:")
        print()

        # Display languages in a numbered list
        languages_list = list(supported_languages.items())
        for i, (code, name) in enumerate(languages_list, 1):
            current_marker = " ← Current" if code == self.config.get("ui", {}).get("language", "en") else ""
            print(f"{i:2d}. {name} ({code}){current_marker}")

        print()
        print("0. Keep current language")

        while True:
            try:
                choice = input("\nEnter your choice (0-13): ").strip()

                if choice == "0":
                    current_lang = self.config.get("ui", {}).get("language", "en")
                    print(f"✅ Keeping current language: {supported_languages.get(current_lang, 'English')}")
                    return True

                choice_num = int(choice)
                if 1 <= choice_num <= len(languages_list):
                    selected_code = languages_list[choice_num - 1][0]
                    selected_name = languages_list[choice_num - 1][1]
                    return self.set_language(selected_code)
                else:
                    print(f"Please enter a number between 0 and {len(languages_list)}")

            except ValueError:
                print("Please enter a valid number")
            except KeyboardInterrupt:
                print("\nLanguage selection cancelled.")
                return False

    def get_language_specific_prompts(self, language_code: str) -> Dict[str, str]:
        """Get language-specific prompt templates"""
        prompts = {
            "en": {
                "concept_generation": """
                Generate 5 unique short story concepts that:
                - Are 2000-5000 words in scope
                - Include an interesting conflict or tension
                - Have clear character motivations
                - Contain a potential twist or revelation

                Format each as: Title | Genre | One-line premise | Central conflict

                Make each concept distinct and compelling.
                """,
                "character_background": "BACKGROUND:\n- Age, occupation, family situation\n- Formative childhood experience\n- Greatest fear and deepest desire",
                "personality": "PERSONALITY:\n- 3 dominant traits with examples\n- Speech patterns and mannerisms\n- How they handle conflict",
                "story_opening": "The scene opens on"
            },
            "zh": {
                "concept_generation": """
                生成5个独特的短篇小说概念，要求：
                - 篇幅在2000-5000字之间
                - 包含有趣的冲突或张力
                - 具有清晰的角色动机
                - 包含潜在的转折或揭示

                格式为：标题 | 类型 | 一行前提 | 核心冲突

                每个概念都要独特而引人入胜。
                """,
                "character_background": "背景：\n- 年龄、职业、家庭情况\n- 童年 formative 经历\n- 最大的恐惧和最深的愿望",
                "personality": "个性：\n- 3个主要特征及例子\n- 说话模式和举止\n- 如何处理冲突",
                "story_opening": "场景从"
            },
            "es": {
                "concept_generation": """
                Genera 5 conceptos únicos de historias cortas que:
                - Tengan un alcance de 2000-5000 palabras
                - Incluyan un conflicto o tensión interesante
                - Tengan motivaciones claras de personajes
                - Contengan un giro o revelación potencial

                Formato: Título | Género | Premisa en una línea | Conflicto central

                Haz que cada concepto sea distintivo y atractivo.
                """,
                "character_background": "ANTECEDENTES:\n- Edad, ocupación, situación familiar\n- Experiencia formativa de la infancia\n- Mayor miedo y deseo más profundo",
                "personality": "PERSONALIDAD:\n- 3 rasgos dominantes con ejemplos\n- Patrones de habla y manieras\n- Cómo manejan el conflicto",
                "story_opening": "La escena se abre en"
            }
        }

        # Return language-specific prompts or fall back to English
        return prompts.get(language_code, prompts["en"])

    def get_cultural_context(self, language_code: str) -> str:
        """Get cultural context for story generation"""
        contexts = {
            "zh": "Consider Chinese cultural elements, values, and storytelling traditions. Adapt the story to resonate with Chinese readers while maintaining universal appeal.",
            "ja": "Consider Japanese cultural elements, social dynamics, and aesthetic sensibilities. Include appropriate honorifics and cultural context.",
            "ko": "Consider Korean cultural elements, social hierarchy, and emotional expression. Adapt to Korean storytelling conventions.",
            "ar": "Consider Arabic cultural values, storytelling traditions, and social dynamics. Respect cultural sensitivities and norms.",
            "hi": "Consider Indian cultural elements, social structures, and philosophical perspectives. Incorporate relevant cultural context.",
            "es": "Consider Spanish-speaking cultural elements, regional differences, and emotional expression in storytelling.",
            "fr": "Consider French cultural elements, philosophical perspectives, and narrative style preferences.",
            "de": "Consider German cultural elements, attention to detail, and narrative precision.",
            "it": "Consider Italian cultural elements, family dynamics, and expressive storytelling style.",
            "pt": "Consider Portuguese-speaking cultural elements, particularly Brazilian perspectives and social dynamics.",
            "ru": "Consider Russian cultural elements, philosophical depth, and emotional intensity in storytelling."
        }

        return contexts.get(language_code, "")

    async def generate_content(self, prompt: str, max_tokens: int = 4000) -> str:
        """Generate content using Gemini API"""
        # Get language setting and instruction
        language = self.config.get("ui", {}).get("language", "en")
        language_instruction = self.get_language_instruction(language)

        # Add language instruction to prompt
        full_prompt = f"{language_instruction}\n\n{prompt}"

        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.model.generate_content(
                    full_prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=max_tokens,
                        temperature=self.config["api"]["temperature"]
                    )
                )
            )

            # Check if response has valid parts
            if response and response.parts:
                return response.text
            else:
                print(f"Warning: API returned empty response for prompt: {prompt[:100]}...")
                return ""

        except Exception as e:
            print(f"Error generating content: {e}")
            return ""

    def save_project(self, filename: str = None):
        """Save current project state"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"story_project_{timestamp}.json"

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

        with open(filename, 'w') as f:
            json.dump(project_dict, f, indent=2)

        print(f"Project saved to {filename}")

    def load_project(self, filename: str):
        """Load project from file"""
        if not os.path.exists(filename):
            print(f"❌ Error: File '{filename}' not found.")

            # Look for existing project files
            project_files = [f for f in os.listdir('.') if f.endswith('.json') and f != 'config.json']
            if project_files:
                print("\n📁 Available project files:")
                for i, f in enumerate(project_files, 1):
                    print(f"  {i}. {f}")
                print("\nPlease enter one of the above filenames or create a new project.")
            else:
                print("No project files found. You can create a new project by not loading an existing one.")

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

            print(f"✅ Project loaded from {filename}")
            return True

        except json.JSONDecodeError as e:
            print(f"❌ Error: Invalid JSON format in '{filename}': {e}")
            return False
        except Exception as e:
            print(f"❌ Error loading project from '{filename}': {e}")
            return False

    def display_concept_options(self, concepts: List[StoryConcept]):
        """Display concept options for user selection"""
        print("\n=== STORY CONCEPT OPTIONS ===")
        for i, concept in enumerate(concepts, 1):
            print(f"\n{i}. {concept.title}")
            print(f"   Genre: {concept.genre}")
            print(f"   Premise: {concept.premise}")
            print(f"   Conflict: {concept.central_conflict}")

    def get_user_choice(self, options: List, prompt: str) -> int:
        """Get user choice from options"""
        while True:
            try:
                choice = int(input(f"\n{prompt} (1-{len(options)}): "))
                if 1 <= choice <= len(options):
                    return choice - 1
                else:
                    print(f"Please enter a number between 1 and {len(options)}")
            except ValueError:
                print("Please enter a valid number")
    async def stage1_concept_generation(self):
        """Stage 1: Concept Generation & Ideation"""
        print("\n📝 Stage 1: Concept Generation & Ideation")

        # Get language-specific prompts
        language = self.config.get("ui", {}).get("language", "en")
        language_prompts = self.get_language_specific_prompts(language)
        cultural_context = self.get_cultural_context(language)

        # Generate initial story concepts
        concept_prompt = f"{language_prompts['concept_generation']}"

        if cultural_context:
            concept_prompt += f"\n\nCultural Context: {cultural_context}"

        print("Generating story concepts...")
        concepts_response = await self.generate_content(concept_prompt)

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
            print("Error: Could not generate concepts. Please try again.")
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

        print(f"\nRefining concept: {selected_concept.title}")
        refinement_response = await self.generate_content(refinement_prompt)

        print("\n=== CONCEPT REFINEMENT ===")
        print(refinement_response)

        # Store the selected concept
        self.project.concept = selected_concept
        print(f"\n✅ Selected concept: {selected_concept.title}")

    async def stage2_structure_planning(self):
        """Stage 2: Story Structure Planning"""
        print("\n📋 Stage 2: Story Structure Planning")

        if not self.project.concept:
            print("Error: No concept selected. Please run Stage 1 first.")
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

        print("Creating story structure...")
        structure_response = await self.generate_content(structure_prompt)

        # Parse and store structure
        structure_data = {
            'act1_setup': {'description': structure_response},  # Simplified for now
            'act2_confrontation': {'description': ''},
            'act3_resolution': {'description': ''},
            'character_arc': {'description': ''}
        }

        self.project.structure = StoryStructure(**structure_data)
        print("✅ Story structure created")

    async def stage3_character_development(self):
        """Stage 3: Character Development Deep-Dive"""
        print("\n👤 Stage 3: Character Development Deep-Dive")

        if not self.project.concept:
            print("Error: No concept selected.")
            return

        # Get protagonist name from user or generate one
        protagonist_name = input("Enter protagonist name (or press Enter to generate): ").strip()
        if not protagonist_name:
            name_prompt = f"Generate a fitting name for the protagonist of this story: {self.project.concept.title}"
            protagonist_name = (await self.generate_content(name_prompt)).strip()

        # Get language-specific prompts and cultural context
        language = self.config.get("ui", {}).get("language", "en")
        language_prompts = self.get_language_specific_prompts(language)
        cultural_context = self.get_cultural_context(language)

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

        print(f"Developing character: {protagonist_name}")
        character_response = await self.generate_content(character_prompt)

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
        print(f"✅ Character profile created for {protagonist_name}")

    async def stage4_setting_creation(self):
        """Stage 4: Setting & Atmosphere Creation"""
        print("\n🌍 Stage 4: Setting & Atmosphere Creation")

        if not self.project.concept:
            print("Error: No concept selected.")
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

        print("Creating immersive setting...")
        setting_response = await self.generate_content(setting_prompt)

        # Create setting object
        setting_data = {
            'physical_space': {'description': setting_response},
            'atmosphere': {'description': ''},
            'world_building': {'description': ''},
            'sensory_details': {'description': ''}
        }

        self.project.setting = StorySetting(**setting_data)
        print("✅ Setting and atmosphere created")

    async def stage5_draft_generation(self):
        """Stage 5: First Draft Generation"""
        print("\n✍️ Stage 5: First Draft Generation")

        if not all([self.project.concept, self.project.structure, self.project.characters, self.project.setting]):
            print("Error: Complete previous stages first.")
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

        print("Writing first draft...")
        self.project.draft = await self.generate_content(draft_prompt, max_tokens=8000)
        print("✅ First draft completed")

    async def stage6_revision_enhancement(self):
        """Stage 6: Revision and Enhancement"""
        print("\n🔄 Stage 6: Revision and Enhancement")

        if not self.project.draft:
            print("Error: No draft to revise.")
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

        print("Analyzing draft for revisions...")
        analysis = await self.generate_content(revision_prompt)
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

        print("Applying revisions...")
        revised_draft = await self.generate_content(improvement_prompt, max_tokens=8000)
        self.project.draft = revised_draft
        print("✅ Revisions completed")

    async def stage7_final_polish(self):
        """Stage 7: Final Polish and Proofing"""
        print("\n✨ Stage 7: Final Polish and Proofing")

        if not self.project.draft:
            print("Error: No draft to polish.")
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

        print("Polishing final draft...")
        self.project.final_story = await self.generate_content(polish_prompt, max_tokens=8000)
        print("✅ Final polish completed")

    async def run_workflow(self):
        """Run the complete short story writing workflow"""
        print("🚀 Starting AI Short Story Writing Workflow")

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

        print("✅ Workflow completed! Your story is ready.")
        self.save_project()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="AI Short Story Writer Agent")
    parser.add_argument("--config", "-c", default="config.json", help="Path to configuration file")
    parser.add_argument("--load", "-l", help="Load existing project file")
    parser.add_argument("--api-key", help="Google Gemini API key (or set GEMINI_API_KEY env var)")
    args = parser.parse_args()

    # Get API key from environment variable (loaded from .env file)
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        api_key = input("Enter your Google Gemini API key: ")

    # Initialize agent with configuration
    try:
        agent = StoryWriterAgent(api_key, args.config)
    except ValueError as e:
        print(f"Error initializing agent: {e}")
        return

    # Load existing project if specified
    if args.load:
        try:
            agent.load_project(args.load)
        except FileNotFoundError:
            print(f"Project file {args.load} not found.")
            return
        except Exception as e:
            print(f"Error loading project: {e}")
            return
    elif os.path.exists("auto_save_project.json"):
        load_auto = input("Found auto-saved project. Load it? (y/n): ").lower().strip()
        if load_auto == 'y':
            agent.load_project("auto_save_project.json")

    # Run the workflow
    try:
        asyncio.run(agent.run_workflow())
    except KeyboardInterrupt:
        print("\nWorkflow interrupted by user.")
        # Auto-save current progress
        agent.save_project("auto_save_project.json")
        print("Progress auto-saved.")
    except Exception as e:
        print(f"Error during workflow execution: {e}")
        # Auto-save current progress
        agent.save_project("auto_save_project.json")
        print("Progress auto-saved due to error.")
def main():
    """Main entry point"""
    # Load environment variables from .env file
    load_dotenv()

    # Get API key from environment or user input
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        api_key = input("Enter your Google Gemini API key: ")

    agent = StoryWriterAgent(api_key)

    # Language selection
    print("\n🌐 Language Selection")
    print("Supported languages:")
    supported_languages = {
        "en": "English",
        "zh": "Chinese (Simplified)",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "ja": "Japanese",
        "ko": "Korean"
    }
    for code, name in supported_languages.items():
        print(f"  {code}: {name}")

    current_language = agent.config.get("ui", {}).get("language", "en")
    print(f"Current language: {supported_languages.get(current_language, 'English')} ({current_language})")

    change_language = input("Change language? (y/n): ").lower().strip()
    if change_language == 'y':
        while True:
            new_language = input("Enter language code (e.g., zh, es, fr): ").lower().strip()
            if agent.set_language(new_language):
                break
            else:
                continue_changing = input("Try a different language? (y/n): ").lower().strip()
                if continue_changing != 'y':
                    break

    # Check if user wants to load existing project
    load_existing = input("Load existing project? (y/n): ").lower().strip()
    if load_existing == 'y':
        while True:
            filename = input("Enter project filename: ")
            if agent.load_project(filename):
                break
            else:
                continue_loading = input("Try a different file? (y/n): ").lower().strip()
                if continue_loading != 'y':
                    break

    # Run the workflow
    asyncio.run(agent.run_workflow())


if __name__ == "__main__":
    main()