#!/usr/bin/env python3
"""
Optimized AI Short Story Writer Agent
Enhanced with better architecture, error handling, and performance optimizations
"""

import argparse
import asyncio
import logging
import os
import re
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn
from rich.table import Table

from config_manager import AppConfig, config_manager
from data_models import CharacterProfile, StoryConcept, StoryProject, StorySetting, StoryStructure
from gemini_service import GeminiService, GenerationResponse, GeminiServiceError
from localization import (
    get_cultural_context,
    get_language_instruction,
    get_language_specific_prompts,
    get_supported_languages,
)
from workflow_manager import WorkflowManager, WorkflowState, StageStatus, DefaultStageHandlers

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('story_writer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class StoryWriterError(Exception):
    """Base exception for story writer errors"""
    pass


class ValidationError(StoryWriterError):
    """Raised when validation fails"""
    pass


class StoryWriterAgent:
    """Optimized AI agent for automated short story writing"""

    def __init__(self, api_key: Optional[str] = None, config_path: Union[str, Path] = "config.json"):
        """Initialize the agent with enhanced configuration and services"""
        # Load environment variables
        load_dotenv()

        # Initialize console for rich output
        self.console = Console()

        # Load configuration
        self.config = self._load_and_validate_config(config_path)

        # Initialize API key
        self.api_key = self._get_api_key(api_key)

        # Initialize services
        self.gemini_service = GeminiService(self.api_key, asdict(self.config))
        self.workflow_manager = WorkflowManager(self.config, self)

        # Initialize project state
        self.project = StoryProject()
        self.project_file: Optional[Path] = None

        # Setup workflow handlers
        self._setup_workflow_handlers()

        # Performance tracking
        self._start_time: Optional[datetime] = None
        self._stage_times: Dict[str, float] = {}

        logger.info("StoryWriterAgent initialized successfully")

    def _load_and_validate_config(self, config_path: Union[str, Path]) -> AppConfig:
        """Load and validate configuration"""
        try:
            config = config_manager.load_config(config_path)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            self.console.print(f"[red]Error loading configuration: {e}[/red]")
            raise StoryWriterError(f"Configuration error: {e}") from e

    def _get_api_key(self, provided_key: Optional[str]) -> str:
        """Get and validate API key from various sources"""
        api_key = provided_key or os.getenv('GEMINI_API_KEY')

        if not api_key:
            api_key = self.console.input("[yellow]Enter your Google Gemini API key: [/yellow]")

        if not api_key or not api_key.strip():
            raise StoryWriterError("API key is required but not provided")

        return api_key.strip()

    def _setup_workflow_handlers(self):
        """Setup workflow stage handlers and callbacks"""
        # Register stage handlers
        stage_methods = {
            "concept_generation": self.stage1_concept_generation,
            "structure_planning": self.stage2_structure_planning,
            "character_development": self.stage3_character_development,
            "setting_creation": self.stage4_setting_creation,
            "draft_generation": self.stage5_draft_generation,
            "revision_enhancement": self.stage6_revision_enhancement,
            "final_polish": self.stage7_final_polish
        }

        for stage_name, handler in stage_methods.items():
            self.workflow_manager.register_stage_handler(stage_name, handler)

        # Register recovery handlers
        self.workflow_manager.register_recovery_handler(
            "concept_generation",
            DefaultStageHandlers.create_concept_recovery_handler([{
                "title": "The Unexpected Journey",
                "genre": "Literary Fiction",
                "premise": "A character faces an unexpected life change",
                "central_conflict": "Internal struggle between comfort and growth"
            }])
        )

        self.workflow_manager.register_recovery_handler(
            "structure_planning",
            DefaultStageHandlers.create_structure_recovery_handler()
        )

        # Setup callbacks
        self.workflow_manager.set_progress_callback(self._on_progress_update)
        self.workflow_manager.set_checkpoint_callback(self._on_checkpoint)
        self.workflow_manager.add_stage_start_callback(self._on_stage_start)
        self.workflow_manager.add_stage_complete_callback(self._on_stage_complete)
        self.workflow_manager.add_error_callback(self._on_workflow_error)

    async def _on_progress_update(self, state: WorkflowState):
        """Handle progress updates"""
        if self.config.ui.show_progress:
            self.console.print(
                f"[blue]Progress: {state.progress_percentage:.1f}% "
                f"({len(state.completed_stages)}/{state.total_stages})[/blue]"
            )

    async def _on_checkpoint(self, state: WorkflowState):
        """Handle checkpoint saves"""
        if self.config.story.auto_save and self.project_file:
            await self._save_project_async(self.project_file)

    async def _on_stage_start(self, stage_name: str):
        """Handle stage start events"""
        self._stage_times[stage_name] = datetime.now().timestamp()
        self.console.print(f"\n[green]🚀 Starting stage: {stage_name.replace('_', ' ').title()}[/green]")

    async def _on_stage_complete(self, result):
        """Handle stage completion events"""
        stage_name = result.stage_name
        if stage_name in self._stage_times:
            duration = datetime.now().timestamp() - self._stage_times[stage_name]
            self.console.print(f"[green]✅ Stage {stage_name} completed in {duration:.2f}s[/green]")

    async def _on_workflow_error(self, error: Exception, context: str):
        """Handle workflow errors"""
        self.console.print(f"[red]❌ Error in {context}: {error}[/red]")
        logger.error(f"Workflow error in {context}: {error}")

    def set_language(self, language_code: str) -> bool:
        """Set the language for story generation with validation"""
        supported_languages = get_supported_languages()

        if language_code not in supported_languages:
            self.console.print(f"[red]❌ Unsupported language: {language_code}[/red]")
            self.console.print("Supported languages:")
            for code, name in supported_languages.items():
                self.console.print(f"  {code}: {name}")
            return False

        self.config.ui.language = language_code
        self.console.print(
            f"[green]✅ Language set to {supported_languages[language_code]} ({language_code})[/green]"
        )
        logger.info(f"Language changed to: {language_code}")
        return True

    def select_language_interactive(self) -> bool:
        """Interactive language selection with enhanced UI"""
        supported_languages = get_supported_languages()

        # Create a table for better display
        table = Table(title="🌐 Language Selection", show_header=True, header_style="bold blue")
        table.add_column("No.", style="dim", width=4)
        table.add_column("Language", style="green")
        table.add_column("Code", style="yellow")
        table.add_column("Status", style="cyan")

        languages_list = list(supported_languages.items())
        current_lang = getattr(self.config.ui, 'language', 'en')

        for i, (code, name) in enumerate(languages_list, 1):
            status = "← Current" if code == current_lang else ""
            table.add_row(str(i), name, code, status)

        table.add_row("0", "Keep current", current_lang, "← Current")

        self.console.print(table)

        while True:
            try:
                choice = self.console.input("\n[bold cyan]Enter your choice (0-13): [/bold cyan]").strip()

                if choice == "0":
                    self.console.print(
                        f"[green]✅ Keeping current language: {supported_languages.get(current_lang, 'English')}[/green]"
                    )
                    return True

                choice_num = int(choice)
                if 1 <= choice_num <= len(languages_list):
                    selected_code = languages_list[choice_num - 1][0]
                    return self.set_language(selected_code)

                self.console.print(f"[red]Please enter a number between 0 and {len(languages_list)}[/red]")

            except ValueError:
                self.console.print("[red]Please enter a valid number[/red]")
            except KeyboardInterrupt:
                self.console.print("\n[yellow]Language selection cancelled.[/yellow]")
                return False

    async def _generate_content_with_language(self, prompt: str, max_tokens: int = 4000) -> GenerationResponse:
        """Generate content with language-specific instructions"""
        language = getattr(self.config.ui, 'language', 'en')
        language_instruction = get_language_instruction(language)
        cultural_context = get_cultural_context(language)

        # Enhance prompt with language and cultural context
        enhanced_prompt = f"{language_instruction}\n\n"
        if cultural_context:
            enhanced_prompt += f"Cultural Context: {cultural_context}\n\n"
        enhanced_prompt += prompt

        try:
            response = await self.gemini_service.generate_content(
                enhanced_prompt,
                max_tokens,
                temperature=self.config.api.temperature
            )
            
            # Log response for debugging (first 500 characters)
            logger.debug(f"AI response received ({len(response.content)} chars): {response.content[:500]}{'...' if len(response.content) > 500 else ''}")
            
            return response
        except GeminiServiceError as e:
            logger.error(f"Content generation failed: {e}")
            raise StoryWriterError(f"Failed to generate content: {e}") from e

    async def _save_project_async(self, filename: Optional[Path] = None) -> bool:
        """Asynchronously save project with better error handling"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = Path(f"story_project_{timestamp}.json")

        try:
            # Convert project to dictionary for JSON serialization
            project_dict = {
                'concept': asdict(self.project.concept) if self.project.concept else None,
                'structure': asdict(self.project.structure) if self.project.structure else None,
                'characters': [asdict(char) for char in self.project.characters],
                'setting': asdict(self.project.setting) if self.project.setting else None,
                'draft': self.project.draft,
                'revisions': self.project.revisions,
                'final_story': self.project.final_story,
                'metadata': {
                    'created_at': datetime.now().isoformat(),
                    'language': getattr(self.config.ui, 'language', 'en'),
                    'model': self.config.api.model,
                    'version': '2.0'
                }
            }

            # Write atomically
            temp_file = filename.with_suffix('.tmp')
            loop = asyncio.get_event_loop()

            def write_file():
                import json
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(project_dict, f, indent=2, ensure_ascii=False)
                temp_file.replace(filename)

            await loop.run_in_executor(None, write_file)

            self.project_file = filename
            self.console.print(f"[green]✅ Project saved to {filename}[/green]")
            logger.info(f"Project saved to {filename}")
            return True

        except Exception as e:
            self.console.print(f"[red]❌ Error saving project: {e}[/red]")
            logger.error(f"Failed to save project: {e}")
            return False

    async def load_project(self, filename: Union[str, Path]) -> bool:
        """Load project with enhanced validation and error recovery"""
        filename = Path(filename)

        if not filename.exists():
            self.console.print(f"[red]❌ File '{filename}' not found.[/red]")

            # Suggest available project files
            project_files = list(Path('.').glob('story_project_*.json'))
            if project_files:
                self.console.print("\n[yellow]📁 Available project files:[/yellow]")
                for i, f in enumerate(project_files, 1):
                    self.console.print(f"  {i}. {f.name}")
                self.console.print("\n[yellow]Use one of the above filenames.[/yellow]")

            return False

        try:
            import json

            with open(filename, 'r', encoding='utf-8') as f:
                project_dict = json.load(f)

            # Validate project structure
            if not isinstance(project_dict, dict):
                raise ValidationError("Invalid project file format")

            # Load project components with validation
            self.project = StoryProject()

            if project_dict.get('concept'):
                self.project.concept = StoryConcept(**project_dict['concept'])

            if project_dict.get('structure'):
                self.project.structure = StoryStructure(**project_dict['structure'])

            if project_dict.get('setting'):
                self.project.setting = StorySetting(**project_dict['setting'])

            # Load characters with validation
            self.project.characters = []
            for char_dict in project_dict.get('characters', []):
                if isinstance(char_dict, dict):
                    self.project.characters.append(CharacterProfile(**char_dict))

            self.project.draft = project_dict.get('draft')
            self.project.revisions = project_dict.get('revisions', [])
            self.project.final_story = project_dict.get('final_story')

            # Load metadata if available
            metadata = project_dict.get('metadata', {})
            if 'language' in metadata:
                self.set_language(metadata['language'])

            self.project_file = filename
            self.console.print(f"[green]✅ Project loaded from {filename}[/green]")
            logger.info(f"Project loaded from {filename}")
            return True

        except json.JSONDecodeError as e:
            self.console.print(f"[red]❌ Invalid JSON format in '{filename}': {e}[/red]")
            return False
        except Exception as e:
            self.console.print(f"[red]❌ Error loading project: {e}[/red]")
            logger.error(f"Failed to load project from {filename}: {e}")
            return False

    def display_concept_options(self, concepts: List[StoryConcept]):
        """Display concept options with enhanced formatting"""
        self.console.print("\n[bold blue]📚 STORY CONCEPT OPTIONS[/bold blue]")

        for i, concept in enumerate(concepts, 1):
            panel_content = (
                f"[bold]Genre:[/bold] {concept.genre}\n"
                f"[bold]Premise:[/bold] {concept.premise}\n"
                f"[bold]Conflict:[/bold] {concept.central_conflict}"
            )

            self.console.print(Panel(
                panel_content,
                title=f"{i}. {concept.title}",
                border_style="green",
                expand=False
            ))

    def get_user_choice(self, options: List, prompt: str, allow_cancel: bool = True) -> Optional[int]:
        """Get user choice with enhanced validation and cancel option"""
        while True:
            try:
                choice_prompt = f"\n[bold cyan]{prompt} (1-{len(options)})"
                if allow_cancel:
                    choice_prompt += " or 'c' to cancel"
                choice_prompt += ": [/bold cyan]"

                choice = self.console.input(choice_prompt).strip().lower()

                if allow_cancel and choice in ('c', 'cancel'):
                    return None

                choice_num = int(choice)
                if 1 <= choice_num <= len(options):
                    return choice_num - 1

                self.console.print(f"[red]Please enter a number between 1 and {len(options)}[/red]")

            except ValueError:
                self.console.print("[red]Please enter a valid number[/red]")
            except KeyboardInterrupt:
                if allow_cancel:
                    return None
                raise

    # Stage implementations with enhanced error handling and validation

    async def stage1_concept_generation(self):
        """Stage 1: Enhanced concept generation with better parsing"""
        self.console.print("\n[bold blue]📝 Stage 1: Concept Generation & Ideation[/bold blue]")

        # Get language-specific prompts
        language = getattr(self.config.ui, 'language', 'en')
        language_prompts = get_language_specific_prompts(language)

        concept_prompt = language_prompts['concept_generation']

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task("Generating story concepts...", total=None)

            try:
                response = await self._generate_content_with_language(concept_prompt)
                concepts_text = response.content
            except Exception as e:
                raise StoryWriterError(f"Failed to generate concepts: {e}") from e

            progress.update(task, completed=True)

        # Parse concepts with improved error handling
        concepts = self._parse_concepts(concepts_text)

        if not concepts:
            # Fallback: generate simpler concepts
            self.console.print("[yellow]⚠️ Initial generation format didn't match expectations, trying alternative parsing...[/yellow]")
            concepts = self._parse_concepts_fallback(concepts_text)

        if not concepts:
            # Second fallback: generate simpler concepts
            self.console.print("[yellow]⚠️ Concept parsing failed, trying simpler approach...[/yellow]")
            simple_prompt = "Generate 3 simple story ideas with title, genre, and basic conflict."

            try:
                response = await self._generate_content_with_language(simple_prompt)
                concepts = self._parse_concepts_fallback(response.content)
            except Exception as e:
                raise StoryWriterError(f"Concept generation completely failed: {e}") from e

        if not concepts:
            # Log the actual response for debugging
            logger.warning(f"AI response was: {concepts_text[:500]}{'...' if len(concepts_text) > 500 else ''}")
            raise StoryWriterError("Could not generate any valid concepts. The AI response may be empty or in an unexpected format.")

        # Display and get selection
        self.display_concept_options(concepts)
        if not concepts:
            raise StoryWriterError("No concepts available for selection.")
            
        selected_idx = self.get_user_choice(concepts, "Select a concept to develop")

        if selected_idx is None:
            raise StoryWriterError("Concept selection cancelled by user")

        self.project.concept = concepts[selected_idx]

        # Generate refinement
        await self._refine_selected_concept()

        self.console.print(f"[green]✅ Selected and refined concept: {self.project.concept.title}[/green]")

    def _parse_concepts(self, concepts_text: str) -> List[StoryConcept]:
        """Parse concepts from AI response with improved error handling"""
        concepts = []

        # Try different parsing strategies
        for line in concepts_text.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Try pipe-separated format first
            if '|' in line:
                try:
                    parts = [part.strip() for part in line.split('|')]
                    if len(parts) >= 4:
                        concept = StoryConcept(
                            title=parts[0],
                            genre=parts[1],
                            premise=parts[2],
                            central_conflict=parts[3]
                        )
                        concepts.append(concept)
                        continue
                except Exception as e:
                    logger.warning(f"Failed to parse pipe-separated concept line: {line}. Error: {e}")
            
            # Try numbered list format (e.g., "1. Title - Genre: Premise (Conflict)")
            if re.match(r'^\d+\.', line):
                try:
                    # Extract title (everything before " - ")
                    title_match = re.match(r'^\d+\.\s*([^-\n]+?)\s*-\s*', line)
                    if title_match:
                        title = title_match.group(1).strip()
                        rest = line[title_match.end():].strip()
                        
                        # Extract genre (between "Genre:" and next colon or parenthesis)
                        genre_match = re.search(r'Genre:\s*([^:\n]*?)(?:\s*\(|$)', rest)
                        genre = genre_match.group(1).strip() if genre_match else "Fiction"
                        
                        # Extract premise and conflict
                        premise_conflict_match = re.search(r'([^(\n]*?)\s*(?:$$(.*?)$$)?', rest)
                        if premise_conflict_match:
                            premise = premise_conflict_match.group(1).strip()
                            conflict = premise_conflict_match.group(2) if premise_conflict_match.group(2) else premise
                            
                            concept = StoryConcept(
                                title=title,
                                genre=genre,
                                premise=premise,
                                central_conflict=conflict
                            )
                            concepts.append(concept)
                            continue
                except Exception as e:
                    logger.warning(f"Failed to parse numbered concept line: {line}. Error: {e}")

        return concepts

    def _parse_concepts_fallback(self, concepts_text: str) -> List[StoryConcept]:
        """Fallback concept parsing for simpler formats"""
        concepts = []
        current_concept = {}

        for line in concepts_text.strip().split('\n'):
            line = line.strip()
            if not line:
                continue

            # Look for numbered items or titles
            if re.match(r'^\d+\.', line) or line.isupper():
                if current_concept and 'title' in current_concept:
                    try:
                        concept = StoryConcept(
                            title=current_concept.get('title', 'Untitled'),
                            genre=current_concept.get('genre', 'Fiction'),
                            premise=current_concept.get('premise', 'A story about human nature'),
                            central_conflict=current_concept.get('conflict', 'Internal struggle')
                        )
                        concepts.append(concept)
                    except Exception as e:
                        logger.warning(f"Failed to create concept from {current_concept}. Error: {e}")

                # Extract title from numbered items
                if re.match(r'^\d+\.', line):
                    current_concept = {'title': re.sub(r'^\d+\.\s*', '', line)}
                else:
                    current_concept = {'title': line}

            elif 'genre' in line.lower():
                current_concept['genre'] = line.split(':', 1)[-1].strip()
            elif 'premise' in line.lower() or 'story' in line.lower():
                current_concept['premise'] = line.split(':', 1)[-1].strip()
            elif 'conflict' in line.lower():
                current_concept['conflict'] = line.split(':', 1)[-1].strip()
            elif 'title' in line.lower() and 'title' not in current_concept:
                current_concept['title'] = line.split(':', 1)[-1].strip()

        # Add the last concept
        if current_concept and 'title' in current_concept:
            try:
                concept = StoryConcept(
                    title=current_concept.get('title', 'Untitled'),
                    genre=current_concept.get('genre', 'Fiction'),
                    premise=current_concept.get('premise', 'A story about human nature'),
                    central_conflict=current_concept.get('conflict', 'Internal struggle')
                )
                concepts.append(concept)
            except Exception as e:
                logger.warning(f"Failed to create final concept from {current_concept}. Error: {e}")

        return concepts

    async def _refine_selected_concept(self):
        """Refine the selected concept with additional details"""
        if not self.project.concept:
            return

        refinement_prompt = f"""
        Expand and refine this story concept:

        Title: {self.project.concept.title}
        Genre: {self.project.concept.genre}
        Premise: {self.project.concept.premise}
        Conflict: {self.project.concept.central_conflict}

        Provide:
        1. Three potential protagonists with different backgrounds
        2. Three possible settings that enhance the theme
        3. A clear thematic direction
        4. Potential story hooks and twists

        Format your response clearly with numbered sections.
        """

        try:
            response = await self._generate_content_with_language(refinement_prompt)
            self.console.print(Panel(
                response.content,
                title="[bold green]Concept Refinement[/bold green]",
                border_style="green"
            ))
        except Exception as e:
            logger.warning(f"Concept refinement failed: {e}")
            # Continue without refinement

    async def stage2_structure_planning(self):
        """Stage 2: Enhanced story structure planning"""
        self.console.print("\n[bold blue]📋 Stage 2: Story Structure Planning[/bold blue]")

        if not self.project.concept:
            raise StoryWriterError("No concept available for structure planning")

        structure_prompt = f"""
        Create a detailed three-act structure for this story:

        Title: {self.project.concept.title}
        Genre: {self.project.concept.genre}
        Premise: {self.project.concept.premise}
        Conflict: {self.project.concept.central_conflict}

        Provide a complete structure with:

        ACT I (25% - Setup):
        - Hook: Opening scene that grabs attention
        - Character Introduction: Protagonist and key supporting characters
        - Setting Establishment: Time, place, and world
        - Inciting Incident: Event that starts the main conflict

        ACT II (50% - Confrontation):
        - Rising Action: Obstacles and complications
        - Character Development: Growth and revelations
        - Midpoint Twist: Major revelation or plot turn
        - Crisis: Lowest point for protagonist

        ACT III (25% - Resolution):
        - Climax: Final confrontation or decision
        - Falling Action: Consequences of climax
        - Resolution: How conflicts are resolved
        - Denouement: Final state of characters and world

        Format each section clearly with specific scene descriptions.
        """

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task("Creating story structure...", total=None)

            try:
                response = await self._generate_content_with_language(structure_prompt)
                structure_content = response.content
            except Exception as e:
                raise StoryWriterError(f"Structure planning failed: {e}") from e

            progress.update(task, completed=True)

        # Create structure object
        self.project.structure = StoryStructure(
            act1_setup={'description': structure_content},
            act2_confrontation={'description': ''},
            act3_resolution={'description': ''},
            character_arc={'description': ''}
        )

        self.console.print("[green]✅ Story structure created[/green]")

    async def stage3_character_development(self):
        """Stage 3: Enhanced character development"""
        self.console.print("\n[bold blue]👤 Stage 3: Character Development Deep-Dive[/bold blue]")

        if not self.project.concept:
            raise StoryWriterError("No concept available for character development")

        # Get protagonist name
        protagonist_name = self.console.input(
            "[cyan]Enter protagonist name (or press Enter to generate): [/cyan]"
        ).strip()

        if not protagonist_name:
            name_prompt = f"""
            Generate a fitting name for the protagonist of this {self.project.concept.genre} story:
            Title: {self.project.concept.title}
            Setting: Contemporary/Modern
            """

            try:
                response = await self._generate_content_with_language(name_prompt, max_tokens=50)
                protagonist_name = response.content.strip().split('\n')[0].strip()
            except Exception:
                protagonist_name = "Alex"  # Fallback name

        # Generate comprehensive character profile
        language = getattr(self.config.ui, 'language', 'en')
        language_prompts = get_language_specific_prompts(language)

        character_prompt = f"""
        Create a comprehensive character profile for the protagonist of "{self.project.concept.title}":

        Character Name: {protagonist_name}
        Story Context: {self.project.concept.premise}
        Central Conflict: {self.project.concept.central_conflict}

        {language_prompts.get('character_background', 'BACKGROUND: Age, occupation, family, formative experiences, fears, and desires')}

        {language_prompts.get('personality', 'PERSONALITY: 3 dominant traits with examples, speech patterns, conflict handling')}

        RELATIONSHIPS:
        - Key relationships that shaped them
        - Current social connections
        - Trust issues or social patterns
        - How they interact with others

        INTERNAL CONFLICT:
        - What they want vs. what they need
        - Self-deception or blind spots
        - Character flaw that drives the plot
        - Growth arc throughout the story

        DIALOGUE SAMPLES:
        - 3 example lines showing their voice
        - How they speak under stress
        - Unique phrases or expressions

        Format clearly with distinct sections and bullet points.
        """

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task(f"Developing character: {protagonist_name}...", total=None)

            try:
                response = await self._generate_content_with_language(character_prompt)
                character_content = response.content
            except Exception as e:
                raise StoryWriterError(f"Character development failed: {e}") from e

            progress.update(task, completed=True)

        # Create character profile
        character = CharacterProfile(
            name=protagonist_name,
            background={'description': character_content},
            personality={'description': ''},
            relationships={'description': ''},
            internal_conflict={'description': ''},
            dialogue_samples=[]
        )

        self.project.characters.append(character)
        self.console.print(f"[green]✅ Character profile created for {protagonist_name}[/green]")

    async def stage4_setting_creation(self):
        """Stage 4: Enhanced setting and atmosphere creation"""
        self.console.print("\n[bold blue]🌍 Stage 4: Setting & Atmosphere Creation[/bold blue]")

        if not self.project.concept:
            raise StoryWriterError("No concept available for setting creation")

        setting_prompt = f"""
        Create a vivid, immersive setting for this story:

        Title: {self.project.concept.title}
        Genre: {self.project.concept.genre}
        Premise: {self.project.concept.premise}
        Protagonist: {self.project.characters[0].name if self.project.characters else 'The protagonist'}

        PHYSICAL SPACE:
        - Primary location with detailed sensory descriptions
        - Secondary locations that support the story
        - How environments reflect character emotions
        - Symbolic elements that reinforce themes
        - Layout and geography that affects plot

        ATMOSPHERE:
        - Overall mood and tone descriptors
        - Time of day/season that enhances the story
        - Weather patterns as narrative elements
        - Lighting and color palettes
        - Sounds and ambient noise

        WORLD-BUILDING:
        - Cultural and social context
        - Historical background if relevant
        - Rules or constraints of this world
        - Economic and political factors
        - Technology level and daily life details

        SENSORY DETAILS:
        - 5 specific visual details that create mood
        - 4 distinctive sounds that enhance atmosphere
        - 3 smells or tastes that trigger emotions
        - 3 tactile sensations that ground the reader
        - How these details change with story progression

        Format with clear headings and rich, specific descriptions.
        """

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task("Creating immersive setting...", total=None)

            try:
                response = await self._generate_content_with_language(setting_prompt)
                setting_content = response.content
            except Exception as e:
                raise StoryWriterError(f"Setting creation failed: {e}") from e

            progress.update(task, completed=True)

        # Create setting object
        self.project.setting = StorySetting(
            physical_space={'description': setting_content},
            atmosphere={'description': ''},
            world_building={'description': ''},
            sensory_details={'description': ''}
        )

        self.console.print("[green]✅ Setting and atmosphere created[/green]")

    async def stage5_draft_generation(self):
        """Stage 5: Enhanced first draft generation"""
        self.console.print("\n[bold blue]✍️ Stage 5: First Draft Generation[/bold blue]")

        # Validate prerequisites
        required_components = [
            (self.project.concept, "concept"),
            (self.project.structure, "structure"),
            (self.project.characters, "characters"),
            (self.project.setting, "setting")
        ]

        missing = [name for component, name in required_components if not component]
        if missing:
            raise StoryWriterError(f"Missing required components: {', '.join(missing)}")

        # Gather all story elements
        protagonist = self.project.characters[0] if self.project.characters else None
        protagonist_name = protagonist.name if protagonist else "The protagonist"

        draft_prompt = f"""
        Write a complete short story draft based on these elements:

        STORY CONCEPT:
        Title: {self.project.concept.title}
        Genre: {self.project.concept.genre}
        Premise: {self.project.concept.premise}
        Central Conflict: {self.project.concept.central_conflict}

        PROTAGONIST:
        Name: {protagonist_name}
        {protagonist.background.get('description', '')[:500] if protagonist else ''}

        STRUCTURE:
        {self.project.structure.act1_setup.get('description', '')[:1000]}

        SETTING:
        {self.project.setting.physical_space.get('description', '')[:800]}

        REQUIREMENTS:
        - Target length: {self.config.story.target_word_count} words
        - Follow three-act structure
        - Include vivid sensory details
        - Show character development through actions
        - Use natural, engaging dialogue
        - Build tension progressively
        - Deliver satisfying resolution

        Write the complete story now, focusing on:
        1. Strong opening hook
        2. Clear character motivation
        3. Rising tension and conflict
        4. Emotional resonance
        5. Satisfying conclusion

        Begin the story immediately without preamble.
        """

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=self.console
        ) as progress:
            task = progress.add_task("Writing first draft...", total=100)

            try:
                # Update progress periodically
                for i in range(0, 90, 10):
                    progress.update(task, completed=i)
                    await asyncio.sleep(0.1)

                response = await self._generate_content_with_language(
                    draft_prompt,
                    max_tokens=self.config.api.max_tokens * 2  # Allow more tokens for full story
                )
                self.project.draft = response.content

                progress.update(task, completed=100)

            except Exception as e:
                progress.stop()
                raise StoryWriterError(f"Draft generation failed: {e}") from e

        # Validate draft length
        word_count = len(self.project.draft.split())
        if word_count < self.config.story.min_word_count:
            self.console.print(f"[yellow]⚠️ Draft is shorter than target ({word_count} words)[/yellow]")
        elif word_count > self.config.story.max_word_count:
            self.console.print(f"[yellow]⚠️ Draft is longer than target ({word_count} words)[/yellow]")
        else:
            self.console.print(f"[green]✅ Draft completed ({word_count} words)[/green]")

    async def stage6_revision_enhancement(self):
        """Stage 6: Enhanced revision and improvement"""
        self.console.print("\n[bold blue]🔄 Stage 6: Revision and Enhancement[/bold blue]")

        if not self.project.draft:
            raise StoryWriterError("No draft available for revision")

        # Analysis phase
        analysis_prompt = f"""
        Analyze this short story draft for improvement opportunities:

        STORY DRAFT:
        {self.project.draft[:3000]}{'...' if len(self.project.draft) > 3000 else ''}

        Evaluate these aspects:

        PACING AND STRUCTURE:
        - Does the opening hook readers effectively?
        - Is the pacing appropriate for the genre?
        - Does tension build naturally toward climax?
        - Is the three-act structure clearly defined?

        CHARACTER DEVELOPMENT:
        - Are character motivations clear and compelling?
        - Does the protagonist show growth/change?
        - Is dialogue natural and character-specific?
        - Are character actions consistent with personality?

        PLOT AND COHERENCE:
        - Are there any plot holes or inconsistencies?
        - Does every scene advance the story?
        - Is the conflict resolution satisfying?
        - Are cause-and-effect relationships clear?

        PROSE AND STYLE:
        - Is the writing style appropriate for the genre?
        - Are descriptions vivid without being excessive?
        - Is the point of view consistent?
        - Are there opportunities to "show don't tell"?

        Provide specific, actionable feedback for each area.
        """

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            analysis_task = progress.add_task("Analyzing draft...", total=None)

            try:
                analysis_response = await self._generate_content_with_language(analysis_prompt)
                analysis = analysis_response.content
            except Exception as e:
                raise StoryWriterError(f"Draft analysis failed: {e}") from e

            progress.update(analysis_task, completed=True)

        # Store analysis
        self.project.revisions.append(f"Analysis: {analysis}")

        # Display analysis to user
        self.console.print(Panel(
            analysis[:1000] + ("..." if len(analysis) > 1000 else ""),
            title="[bold yellow]Draft Analysis[/bold yellow]",
            border_style="yellow"
        ))

        # Ask user if they want to proceed with revision
        if self.config.workflow.require_user_approval:
            proceed = self.console.input(
                "\n[cyan]Proceed with automatic revision based on this analysis? (y/n): [/cyan]"
            ).lower().strip()

            if proceed != 'y':
                self.console.print("[yellow]Revision skipped by user[/yellow]")
                return

        # Revision phase
        revision_prompt = f"""
        Revise this story based on the analysis feedback:

        ORIGINAL DRAFT:
        {self.project.draft}

        ANALYSIS AND FEEDBACK:
        {analysis}

        Instructions for revision:
        1. Address the specific issues identified in the analysis
        2. Maintain the core story concept and character
        3. Improve pacing, dialogue, and descriptions
        4. Enhance character development and emotional impact
        5. Ensure plot coherence and satisfying resolution
        6. Keep the story within the target word count range

        Provide the complete revised story.
        """

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            revision_task = progress.add_task("Applying revisions...", total=None)

            try:
                revision_response = await self._generate_content_with_language(
                    revision_prompt,
                    max_tokens=self.config.api.max_tokens * 2
                )
                self.project.draft = revision_response.content

                progress.update(revision_task, completed=True)

            except Exception as e:
                progress.stop()
                raise StoryWriterError(f"Revision failed: {e}") from e

        # Store revision record
        self.project.revisions.append(f"Revision applied: {datetime.now().isoformat()}")

        word_count = len(self.project.draft.split())
        self.console.print(f"[green]✅ Revisions completed ({word_count} words)[/green]")

    async def stage7_final_polish(self):
        """Stage 7: Final polish and proofing"""
        self.console.print("\n[bold blue]✨ Stage 7: Final Polish and Proofing[/bold blue]")

        if not self.project.draft:
            raise StoryWriterError("No draft available for polishing")

        polish_prompt = f"""
        Perform final polish and proofing on this story:

        STORY DRAFT:
        {self.project.draft}

        Focus on these final improvements:

        GRAMMAR AND MECHANICS:
        - Correct any grammar, punctuation, or spelling errors
        - Fix sentence fragments or run-on sentences
        - Ensure consistent verb tense and point of view
        - Proper dialogue formatting and attribution

        STYLE AND FLOW:
        - Improve word choice for clarity and impact
        - Eliminate redundancy and wordiness
        - Enhance sentence variety and rhythm
        - Ensure smooth transitions between scenes

        FINAL TOUCHES:
        - Strengthen the opening and closing lines
        - Polish dialogue to sound more natural
        - Add final sensory details where appropriate
        - Ensure emotional beats land effectively

        Provide the final, polished version of the story.
        """

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console
        ) as progress:
            task = progress.add_task("Polishing final draft...", total=None)

            try:
                response = await self._generate_content_with_language(
                    polish_prompt,
                    max_tokens=self.config.api.max_tokens * 2
                )
                self.project.final_story = response.content

                progress.update(task, completed=True)

            except Exception as e:
                progress.stop()
                raise StoryWriterError(f"Final polishing failed: {e}") from e

        # Final validation
        if self.project.final_story:
            word_count = len(self.project.final_story.split())
            self.console.print(f"[green]✅ Final polish completed ({word_count} words)[/green]")

            # Show completion summary
            self.console.print(Panel(
                f"[bold green]Story Complete![/bold green]\n\n"
                f"Title: {self.project.concept.title if self.project.concept else 'Untitled'}\n"
                f"Genre: {self.project.concept.genre if self.project.concept else 'Unknown'}\n"
                f"Word Count: {word_count}\n"
                f"Characters: {len(self.project.characters)}\n"
                f"Revisions: {len(self.project.revisions)}",
                title="[bold blue]🎉 Story Creation Summary[/bold blue]",
                border_style="green"
            ))
        else:
            self.console.print("[red]❌ Final polishing produced no output[/red]")

    async def run_workflow(self) -> WorkflowState:
        """Run the complete workflow using the workflow manager"""
        self.console.print("[bold blue]🚀 Starting AI Short Story Writing Workflow[/bold blue]")
        self._start_time = datetime.now()

        try:
            # Set workflow project
            self.workflow_manager.state.project = self.project

            # Run the workflow
            final_state = await self.workflow_manager.run_workflow(self.project)

            # Display final results
            if final_state.status.value == "completed":
                self.console.print("\n[bold green]🎉 Workflow completed successfully![/bold green]")

                # Save the project
                if self.config.story.auto_save:
                    await self._save_project_async()

                # Display statistics
                stats = self.workflow_manager.get_workflow_summary()
                self._display_completion_stats(stats)
            else:
                self.console.print(f"\n[yellow]Workflow ended with status: {final_state.status.value}[/yellow]")

            return final_state

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            self.console.print(f"\n[red]❌ Workflow failed: {e}[/red]")

            # Auto-save progress
            if self.config.story.auto_save:
                await self._save_project_async(Path("emergency_save_project.json"))
                self.console.print("[yellow]Progress auto-saved to emergency_save_project.json[/yellow]")

            raise

    def _display_completion_stats(self, stats: Dict[str, Any]):
        """Display workflow completion statistics"""
        table = Table(title="📊 Workflow Statistics", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Duration", f"{stats.get('total_duration', 0):.2f}s")
        table.add_row("Stages Completed", f"{stats['stages']['completed']}/{stats['stages']['total']}")
        table.add_row("Success Rate", f"{(stats['stages']['completed']/stats['stages']['total']*100):.1f}%")

        if stats.get('stage_durations'):
            slowest_stage = max(stats['stage_durations'], key=stats['stage_durations'].get)
            table.add_row("Slowest Stage", f"{slowest_stage} ({stats['stage_durations'][slowest_stage]:.2f}s)")

        self.console.print(table)


def create_agent_from_args(args) -> StoryWriterAgent:
    """Create agent instance from command line arguments"""
    try:
        agent = StoryWriterAgent(args.api_key, args.config)
        logger.info("Agent created successfully")
        return agent
    except Exception as e:
        console = Console()
        console.print(f"[red]❌ Error initializing agent: {e}[/red]")
        sys.exit(1)


async def setup_language(agent: StoryWriterAgent, args) -> bool:
    """Setup language configuration"""
    if args.language:
        if not agent.set_language(args.language):
            return False
    elif not agent.select_language_interactive():
        return False
    return True


async def handle_project_loading(agent: StoryWriterAgent, args) -> bool:
    """Handle project loading logic"""
    if args.load:
        if not await agent.load_project(args.load):
            return False
    else:
        # Check for auto-save
        auto_save_path = Path("auto_save_project.json")
        if auto_save_path.exists():
            load_auto = agent.console.input(
                "[cyan]Found auto-saved project. Load it? (y/n): [/cyan]"
            ).lower().strip()
            if load_auto == 'y':
                await agent.load_project(auto_save_path)
    return True


def main():
    """Enhanced main entry point with better error handling"""
    console = Console()

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Optimized AI Short Story Writer Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Interactive mode with language selection
  %(prog)s --language zh            # Start with Chinese language
  %(prog)s --load story.json        # Load existing project
  %(prog)s --config custom.json     # Use custom configuration
  %(prog)s --api-key YOUR_KEY       # Provide API key directly
        """
    )

    parser.add_argument(
        "--config", "-c",
        default="config.json",
        help="Path to configuration file (default: config.json)"
    )
    parser.add_argument(
        "--load", "-l",
        help="Load existing project file"
    )
    parser.add_argument(
        "--api-key",
        help="Google Gemini API key (or set GEMINI_API_KEY env var)"
    )
    parser.add_argument(
        "--language",
        help="Language code for story generation (e.g., en, zh, es)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled")

    try:
        console.print("[bold blue]🤖 AI Short Story Writer Agent v2.0[/bold blue]\n")

        # Create agent
        agent = create_agent_from_args(args)

        async def run_async():
            try:
                # Setup language
                if not await setup_language(agent, args):
                    console.print("[yellow]Exiting due to language setup cancellation[/yellow]")
                    return

                # Handle project loading
                if not await handle_project_loading(agent, args):
                    console.print("[red]Exiting due to project loading failure[/red]")
                    return

                # Run the workflow
                workflow_state = await agent.run_workflow()

                # Final status
                if workflow_state.status.value == "completed":
                    console.print("\n[bold green]🎉 Story creation completed successfully![/bold green]")
                    if agent.project_file:
                        console.print(f"[green]📁 Project saved: {agent.project_file}[/green]")
                else:
                    console.print(f"\n[yellow]⚠️ Workflow ended with status: {workflow_state.status.value}[/yellow]")

            except KeyboardInterrupt:
                console.print("\n[yellow]⚠️ Workflow interrupted by user[/yellow]")

                # Emergency save
                emergency_file = Path(f"interrupted_project_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                await agent._save_project_async(emergency_file)
                console.print(f"[green]💾 Progress saved to {emergency_file}[/green]")

            except Exception as e:
                console.print(f"\n[red]❌ Unexpected error: {e}[/red]")
                logger.exception("Unexpected error in main workflow")

                # Emergency save
                try:
                    emergency_file = Path(f"error_save_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                    await agent._save_project_async(emergency_file)
                    console.print(f"[green]💾 Emergency save created: {emergency_file}[/green]")
                except Exception as save_error:
                    logger.error(f"Failed to create emergency save: {save_error}")

                sys.exit(1)

        # Run the async main function
        asyncio.run(run_async())

    except Exception as e:
        console.print(f"[red]❌ Fatal error: {e}[/red]")
        logger.exception("Fatal error in main")
        sys.exit(1)


if __name__ == "__main__":
    main()
