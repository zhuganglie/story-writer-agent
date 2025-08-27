"""
Data models for the Story Writer Agent.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

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
    characters: List[CharacterProfile] = field(default_factory=list)
    setting: Optional[StorySetting] = None
    draft: Optional[str] = None
    revisions: List[str] = field(default_factory=list)
    final_story: Optional[str] = None
