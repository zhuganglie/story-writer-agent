
"""
This module provides language-specific prompts and cultural contexts for the story writer agent.
Enhanced with caching and better structure for maintainability.
"""

import json
from typing import Dict, Optional
from functools import lru_cache

# Language configurations with metadata
LANGUAGE_CONFIGS = {
    "en": {
        "name": "English",
        "rtl": False,
        "family": "germanic"
    },
    "zh": {
        "name": "Chinese (Simplified)",
        "rtl": False,
        "family": "sino-tibetan"
    },
    "zh-tw": {
        "name": "Chinese (Traditional)",
        "rtl": False,
        "family": "sino-tibetan"
    },
    "es": {
        "name": "Spanish",
        "rtl": False,
        "family": "romance"
    },
    "fr": {
        "name": "French",
        "rtl": False,
        "family": "romance"
    },
    "de": {
        "name": "German",
        "rtl": False,
        "family": "germanic"
    },
    "ja": {
        "name": "Japanese",
        "rtl": False,
        "family": "japonic"
    },
    "ko": {
        "name": "Korean",
        "rtl": False,
        "family": "koreanic"
    },
    "it": {
        "name": "Italian",
        "rtl": False,
        "family": "romance"
    },
    "pt": {
        "name": "Portuguese",
        "rtl": False,
        "family": "romance"
    },
    "ru": {
        "name": "Russian",
        "rtl": False,
        "family": "slavic"
    },
    "ar": {
        "name": "Arabic",
        "rtl": True,
        "family": "afro-asiatic"
    },
    "hi": {
        "name": "Hindi",
        "rtl": False,
        "family": "indo-european"
    }
}

@lru_cache(maxsize=128)
def get_language_specific_prompts(language_code: str) -> dict:
    """Get language-specific prompt templates with caching"""
    prompts = {
        "en": {
            "concept_generation": (
                'Generate 5 unique short story concepts that:\n'
                '- Are 2000-5000 words in scope\n'
                '- Include an interesting conflict or tension\n'
                '- Have clear character motivations\n'
                '- Contain a potential twist or revelation\n\n'
                'Format each as: Title | Genre | One-line premise | Central conflict\n\n'
                'Make each concept distinct and compelling.'
            ),
            "character_background": "BACKGROUND:\n- Age, occupation, family situation\n- Formative childhood experience\n- Greatest fear and deepest desire",
            "personality": "PERSONALITY:\n- 3 dominant traits with examples\n- Speech patterns and mannerisms\n- How they handle conflict",
            "story_opening": "The scene opens on"
        },
        "zh": {
            "concept_generation": (
                '生成5个独特的短篇小说概念，要求：\n'
                '- 篇幅在2000-5000字之间\n'
                '- 包含有趣的冲突或张力\n'
                '- 具有清晰的角色动机\n'
                '- 包含潜在的转折或揭示\n\n'
                '格式为：标题 | 类型 | 一行前提 | 核心冲突\n\n'
                '每个概念都要独特而引人入胜。'
            ),
            "character_background": "背景：\n- 年龄、职业、家庭情况\n- 童年 formative 经历\n- 最大的恐惧和最深的愿望",
            "personality": "个性：\n- 3个主要特征及例子\n- 说话模式和举止\n- 如何处理冲突",
            "story_opening": "场景从"
        },
        "zh-tw": {
            "concept_generation": (
                '生成5個獨特的短篇小說概念，要求：\n'
                '- 篇幅在2000-5000字之間\n'
                '- 包含有趣的衝突或張力\n'
                '- 具有清晰的角色動機\n'
                '- 包含潛在的轉折或揭示\n\n'
                '格式為：標題 | 類型 | 一行前提 | 核心衝突\n\n'
                '每個概念都要獨特而引人入勝。'
            ),
            "character_background": "背景：\n- 年齡、職業、家庭情況\n- 童年 formative 經歷\n- 最大的恐懼和最深的願望",
            "personality": "個性：\n- 3個主要特徵及例子\n- 說話模式和舉止\n- 如何處理衝突",
            "story_opening": "場景從"
        },
        "es": {
            "concept_generation": (
                'Genera 5 conceptos únicos de historias cortas que:\n'
                '- Tengan un alcance de 2000-5000 palabras\n'
                '- Incluyan un conflicto o tensión interesante\n'
                '- Tengan motivaciones claras de personajes\n'
                '- Contengan un giro o revelación potencial\n\n'
                'Formato: Título | Género | Premisa en una línea | Conflicto central\n\n'
                'Haz que cada concepto sea distintivo y atractivo.'
            ),
            "character_background": "ANTECEDENTES:\n- Edad, ocupación, situación familiar\n- Experiencia formativa de la infancia\n- Mayor miedo y deseo más profundo",
            "personality": "PERSONALIDAD:\n- 3 rasgos dominantes con ejemplos\n- Patrones de habla y manieras\n- Cómo manejan el conflicto",
            "story_opening": "La escena se abre en"
        }
    }

    # Add more language prompts dynamically for better coverage
    if language_code not in prompts and language_code in LANGUAGE_CONFIGS:
        # Generate basic prompts for unsupported languages using English as template
        prompts[language_code] = _generate_fallback_prompts(language_code)

    return prompts.get(language_code, prompts["en"])

def _generate_fallback_prompts(language_code: str) -> dict:
    """Generate fallback prompts for languages without full translation"""
    config = LANGUAGE_CONFIGS.get(language_code, {})
    language_name = config.get("name", language_code.upper())

    return {
        "concept_generation": (
            f'Generate 5 unique short story concepts in {language_name} that:\n'
            '- Are 2000-5000 words in scope\n'
            '- Include an interesting conflict or tension\n'
            '- Have clear character motivations\n'
            '- Contain a potential twist or revelation\n\n'
            'Format each as: Title | Genre | One-line premise | Central conflict\n\n'
            'Make each concept distinct and compelling.'
        ),
        "character_background": f"BACKGROUND (in {language_name}):\n- Age, occupation, family situation\n- Formative childhood experience\n- Greatest fear and deepest desire",
        "personality": f"PERSONALITY (in {language_name}):\n- 3 dominant traits with examples\n- Speech patterns and mannerisms\n- How they handle conflict",
        "story_opening": f"The scene opens on (in {language_name})"
    }

@lru_cache(maxsize=64)
def get_cultural_context(language_code: str) -> str:
    """Get cultural context for story generation with caching"""
    contexts = {
        "zh": "Consider Chinese cultural elements, values, and storytelling traditions. Adapt the story to resonate with Chinese readers while maintaining universal appeal.",
        "zh-tw": "Consider Traditional Chinese cultural elements, values, and storytelling traditions. Adapt the story to resonate with Taiwanese/Traditional Chinese readers while maintaining universal appeal.",
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

def get_supported_languages() -> Dict[str, str]:
    """Get all supported languages with their display names"""
    return {code: config["name"] for code, config in LANGUAGE_CONFIGS.items()}

def get_language_instruction(language_code: str) -> str:
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
