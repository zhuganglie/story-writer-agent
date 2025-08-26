#!/usr/bin/env python3
"""
Demo script showing the enhanced language functionality for story generation
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from story_writer_agent import StoryWriterAgent

def demo_language_story_generation():
    """Demonstrate story generation in different languages"""
    print("🎭 Language-Based Story Generation Demo")
    print("=" * 50)

    try:
        # Initialize the agent
        agent = StoryWriterAgent()
        print("✅ Agent initialized successfully")

        # Show current language
        current_lang = agent.config.get("ui", {}).get("language", "en")
        print(f"📝 Current language: {current_lang}")

        # Demonstrate language selection
        print("\n🌐 Language Selection Demo:")
        print("The system now supports interactive language selection.")
        print("Supported languages include:")
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

        for code, name in supported_languages.items():
            marker = " ← Current" if code == current_lang else ""
            print(f"  {code}: {name}{marker}")

        print("\n🔧 Enhanced Features:")
        print("✅ Expanded language support (13 languages)")
        print("✅ Cultural context awareness")
        print("✅ Language-specific prompt templates")
        print("✅ Interactive language selection menu")
        print("✅ Enhanced configuration options")

        # Demonstrate prompt templates
        print("\n📋 Language-Specific Templates Demo:")
        for lang_code in ["en", "zh", "es"]:
            prompts = agent.get_language_specific_prompts(lang_code)
            cultural_context = agent.get_cultural_context(lang_code)
            print(f"\n{lang_code.upper()}:")
            print(f"  - Concept generation prompt available: {'Yes' if 'concept_generation' in prompts else 'No'}")
            print(f"  - Cultural context: {'Yes' if cultural_context else 'No'}")
            if cultural_context:
                print(f"  - Context preview: {cultural_context[:80]}...")

        print("\n🎯 Usage Instructions:")
        print("1. Run the main story writer: python story_writer_agent.py")
        print("2. Select your preferred language from the interactive menu")
        print("3. The entire story generation process will use your chosen language")
        print("4. Stories will be generated with culturally appropriate content")

        print("\n✨ Benefits:")
        print("• Stories generated in the user's preferred language")
        print("• Culturally adapted content and themes")
        print("• Enhanced user experience with native language support")
        print("• Consistent language throughout the entire creative process")

        return True

    except Exception as e:
        print(f"❌ Error during demo: {e}")
        return False

if __name__ == "__main__":
    success = demo_language_story_generation()
    print(f"\n{'✅ Demo completed successfully!' if success else '❌ Demo failed!'}")
    sys.exit(0 if success else 1)