#!/usr/bin/env python3
"""
Test script for the enhanced language functionality
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from story_writer_agent import StoryWriterAgent

def test_language_functionality():
    """Test the language selection and story generation functionality"""
    print("🧪 Testing Language Functionality")
    print("=" * 40)

    try:
        # Initialize the agent
        agent = StoryWriterAgent()
        print("✅ Agent initialized successfully")

        # Test language selection
        print("\n🌐 Testing Language Selection...")
        result = agent.select_language_interactive()

        if result:
            current_lang = agent.config.get("ui", {}).get("language", "en")
            print(f"✅ Language set to: {current_lang}")

            # Test language-specific prompts
            prompts = agent.get_language_specific_prompts(current_lang)
            print(f"✅ Retrieved prompts for language: {current_lang}")
            print(f"   Available prompt keys: {list(prompts.keys())}")

            # Test cultural context
            cultural_context = agent.get_cultural_context(current_lang)
            if cultural_context:
                print(f"✅ Cultural context available for {current_lang}")
                print(f"   Context: {cultural_context[:100]}...")
            else:
                print(f"ℹ️  No specific cultural context for {current_lang}")

            print("\n🎉 Language functionality test completed successfully!")
            return True
        else:
            print("❌ Language selection failed")
            return False

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False

if __name__ == "__main__":
    success = test_language_functionality()
    sys.exit(0 if success else 1)