#!/usr/bin/env python3
"""
Test script for the enhanced language functionality
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from story_writer_agent import StoryWriterAgent
from localization import get_language_specific_prompts, get_cultural_context

from unittest.mock import patch


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
        with patch('builtins.input', return_value='1'):
            result = agent.select_language_interactive()
        assert result

        current_lang = agent.config.get("ui", {}).get("language", "en")
        print(f"✅ Language set to: {current_lang}")

        prompts = get_language_specific_prompts(current_lang)
        print(f"✅ Retrieved prompts for language: {current_lang}")
        print(f"   Available prompt keys: {list(prompts.keys())}")

        # Test cultural context
        cultural_context = get_cultural_context(current_lang)
        if cultural_context:
            print(f"✅ Cultural context available for {current_lang}")
            print(f"   Context: {cultural_context[:100]}...")
        else:
            print(f"ℹ️  No specific cultural context for {current_lang}")

        print("\n🎉 Language functionality test completed successfully!")

    except Exception as e:
        print(f"❌ Error during testing: {e}")
        assert False, e
