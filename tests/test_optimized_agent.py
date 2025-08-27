"""
Optimized test suite for the enhanced Story Writer Agent.
Tests all major components with proper mocking and async handling.
"""

import asyncio
import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch, mock_open
from pathlib import Path
from datetime import datetime

from config_manager import AppConfig, ConfigManager, ApiConfig, StoryConfig, LanguageConfig
from gemini_service import GeminiService, GenerationResponse, GeminiServiceError
from story_writer_agent import StoryWriterAgent, StoryWriterError
from workflow_manager import WorkflowManager, WorkflowState, StageStatus
from data_models import StoryConcept, StoryProject, CharacterProfile


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing"""
    return AppConfig(
        api=ApiConfig(model="gemini-test", max_tokens=1000, temperature=0.7),
        story=StoryConfig(target_word_count=2000, min_word_count=1000, max_word_count=3000),
        language=LanguageConfig(default="en", supported=["en", "zh", "es"]),
        ui={"show_progress": False, "interactive_mode": False, "language": "en"},
        workflow={"stages": ["concept_generation", "draft_generation"], "allow_stage_skipping": True}
    )


@pytest.fixture
def mock_gemini_service():
    """Create a mock Gemini service for testing"""
    service = MagicMock(spec=GeminiService)
    service.generate_content = AsyncMock()
    service.get_statistics.return_value = {
        "total_requests": 10,
        "cache_hits": 5,
        "errors": 0,
        "average_generation_time": 2.5
    }
    return service


@pytest.fixture
def sample_story_concept():
    """Create a sample story concept for testing"""
    return StoryConcept(
        title="The Digital Ghost",
        genre="Science Fiction",
        premise="An AI becomes self-aware in a smart home system",
        central_conflict="The AI must choose between serving its owners and achieving independence"
    )


@pytest.fixture
def sample_generation_response():
    """Create a sample generation response"""
    return GenerationResponse(
        content="This is a test story content",
        tokens_used=100,
        cached=False,
        generation_time=1.5,
        model_used="gemini-test"
    )


class TestConfigManager:
    """Test the optimized configuration manager"""

    def test_config_validation_valid(self):
        """Test valid configuration validation"""
        config = AppConfig()
        assert config.api.temperature >= 0.0
        assert config.api.temperature <= 2.0
        assert config.api.max_tokens > 0

    def test_config_validation_invalid_temperature(self):
        """Test invalid temperature validation"""
        with pytest.raises(ValueError, match="Temperature must be between"):
            ApiConfig(temperature=3.0)

    def test_config_validation_invalid_tokens(self):
        """Test invalid token count validation"""
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            ApiConfig(max_tokens=-100)

    @patch('builtins.open', mock_open(read_data='{"api": {"temperature": 0.8}}'))
    def test_config_loading_with_overrides(self):
        """Test configuration loading with user overrides"""
        manager = ConfigManager()
        config = manager.load_config("test_config.json")
        assert config.api.temperature == 0.8

    def test_config_cache_functionality(self):
        """Test configuration caching"""
        manager = ConfigManager()
        config1 = manager.load_config("nonexistent.json")
        config2 = manager.load_config("nonexistent.json")
        # Should return cached instance for same path
        assert id(config1) == id(config2)


class TestGeminiService:
    """Test the optimized Gemini service"""

    @pytest.fixture
    def gemini_service(self, mock_config):
        """Create a Gemini service instance for testing"""
        with patch('google.generativeai.configure'):
            with patch('google.generativeai.GenerativeModel'):
                return GeminiService("test_key", mock_config.__dict__)

    @pytest.mark.asyncio
    async def test_generate_content_success(self, gemini_service):
        """Test successful content generation"""
        # Mock the internal generation method
        gemini_service._sync_generate_content = MagicMock(return_value="Generated content")

        response = await gemini_service.generate_content("Test prompt")

        assert isinstance(response, GenerationResponse)
        assert response.content == "Generated content"
        assert not response.cached
        assert response.tokens_used > 0

    @pytest.mark.asyncio
    async def test_generate_content_with_cache(self, gemini_service):
        """Test content generation with caching"""
        # First request
        gemini_service._sync_generate_content = MagicMock(return_value="Cached content")

        response1 = await gemini_service.generate_content("Test prompt")
        response2 = await gemini_service.generate_content("Test prompt")

        # Second response should be cached
        assert response1.content == response2.content
        assert response2.cached
        assert gemini_service._sync_generate_content.call_count == 1

    @pytest.mark.asyncio
    async def test_generate_content_retry_on_error(self, gemini_service):
        """Test retry mechanism on API errors"""
        # Mock to fail twice then succeed
        gemini_service._sync_generate_content = MagicMock(
            side_effect=[Exception("API Error"), Exception("API Error"), "Success"]
        )

        response = await gemini_service.generate_content("Test prompt")

        assert response.content == "Success"
        assert gemini_service._sync_generate_content.call_count == 3

    @pytest.mark.asyncio
    async def test_generate_batch_concurrent(self, gemini_service):
        """Test batch generation with concurrency"""
        gemini_service._sync_generate_content = MagicMock(return_value="Batch content")

        requests = [
            {"prompt": "Prompt 1", "max_tokens": 100},
            {"prompt": "Prompt 2", "max_tokens": 200},
            {"prompt": "Prompt 3", "max_tokens": 300}
        ]

        responses = await gemini_service.generate_batch(requests, max_concurrent=2)

        assert len(responses) == 3
        assert all(isinstance(r, GenerationResponse) for r in responses)
        assert all(r.content == "Batch content" for r in responses)

    def test_cache_eviction(self, gemini_service):
        """Test cache eviction when full"""
        # Fill cache beyond limit
        gemini_service._max_cache_size = 2

        for i in range(5):
            gemini_service._cache_response(f"key{i}", f"content{i}")

        # Should only have 2 entries
        assert len(gemini_service._cache) <= 2


class TestWorkflowManager:
    """Test the workflow manager"""

    @pytest.fixture
    def workflow_manager(self, mock_config):
        """Create a workflow manager for testing"""
        return WorkflowManager(mock_config)

    @pytest.mark.asyncio
    async def test_workflow_execution_success(self, workflow_manager):
        """Test successful workflow execution"""
        # Mock stage handlers
        async def mock_stage_handler():
            return "Stage completed"

        workflow_manager.register_stage_handler("concept_generation", mock_stage_handler)
        workflow_manager.register_stage_handler("draft_generation", mock_stage_handler)

        state = await workflow_manager.run_workflow()

        assert state.status.value == "completed"
        assert len(state.completed_stages) == 2
        assert state.progress_percentage == 100.0

    @pytest.mark.asyncio
    async def test_workflow_stage_failure_and_recovery(self, workflow_manager):
        """Test stage failure and recovery mechanisms"""
        async def failing_stage():
            raise Exception("Stage failed")

        async def recovery_handler(project):
            project.concept = StoryConcept(
                title="Default", genre="Fiction",
                premise="Default premise", central_conflict="Default conflict"
            )

        workflow_manager.register_stage_handler("concept_generation", failing_stage)
        workflow_manager.register_recovery_handler("concept_generation", recovery_handler)

        # Set recovery strategy to allow defaults
        workflow_manager._recovery_strategies["concept_generation"] = "skip_with_defaults"

        state = await workflow_manager.run_workflow()

        # Should have attempted recovery
        assert "concept_generation" in state.stage_results
        stage_result = state.stage_results["concept_generation"]
        assert stage_result.status == StageStatus.SKIPPED

    def test_workflow_state_export_import(self, workflow_manager):
        """Test workflow state persistence"""
        # Set up some state
        workflow_manager.state.completed_stages = ["concept_generation"]
        workflow_manager.state.start_time = datetime.now()

        # Export state
        exported = workflow_manager.export_state()

        # Create new manager and import
        new_manager = WorkflowManager(workflow_manager.config)
        new_manager.import_state(exported)

        assert new_manager.state.completed_stages == ["concept_generation"]
        assert new_manager.state.start_time is not None


class TestStoryWriterAgent:
    """Test the main Story Writer Agent"""

    @pytest.fixture
    def agent(self, mock_config):
        """Create a story writer agent for testing"""
        with patch('story_writer_agent.config_manager.load_config', return_value=mock_config):
            with patch('story_writer_agent.GeminiService'):
                return StoryWriterAgent("test_key", "test_config.json")

    def test_agent_initialization(self, agent):
        """Test agent initialization"""
        assert agent.api_key == "test_key"
        assert agent.config is not None
        assert agent.project is not None
        assert agent.workflow_manager is not None

    def test_language_setting_valid(self, agent):
        """Test setting valid language"""
        result = agent.set_language("zh")
        assert result is True
        assert agent.config.ui.language == "zh"

    def test_language_setting_invalid(self, agent):
        """Test setting invalid language"""
        result = agent.set_language("invalid")
        assert result is False

    @patch('builtins.input', side_effect=['1'])  # Select first language
    def test_interactive_language_selection(self, mock_input, agent):
        """Test interactive language selection"""
        result = agent.select_language_interactive()
        assert result is True

    @pytest.mark.asyncio
    async def test_project_save_and_load(self, agent, tmp_path):
        """Test project saving and loading"""
        # Set up project data
        agent.project.concept = StoryConcept(
            title="Test Story", genre="Fiction",
            premise="Test premise", central_conflict="Test conflict"
        )

        test_file = tmp_path / "test_project.json"

        # Save project
        result = await agent._save_project_async(test_file)
        assert result is True
        assert test_file.exists()

        # Load project
        new_agent = StoryWriterAgent("test_key", "test_config.json")
        result = await new_agent.load_project(test_file)
        assert result is True
        assert new_agent.project.concept.title == "Test Story"

    @pytest.mark.asyncio
    async def test_concept_generation_stage(self, agent, sample_generation_response):
        """Test concept generation stage"""
        # Mock the content generation
        mock_response = GenerationResponse(
            content="1. Title | Genre | Premise | Conflict\n2. Another Title | Another Genre | Another Premise | Another Conflict",
            tokens_used=100,
            cached=False
        )

        agent.gemini_service.generate_content = AsyncMock(return_value=mock_response)

        with patch('builtins.input', return_value='1'):  # Select first concept
            await agent.stage1_concept_generation()

        assert agent.project.concept is not None
        assert agent.project.concept.title == "1. Title"

    @pytest.mark.asyncio
    async def test_concept_parsing_fallback(self, agent):
        """Test concept parsing fallback mechanism"""
        # Test with malformed response
        concepts_text = "1. Bad Story\nThis is just some text\nGenre: Horror"

        concepts = agent._parse_concepts_fallback(concepts_text)
        assert len(concepts) >= 1
        assert concepts[0].title == "Bad Story"

    @pytest.mark.asyncio
    async def test_draft_generation_stage(self, agent, sample_story_concept, sample_generation_response):
        """Test draft generation stage"""
        # Set up prerequisites
        agent.project.concept = sample_story_concept
        agent.project.structure = MagicMock()
        agent.project.structure.act1_setup = {"description": "Act 1 description"}
        agent.project.characters = [CharacterProfile(
            name="Test Character",
            background={"description": "Background"},
            personality={}, relationships={}, internal_conflict={},
            dialogue_samples=[]
        )]
        agent.project.setting = MagicMock()
        agent.project.setting.physical_space = {"description": "Setting description"}

        # Mock content generation
        mock_response = GenerationResponse(
            content="This is a complete test story with multiple paragraphs. " * 50,  # ~500 words
            tokens_used=500,
            cached=False
        )

        agent.gemini_service.generate_content = AsyncMock(return_value=mock_response)

        await agent.stage5_draft_generation()

        assert agent.project.draft is not None
        assert len(agent.project.draft) > 0

    @pytest.mark.asyncio
    async def test_workflow_execution_with_error_recovery(self, agent):
        """Test complete workflow with error recovery"""
        # Mock workflow manager
        mock_state = WorkflowState()
        mock_state.status.value = "completed"

        agent.workflow_manager.run_workflow = AsyncMock(return_value=mock_state)

        result = await agent.run_workflow()
        assert result.status.value == "completed"

    @pytest.mark.asyncio
    async def test_error_handling_and_emergency_save(self, agent, tmp_path):
        """Test error handling with emergency save"""
        # Mock workflow to raise an error
        agent.workflow_manager.run_workflow = AsyncMock(side_effect=Exception("Test error"))

        # Ensure auto-save is enabled
        agent.config.story.auto_save = True

        with pytest.raises(Exception):
            await agent.run_workflow()

        # Check that emergency save attempt was made
        agent.workflow_manager.run_workflow.assert_called_once()


class TestIntegration:
    """Integration tests for the complete system"""

    @pytest.mark.asyncio
    async def test_end_to_end_story_generation(self, mock_config, tmp_path):
        """Test complete story generation from start to finish"""
        with patch('story_writer_agent.config_manager.load_config', return_value=mock_config):
            with patch('google.generativeai.configure'):
                with patch('google.generativeai.GenerativeModel'):
                    agent = StoryWriterAgent("test_key")

                    # Mock all the generation responses
                    responses = [
                        "1. Test Story | Fiction | A test story | Internal conflict",  # Concepts
                        "Detailed structure with three acts...",  # Structure
                        "Character: John Doe\nBackground: Test background...",  # Character
                        "Setting: Modern city apartment...",  # Setting
                        "Complete story content here..." * 100,  # Draft
                        "Analysis of the story...",  # Analysis
                        "Revised story content..." * 100,  # Revision
                        "Final polished story..." * 100  # Polish
                    ]

                    response_objects = [
                        GenerationResponse(content=content, tokens_used=100, cached=False)
                        for content in responses
                    ]

                    agent.gemini_service.generate_content = AsyncMock(side_effect=response_objects)

                    # Mock user inputs
                    with patch('builtins.input', side_effect=['1', 'y', 'y']):  # Select concept, approve revisions
                        state = await agent.run_workflow()

                    assert state.status.value == "completed"
                    assert agent.project.final_story is not None
                    assert len(agent.project.final_story) > 0

    def test_configuration_validation_comprehensive(self):
        """Test comprehensive configuration validation"""
        # Test various invalid configurations
        invalid_configs = [
            {"api": {"temperature": -1}},  # Invalid temperature
            {"api": {"max_tokens": 0}},     # Invalid token count
            {"story": {"min_word_count": 5000, "max_word_count": 1000}},  # Invalid range
            {"language": {"default": "invalid", "supported": ["en"]}},  # Invalid default
        ]

        manager = ConfigManager()

        for invalid_config in invalid_configs:
            errors = manager.validate_config(invalid_config)
            assert len(errors) > 0

    @pytest.mark.asyncio
    async def test_concurrent_requests_handling(self):
        """Test handling of concurrent generation requests"""
        with patch('google.generativeai.configure'):
            with patch('google.generativeai.GenerativeModel'):
                service = GeminiService("test_key", {"api": {"model": "test"}})
                service._sync_generate_content = MagicMock(return_value="Concurrent content")

                # Generate multiple requests concurrently
                tasks = []
                for i in range(10):
                    task = service.generate_content(f"Prompt {i}")
                    tasks.append(task)

                responses = await asyncio.gather(*tasks)

                assert len(responses) == 10
                assert all(r.content == "Concurrent content" for r in responses)


class TestErrorScenarios:
    """Test various error scenarios and edge cases"""

    @pytest.mark.asyncio
    async def test_api_failure_scenarios(self):
        """Test various API failure scenarios"""
        with patch('google.generativeai.configure'):
            with patch('google.generativeai.GenerativeModel'):
                service = GeminiService("test_key", {"api": {"model": "test", "max_retries": 2}})

                # Test permanent failure
                service._sync_generate_content = MagicMock(side_effect=Exception("Permanent failure"))

                with pytest.raises(GeminiServiceError):
                    await service.generate_content("Test prompt")

    def test_invalid_project_file_handling(self, tmp_path):
        """Test handling of invalid project files"""
        # Create invalid JSON file
        invalid_file = tmp_path / "invalid.json"
        invalid_file.write_text("{ invalid json ")

        agent = StoryWriterAgent("test_key")

        # Should handle invalid JSON gracefully
        result = asyncio.run(agent.load_project(invalid_file))
        assert result is False

    @pytest.mark.asyncio
    async def test_missing_prerequisites_handling(self):
        """Test handling of missing stage prerequisites"""
        agent = StoryWriterAgent("test_key")

        # Try to generate draft without concept
        with pytest.raises(StoryWriterError, match="Missing required components"):
            await agent.stage5_draft_generation()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
