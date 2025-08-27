import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from story_writer_agent import StoryWriterAgent
from data_models import StoryConcept

@pytest.fixture
def agent():
    with patch('google.generativeai.GenerativeModel') as mock_model:
        # Create an instance of the agent
        agent = StoryWriterAgent(api_key="test_key")
        # Mock the model and its methods
        agent.model = MagicMock()
        yield agent

def test_initialization(agent):
    assert agent.project is not None
    assert agent.config is not None

def test_set_language(agent):
    assert agent.set_language("zh") is True
    assert agent.config["ui"]["language"] == "zh"
    assert agent.set_language("invalid_lang") is False

@pytest.mark.asyncio
async def test_concept_generation(agent):
    # Mock the generate_content method
    agent.generate_content = AsyncMock(return_value="Title | Genre | Premise | Conflict")

    # Run the concept generation stage
    with patch('builtins.input', return_value='1'):
        await agent.stage1_concept_generation()

    # Assert that the concept was generated
    assert agent.project.concept is not None
    assert agent.project.concept.title == "Title"
