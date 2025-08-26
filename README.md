# AI Short Story Writer Agent

An automated AI-powered agent that creates compelling short stories through a structured 7-stage workflow using Google Gemini AI. Now featuring multi-language support for global creative writing.

## 🚀 Features

- **Automated Workflow**: Complete 7-stage story development process
- **AI-Powered**: Uses Google Gemini for intelligent content generation
- **Interactive**: User-guided concept selection and character development
- **Structured Output**: Organized story components and project management
- **Configurable**: Customizable settings for different writing styles
- **Progress Saving**: Automatic project state persistence
- **🌐 Multi-Language Support**: Generate stories in 13 languages with cultural adaptation
- **🎭 Cultural Context Awareness**: Language-specific prompts and culturally appropriate content
- **🧪 Demo & Testing Tools**: Included demonstration and testing scripts

## 📋 Workflow Stages

1. **Concept Generation & Ideation**
   - Generate 5 unique story concepts
   - User selects and refines chosen concept
   - Develop protagonists, settings, and thematic direction

2. **Story Structure Planning**
   - Create three-act narrative framework
   - Define key plot points and character arcs
   - Establish scene-by-scene outline

3. **Character Development Deep-Dive**
   - Rich character psychology and background
   - Personality traits and motivations
   - Dialogue voice development

4. **Setting & Atmosphere Creation**
   - Immersive world-building
   - Sensory details and mood establishment
   - Environmental storytelling elements

5. **First Draft Generation**
   - Complete story draft (2000-5000 words)
   - Scene-by-scene writing with AI assistance
   - Natural dialogue and compelling narrative

6. **Revision and Enhancement**
   - Pacing and tension analysis
   - Character consistency checks
   - Plot coherence improvements

7. **Final Polish and Proofing**
   - Grammar and style corrections
   - Final quality assessment
   - Publication-ready manuscript

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Google Gemini API key

### Setup

1. **Clone or download the project files**

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment (choose one method):**

   **Method A: Using .env file (Recommended)**
   ```bash
   # Edit the .env file with your API key
   GEMINI_API_KEY="your_api_key_here"
   ```

   **Method B: Environment variable**
   ```bash
   export GEMINI_API_KEY="your_api_key_here"
   ```

   **Method C: Interactive input**
   - Run the application and enter your API key when prompted

4. **Get Google Gemini API key:**
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Add it to your `.env` file or set as environment variable

## 🎯 Usage

### Basic Usage

```bash
python story_writer_agent.py
```

### Language Features

1. **Demo language functionality:**
   ```bash
   python demo_language_story.py
   ```

2. **Test language features:**
   ```bash
   python test_language.py
   ```

### Advanced Usage

1. **Run with custom config:**
   ```bash
   python story_writer_agent.py --config custom_config.json
   ```

2. **Load existing project:**
   ```bash
   python story_writer_agent.py --load story_project_20241201_120000.json
   ```

3. **Run specific stages:**
   ```bash
   python story_writer_agent.py --stages 1,3,5
   ```

## 📁 Project Structure

```
├── story_writer_agent.py     # Main agent application
├── demo_language_story.py    # Language functionality demonstration
├── test_language.py          # Language feature testing
├── config.json              # Configuration settings
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables (API keys)
├── README.md                # This documentation
└── *.json                   # Saved project files
```

## ⚙️ Configuration

Edit `config.json` to customize:

- **API Settings**: Model, tokens, temperature, timeout
- **Story Parameters**: Word count targets, auto-save frequency
- **UI Preferences**: Progress display, interaction mode, colors, logging
- **🌐 Language Settings**: Default language, supported languages, cultural adaptation
- **Workflow Control**: Stage skipping, user approval requirements

### Language Configuration

The system supports 13 languages with cultural context awareness:

```json
{
  "language": {
    "default": "en",
    "supported": ["en", "zh", "zh-tw", "es", "fr", "de", "ja", "ko", "it", "pt", "ru", "ar", "hi"],
    "auto_detect": false,
    "preserve_original_names": true,
    "cultural_adaptation": true
  }
}
```

**Supported Languages:**
- English (en) - Default
- Chinese Simplified (zh), Traditional (zh-tw)
- Spanish (es), French (fr), German (de)
- Japanese (ja), Korean (ko)
- Italian (it), Portuguese (pt), Russian (ru)
- Arabic (ar), Hindi (hi)

## 🌐 Multi-Language Features

### Language Selection
- **Interactive Menu**: Choose your preferred language at startup
- **Persistent Settings**: Language preference saved in configuration
- **Dynamic Switching**: Change language during story development

### Cultural Adaptation
- **Context-Aware Prompts**: Language-specific prompt templates
- **Cultural References**: Appropriate themes and settings for each culture
- **Localized Content**: Culturally relevant story elements and character names

### Demo and Testing
- **Language Demo**: `python demo_language_story.py` - See all supported features
- **Functionality Test**: `python test_language.py` - Test language selection and prompts

### Benefits
- **Global Accessibility**: Create stories in users' native languages
- **Cultural Authenticity**: Stories with appropriate cultural context
- **Enhanced Creativity**: Language-specific creative approaches
- **Professional Quality**: Native language storytelling experience

## 💡 Tips for Best Results

### AI Collaboration Best Practices
- **Be Specific**: Provide clear, detailed prompts
- **Iterate**: Refine AI output through multiple passes
- **Maintain Control**: Add your unique voice and perspective
- **Edit Actively**: Don't accept AI output blindly

### Story Development Tips
- **Start Simple**: Let the AI generate broad concepts first
- **Layer Details**: Build complexity gradually through stages
- **Trust Your Instincts**: Override AI suggestions when they don't feel right
- **Read Aloud**: Check flow and natural dialogue

## 🔧 Troubleshooting

### Common Issues

1. **API Key Errors**
   - Verify your Gemini API key is correct
   - Check your API quota limits
   - Ensure the key has proper permissions

2. **Content Generation Issues**
   - Check your internet connection
   - Try reducing the token limit in config
   - Verify the model name is correct

3. **File Permission Errors**
   - Ensure write permissions in the working directory
   - Check if project files are corrupted

### Getting Help

If you encounter issues:
1. Check the error messages for specific details
2. Verify your configuration settings
3. Try running with verbose logging enabled
4. Check your API key validity and quota

## 📊 Example Workflow Output

```
🚀 Starting AI Short Story Writing Workflow

📝 Stage 1: Concept Generation & Ideation
Generating story concepts...

=== STORY CONCEPT OPTIONS ===

1. Shadows of Tomorrow
   Genre: Science Fiction
   Premise: A scientist discovers a way to communicate with parallel universes
   Conflict: The discovery threatens to unravel reality itself

2. The Forgotten Melody
   Genre: Literary Fiction
   Premise: A retired musician hears a song that unlocks repressed memories
   Conflict: The memories reveal a long-buried family secret

[... more options ...]

Select a concept to develop (1-5): 2

✅ Selected concept: The Forgotten Melody
```

## 🤝 Contributing

This project demonstrates AI-assisted creative writing. To contribute:

1. Fork the repository
2. Create a feature branch
3. Make your improvements
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source. Feel free to use, modify, and distribute.

## 🔗 Resources

- [Google Gemini AI](https://ai.google.dev/)
- [AI Writing Best Practices](https://www.writersdigest.com/write-better-fiction/ai-writing-tools)
- [Short Story Structure Guide](https://www.masterclass.com/articles/how-to-write-a-short-story)
- [International Writing Techniques](https://www.writersdigest.com/international-writing)
- [Cultural Context in Storytelling](https://www.masterclass.com/articles/cultural-context-in-storytelling)

---

**Happy Writing!** 🎨✍️

The AI agent handles the heavy lifting, but your creativity and judgment remain essential for crafting truly compelling stories.