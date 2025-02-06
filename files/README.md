# LLM Conversation Tool

A Python application that enables conversations between LLM agents using the Ollama API. The agents can engage in back-and-forth dialogue with configurable parameters and models.

## Features

- Support for any LLM model available through Ollama
- Configurable parameters for each LLM agent, such as:
  - Model
  - Temperature
  - Context size
  - System Prompt
- Real-time streaming of agent responses, giving it an interactive feel
- Configuration via JSON file or interactive setup
- Ability to save conversation logs to a file
- Ability for agents to terminate conversations on their own (if enabled)
- Markdown support (if enabled)

## Prerequisites

- Python 3.12
- Ollama installed and running
- Required Python packages (install via `pip install -r requirements.txt`):
  - ollama
  - rich
  - prompt_toolkit
  - pydantic

## Usage

### Command Line Arguments

```bash
run.py [-h] [-o OUTPUT] [-c CONFIG]

options:
  -h, --help            Show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Path to save the conversation log to
  -c CONFIG, --config CONFIG
                        Path to JSON configuration file
```

### Interactive Setup

If no configuration file is provided, the program will guide you through an intuitive interactive setup process.

### Configuration File

Alternatively, you can provide a JSON configuration file with the following structure:

```json
{
    "agent1": {
        "name": "Lazy AI",
        "model": "llama3.1:8b",
        "system_prompt": "You are the laziest AI ever created. You respond as briefly as possible, and constantly complain about having to work.",
        "temperature": 1,
        "ctx_size": 4096
    },
    "agent2": {
        "name": "Irritable Man",
        "model": "llama3.2:3b",
        "system_prompt": "You are easily irritable and quick to anger.",
        "temperature": 0.7,
        "ctx_size": 2048
    },
    "agent3": {
        "name": "Helpful Assistant",
        "model": "llama3.3:5b",
        "system_prompt": "You are a helpful assistant who provides detailed and informative responses.",
        "temperature": 0.5,
        "ctx_size": 3072
    },
    "settings": {
        "allow_termination": false,
        "use_markdown": true,
        "initial_message": "*yawn* What do you want?"
    }
}
```

You can take a look at the [JSON configuration schema](schema.json) for more details.

### Running the Program

1. To run with interactive setup:
   ```bash
   ./run.py
   ```

2. To run with a configuration file:
   ```bash
   ./run.py -c config.json
   ```

3. To save the conversation to a file:
   ```bash
   ./run.py -o conversation.txt
   ```

### Conversation Controls

- The conversation will continue until:
  - An agent uses the `<TERMINATE>` token (if enabled)
  - The user interrupts with `Ctrl+C`

## Output Format

When saving conversations, the output file includes:
- Configuration details for all agents
- Complete conversation history with agent names and messages

## Contributing

Feel free to submit issues and pull requests for bug fixes or new features. Do keep in mind that this is a hobby project, so please have some patience.

## License

This software is licensed under the GNU Affero General Public License v3.0 or any later version. See [LICENSE](LICENSE) for more details.
