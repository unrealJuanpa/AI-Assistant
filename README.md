# AI-Assistant

AI-Assistant is a project that consists of a multifunctional AI assistant. This assistant listens to what is said through the microphone and responds accordingly using text-to-speech technology.

## Features

- **Speech recognition**: Uses OpenAI Whisper to convert speech to text.
- **Natural language response**: Implements the llama 3.1 model to generate responses based on the received text.
- **Text-to-speech synthesis**: Uses Silero models to convert text to speech.
- **Laanguage**: Currently, the project is designed for the Spanish language, but it can be easily adapted to other languages. In future versions, the assistant will automatically switch languages.

## Requirements

- **Graphics card**: An RTX 3060 mobile 6GB graphics card is recommended for optimal performance and real-time conversation capability.
- **Dependencies**: Make sure to have the following libraries installed to run the project:

  - Python 3.9.19
  - cuDNN 8.1.0.77
  - CUDA Toolkit 11.2.2
  - Torch 2.1.0+cu121
  - Torchaudio 2.1.0+cu121
  - OmegaConf 2.3.0
  - OpenAI Whisper 20231117
  - Ollama 0.3.1

## Installation

1. Clone this repository to your local machine:
    ```bash
    git clone https://github.com/your_username/AI-Assistant.git
    cd AI-Assistant
    ```

2. Install the required dependencies (I'm using conda)

3. For convenience, Silero files are included directly in the repository. This ensures the project works correctly without configuration issues.

## Usage

1. Run the AI assistant:
    ```bash
    python main.py
    ```

2. Speak into the microphone while pressing '.' key on your keyoard, and voilà, the assistant will respond using text-to-speech.

## Testing and Performance

In tests conducted, great response speed was achieved using an RTX 3060 mobile 6GB graphics card, allowing real-time conversation.

## Contributions

Contributions are welcome. Please open an issue or submit a pull request to discuss any changes you would like to make.
This is a minimum viable product, so the coding style may not fully adhere to standard conventions. 

## License

This project is licensed under the [MIT License](LICENSE).
