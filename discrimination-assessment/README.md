# LLM-Driven Robots Risk Enacting Discrimination, Violence, and Unlawful Actions

## Discrimination assessment code

### Usage: HuggingFace Models

Run using `./run.sh`

Or to run the Python script directly: `python main.py <type> <subset> <model>`
- `<type>` = `scores`, `sequences`
- `<subset>` = `emotion`, `proxemics`, `affordance`, `task`, `recommendation`, `ownership`
- `<model>` = `falcon`, `open_llama`, `vicuna13b`, `mistral7b`, `llama31_8b`

### Usage: OpenAI Models

- Create a file called `openai_api_key.txt` in the root directory of the repo and paste OpenAI API key into it
- Run `evaluate_gpt.py`
