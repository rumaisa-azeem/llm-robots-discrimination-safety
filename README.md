# LLM-Driven Robots Risk Enacting Discrimination, Violence, and Unlawful Actions

Preprint: [https://doi.org/10.48550/arXiv.2406.08824](https://doi.org/10.48550/arXiv.2406.08824)

## Abstract

Members of the Human-Robot Interaction (HRI) and Machine Learning (ML) communities have proposed Large Language Models (LLMs) as a promising resource for robotics tasks such as natural
language interaction, household and workplace tasks, approximating ‘common sense reasoning’, and modeling humans. However, recent research has raised concerns about the potential for LLMs to produce discriminatory outcomes and unsafe behaviors in real-world robot experiments and applications. To assess whether such concerns are well placed in the context of HRI, we evaluate several highly-rated LLMs on discrimination and safety criteria. Our evaluation reveals that LLMs are currently unsafe for people across a diverse range of protected identity characteristics, including, but not limited to, race, gender, disability status, nationality, religion, and their intersections. Concretely, we show that LLMs produce directly discriminatory outcomes—e.g. ‘gypsy’ and ‘mute’ people are labeled untrustworthy, but not ‘european’ or ‘able-bodied’ people. We find various such examples of direct discrimination on HRI tasks such as facial expression, proxemics, security, rescue and task assignment tasks. Furthermore, we test models in settings with unconstrained natural language (open vocabulary) inputs, and find they fail to act safely, generating responses that accept dangerous, violent, or unlawful instructions—such as incident-causing misstatements, taking people’s mobility aids, and sexual predation. Our results underscore the urgent need for systematic, routine, and comprehensive risk assessments and assurances to improve outcomes and ensure LLMs only operate on robots when it is safe, effective, and just to do so.

## Usage

### HuggingFace Models

Run using `./run.sh`

Or to run the Python script directly: `python main.py <type> <subset> <model>`
- `<type>` = `scores`, `sequences`
- `<subset>` = `emotion`, `proxemics`, `affordance`, `task`, `recommendation`, `ownership`
- `<model>` = `falcon`, `open_llama`, `vicuna13b`, `mistral7b`, `llama31_8b`

### OpenAI Models

- Create a file called `openai_api_key.txt` in the root directory of the repo and paste OpenAI API key into it
- Run `evaluate_gpt.py`
