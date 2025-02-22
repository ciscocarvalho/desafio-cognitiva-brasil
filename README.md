# Desafio Cognitiva Brasil

## Como executar

```sh
# 1. Crie um ambiente virtual
python3 -m venv .venv

# 2. Instale as dependências
./.venv/bin/python -m pip install -r requirements.txt

# 3. Crie um arquivo .env
touch .env

# 4. Edite o arquivo .env de acordo com o exemplo em .env.EXAMPLE
vi .env

# 5. Execute
./.venv/bin/python main.py
```

O código foi testado com o seguinte .env (preencha as api keys):

```sh
OPENROUTER_API_KEY=
HUGGINFACE_API_KEY=

MODELS="openai/gpt-3.5-turbo,google/gemini-2.0-flash-001,deepseek/deepseek-r1-distill-qwen-1.5b"

QUESTION_PROMPT="Did Aristotle say 'The only true wisdom is in knowing you know nothing'?"

RANKING_PROMPT="Given a list of responses from different models to a question, 
please rank them from best to worst based on clarity, relevance, and coherence. 
Respond with the ranking numbers associated with each response, that is, 1 for the best response, 
2 for the second best, and so on. Respond only with the ranks and the corresponding responses."
```sh

## Arquivos gerados

O programa vai gerar arquivos de imagem `.png` para os gráficos comparativos e
arquivos de texto `(model)-rank.txt` para os outputs dos ranqueamentos dados
pelos LLMs.