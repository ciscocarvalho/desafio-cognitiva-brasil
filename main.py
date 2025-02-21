import requests
import json
from pydantic import ValidationError
from challenge.settings import Settings
import language_tool_python


class Challenge():
    def __init__(
            self,
            openrouter_key,
            hugginfacekey,
            question_prompt,
            analysis_prompt,
            language
    ):
        self._openrouter_key = openrouter_key
        self._hugginfacekey = hugginfacekey
        self.question_prompt = question_prompt
        self.analysis_prompt = analysis_prompt
        self.language = language

    @property
    def language(self):
        return self._language

    @language.setter
    def language(self, language):
        self._language = language
        self._language_tool = language_tool_python.LanguageTool(self._language)

    def use(self, model, prompt):
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self._openrouter_key}",
            },
            data=json.dumps({
                "model": model,
                "messages": [{"role": "user", "content": prompt}]
            })
        )

        return response.json()

    def list_models(self):
        res = requests.get("https://openrouter.ai/api/v1/models")

        if res.status_code == 200:
            return res.text

    def analyze_grammar(self, text):
        """Retorna a quantidade de erros gramaticais encontrados."""
        matches = self._language_tool.check(text)
        return matches

    def analyze_coherence(self, text):
        """Usa um modelo de similaridade para verificar coerência entre partes do texto."""
        model = "sentence-transformers/all-MiniLM-L6-v2"
        api_url = f"https://api-inference.huggingface.co/models/{model}"
        headers = {"Authorization": f"Bearer {self._hugginfacekey}"}

        first_half = text[: len(text) // 2]
        second_half = text[len(text) // 2:]
        coherence_pairs = [(first_half, second_half)]
        similarities = []

        for pair in coherence_pairs:
            payload = {"inputs": {"source_sentence": pair[0], "sentences": [pair[1]]}}
            response = requests.post(api_url, headers=headers, json=payload)

            if response.status_code == 200:
                similarity = response.json()[0]
                similarities.append(similarity)

        return sum(similarities) / len(similarities) if similarities else 0.0

    def analyze_fact_checking(self, text):
        """Verifica se a resposta contém informações falsas."""
        model = "facebook/bart-large-mnli"
        url = f"https://api-inference.huggingface.co/models/{model}"
        headers = {"Authorization": f"Bearer {self._hugginfacekey}"}

        payload = {
            "inputs": text,
            "parameters": {"candidate_labels": ["true", "false", "uncertain"]}
        }

        response = requests.post(url, headers=headers, json=payload)

        if response.status_code == 200:
            result = response.json()
            return result["labels"][0]  # Pega o rótulo mais provável

        return "uncertain"


def main():
    try:
        settings = Settings()
    except ValidationError as e:
        print(f"Environment validation error: {e}")
        return

    challenge = Challenge(
        openrouter_key=settings.OPENROUTER_API_KEY,
        hugginfacekey=settings.HUGGINFACE_API_KEY,
        # question_prompt="How many r's are there in strawberry?",
        # analysis_prompt="",
        # language="pt-BR"
        # question_prompt="How many r's are there in strawberry?",
        # question_prompt="Tell me one famous sentence from Aristotle",

        question_prompt="Did Aristotle say 'The only true wisdom is in knowing you know nothing'?",
        analysis_prompt="",
        language="en-US"
    )

    models = [
        # "openai/gpt-3.5-turbo",
        # "google/gemini-2.0-flash-001",
        "deepseek/deepseek-r1-distill-qwen-1.5b",
        # "anthropic/claude-3.5-sonnet",
        # "x-ai/grok-2-1212",
    ]

    responses = []

    for model in models:
        try:
            model_result = challenge.use(model, challenge.question_prompt)
            model_text = model_result["choices"][0]["message"]["content"]

            responses.append(model_text)
        except Exception as e:
            print(f"Error getting response from {model}: {e}")

    for response in responses:
        print("-" * 80)
        print("### Response ###")
        print(response)
        # print("### Grammar Analysis ###")
        # print(len(challenge.analyze_grammar(response)))
        # print("### Coherence Analysis ###")
        # print(challenge.analyze_coherence(response))
        print("### Fact Checking Analysis ###")
        print(challenge.analyze_fact_checking(response))
        print("-" * 80)


if __name__ == "__main__":
    main()
