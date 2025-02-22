import sys
import requests
import json
import language_tool_python
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class Challenge():
    def __init__(
            self,
            openrouter_key,
            hugginfacekey,
            question_prompt,
            ranking_prompt,
            language
    ):
        self._openrouter_key = openrouter_key
        self._hugginfacekey = hugginfacekey
        self.question_prompt = question_prompt
        self.ranking_prompt = ranking_prompt
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
            return result["labels"][0]

        return "uncertain"

    def plot_grammar_analysis(self, results, figpath="model_comparison_grammar.png"):
        """Plots the numberr of grammar errors for each model based on analysis resuls."""
        errors_count = []

        for model, result in results.items():
            errors_count.append(len(result))

        df = pd.DataFrame({
            "Model": list(results.keys()),
            "Grammar Errors": errors_count
        })

        df.plot(kind="bar", x="Model", y="Grammar Errors", legend=False, color="skyblue")
        plt.title("Number of Grammar Errors per Model")
        plt.ylabel("Amount of Errors")
        plt.xlabel("Models")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(figpath)

    def plot_coherence_analysis(self, results, figpath="model_comparison_coherence.png"):
        """Plots the coherence scores for each model based on analysis results."""
        coherence_scores = []

        for model, result in results.items():
            coherence_scores.append(result)

        df = pd.DataFrame({
            "Model": list(results.keys()),
            "Coherence Score": coherence_scores
        })

        df.plot(kind="bar", x="Model", y="Coherence Score", legend=False, color="lightgreen")
        plt.title("Coherence Score per Model")
        plt.ylabel("Coherence Score")
        plt.xlabel("Models")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(figpath)

    def plot_fact_checking_analysis(self, results, figpath="model_comparison_fact_checking.png"):
        """Creates a table to display fact-checking results per model."""
        
        _, ax = plt.subplots(figsize=(6, len(results) * 0.5 + 1))
        ax.axis("tight")
        ax.axis("off")

        table_data = [[model, label] for model, label in results.items()]
        
        table = ax.table(
            cellText=table_data,
            colLabels=["Model", "Fact-Checking Result"],
            cellLoc="center",
            loc="center"
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.auto_set_column_width([0, 1])

        plt.savefig(figpath)
    
    def rank_responses(self, models_responses):
        """Ranks model responses from best to worst, associating each response with a ranking number."""

        # ranked_responses = []
        responses_text = "\n".join([f"{model}: {response}" for model, response in models_responses.items()])
        prompt = f"{self.ranking_prompt}\n\n{responses_text}"

        for model in models_responses.keys():
            response = self.use(model, prompt)

            if "choices" not in response:
                print(f"Ranking failed for model {model}", file=sys.stderr)
                continue

            ranking = response["choices"][0]["message"]["content"]
            filename = ''.join([char if char.isalpha() else '_' for char in model])
            filename = f"{filename}-rank.txt"

            with open(filename, "w") as file:
                file.write(ranking)
            
            # ranked_responses = ranking.strip().split("\n")
            # return ranked_responses

