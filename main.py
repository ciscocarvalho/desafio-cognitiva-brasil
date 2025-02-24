from pydantic import ValidationError
from challenge.settings import Settings
from challenge.challenge import Challenge

def main():
    try:
        settings = Settings()
    except ValidationError as e:
        print(f"Environment validation error: {e}")
        return

    models = settings.MODELS.split(",")
    models_responses = {}

    challenge = Challenge(
        openrouter_key=settings.OPENROUTER_API_KEY,
        hugginfacekey=settings.HUGGINFACE_API_KEY,
        question_prompt=settings.QUESTION_PROMPT,
        ranking_prompt=settings.RANKING_PROMPT,
        language=settings.PROMPTS_LANGUAGE
    )

    analysis_results = {
        "grammar": {model: {} for model in models},
        "fact_checking": {model: {} for model in models},
        "coherence": {model: {} for model in models}
    }

    for model in models:
        try:
            result = challenge.use(model, challenge.question_prompt)
            text = result["choices"][0]["message"]["content"]
            models_responses[model] = text

            print(f"Analyzing grammar from model {model}")
            analysis_results["grammar"][model] = challenge.analyze_grammar(text)

            print(f"Analyzing fact checks from model {model}")
            analysis_results["fact_checking"][model] = challenge.analyze_fact_checking(text)

            print(f"Analyzing coherence from model {model}")
            analysis_results["coherence"][model] = challenge.analyze_coherence(text)

        except Exception as e:
            print(f"Error getting response from {model}: {e}")
    
    print("Plotting grammar analysis")
    challenge.plot_grammar_analysis(analysis_results["grammar"])

    print("Plotting coherence analysis")
    challenge.plot_coherence_analysis(analysis_results["coherence"])

    print("Plotting fact checking analysis")
    challenge.plot_fact_checking_analysis(analysis_results["fact_checking"])

    print("Ranking responses")
    challenge.rank_responses(models_responses)


if __name__ == "__main__":
    main()