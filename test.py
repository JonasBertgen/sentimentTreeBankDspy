import dspy


def main():
    simpleModelName = "llama3.2"
    simpleLLM = dspy.LM(f"ollama_chat/{simpleModelName}",
                        api_base="http://localhost:11434", api_key="")
    dspy.configure(lm=simpleLLM)
    qa = dspy.Predict("question -> response")
    print(qa(question="how are you today?"))


if __name__ == "__main__":
    main()
