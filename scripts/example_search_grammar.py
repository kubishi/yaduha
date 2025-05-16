from yaduha.chatbot.tools.grammar import search_grammar


def main():
    responses = search_grammar("pronouns")

    print(responses)


if __name__ == "__main__":
    main()