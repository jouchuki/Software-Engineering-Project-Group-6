import ctranslate2
import sentencepiece as spm


def trs_article(input_text):
    # Set up the path to the translation model
    translator = ctranslate2.Translator("C:/Users/vsoko/Downloads/ct_model")
    sp_source = spm.SentencePieceProcessor("C:/Users/vsoko/Downloads/source.model")
    sp_target = spm.SentencePieceProcessor("C:/Users/vsoko/Downloads/target.model")

    input_tokens = sp_source.encode(input_text, out_type=str)

    results = translator.translate_batch([input_tokens])

    output_tokens = results[0].hypotheses[0]
    output_text = sp_target.decode(output_tokens)

    return output_text


def ask_for_translation(graph, selected_idx):
    answer = str(input("Do you want to translate the article summary to Dutch?(y/n)"))
    if answer == "y":
        print(f"Translated Title: {trs_article(graph.metadata[selected_idx]['title'])}")
        print(f"Translated Summary: {trs_article(graph.metadata[selected_idx]['summary'])}")
    else:
        return 0
