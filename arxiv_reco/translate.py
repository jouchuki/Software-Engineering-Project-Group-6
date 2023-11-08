import ctranslate2
import sentencepiece as spm

translator = ctranslate2.Translator("C:/Users/vsoko/Downloads/ct_model")
sp_source = spm.SentencePieceProcessor("C:/Users/vsoko/Downloads/source.model")
sp_target = spm.SentencePieceProcessor("C:/Users/vsoko/Downloads/target.model")

input_text = "Hello world!"
input_tokens = sp_source.encode(input_text, out_type=str)

results = translator.translate_batch([input_tokens])

output_tokens = results[0].hypotheses[0]
output_text = sp_target.decode(output_tokens)

print(output_text)