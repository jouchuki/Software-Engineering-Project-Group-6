from fastapi import FastAPI, HTTPException
import ctranslate2
import sentencepiece as spm

app = FastAPI()


def trs_article(input_text):
    translator = ctranslate2.Translator("/home/ubuntu/ct_model")
    sp_source = spm.SentencePieceProcessor("/home/ubuntu/source.model")
    sp_target = spm.SentencePieceProcessor("/home/ubuntu/target.model")

    input_tokens = sp_source.encode(input_text, out_type=str)
    results = translator.translate_batch([input_tokens])
    output_tokens = results[0].hypotheses[0]
    output_text = sp_target.decode(output_tokens)

    return output_text


@app.post("/translate/")
async def translate(input_text: str):
    try:
        return {"translated_text": trs_article(input_text)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
