import vllm_mbart
from transformers import MBart50TokenizerFast
from vllm import LLM, SamplingParams
from vllm.inputs import ExplicitEncoderDecoderPrompt, TokensPrompt

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

article_hi = "संयुक्त राष्ट्र के प्रमुख का कहना है कि सीरिया में कोई सैन्य समाधान नहीं है"
article_ar = "الأمين العام للأمم المتحدة يقول إنه لا يوجد حل عسكري في سوريا."

tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

tokenizer.src_lang = "hi_IN"
encoded_ar = tokenizer(article_ar)
tokens_prompt1 = TokensPrompt(
    prompt_token_ids=encoded_ar["input_ids"]
)

decoder_prompt1 = TokensPrompt(
    prompt_token_ids=[tokenizer.lang_code_to_id["fr_XX"]]
)

tokenizer.src_lang = "ar_AR"
encoded_ar = tokenizer(article_ar)
tokens_prompt2 = TokensPrompt(
    prompt_token_ids=encoded_ar["input_ids"]
)

decoder_prompt2 = TokensPrompt(
    prompt_token_ids=[tokenizer.lang_code_to_id["en_XX"]]
)

enc_dec_prompt1 = ExplicitEncoderDecoderPrompt(encoder_prompt=tokens_prompt1, decoder_prompt=decoder_prompt1)
enc_dec_prompt2 = ExplicitEncoderDecoderPrompt(encoder_prompt=tokens_prompt2, decoder_prompt=decoder_prompt2)

# Create a BART encoder/decoder model instance
llm = LLM(
    model="facebook/mbart-large-50-many-to-many-mmt",
    dtype="float32",    # if cuda is available, use "float16"
    enforce_eager=True
)

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0, min_tokens=0, max_tokens=50)

# Generate output tokens from the prompts. The output is a list of
# RequestOutput objects that contain the prompt, generated text, and other information.
outputs = llm.generate([enc_dec_prompt1, enc_dec_prompt2], sampling_params)

print("------" * 8)

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    encoder_prompt = output.encoder_prompt
    generated_text = output.outputs[0].text
    print(f"Encoder prompt: {encoder_prompt!r}, "
          f"\nDecoder prompt: {prompt!r}, "
          f"\nGenerated text: {generated_text!r}")

    print("------" * 8)

# ------------------------------------------------
# Encoder prompt: None,
# Decoder prompt: None,
# Generated text: "Le Secrétaire général de l 'ONU dit qu' il n 'y a solution militaire en Syrie."
# ------------------------------------------------
# Encoder prompt: None,
# Decoder prompt: None,
# Generated text: 'The Secretary-General of the United Nations says there is no military solution in Syria.'
# ------------------------------------------------
