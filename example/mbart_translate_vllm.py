import vllm_mbart
from vllm import LLM, SamplingParams
from vllm.inputs import ExplicitEncoderDecoderPrompt

import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Create a BART encoder/decoder model instance
llm = LLM(
    model="facebook/mbart-large-en-ro",
    dtype="float32",    # if cuda is available, use "float16"
    # enforce_eager=True,
    hf_overrides={"architectures": ["MBartForConditionalGeneration"]}
)

enc_dec_prompt1 = ExplicitEncoderDecoderPrompt(encoder_prompt="42 is the answer", decoder_prompt=None)
enc_dec_prompt2 = ExplicitEncoderDecoderPrompt(encoder_prompt="What is the meaning of life?", decoder_prompt=None)

# Create a sampling params object.
sampling_params = SamplingParams(
    temperature=0,
    min_tokens=0,
    max_tokens=10,
)

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
# Encoder prompt: '42 is the answer',
# Decoder prompt: None,
# Generated text: '42 este răspunsul'
# ------------------------------------------------
# Encoder prompt: 'What is the meaning of life?',
# Decoder prompt: None,
# Generated text: 'Care este sensul vieţii?'
# ------------------------------------------------
