from anthropic import Anthropic
import os
from anthropic._types import NOT_GIVEN

anthropic = Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
    max_retries=50
)

def query(PROMPT, temperature = 0., max_tokens = 500, model = 'claude-2'):
    completion = anthropic.completions.create(
        model=model,
        max_tokens_to_sample=max_tokens,
        prompt=PROMPT,
        temperature = temperature,
        top_k = 1 if temperature == 0 else NOT_GIVEN
    )
    return completion.completion 