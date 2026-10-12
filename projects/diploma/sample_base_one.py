from dotenv import load_dotenv
import tinker
from tinker import types


def main() -> None:
    load_dotenv()
    service_client = tinker.ServiceClient()

    model_name = "Qwen/Qwen3-8B-Base"
    sampling_client = service_client.create_sampling_client(base_model=model_name)
    tokenizer = sampling_client.get_tokenizer()

    problem = (
        "Does there exist a totally ordered field in which the rational numbers "
        "\\( \\mathbb{Q} \\) are not dense?"
    )

    prompt_text = (
        "<|im_start|>user\n"
        "Answer the following math question briefly. "
        "End with a final answer in \\boxed{}.\n\n"
        f"{problem}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    prompt = types.ModelInput.from_ints(tokenizer.encode(prompt_text))
    params = types.SamplingParams(max_tokens=512, temperature=0.2)

    result = sampling_client.sample(
        prompt=prompt,
        sampling_params=params,
        num_samples=1,
    ).result()

    seq = result.sequences[0]
    print(tokenizer.decode(seq.tokens, skip_special_tokens=True))


if __name__ == "__main__":
    main()
