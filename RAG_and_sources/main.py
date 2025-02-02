from transformers import AutoTokenizer
import torch
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings

def main():
    hf_token = ""
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        token=hf_token,
    )

    stopping_ids = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    # generate_kwargs parameters are taken from https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct


    # Optional quantization to 4bit
    # import torch
    # from transformers import BitsAndBytesConfig

    # quantization_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_compute_dtype=torch.float16,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_use_double_quant=True,
    # )

    llm = HuggingFaceLLM(
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        model_kwargs={
            "token": hf_token,
            "torch_dtype": torch.bfloat16,  # comment this line and uncomment below to use 4bit
            # "quantization_config": quantization_config
        },
        generate_kwargs={
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.9,
        },
        tokenizer_name="meta-llama/Meta-Llama-3-8B-Instruct",
        tokenizer_kwargs={"token": hf_token},
        stopping_ids=stopping_ids,
    )


    documents = SimpleDirectoryReader(input_files=["EASY-Schedule-Cheat-Sheet.md", "EN_Guide_Childhood_Diseases.md", "First12Months_rev.md",  "K82-E.md", "My-2-month-old-and-me.md", "Newborn-Care-Booklet_2023_6x8.5.md", "Sleeping-Schedule.md", "Your-guide-to-the-first-12-months.md", "better_sleep.md", "newborncare.md", "nursesguide.md"]).load_data()
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    Settings.embed_model = embed_model
    Settings.llm = llm
    index = VectorStoreIndex.from_documents(documents,)
    query_engine = index.as_query_engine(similarity_top_k=3)
    response = query_engine.query("my child has a rash what to do?")
    print(response)
    print(response.source_nodes[0].metadata['file_path'])


if __name__ == '__main__':
    main()
