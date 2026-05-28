from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    encode_kwargs={"normalize_embeddings": True},
)
text_splitter = SemanticChunker(embeddings=embeddings, chunk_size=600, chunk_overlap=0)
text= "When I read about a cat quietly sitting on a mat, it feels worlds apart from the dense theories of quantum mechanics that probe subatomic particles, yet both remind me of the strange beauty in simplicity and complexity. And then, shifting to something as ordinary as sharing pizza with friends, the contrast becomes even sharper—everyday joy against cosmic mystery—showing how texts can carry similarities that range from almost identical to nearly zero, depending on the worlds they describe."
chunks= text_splitter.split_text(text)
print(chunks)