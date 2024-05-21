import pickle
from pathlib import Path
from dataclasses import dataclass

from langchain_core.embeddings import Embeddings as LCEmbeddings
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings

from .utils import hash_uuid


GET_DEFAULT_EMBEDDINGS = lambda: HuggingFaceEmbeddings(
    model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1",
    model_kwargs={"device": "cpu"},
    show_progress=False,
)


@dataclass
class Embeddings:
    model: LCEmbeddings = None
    use_cache: bool = True
    embeddings_directory: Path = "embeddings"

    def __post_init__(self):
        if self.model is None:
            self.model = GET_DEFAULT_EMBEDDINGS()
        self.filename_suffix = ".hash"
        self.embeddings_directory = Path(self.embeddings_directory)
        self.embeddings_directory.mkdir(parents=True, exist_ok=True)
        
    def hash_text(self, text: str):
        return hash_uuid(self.model.model_name + text)

    def get_file_path(self, text: str):
        return (
            self.embeddings_directory / f"{self.hash_text(text).hex}{self.filename_suffix}"
        )

    def embed_documents_with_caching(self, texts: list[str]):
        text_file_map = {x: self.get_file_path(x) for x in texts}
        file_text_map = {file: text for text, file in text_file_map.items()}
        text_filename_map = {text: file.name for file, text in file_text_map.items()}

        filenames = [x.name for x in file_text_map]

        cached_files: list[Path] = [
            x
            for x in self.embeddings_directory.glob(f"*{self.filename_suffix}")
            if x.name in filenames
        ]
        del filenames

        relevant_filenames = [x.name for x in cached_files]
        cached_filename_embedding_map: dict[str, list[float]] = {
            x.name: pickle.load(x.open("rb")) for x in cached_files
        }

        uncached_file_text_map = {
            file: text
            for file, text in file_text_map.items()
            if file.name not in relevant_filenames
        }

        del relevant_filenames, cached_files

        generated_embeddings = self.model.embed_documents(
            list(uncached_file_text_map.values())
        )
        generated_filename_embedding_map = {
            path.name: embedding
            for path, embedding in zip(uncached_file_text_map, generated_embeddings)
        }
        for path, embedding in zip(uncached_file_text_map, generated_embeddings):
            pickle.dump(embedding, path.open("wb"))
        all_filename_embeddings = {
            **cached_filename_embedding_map,
            **generated_filename_embedding_map,
        }
        del (
            cached_filename_embedding_map,
            generated_filename_embedding_map,
            uncached_file_text_map,
        )
        all_text_embeddings = [
            (text, all_filename_embeddings[text_filename_map[text]]) for text in texts
        ]
        return [embeddings for text, embeddings in all_text_embeddings]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.use_cache:
            return self.embed_documents_with_caching(texts)
        return self.model.embed_documents(texts)
