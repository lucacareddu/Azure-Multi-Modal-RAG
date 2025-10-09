from modules.document import DocumentProcessor
from modules.embedding import Embedder
from modules.index import Index
from modules.storage import Storage
from modules.indexer import Indexer


def create_index(use_vector: bool = True):
    pdf_root_dir = "Contoso Corp."
    pkl_file = "results.pkl"

    doc = DocumentProcessor(root_path=pdf_root_dir)

    if use_vector:
        emb = Embedder(format_content_field="format_content")

    index = Index(title_field="header", use_vector=use_vector)

    stor = Storage(title_field="header")

    indexer = Indexer()

    print("\nRetrieving information...")
    # result = doc.create_from_layout(save_to_file=pkl_file)
    result = doc.create_from_pkl(path_to_file=pkl_file)

    print("\nFormatting text...")
    result = doc.format_result(result)
    # doc.visualize_result(result)
    paragraphs = doc.flatten_result(result)

    if use_vector:
        print("\nAssessing tokens number...")
        assert emb.tokens_number_test(paragraphs) ==  "succeded" # Less than 1k

        print("\nEmbedding text...")
        paragraphs = [emb.get_chunk_object(x) for x in paragraphs] # Add id and vector fields

    print("\nDeleting search index if it exists...")
    index.delete_index_if_exists()

    print("\nCreating new search index...")
    index.create_search_index()

    print("\nCreating new data source connection...")
    stor.connect_to_container()

    print("\nLoading data into the container...")
    stor.upload_to_container(paragraphs, overwrite=True)
    
    print("\nCreating new indexer...")
    indexer.build_indexer()

    print("\nFinished.")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--no-vector", action="store_false", help="not create embeddings")

    args = parser.parse_args()
    use_vector = args.no_vector

    create_index(use_vector=use_vector)
