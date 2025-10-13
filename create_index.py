from index_modules.document import DocumentProcessor
from index_modules.embedding import Embedder
from index_modules.index import Index
from index_modules.storage import Storage
from index_modules.indexer import Indexer


def create_index(use_image: bool = True, use_vector: bool = True):

    pdf_root_dir = "Contoso Corp."
    raw_pkl_file = "data/results.pkl"
    format_pkl_file = "data/sources.pkl"

    doc = DocumentProcessor(root_path=pdf_root_dir, use_images=use_image)

    if use_vector:
        emb = Embedder(format_content_field="format_content")

    index = Index(title_field="header", use_vector=use_vector)

    stor = Storage(title_field="header")

    indexer = Indexer()

    print("\nRetrieving information...")
    # result = doc.create_from_layout(save_to_file=raw_pkl_file)
    result = doc.create_from_pkl(path_to_file=raw_pkl_file)

    print("\nFormatting text...")
    result = doc.format_result(result)
    # doc.visualize_result(result)
    paragraphs = doc.flatten_result(result, save_to_file=format_pkl_file)

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

    print("\nErasing all blobs in the container...")
    stor.erase_container()

    print("\nLoading data into the container...")
    stor.upload_to_container(paragraphs, overwrite=True)
    
    print("\nCreating new indexer...")
    indexer.build_indexer()

    print("\nFinished.")



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-ni", "--no-image", action="store_false", help="not embed images")
    parser.add_argument("-nv", "--no-vector", action="store_false", help="not create embeddings")
    args = parser.parse_args()

    use_image = args.no_image
    use_vector = args.no_vector

    create_index(use_image=use_image, use_vector=use_vector)
