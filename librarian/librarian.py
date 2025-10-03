import hashlib
from pathlib import Path

from librarian import components
from librarian import text_processing
from librarian.constants import MAX_FILE_SIZE_BYTES
from librarian.dao.base_dao import BaseDao
from librarian.dao.library_file_dao import LibraryFileDao
from librarian.dao.schema.dao_schema import LibraryFile
from librarian.vector_store_service import VectorStoreService


def calculate_file_hash(file_path: Path) -> str:
    """Calculate SHA256 hash of a file."""
    with open(file_path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


class Librarian:
    """
    Manage files in the library
    - manage both in DB and in Vector Store
    - avoid duplicate files by hash
    """

    def __init__(self, vector_store=None, dao=None):
        self.vector_store = vector_store or components.get_vector_store()
        self.dao = dao or BaseDao()

    def add_file(self, file_path: str | Path) -> None:
        file_path = Path(file_path)

        # Check if file exists
        if not file_path.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Check file size
        file_size = file_path.stat().st_size
        if file_size > MAX_FILE_SIZE_BYTES:
            raise ValueError(
                f"File size ({file_size} bytes) exceeds maximum allowed size ({MAX_FILE_SIZE_BYTES} bytes)")

        file_hash = calculate_file_hash(file_path)
        print(f"File hash: {file_hash}")

        # Check hash existence in DB
        file_dao = LibraryFileDao()
        if file_dao.exist(file_hash):
            raise ValueError(f"File with hash {file_hash} already exists in library")

        # Add to Vector Store
        text_processing.save_any_to_vectorstore(file_path,
                                                vectorstore=self.vector_store,
                                                metadata={"hash_id": file_hash})

        # Add to DB
        library_file = LibraryFile(hash_id=file_hash, file_name=file_path.name)
        self.dao.add(library_file)

    def find_all(self) -> list[LibraryFile]:
        with self.dao.create_session() as session:
            return list(session.query(LibraryFile).all())

    def find_latest(self, limit: int = 10) -> list[LibraryFile]:
        """Find the latest n documents ordered by creation date."""
        with self.dao.create_session() as session:
            return list(session.query(LibraryFile)
                       .order_by(LibraryFile.created_at.desc())
                       .limit(limit)
                       .all())

    def drop_vector_store(self):
        """Drop the entire vector store collection and clear database records."""
        vector_service = VectorStoreService(self.vector_store)
        vector_service.delete_collection()

        # Clear all library file records from database
        with self.dao.create_session() as session:
            session.query(LibraryFile).delete()
            session.commit()

    def remove_by_hash(self, hash_id: str) -> bool:
        """Remove a library file by its hash ID from both vector store and database."""
        from qdrant_client import models

        # Check if file exists in database
        file_dao = LibraryFileDao()
        if not file_dao.exist(hash_id):
            return False

        # Delete from vector store using metadata filter
        client = self.vector_store.client
        collection_name = self.vector_store.collection_name

        # Create filter for hash_id
        filter_condition = models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.hash_id",
                    match=models.MatchValue(value=hash_id)
                )
            ]
        )

        # Delete from vector store
        client.delete(
            collection_name=collection_name,
            points_selector=models.FilterSelector(filter=filter_condition)
        )

        # Delete from database
        with self.dao.create_session() as session:
            session.query(LibraryFile).filter(LibraryFile.hash_id == hash_id).delete()
            session.commit()

        return True


def main__add_test_file():
    librarian = Librarian()
    librarian.add_file(Path.home() / 'tmp/test-file/deep-learning.pdf')
    librarian.add_file(Path.home() / 'tmp/test-file/rl1.pdf')
    librarian.add_file(Path.home() / 'tmp/test-file/rl2.pdf')
    files = librarian.find_all()
    for f in files:
        print(f"{f.hash_id}, {f.file_name}, {f.created_at}")




if __name__ == '__main__':
    main__add_test_file()
