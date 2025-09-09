import hashlib
from pathlib import Path

from re_mind import components, text_processing
from re_mind.constants import MAX_FILE_SIZE_BYTES
from re_mind.db.dao.base_dao import BaseDao
from re_mind.db.dao.library_file_dao import LibraryFileDao
from re_mind.db.orm.orm_schema import LibraryFile


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


def main():
    librarian = Librarian()
    librarian.add_file(Path.home() / 'tmp/test-file/deep-learning.pdf')
    files = librarian.find_all()
    for f in files:
        print(f"{f.hash_id}, {f.file_name}, {f.created_at}")




if __name__ == '__main__':
    main()
