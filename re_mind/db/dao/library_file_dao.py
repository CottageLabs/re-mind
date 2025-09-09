from re_mind.db.dao.base_dao import BaseDao
from re_mind.db.orm.orm_schema import LibraryFile


class LibraryFileDao(BaseDao):
    def exist(self, hash_id: str) -> bool:
        clause = LibraryFile.hash_id == hash_id
        return self.exists(clause)
