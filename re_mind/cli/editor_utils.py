import os
import subprocess
import tempfile


def open_editor(path: str = None) -> str:
    """
    Opens an editor on the specified file path.

    Args:
        path: File path to edit. If None, creates a temporary file.

    Returns:
        The file path that was edited
    """
    editor = os.environ.get('EDITOR', 'vi')

    if path is None:
        tmp_file = tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False)
        path = tmp_file.name
        tmp_file.close()

    subprocess.call([editor, path])
    return path


def edit_items_in_editor(content: str) -> str:
    """
    Opens an editor to edit text content.

    Args:
        content: Text content to edit

    Returns:
        Updated text content after editing
    """
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False) as tmp_file:
        tmp_file_path = tmp_file.name
        tmp_file.write(content)

    try:
        open_editor(tmp_file_path)

        with open(tmp_file_path, 'r') as f:
            return f.read()
    finally:
        os.unlink(tmp_file_path)
