"""Process-scoped project locks, automatically released by the OS after a crash."""
import hashlib
import errno
import os
from contextlib import contextmanager


class WorkflowBusy(RuntimeError):
    """Another process currently owns this project's execution lock."""


@contextmanager
def project_lock(root, project):
    directory = root / 'locks'
    directory.mkdir(parents=True, exist_ok=True)
    key = hashlib.blake2s(project.encode('utf-8'), digest_size=16).hexdigest()
    # Keep the file: deleting it could let two processes lock different inodes.
    with (directory / (key+'.lock')).open('a+b') as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno not in (errno.EACCES,errno.EAGAIN,errno.EDEADLK) and getattr(exc,'winerror',None) != 33:
                raise
            raise WorkflowBusy('Another workflow owns this project lock') from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
