import shutil
import threading
import uuid
from pathlib import Path
from unittest.mock import MagicMock

from openviking.pyagfs.exceptions import AGFSNotFoundError


class MockAGFSNotFoundError(AGFSNotFoundError, FileNotFoundError):
    """Not-found error matching the real binding (AGFSNotFoundError).

    Also a FileNotFoundError so callers that catch the builtin keep working.
    """


class MockLocalAGFS:
    """
    A mock implementation of the AGFS binding client that operates on a local
    directory. Useful for tests where running the real RAGFS binding is not
    feasible or desired.
    """

    def __init__(self, config=None, root_path=None):
        self.config = config
        self.root = Path(root_path) if root_path else Path("/tmp/viking_data")
        self.root.mkdir(parents=True, exist_ok=True)
        self._pathlocks_guard = threading.Lock()
        self._pathlocks = {}
        self._pathlock_leases = {}

    def _resolve(self, path):
        if str(path).startswith("viking://"):
            path = str(path).replace("viking://", "")
        if str(path).startswith("/"):
            path = str(path)[1:]
        return self.root / path

    def exists(self, path, ctx=None):
        return self._resolve(path).exists()

    def mkdir(self, path, ctx=None, parents=True, exist_ok=True):
        self._resolve(path).mkdir(parents=parents, exist_ok=exist_ok)

    def ls(self, path, ctx=None, **kwargs):
        """List test entries with the same sorting and pagination contract as RagFS."""
        p = self._resolve(path)
        if not p.exists():
            return []
        res = []
        for item in p.iterdir():
            res.append(
                {
                    "name": item.name,
                    "isDir": item.is_dir(),  # Note: JS style camelCase for some APIs
                    "type": "directory" if item.is_dir() else "file",
                    "size": item.stat().st_size if item.is_file() else 0,
                    "mtime": item.stat().st_mtime,
                    "uri": f"viking://{path}/{item.name}".replace("//", "/"),
                }
            )
        sort_by = kwargs.get("sort_by")
        reverse = kwargs.get("sort_order", "asc") == "desc"
        if sort_by:
            key = (
                (lambda entry: (str(entry["name"]).lower(), str(entry["name"])))
                if sort_by == "name"
                else (
                    lambda entry: (
                        entry["mtime"],
                        str(entry["name"]).lower(),
                        str(entry["name"]),
                    )
                )
            )
            directories = sorted(
                (entry for entry in res if entry["isDir"]), key=key, reverse=reverse
            )
            files = sorted((entry for entry in res if not entry["isDir"]), key=key, reverse=reverse)
            res = directories + files
        offset = kwargs.get("offset", 0)
        limit = kwargs.get("limit")
        return res[offset:] if limit is None else res[offset : offset + limit]

    def glob_directory(
        self,
        path,
        pattern,
        show_hidden=False,
        page_size=None,
        level_limit=None,
        continuation_token=None,
        ctx=None,
    ):
        del ctx
        root = self._resolve(path)
        matches = []
        for item in sorted(root.glob(pattern)):
            relative = item.relative_to(root)
            if not show_hidden and any(part.startswith(".") for part in relative.parts):
                continue
            if level_limit is not None and len(relative.parts) > level_limit:
                continue
            matches.append(
                {
                    "path": f"{str(path).rstrip('/')}/{relative.as_posix()}",
                    "rel_path": relative.as_posix(),
                    "name": item.name,
                    "is_dir": item.is_dir(),
                }
            )

        start = int(continuation_token or 0)
        end = len(matches) if not page_size else start + page_size
        return {
            "entries": matches[start:end],
            "next_token": str(end) if end < len(matches) else None,
        }

    def writeto(self, path, content, ctx=None, **kwargs):
        p = self._resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, str):
            p.write_text(content, encoding="utf-8")
        elif isinstance(content, (bytes, bytearray, memoryview)):
            p.write_bytes(content)
        else:
            # Streaming writers pass an iterable of byte chunks.
            p.write_bytes(b"".join(content))
        return str(p)

    def write(self, path, content, ctx=None, **kwargs):
        return self.writeto(path, content, ctx, **kwargs)

    def write_file(self, path, content, ctx=None, **kwargs):
        return self.writeto(path, content, ctx, **kwargs)

    def read_file(self, path, ctx=None, **kwargs):
        p = self._resolve(path)
        if not p.exists():
            raise MockAGFSNotFoundError(path)
        return p.read_bytes()

    def read(self, path, ctx=None, **kwargs):
        return self.read_file(path, ctx, **kwargs)

    def cat(self, path, ctx=None, **kwargs):
        return self.read_file(path, ctx, **kwargs)

    def rm(self, path, recursive=False, ctx=None):
        p = self._resolve(path)
        if p.exists():
            if p.is_dir():
                if recursive:
                    shutil.rmtree(p)
                else:
                    p.rmdir()
            else:
                p.unlink()

    def delete_temp(self, path, ctx=None):
        self.rm(path, recursive=True, ctx=ctx)

    def mv(self, src, dst, ctx=None):
        s = self._resolve(src)
        d = self._resolve(dst)
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(s), str(d))

    def stat(self, path, ctx=None):
        p = self._resolve(path)
        if not p.exists():
            raise MockAGFSNotFoundError(path)
        s = p.stat()
        return {
            "size": s.st_size,
            "mtime": s.st_mtime,
            "isDir": p.is_dir(),
            "is_dir": p.is_dir(),
        }

    def bind_request_context(self, ctx):
        return MagicMock(__enter__=lambda x: None, __exit__=lambda x, y, z: None)

    # -- pathlock -------------------------------------------------------------
    # One non-reentrant lock per path. A lease owns the locks for its paths;
    # acquiring with ``owner_lease_ref`` of a live lease is re-entrant and
    # takes no new locks (the parent lease keeps holding them).

    def _new_lease(self, paths, locks):
        lease_ref = str(uuid.uuid4())
        lease = {
            "lease_ref": lease_ref,
            "ownership_ref": str(uuid.uuid4()),
            "owner_id": "mock-local-agfs",
            "owned": True,
        }
        with self._pathlocks_guard:
            self._pathlock_leases[lease_ref] = {"paths": list(paths), "locks": list(locks)}
        return lease

    def _pathlock_acquire_paths(self, paths, timeout_secs=0.0, owner_lease_ref=None):
        if owner_lease_ref is not None:
            with self._pathlocks_guard:
                reentrant = owner_lease_ref.get("lease_ref") in self._pathlock_leases
            if reentrant:
                return self._new_lease(paths, [])

        acquired = []
        for path in sorted(set(paths)):
            with self._pathlocks_guard:
                lock = self._pathlocks.setdefault(path, threading.Lock())
            if not lock.acquire(timeout=timeout_secs):
                for held in acquired:
                    held.release()
                raise TimeoutError(f"timed out acquiring test path lock: {path}")
            acquired.append(lock)
        return self._new_lease(paths, acquired)

    def pathlock_acquire_tree(self, ctx, path, timeout_secs=0.0, owner_lease_ref=None):
        del ctx
        return self._pathlock_acquire_paths([path], timeout_secs, owner_lease_ref)

    pathlock_acquire_exact = pathlock_acquire_tree

    def pathlock_acquire_tree_batch(self, ctx, paths, timeout_secs=0.0, owner_lease_ref=None):
        del ctx
        return self._pathlock_acquire_paths(paths, timeout_secs, owner_lease_ref)

    pathlock_acquire_exact_batch = pathlock_acquire_tree_batch

    def pathlock_acquire_exact_tree_batch(
        self, ctx, exact_paths, tree_paths, timeout_secs=0.0, owner_lease_ref=None
    ):
        del ctx
        return self._pathlock_acquire_paths(
            [*exact_paths, *tree_paths], timeout_secs, owner_lease_ref
        )

    def pathlock_acquire_batch(self, ctx, requests, timeout_secs=0.0, owner_lease_ref=None):
        del ctx
        return self._pathlock_acquire_paths(
            [request["path"] for request in requests], timeout_secs, owner_lease_ref
        )

    def pathlock_release(self, ctx, owned_lease_ref):
        del ctx
        with self._pathlocks_guard:
            entry = self._pathlock_leases.pop(owned_lease_ref["lease_ref"], None)
        for lock in (entry or {}).get("locks", []):
            lock.release()

    def pathlock_release_selected(self, ctx, owned_lease_ref, lock_paths):
        del ctx, owned_lease_ref, lock_paths

    def pathlock_as_borrowed(self, ctx, owned_lease_ref):
        del ctx
        return {**owned_lease_ref, "owned": False}

    def pathlock_refresh(self, ctx, owned_lease_ref):
        del ctx
        with self._pathlocks_guard:
            alive = owned_lease_ref.get("lease_ref") in self._pathlock_leases
        return "refreshed" if alive else "lost"

    def pathlock_to_handoff(self, ctx, owned_lease_ref):
        del ctx
        lease_ref = owned_lease_ref["lease_ref"]
        with self._pathlocks_guard:
            entry = self._pathlock_leases.get(lease_ref, {"paths": []})
        return {
            "lease_ref": lease_ref,
            "owner_id": owned_lease_ref.get("owner_id", "mock-local-agfs"),
            "lock_paths": list(entry["paths"]),
            "covered_paths": [],
        }

    def pathlock_handoff(self, ctx, owned_lease_ref):
        # Real handoffs are adopted (and released) by a queue worker; the mock's
        # queues never run, so free the locks now to avoid holding them forever.
        del ctx
        with self._pathlocks_guard:
            entry = self._pathlock_leases.get(owned_lease_ref["lease_ref"])
            locks = entry["locks"] if entry else []
            if entry:
                entry["locks"] = []
        for lock in locks:
            lock.release()

    def pathlock_adopt(self, ctx, handoff_ref):
        del ctx
        with self._pathlocks_guard:
            entry = self._pathlock_leases.pop(handoff_ref.get("lease_ref"), None)
        if entry is None:
            raise ValueError("pathlock handoff ref is not live")
        return self._new_lease(entry["paths"], entry["locks"])

    def pathlock_is_locked(self, ctx, path, ignore_stale=True):
        del ctx, ignore_stale
        with self._pathlocks_guard:
            lock = self._pathlocks.get(path)
        return lock is not None and lock.locked()

    def pathlock_observe(self, ctx):
        del ctx
        with self._pathlocks_guard:
            return {"active_leases": len(self._pathlock_leases)}
