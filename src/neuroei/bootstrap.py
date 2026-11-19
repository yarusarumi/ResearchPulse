"""Build a pointer-complete seed bundle from the approved project state.

The bundle is deliberately constructed in memory: callers decide when and how to
transfer it.  ``active_profile`` is authoritative; ``latest_profile`` is only a
reference and must never be promoted here.
"""
from .models import ProjectProfile
from .paperflow_adapter import validate_mapping


def _get(store, kind: str, key: str):
    try:
        body = store.get(kind, key)
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Seed source missing {kind}: {key}") from exc
    if body is None:
        raise ValueError(f"Seed source missing {kind}: {key}")
    return body


def _key_for(project: str, version: str) -> str:
    return f"{project}:{version}"


def _matches_project(key: str, project: str) -> bool:
    return key.startswith(project + ":") and len(key) > len(project) + 1


def _pointer(store, kind: str, project: str) -> dict:
    body = _get(store, kind, project)
    if not isinstance(body, dict):
        raise ValueError(f"Invalid {kind} pointer for {project}")  # noqa: TRY004
    key = body.get("key")
    if not isinstance(key, str) or not key.strip():
        raise ValueError(f"Invalid {kind} pointer for {project}")
    return body


def _active_profile(store, project: str) -> tuple[str, ProjectProfile, dict]:
    pointer = _pointer(store, "active_profile", project)
    key = pointer["key"]
    if not _matches_project(key, project):
        raise ValueError("active_profile pointer project mismatch")
    body = _get(store, "profile", key)
    profile = ProjectProfile.model_validate(body)
    if profile.project_id != project:
        raise ValueError("Active profile project mismatch")
    if key != _key_for(project, profile.version):
        raise ValueError("Active profile key/version mismatch")
    return key, profile, body


def _latest_reference(store, project: str, active_key: str):
    """Return (key, profile body, snapshot) for a usable latest pointer, or None.

    The latest pointer is optional and only a reference.  If it is absent,
    malformed, or incomplete, omit it instead of exporting a dangling pointer.
    A pointer that resolves to the active key is returned so the caller can keep
    it; the caller must not export duplicate profile/snapshot objects for it.
    """
    try:
        pointer = store.get("latest_profile", project)
    except (KeyError, ValueError):
        return None
    if not isinstance(pointer, dict):
        return None
    key = pointer.get("key")
    if not isinstance(key, str) or not key.strip():
        return None
    if not _matches_project(key, project):
        return None
    try:
        body = _get(store, "profile", key)
        profile = ProjectProfile.model_validate(body)
        snapshot = _get(store, "snapshot", key)
    except ValueError:
        return None
    if not isinstance(snapshot, dict):
        return None
    if "project_id" in snapshot and snapshot["project_id"] != project:
        return None
    if profile.project_id != project or key != _key_for(project, profile.version):
        return None
    return key, body, snapshot


def build_seed_objects(
    store, project: str, pool_id: str, interest_mapping=None
) -> list[tuple[str, str, dict]]:
    """Return the exact objects needed to import an approved project seed.

    No network, model, or store writes occur.  The active profile, its source
    snapshot and its approval are mandatory and must all be identifiable from the
    active pointer.  A usable latest pointer is preserved, including when it points
    at the active key; for a different latest key its complete profile/snapshot pair
    is exported as a reference.  The latest pointer never changes the active choice.
    """
    if not isinstance(project, str) or not project.strip():
        raise ValueError("Project must be a non-empty string")
    if not isinstance(pool_id, str) or not pool_id.strip():
        raise ValueError("Pool must be a non-empty string")

    active_key, active, active_body = _active_profile(store, project)
    snapshot = _get(store, "snapshot", active_key)
    if not isinstance(snapshot, dict):
        raise ValueError("Active snapshot is not an object")  # noqa: TRY004
    if "project_id" in snapshot and snapshot["project_id"] != project:
        raise ValueError("Active snapshot project mismatch")
    approval = _get(store, "approval", active_key)
    if not isinstance(approval, dict):
        raise ValueError("Active approval is not an object")  # noqa: TRY004
    for field in ("reviewer", "at"):
        value = approval.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Active approval has empty {field}")

    pool = _get(store, "pool", pool_id)
    if not isinstance(pool, dict):
        raise ValueError("Pool is not an object")  # noqa: TRY004
    if pool.get("project_id") != project:
        raise ValueError("Pool does not belong to project")

    latest = _latest_reference(store, project, active_key)
    mapping = None
    if interest_mapping is not None:
        mapping = validate_mapping(active, interest_mapping)
        mapping_key = f"{mapping.project_id}:{mapping.profile_version}:{mapping.mapping_version}"
        mapping = (mapping_key, mapping.model_dump())

    objects: list[tuple[str, str, dict]] = []
    seen: set[tuple[str, str]] = set()

    def add(kind: str, key: str, body: dict):
        identity = (kind, key)
        if identity in seen:
            raise ValueError(f"Duplicate seed object: {kind}:{key}")
        seen.add(identity)
        objects.append((kind, key, body))

    add("profile", active_key, active_body)
    add("snapshot", active_key, snapshot)
    add("approval", active_key, approval)
    add("active_profile", project, {"key": active_key})
    if latest is not None:
        latest_key, latest_body, latest_snapshot = latest
        if latest_key != active_key:
            add("profile", latest_key, latest_body)
            add("snapshot", latest_key, latest_snapshot)
        add("latest_profile", project, {"key": latest_key})
    add("pool", pool_id, pool)
    if mapping is not None:
        add("interest_mapping", mapping[0], mapping[1])
    return objects
