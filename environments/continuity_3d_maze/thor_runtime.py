import json
import os
import shutil
import sys
from pathlib import Path


DEFAULT_BRANCH = "nanna"
REPO_ROOT = Path(__file__).resolve().parent
RUNTIME_HOME = REPO_ROOT / ".thor_runtime"
ORIGINAL_HOME = Path(os.environ.get("THOR_ORIGINAL_HOME", str(Path.home()))).resolve()
HEX_DIGITS = set("0123456789abcdef")


def _user_site_path(home: Path) -> Path:
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    return home / ".local" / "lib" / version / "site-packages"


def _ensure_symlink(src: Path, dst: Path) -> None:
    try:
        real_src = src.resolve(strict=True)
    except FileNotFoundError:
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink():
        try:
            if dst.resolve(strict=True) == real_src:
                return
            dst.unlink()
        except FileNotFoundError:
            dst.unlink(missing_ok=True)
    elif dst.exists():
        return

    try:
        dst.symlink_to(real_src, target_is_directory=real_src.is_dir())
    except FileExistsError:
        if dst.is_symlink() and dst.resolve() == real_src:
            return
        if dst.exists():
            return
        raise


def _clean_stale_release_links(releases_dir: Path) -> None:
    if not releases_dir.exists():
        return

    for entry in releases_dir.iterdir():
        if not entry.is_symlink():
            continue
        try:
            entry.resolve(strict=True)
        except FileNotFoundError:
            entry.unlink(missing_ok=True)
            continue
        if entry.name.endswith(".lock"):
            # Locks are runtime-local state; linking stale locks between runtime homes can
            # make AI2-THOR try to open a path that no longer exists.
            entry.unlink(missing_ok=True)


def _should_link_release_entry(path: Path) -> bool:
    return not path.name.endswith(".lock")


def _copy_commit_cache(src_dir: Path, dst_dir: Path) -> None:
    if not src_dir.exists():
        return

    dst_dir.mkdir(parents=True, exist_ok=True)
    for src_file in src_dir.glob("*.json"):
        dst_file = dst_dir / src_file.name
        if dst_file.exists() and dst_file.resolve() == src_file.resolve():
            continue
        shutil.copy2(src_file, dst_file)


def _release_sources() -> list[Path]:
    sources = [
        ORIGINAL_HOME / ".ai2thor" / "releases",
        RUNTIME_HOME / ".ai2thor" / "releases",
    ]
    if RUNTIME_HOME.exists():
        for candidate in RUNTIME_HOME.iterdir():
            releases = candidate / ".ai2thor" / "releases"
            if releases.exists():
                sources.append(releases)
    return sources


def _commit_cache_sources() -> list[Path]:
    sources = [
        ORIGINAL_HOME / ".ai2thor" / "cache" / "commits",
        RUNTIME_HOME / ".ai2thor" / "cache" / "commits",
    ]
    if RUNTIME_HOME.exists():
        for candidate in RUNTIME_HOME.iterdir():
            commits = candidate / ".ai2thor" / "cache" / "commits"
            if commits.exists():
                sources.append(commits)
    return sources


def _parse_release_name(name: str) -> tuple[str, str] | None:
    if not name.startswith("thor-"):
        return None

    try:
        platform_name, commit_id = name[len("thor-") :].rsplit("-", 1)
    except ValueError:
        return None

    if len(commit_id) != 40 or any(char not in HEX_DIGITS for char in commit_id):
        return None
    return platform_name, commit_id


def _platform_name(platform: object | None) -> str | None:
    if platform is None:
        return None
    if isinstance(platform, str):
        return platform
    name = getattr(platform, "name", None)
    if callable(name):
        return str(name())
    return getattr(platform, "__name__", None)


def _local_release_commits(platform: object | None = None) -> set[str]:
    platform_name = _platform_name(platform)
    commits: set[str] = set()

    for releases_dir in _release_sources():
        if not releases_dir.exists():
            continue
        for release in releases_dir.iterdir():
            parsed = _parse_release_name(release.name)
            if parsed is None:
                continue

            release_platform, commit_id = parsed
            if platform_name is not None and release_platform != platform_name:
                continue

            try:
                if release.resolve(strict=True).is_dir():
                    commits.add(commit_id)
            except FileNotFoundError:
                continue

    return commits


def _cached_branch_commits(branch: str = DEFAULT_BRANCH) -> list[str]:
    for commits_dir in _commit_cache_sources():
        cache_path = commits_dir / f"{branch}.json"
        if not cache_path.exists():
            continue
        try:
            payload = json.loads(cache_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue

        commits = [
            item.get("sha")
            for item in payload
            if isinstance(item, dict) and isinstance(item.get("sha"), str)
        ]
        if commits:
            return commits

    return []


def resolve_local_ai2thor_commit(
    platform: object | None = None,
    branch: str = DEFAULT_BRANCH,
) -> str | None:
    local_commits = _local_release_commits(platform)
    if not local_commits:
        return None

    for commit_id in _cached_branch_commits(branch):
        if commit_id in local_commits:
            return commit_id

    return sorted(local_commits)[0]


def prepare_ai2thor_runtime(runtime_name: str | None = None) -> Path:
    runtime_home = (
        RUNTIME_HOME.resolve()
        if runtime_name is None
        else (RUNTIME_HOME / runtime_name).resolve()
    )
    ai2thor_home = runtime_home / ".ai2thor"
    releases_dir = ai2thor_home / "releases"
    commits_dir = ai2thor_home / "cache" / "commits"

    for path in (
        releases_dir,
        commits_dir,
        ai2thor_home / "log",
        ai2thor_home / "tmp",
        runtime_home / ".cache",
        runtime_home / ".config" / "matplotlib",
    ):
        path.mkdir(parents=True, exist_ok=True)

    _clean_stale_release_links(releases_dir)

    for source_releases in _release_sources():
        if source_releases.resolve() == releases_dir.resolve():
            continue
        for release in source_releases.iterdir():
            if not _should_link_release_entry(release):
                continue
            _ensure_symlink(release, releases_dir / release.name)

    for source_commits in _commit_cache_sources():
        if source_commits.resolve() == commits_dir.resolve():
            continue
        _copy_commit_cache(source_commits, commits_dir)

    user_site = _user_site_path(ORIGINAL_HOME)
    if user_site.exists():
        user_site_str = str(user_site)
        if user_site_str not in sys.path:
            sys.path.insert(0, user_site_str)

    os.environ.setdefault("THOR_ORIGINAL_HOME", str(ORIGINAL_HOME))
    os.environ["HOME"] = str(runtime_home)
    os.environ["XDG_CACHE_HOME"] = str(runtime_home / ".cache")
    os.environ["MPLCONFIGDIR"] = str(runtime_home / ".config" / "matplotlib")
    return runtime_home
