"""Copy completed run evidence to persistent storage; never delete the source."""
import hashlib
import json
from pathlib import Path
import shutil
import time


def validate_archive_paths(source, destination, *, before_run=False):
    source,destination=Path(source).resolve(),Path(destination).resolve()
    if source==destination or source in destination.parents or destination in source.parents:
        raise ValueError('Run output and archive must be disjoint directories')
    if destination.exists():
        raise FileExistsError('Archive destination already exists: '+str(destination))
    if before_run and source.exists() and (not source.is_dir() or any(source.iterdir())):
        raise FileExistsError('Archived runs require a new or empty output directory')
    return source,destination


def digest(path):
    sha=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):sha.update(block)
    return sha.hexdigest()


def archive_run(source,destination):
    """The final manifest is the completion marker; partial copies stay visible.

    Run this only after task writers/encoders close. Copy or checksum failure
    raises and retains local files; an existing archive is never overwritten.
    """
    started=time.perf_counter()
    source,destination=validate_archive_paths(source,destination)
    if not source.is_dir():raise FileNotFoundError('Run output missing: '+str(source))
    paths=sorted(source.rglob('*'))
    if any(p.is_symlink() for p in paths):
        raise ValueError('Run archive cannot follow symlinked evidence')
    files=[p for p in paths if p.is_file()]
    if not files:raise ValueError('Run archive cannot be empty')
    marker='_archive_manifest.json'
    if (source/marker).exists():raise ValueError('Run already contains an archive manifest')
    destination.mkdir(parents=True,exist_ok=False)
    records=[]
    for path in files:
        relative=path.relative_to(source);target=destination/relative
        before=path.stat();checksum=digest(path)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,target)
        after=path.stat()
        if (before.st_size!=after.st_size or before.st_mtime_ns!=after.st_mtime_ns
                or target.stat().st_size!=before.st_size or digest(target)!=checksum):
            raise RuntimeError('Run archive verification failed: '+str(relative))
        records.append(dict(path=relative.as_posix(),bytes=before.st_size,sha256=checksum))
    report=dict(status='verified',source=str(source),destination=str(destination),
        source_retained=True,files=records,total_bytes=sum(r['bytes'] for r in records),
        archive_wall_s=time.perf_counter()-started,
        timing_scope='after task and environment close; excluded from PREGRASP-to-DONE')
    (destination/marker).write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report
