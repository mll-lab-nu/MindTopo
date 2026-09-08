"""Replay saved scenes, retaining question.jsonl bytes and all image paths."""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import shutil
import struct
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from playwright.async_api import async_playwright
from vite_server import ViteFrontendServer

PROJECT = Path(__file__).resolve().parent.parent
SCENE_FIELDS = (
    'bead_sequence', 'bead_colors_hex', 'num_beads', 'curve_type',
    'curve_complexity', 'bead_size', 'rope_thickness', 'is_ring',
    'seed', 'bead_positions',
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


async def run(args):
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite existing directory: {output}')
    question_bytes = (source / 'question.jsonl').read_bytes()
    rows = [json.loads(line) for line in question_bytes.splitlines() if line.strip()]
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Duplicate question IDs')
    samples = json.loads((source / 'dataset_metadata.json').read_text())['samples']
    by_name = {sample['raw_filename']: sample for sample in samples}
    if len(by_name) != len(samples):
        raise ValueError('Ambiguous raw filenames in metadata')
    references = defaultdict(list)
    original_hashes = {}
    for row in rows:
        for rel in row['images']:
            path = Path(rel)
            if path.is_absolute() or '..' in path.parts or path.parts[0] != 'images':
                raise ValueError(f'Unsafe image path: {rel}')
            sample = by_name[path.name]
            if not references[path.name]:
                for field in (*SCENE_FIELDS, 'camera_angle', 'image_size'):
                    if field not in sample:
                        raise ValueError(f'Missing {field} in {path.name}')
            references[path.name].append(rel)
            original_hashes[rel] = digest((source / rel).read_bytes())
    png_paths = {str(p.relative_to(source)) for p in (source / 'images').rglob('*.png')}
    if png_paths != set(original_hashes):
        raise ValueError('Source PNG inventory differs from question references')
    for name, refs in references.items():
        if len({original_hashes[rel] for rel in refs}) != 1:
            raise ValueError(f'Different source images share basename {name}')

    output.mkdir(parents=True)
    queue = asyncio.Queue()
    for name in references:
        queue.put_nowait(name)
    completed = {}
    source_code_hashes = {
        str(path.relative_to(PROJECT)): digest(path.read_bytes())
        for path in [
            PROJECT / 'frontend/src/bead-gallery.js',
            PROJECT / 'frontend/src/bead-renderer.js',
            PROJECT / 'frontend/src/bead-curve-library.js',
        ]
    }
    print(f'Replaying {len(references)} scenes -> {len(original_hashes)} PNG paths; {len(rows)} fixed questions', flush=True)
    with ViteFrontendServer(frontend_dir=PROJECT / 'frontend') as server:
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=True)
            browser_version = browser.version

            async def worker():
                page = await browser.new_page(viewport={'width': 1440, 'height': 1400})
                try:
                    # capturePNG renders synchronously; continuous gallery animation
                    # would spend GPU time redrawing scenes between captures.
                    await page.add_init_script('window.requestAnimationFrame = () => 0;')
                    await page.goto(server.base_url, wait_until='networkidle')
                    await page.evaluate("async () => { window.__replay = (await import('/src/bead-gallery.js')).renderSavedSample; }")
                    while not queue.empty():
                        name = queue.get_nowait()
                        sample = by_name[name]
                        result = await page.evaluate('(s) => window.__replay(s)', sample)
                        for field in SCENE_FIELDS:
                            if result['metadata'][field] != sample[field]:
                                raise ValueError(f'Scene drift: {name}: {field}')
                        if result['beadCount'] != sample['num_beads']:
                            raise ValueError(f'Bead count mismatch: {name}')
                        data = base64.b64decode(result['image_data_url'].split(',', 1)[1], validate=True)
                        if data[:8] != b'\x89PNG\r\n\x1a\n':
                            raise ValueError(f'Invalid PNG: {name}')
                        if list(struct.unpack('>II', data[16:24])) != sample['image_size']:
                            raise ValueError(f'Image size mismatch: {name}')
                        new_hash = digest(data)
                        for rel in references[name]:
                            dest = output / rel
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            dest.write_bytes(data)
                            if digest(dest.read_bytes()) != new_hash:
                                raise ValueError(f'Write verification failed: {rel}')
                        completed[name] = new_hash
                        if len(completed) == 1 or len(completed) % 25 == 0:
                            print(f'{len(completed)}/{len(references)} scenes verified', flush=True)
                        queue.task_done()
                finally:
                    await page.close()

            try:
                await asyncio.gather(*(worker() for _ in range(max(1, args.workers))))
            finally:
                await browser.close()

    # Publish the question file only after every referenced image is ready.
    for name in ['dataset_metadata.json', 'summary.json', 'question.jsonl']:
        shutil.copy2(source / name, output / name)
        if (source / name).read_bytes() != (output / name).read_bytes():
            raise ValueError(f'Copied file differs: {name}')
    if (source / 'question.jsonl').read_bytes() != question_bytes:
        raise ValueError('Source question.jsonl changed during rendering')
    output_pngs = {str(p.relative_to(output)) for p in (output / 'images').rglob('*.png')}
    if output_pngs != png_paths:
        raise ValueError('Output PNG paths differ from source')
    image_records = [
        {'path': rel, 'source_sha256': old_hash, 'output_sha256': completed[Path(rel).name]}
        for rel, old_hash in original_hashes.items()
    ]
    report = {
        'status': 'complete',
        'source': str(source), 'output': str(output),
        'rendered_at': datetime.now(timezone.utc).isoformat(),
        'browser_version': browser_version,
        'source_code_sha256': source_code_hashes,
        'question_sha256': digest(question_bytes),
        'question_bytes_identical': True,
        'questions': len(rows), 'unique_scenes': len(completed),
        'png_files': len(image_records),
        'changed_png_files': sum(r['source_sha256'] != r['output_sha256'] for r in image_records),
        'scene_fields_verified': list(SCENE_FIELDS),
        'render_changes': {'thread_guide': False, 'bead_opacity': 0.9, 'shadow_plane_depth_write': False},
        'legacy_framing_preserved': True,
        'copied_metadata_note': 'Original metadata and summary retained byte-for-byte; this report records the new rendering.',
        'model_answers_note': 'Old model_answer.jsonl remains in the source directory; it does not evaluate these new images.',
        'images': image_records,
    }
    (output / 'render_manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'images'}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=4)
    asyncio.run(run(parser.parse_args()))
