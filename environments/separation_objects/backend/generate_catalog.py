from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[2]
OBJECT_ROOT = ROOT / "object_dataset"
PARTS_ROOT = OBJECT_ROOT / "parts"
MASK_ROOT = OBJECT_ROOT / "mask"
LINE_ROOT = OBJECT_ROOT / "line_seg"
PDF_ROOT = OBJECT_ROOT / "pdfs"
DATA_JSON = OBJECT_ROOT / "main_data.json"
OUTPUT_PATH = ROOT / "separation_objects" / "frontend" / "catalog.json"


def is_connected_graph(part_count: int, edges: List[List[int]]) -> bool:
    if part_count <= 0:
        return False
    if part_count == 1:
        return True

    adjacency = [set() for _ in range(part_count)]
    for edge in edges:
        if not isinstance(edge, list) or len(edge) < 2:
            continue
        left = int(edge[0])
        right = int(edge[1])
        if left < 0 or right < 0 or left >= part_count or right >= part_count:
            continue
        adjacency[left].add(right)
        adjacency[right].add(left)

    seen = {0}
    stack = [0]
    while stack:
        node = stack.pop()
        for neighbor in adjacency[node]:
            if neighbor in seen:
                continue
            seen.add(neighbor)
            stack.append(neighbor)
    return len(seen) == part_count


def load_metadata() -> Dict[Tuple[str, str], Dict[str, Any]]:
    rows = json.loads(DATA_JSON.read_text(encoding="utf-8"))
    mapping: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        mapping[(row["category"], row["name"])] = row
    return mapping


def build_catalog() -> Dict[str, Any]:
    metadata = load_metadata()
    categories: Dict[str, List[Dict[str, Any]]] = {}

    for category_dir in sorted(path for path in PARTS_ROOT.iterdir() if path.is_dir()):
        entries: List[Dict[str, Any]] = []
        for object_dir in sorted(path for path in category_dir.iterdir() if path.is_dir()):
            key = (category_dir.name, object_dir.name)
            row = metadata.get(key, {})
            part_files = sorted(path.name for path in object_dir.glob("*.obj"))
            mask_files = sorted(path.name for path in (MASK_ROOT / category_dir.name / object_dir.name).glob("step_*_mask.png"))
            line_files = sorted(path.name for path in (LINE_ROOT / category_dir.name / object_dir.name).glob("step_*.svg"))
            pdf_files = sorted(path.name for path in (PDF_ROOT / category_dir.name / object_dir.name).glob("*.pdf"))
            connection_relation = row.get("connection_relation", [])
            part_count = len(part_files)

            entry = {
                "category": category_dir.name,
                "name": object_dir.name,
                "part_count": part_count,
                "parts_ct_json": row.get("parts_ct"),
                "steps_count": len(row.get("steps", [])),
                "pdf_count": len(pdf_files),
                "assembly_tree": row.get("assembly_tree"),
                "connection_relation": connection_relation,
                "connection_relation_count": len(connection_relation),
                "is_graph_connected": is_connected_graph(part_count, connection_relation),
                "part_dir": f"/object_dataset/parts/{category_dir.name}/{object_dir.name}",
                "parts": [
                    f"/object_dataset/parts/{category_dir.name}/{object_dir.name}/{filename}"
                    for filename in part_files
                ],
                "part_files": part_files,
                "mask_files": mask_files,
                "line_files": line_files,
                "pdfs": [
                    f"/object_dataset/pdfs/{category_dir.name}/{object_dir.name}/{filename}"
                    for filename in pdf_files
                ],
            }
            entries.append(entry)

        categories[category_dir.name] = entries

    return {
        "source": "object_dataset",
        "object_count": sum(len(entries) for entries in categories.values()),
        "categories": categories,
    }


def main() -> None:
    catalog = build_catalog()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote catalog to {OUTPUT_PATH}")
    print(f"Objects: {catalog['object_count']}")
    print(f"Categories: {', '.join(sorted(catalog['categories'].keys()))}")


if __name__ == "__main__":
    main()
