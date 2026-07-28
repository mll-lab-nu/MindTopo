# IKEA-Manual Dataset (TopoBench subset)

This is the minimal subset of the upstream IKEA-Manual dataset that the
`separation_objects` task actually uses. The upstream release additionally ships
`pdfs/`, `code/`, `line_seg/`, and `seg/`; those are not needed at runtime and
are not included in this snapshot.

## Dataset Structure
```
- README.md
- main_data.json   # Core data json (source metadata for the catalog)
- parts            # Assembly part decomposition (.obj meshes rendered by the frontend)
```

## main_data.json File Structure
`main_data.json` contains annotations for each objects in the following format:
```
{
    // Object metadata.
    'category': 'Bench',
    'name': 'applaro',

     // Annotations for each step.
    'steps': [...],

     // Connection relation between primitive assembly parts.
    'connection_relation': [...],

     // Geometric equivalence relation between primitive assembly parts.
    'geometric_equivalence_relation': [...],

     // Tree-structured assembly plan in list format.
    'assembly_tree': [...],

     // Number of primitive assembly parts.
    'parts_ct': 4,
    }
```

The `steps` field contains a list of step annotation in the following format:

```
{
    // List of involved assembly parts in this step.
    'parts': ['0,1,2', '3']

    // List of pairwise connection relationships.
    'connections': [['0,1,2', '3']]

    // The page where this step is in.
    'page_id': 4,

    // The step id.
    'step_id': 1,

    // A list of masks for each assembly parts.
    // Use pycocotools.masks.decode to decode each mask
    // into an numpy array.
    'masks': masks,

    // A list of intrinsic matrices for each assembly part.
    'intrinsics': [...],

    // A list of extrinsic matrices for each assembly part.
    'extrinsics': [...],

    // The split for the segmentation task.
    'part_segmentation_split': 'test',

    // The index of the step in the whole dataset.
    'step_id_global': 10,
}
```
