from scip.segmentation import mask_apply

def apply_mask_partition(part):
    return [mask_apply.apply(p, "intermediate") for p in part]
