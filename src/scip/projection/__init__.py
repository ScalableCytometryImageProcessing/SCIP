
def project_block_partition(part, proj, **proj_kw):
    return [proj(p, **proj_kw) for p in part]
