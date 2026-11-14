"""Shared activation boundary for tail-only denoising experiments."""


def tail_steps(backend):
    count=getattr(backend,'settings',{}).get('tail_steps',5)
    if isinstance(count,bool) or int(count)!=count or not 1<=count<len(backend.grid):
        raise ValueError('tail_steps must be an integer between 1 and the denoise step count')
    return int(count)


def tail_start(backend):
    return len(backend.grid)-1-tail_steps(backend)
