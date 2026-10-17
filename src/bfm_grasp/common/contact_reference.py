"""Canonical ten-body contact schema shared by PPO and expert export."""
import numpy as np

LINKS = tuple(f'{finger}_{part}' for finger in ('index','middle','ring','pinky','thumb') for part in ('PP','DP'))

def load_direct_contacts(file, frames, objects, hands):
    names = [x.decode() if isinstance(x, bytes) else str(x) for x in file['contacts/link_names'][:]]
    if names != list(LINKS):
        raise ValueError(f'Incorrect ten-body contact order: {names}')
    flags = np.stack([file[f'contacts/{h}/is_contact'][:] for h in range(hands)], axis=2)
    points = np.stack([file[f'contacts/{h}/points'][:] for h in range(hands)], axis=2)
    if flags.shape != (frames, objects, hands, 10) or points.shape != (*flags.shape, 3):
        raise ValueError(f'Invalid contact shapes: {flags.shape}, {points.shape}')
    if not np.isin(flags, [0,1]).all() or not np.isfinite(points[flags.astype(bool)]).all():
        raise ValueError('Invalid contact values')
    return flags.astype(np.uint8), np.where(flags[...,None],points,0).astype(np.float32)
