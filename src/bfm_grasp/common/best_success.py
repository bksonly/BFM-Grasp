"""Selection policy for the training-success checkpoint (not independent eval)."""
import math

def improves(candidate, incumbent):
    if candidate is None:return False
    if not math.isfinite(candidate['macro_rate']):raise ValueError('Nonfinite success rate')
    if incumbent is not None:
        if candidate['metric']!=incumbent['metric'] or candidate['object_names']!=incumbent['object_names']:
            raise ValueError('Best-checkpoint metric/object set changed; use a new output directory')
    return incumbent is None or candidate['macro_rate']>incumbent['macro_rate']
