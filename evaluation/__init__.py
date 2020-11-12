from evaluation.coco_evaler import COCOEvaler
from evaluation.AIC_evaler import AICEvaler

__factory = {
    'COCO': COCOEvaler,
    'AIC': AICEvaler
}

def names():
    return sorted(__factory.keys())

def create(name, *args, **kwargs):
    if name not in __factory:
        raise KeyError("Unknown Evaler:", name)
    return __factory[name](*args, **kwargs)