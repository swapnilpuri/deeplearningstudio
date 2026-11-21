"""Framework-neutral, JSON-safe CNN architecture specification (v2).

Does not modify or reinterpret the v4 `ModelSpec` (v1) format.
"""
from dataclasses import dataclass, asdict
from typing import Literal
import math

Kind = Literal['conv2d', 'relu', 'maxpool2d', 'avgpool2d', 'flatten', 'linear']


def pair(value):
    if isinstance(value, int) and not isinstance(value, bool):
        return (value, value)
    if isinstance(value, (list, tuple)) and len(value) == 2 and all(isinstance(v, int) and not isinstance(v, bool) for v in value):
        return tuple(value)
    raise ValueError('Expected integer or (height, width) integer pair')


def output_axis(size: int, kernel: int, stride: int = 1, padding: int = 0, dilation: int = 1) -> int:
    if min(size, kernel, stride, dilation) < 1 or padding < 0:
        raise ValueError('Positive dimensions, kernel, stride, dilation and nonnegative padding required')
    result = (size + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1
    if result < 1:
        raise ValueError('Kernel/window is too large for the padded input')
    return result


@dataclass(frozen=True)
class CNNLayerSpec:
    kind: Kind
    out_channels: int | None = None
    out_features: int | None = None
    kernel_size: tuple[int, int] = (3, 3)
    stride: tuple[int, int] = (1, 1)
    padding: tuple[int, int] = (0, 0)
    dilation: tuple[int, int] = (1, 1)
    bias: bool = True

    def __post_init__(self):
        if self.kind not in ('conv2d','relu','maxpool2d','avgpool2d','flatten','linear'):
            raise ValueError('Unknown CNN operator')
        for field in ('kernel_size','stride','padding','dilation'):
            value = pair(getattr(self, field))
            object.__setattr__(self, field, value)
            if any(v < (0 if field == 'padding' else 1) for v in value):
                raise ValueError(f'Invalid {field}')
        if self.kind == 'conv2d' and (not isinstance(self.out_channels,int) or isinstance(self.out_channels,bool) or not 1 <= self.out_channels <= 128):
            raise ValueError('Convolution needs 1..128 output channels')
        if self.kind == 'linear' and (not isinstance(self.out_features,int) or isinstance(self.out_features,bool) or not 1 <= self.out_features <= 1024):
            raise ValueError('Linear needs 1..1024 output features')
        if self.kind in ('maxpool2d', 'avgpool2d') and self.padding != (0,0):
            raise ValueError('Pooling v5 supports zero padding')
        if self.kind in ('maxpool2d', 'avgpool2d') and self.dilation != (1,1):
            raise ValueError('Pooling v5 supports dilation=1')
        if self.kind == 'conv2d' and self.out_features is not None:
            raise ValueError('Conv2d cannot set out_features')

    def as_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)


@dataclass(frozen=True)
class CNNModelSpec:
    input_channels: int
    image_height: int
    image_width: int
    layers: tuple[CNNLayerSpec, ...]
    seed: int = 42
    framework: str = 'pytorch'
    schema_version: int = 2
    architecture: str = 'cnn'

    def __post_init__(self):
        if self.schema_version != 2 or self.architecture != 'cnn':
            raise ValueError('Unsupported CNN schema version/architecture')
        if any(not isinstance(v,int) or isinstance(v,bool) for v in (self.input_channels,self.image_height,self.image_width)) or self.input_channels not in (1,3) or not 4 <= self.image_height <= 128 or not 4 <= self.image_width <= 128:
            raise ValueError('CNN input must be 1/3 channels and spatial dimensions 4..128')
        if not 1 <= len(self.layers) <= 32:
            raise ValueError('CNN needs 1..32 operators')
        if not isinstance(self.seed,int) or isinstance(self.seed,bool) or not 0 <= self.seed <= 2147483647:
            raise ValueError('Invalid seed')
        if self.layers[-1].kind != 'linear':
            raise ValueError('The final operation must emit classification logits (linear)')
        self.infer_shapes()

    def infer_shapes(self):
        shape = (1, self.input_channels, self.image_height, self.image_width)
        rows = [{'index': 0, 'operator':'Input', 'shape': shape, 'parameters':0, 'weights':None}]
        for i, layer in enumerate(self.layers,1):
            old = shape
            count = 0; weight_shape = None
            if layer.kind == 'conv2d':
                assert layer.out_channels is not None  # validated by CNNLayerSpec
                if len(shape) != 4: raise ValueError('Conv2d expects NCHW input')
                n, cin, h, w = shape
                kh, kw = layer.kernel_size; sh,sw=layer.stride; ph,pw=layer.padding; dh,dw=layer.dilation
                shape = (n, layer.out_channels,output_axis(h,kh,sh,ph,dh),output_axis(w,kw,sw,pw,dw))
                weight_shape = (layer.out_channels,cin,kh,kw)
                count = math.prod(weight_shape) + (layer.out_channels if layer.bias else 0)
            elif layer.kind in ('maxpool2d','avgpool2d'):
                if len(shape) != 4: raise ValueError('Pooling expects NCHW input')
                n, c,h,w=shape; kh,kw=layer.kernel_size;sh,sw=layer.stride
                shape = (n,c,output_axis(h,kh,sh),output_axis(w,kw,sw))
            elif layer.kind == 'relu':
                pass
            elif layer.kind == 'flatten':
                if len(shape) != 4: raise ValueError('Flatten expects NCHW input')
                shape = (shape[0],math.prod(shape[1:]))
            elif layer.kind == 'linear':
                assert layer.out_features is not None  # validated by CNNLayerSpec
                if len(shape) != 2: raise ValueError('Linear requires Flatten first')
                weight_shape = (layer.out_features, shape[1])
                count = math.prod(weight_shape)+(layer.out_features if layer.bias else 0)
                shape = (shape[0],layer.out_features)
            rows.append({'index':i,'operator':layer.kind,'input_shape':old,'shape':shape,'parameters':count,'weights':weight_shape,
                         'bias':(weight_shape[0],) if weight_shape and layer.bias else None})
        return rows

    def as_dict(self):
        return dict(schema_version=2, architecture='cnn', framework=self.framework, seed=self.seed,
                    input_channels=self.input_channels, image_height=self.image_height, image_width=self.image_width,
                    layers=[l.as_dict() for l in self.layers])

    @classmethod
    def from_dict(cls,data):
        if data.get('schema_version')!=2 or data.get('architecture')!='cnn':
            raise ValueError('Expected v2 CNN specification; v1 dense specs must use ModelSpec.from_dict')
        return cls(input_channels=data['input_channels'],image_height=data['image_height'],image_width=data['image_width'],
                   layers=tuple(CNNLayerSpec.from_dict(x) for x in data['layers']),
                   seed=data.get('seed',42),framework=data.get('framework','pytorch'))


def digits_spec(filters1=8, filters2=16, pooling='maxpool2d', seed=42):
    return CNNModelSpec(1,8,8,(
        CNNLayerSpec('conv2d',out_channels=filters1,padding=(1,1)), CNNLayerSpec('relu'),
        CNNLayerSpec(pooling,kernel_size=(2,2),stride=(2,2)),
        CNNLayerSpec('conv2d',out_channels=filters2,padding=(1,1)), CNNLayerSpec('relu'),
        CNNLayerSpec(pooling,kernel_size=(2,2),stride=(2,2)),
        CNNLayerSpec('flatten'),CNNLayerSpec('linear',out_features=10)
    ), seed=seed)
