import torchvision.transforms as T

def transform(params, to_tensor=True):
    transform_list = []

    if to_tensor:
        transform_list.append(T.ToTensor())

    if params.resize:
        transform_list.append(T.Resize(params.size, T.InterpolationMode.BICUBIC))

    if params.grayscale:
        transform_list.append(T.Grayscale(num_output_channels=1))

    if params.normalize:
        if params.grayscale:
            transform_list.append(T.Normalize((0.5,), (0.5,)))
        else:
            transform_list.append(T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    return T.Compose(transform_list)