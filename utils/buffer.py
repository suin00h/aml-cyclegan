import random
import torch

class ImageBuffer:
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.buffer = []

    def sample(self, images):
        '''
        Second, to reduce model oscillation [15], we follow Shrivastava et al.’s strategy [46]
        and update the discriminators using a history of generated images rather than the ones
        produced by the latest generators. We keep an image buffer that stores the 50 previously created images.
        '''

        return_images = []

        # mini-batch
        for image in images:
            image = image.unsqueeze(0)
            # if buffer is not full
            if len(self.buffer) < self.max_size:
                self.buffer.append(image)
                return_images.append(image)

            # if buffer is full
            else:
                if random.uniform(0, 1) > 0.5:
                    # use previously stored image
                    idx = random.randint(0, self.max_size - 1)
                    old_image = self.buffer[idx].clone()
                    self.buffer[idx] = image
                    return_images.append(old_image)
                else:
                    # use current image
                    return_images.append(image)

        return torch.cat(return_images, dim=0)