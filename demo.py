from segment_anything import sam_model_registry
from sam_refiner import sam_refiner
import numpy as np
from PIL import Image

if __name__ == "__main__":
    sam_checkpoint = "/path/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    print(sam.device)
    
    image_path = 'examples/2007_000256.jpg'
    mask_path = 'examples/2007_000256_init_mask.png'
    init_masks = np.asarray(Image.open(mask_path), dtype=np.uint8)
    
    if np.max(init_masks) == 255:
        init_masks = init_masks / 255
    
    refined_masks = sam_refiner(image_path, 
                [init_masks],
                sam)[0]
                                
    print(refined_masks.shape)

    Image.fromarray(255*refined_masks[0].astype(np.uint8)).save('examples/2007_000256_refined_mask.png')