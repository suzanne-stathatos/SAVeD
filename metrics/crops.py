
import numpy as np

def get_noise_crops_not_in_bboxes(crop_h, crop_w, gt_img, pred_img, bboxes, num_locations=10):
    # Create mask using vectorized operations
    h, w = gt_img.shape[:2]
    valid_mask = np.ones((h, w), dtype=bool)
    
    # Exclude edges efficiently
    valid_mask[-crop_h:, :] = 0
    valid_mask[:, -crop_w:] = 0
    
    # Vectorized bbox exclusion
    for x1, y1, w, h in bboxes:
        x1, y1, w, h = map(int, [x1+1, y1+1, w, h])
        x1_expanded = max(0, x1 - crop_w)
        y1_expanded = max(0, y1 - crop_h)
        x2_expanded = min(gt_img.shape[1], x1 + w + crop_w)
        y2_expanded = min(gt_img.shape[0], y1 + h + crop_h)
        valid_mask[y1_expanded:y2_expanded, x1_expanded:x2_expanded] = 0

    valid_y, valid_x = np.nonzero(valid_mask)
    
    if len(valid_y) < num_locations:
        num_samples = len(valid_y)
    else:
        num_samples = num_locations
    
    if num_samples == 0:
        return [], [], []

    # Vectorized crop extraction
    indices = np.random.choice(len(valid_y), size=num_samples, replace=False)
    y_coords = valid_y[indices]
    x_coords = valid_x[indices]
    
    # Extract all crops at once using advanced indexing
    gt_crops = []
    pred_crops = []
    random_locations = []
    
    for y, x in zip(y_coords, x_coords):
        gt_crop = gt_img[y:y+crop_h, x:x+crop_w].astype(np.float32)
        pred_crop = pred_img[y:y+crop_h, x:x+crop_w].astype(np.float32)
        
        # Normalize crops efficiently
        if gt_crop.max() > 1.0:
            gt_crop /= 255.0
        if pred_crop.max() > 1.0:
            pred_crop /= 255.0
        
        gt_crop = np.clip(gt_crop, 0.0, 1.0)
        pred_crop = np.clip(pred_crop, 0.0, 1.0)

        if np.any(gt_crop): # Make sure crop is within the cone
            gt_crops.append(gt_crop)
            pred_crops.append(pred_crop)
            random_locations.append((y, x))
    
    return gt_crops, pred_crops, random_locations


def get_crops ( gt_img, pred_img, bboxes, num_noise_crops_per_bbox=1):
    # Preallocate lists
    gt_crops = []
    pred_crops = []
    gt_noise_crops_list = []
    pred_noise_crops_list = []
    random_locations_list = []
    
    # Process all bboxes
    for x1, y1, w, h in bboxes:
        if int(w) <= 1 or int(h) <= 1:
            continue
        
        # Extract and normalize crops
        y1, x1, h, w = map(int, [y1+1, x1+1, h, w])
        gt_crop = gt_img[y1:y1+h, x1:x1+w].astype(np.float32)
        pred_crop = pred_img[y1:y1+h, x1:x1+w].astype(np.float32)
        
        if gt_crop.max() > 1.0:
            gt_crop /= 255.0
        if pred_crop.max() > 1.0:
            pred_crop /= 255.0
        
        gt_crop = np.clip(gt_crop, 0.0, 1.0)
        pred_crop = np.clip(pred_crop, 0.0, 1.0)
        
        gt_crops.append(gt_crop)
        pred_crops.append(pred_crop)
        
        # Get noise crops
        gt_noise_crops, pred_noise_crops, random_locations = get_noise_crops_not_in_bboxes(h, w, gt_img, pred_img, bboxes, num_locations=num_noise_crops_per_bbox)
        if len(gt_noise_crops) > 0:
            gt_noise_crops_list.append(gt_noise_crops)
            pred_noise_crops_list.append(pred_noise_crops)
            random_locations_list.append(random_locations)

    return gt_crops, pred_crops, gt_noise_crops_list, pred_noise_crops_list, random_locations_list