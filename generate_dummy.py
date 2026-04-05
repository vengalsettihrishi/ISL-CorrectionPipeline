import cv2
import numpy as np
import os

classes = ['Hello', 'ThankYou']
base_dir = './dummy_videos'

os.makedirs(base_dir, exist_ok=True)

for cls in classes:
    cls_dir = os.path.join(base_dir, cls)
    os.makedirs(cls_dir, exist_ok=True)
    
    for i in range(10): # 10 videos per class
        filepath = os.path.join(cls_dir, f'video_{i:03d}.mp4')
        
        # Create a VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filepath, fourcc, 30.0, (640, 480))
        
        # Generate 30 frames
        for _ in range(30):
            frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            out.write(frame)
            
        out.release()
print(f"Generated dummy dataset at {base_dir}")
