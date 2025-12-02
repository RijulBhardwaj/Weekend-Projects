"""
Standalone demo script to test filters without Streamlit
"""

import cv2
import numpy as np
from rnn_filter_model import ImageFilterPresets
import matplotlib.pyplot as plt

def demo_all_filters(image_path):
    """
    Apply all filters to an image and display results
    
    Args:
        image_path: Path to input image
    """
    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Initialize filter presets
    filters = ImageFilterPresets()
    
    # Apply all filters
    results = {
        'Original': image,
        'Sketch': filters.apply_sketch_filter(image),
        'Watercolor': filters.apply_watercolor_filter(image),
        'Oil Painting': filters.apply_oil_painting_filter(image),
        'Cartoon': filters.apply_cartoon_filter(image),
        'Vintage': filters.apply_vintage_filter(image),
        'Edge Enhanced': filters.apply_edge_enhancement(image),
        'Emboss': filters.apply_emboss_filter(image)
    }
    
    # Display results
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, (name, img) in enumerate(results.items()):
        axes[idx].imshow(img)
        axes[idx].set_title(name, fontsize=12, fontweight='bold')
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig('filter_results.png', dpi=150, bbox_inches='tight')
    print("✅ Results saved to 'filter_results.png'")
    plt.show()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python demo_filters.py <image_path>")
        print("Example: python demo_filters.py sample.jpg")
    else:
        demo_all_filters(sys.argv[1])
