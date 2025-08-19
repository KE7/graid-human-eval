"""
Image visualization utilities for GRAID human evaluation.

This module provides functionality to create side-by-side image displays with
bounding box overlays, category labels, and confidence scores for human evaluation.
"""

import logging
from typing import Dict, List, Any, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.colors as mcolors

logger = logging.getLogger(__name__)


class VisualizationUtils:
    """
    Utilities for creating annotated images with bounding box overlays.
    
    This class handles the creation of side-by-side image displays showing
    the original image and the same image with bounding box annotations,
    category labels, and confidence scores overlaid.
    """
    
    def __init__(self):
        """Initialize visualization utilities with color mapping."""
        self.category_colors = {}
        self._setup_colors()
        
    def _setup_colors(self):
        """Setup distinct colors for object categories."""
        # Use matplotlib's tab20 colormap for distinct colors
        colors = list(mcolors.TABLEAU_COLORS.values())
        # Add more colors if needed
        colors.extend(list(mcolors.CSS4_COLORS.values())[:20])
        self.base_colors = colors
        
    def _get_category_color(self, category: str) -> str:
        """Get consistent color for a category."""
        if category not in self.category_colors:
            # Use hash of category name to get consistent color
            color_idx = hash(category) % len(self.base_colors)
            self.category_colors[category] = self.base_colors[color_idx]
        return self.category_colors[category]
    
    def create_side_by_side_display(self, sample: Dict[str, Any]) -> Tuple[Image.Image, Image.Image]:
        """
        Create side-by-side display of original and annotated images.
        
        Args:
            sample: Dataset sample containing image and annotations
            
        Returns:
            Tuple of (original_image, annotated_image)
        """
        try:
            # Get the original image
            original_image = sample['image']
            if not isinstance(original_image, Image.Image):
                # Convert if needed (e.g., from PIL format in dataset)
                original_image = Image.fromarray(np.array(original_image))
            
            # Ensure RGB mode
            if original_image.mode != 'RGB':
                original_image = original_image.convert('RGB')
            
            # Create annotated version
            annotated_image = self._draw_annotations(original_image.copy(), sample['annotations'])
            
            return original_image, annotated_image
            
        except Exception as e:
            logger.error(f"Failed to create image display: {e}")
            # Return placeholder images on error
            placeholder = Image.new('RGB', (400, 300), color='lightgray')
            return placeholder, placeholder
    
    def _draw_annotations(self, image: Image.Image, annotations: List[Dict[str, Any]]) -> Image.Image:
        """
        Draw bounding box annotations on image.
        
        Args:
            image: PIL Image to annotate
            annotations: List of COCO-style annotation dictionaries
            
        Returns:
            Annotated PIL Image
        """
        if not annotations:
            return image
        
        draw = ImageDraw.Draw(image)
        
        # Try to load a font, fallback to default
        try:
            # Try to use a reasonable font size based on image dimensions
            font_size = max(12, min(image.width, image.height) // 40)
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
        except (OSError, IOError):
            try:
                font = ImageFont.load_default()
            except:
                font = None
        
        for annotation in annotations:
            try:
                # Extract bounding box (COCO format: [x, y, width, height])
                bbox = annotation['bbox']
                x, y, w, h = bbox
                
                # Convert to corner coordinates
                x1, y1 = int(x), int(y)
                x2, y2 = int(x + w), int(y + h)
                
                # Get category and color
                category = annotation.get('category', 'unknown')
                color = self._get_category_color(category)
                
                # Draw bounding box
                box_width = max(2, min(image.width, image.height) // 200)
                draw.rectangle([x1, y1, x2, y2], outline=color, width=box_width)
                
                # Prepare label text
                confidence = annotation.get('score', 1.0)
                if confidence < 1.0:
                    label_text = f"{category} ({confidence:.2f})"
                else:
                    label_text = category
                
                # Draw label background and text
                if font:
                    # Get text bounding box
                    bbox_text = draw.textbbox((0, 0), label_text, font=font)
                    text_width = bbox_text[2] - bbox_text[0]
                    text_height = bbox_text[3] - bbox_text[1]
                else:
                    # Estimate text size for default font
                    text_width = len(label_text) * 8
                    text_height = 12
                
                # Position label (above box if space, otherwise inside)
                label_x = x1
                label_y = y1 - text_height - 4 if y1 > text_height + 4 else y1 + 2
                
                # Ensure label stays within image bounds
                label_x = max(0, min(label_x, image.width - text_width - 4))
                label_y = max(0, min(label_y, image.height - text_height - 4))
                
                # Draw label background
                bg_color = self._lighten_color(color, 0.8)
                draw.rectangle(
                    [label_x - 2, label_y - 2, label_x + text_width + 2, label_y + text_height + 2],
                    fill=bg_color,
                    outline=color,
                    width=1
                )
                
                # Draw label text
                text_color = self._get_text_color(bg_color)
                if font:
                    draw.text((label_x, label_y), label_text, fill=text_color, font=font)
                else:
                    draw.text((label_x, label_y), label_text, fill=text_color)
                    
            except Exception as e:
                logger.warning(f"Failed to draw annotation {annotation}: {e}")
                continue
        
        return image
    
    def _lighten_color(self, color: str, factor: float = 0.7) -> str:
        """Lighten a color for background use."""
        try:
            # Convert color to RGB
            if color.startswith('#'):
                color = color[1:]
            
            # Parse hex color
            if len(color) == 6:
                r = int(color[:2], 16)
                g = int(color[2:4], 16)
                b = int(color[4:6], 16)
            else:
                # Named color - convert via matplotlib
                rgb = mcolors.to_rgb(color)
                r, g, b = [int(c * 255) for c in rgb]
            
            # Lighten by blending with white
            r = int(r + (255 - r) * factor)
            g = int(g + (255 - g) * factor)
            b = int(b + (255 - b) * factor)
            
            return f"#{r:02x}{g:02x}{b:02x}"
            
        except Exception:
            return "#f0f0f0"  # Light gray fallback
    
    def _get_text_color(self, bg_color: str) -> str:
        """Get appropriate text color (black/white) for background."""
        try:
            # Convert background color to RGB
            if bg_color.startswith('#'):
                color = bg_color[1:]
                r = int(color[:2], 16)
                g = int(color[2:4], 16)
                b = int(color[4:6], 16)
            else:
                # Fallback
                return "black"
            
            # Calculate luminance
            luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
            
            # Return black for light backgrounds, white for dark
            return "black" if luminance > 0.5 else "white"
            
        except Exception:
            return "black"  # Safe fallback
    
    def get_annotation_summary(self, annotations: List[Dict[str, Any]]) -> str:
        """
        Get a text summary of annotations for display.
        
        Args:
            annotations: List of annotation dictionaries
            
        Returns:
            Human-readable summary string
        """
        if not annotations:
            return "No objects detected"
        
        # Count objects by category
        category_counts = {}
        for ann in annotations:
            category = ann.get('category', 'unknown')
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Create summary
        total_objects = len(annotations)
        unique_categories = len(category_counts)
        
        summary_parts = [f"{total_objects} objects detected"]
        
        if unique_categories <= 5:
            # List all categories if not too many
            category_list = [f"{count} {cat}" for cat, count in category_counts.items()]
            summary_parts.append("(" + ", ".join(category_list) + ")")
        else:
            summary_parts.append(f"({unique_categories} different categories)")
        
        return " ".join(summary_parts)
