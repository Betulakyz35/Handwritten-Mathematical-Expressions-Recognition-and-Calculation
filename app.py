import streamlit as st
import cv2
import numpy as np
import sympy as sp
import os
from PIL import Image, ImageDraw
import io
import tempfile

import base64

from streamlit_vertical_slider import vertical_slider

st.set_page_config(
    page_title="Math OCR",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>

@import url('https://fonts.googleapis.com/css2?family=Noto+Sans&display=swap');

body, .main, .math-expression, .math-solution, .result-container {
    font-family: 'Noto Sans', sans-serif !important;
}

    .main .block-container {
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
        max-width: 1200px;
    }
    .stButton button {
        width: 100%;
        border-radius: 2px;
        height: 2.2rem;
    }
    .upload-header {
        font-size: 1.2rem;
        font-weight: 500;
        margin-bottom: 0.2rem;
        color: #333;
        text-align: center;
    }
    .stImage img {
        border-radius: 2px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        object-fit: contain;
        width: 100%;
        max-height: 500px; /* Limit image height */
    }
    .css-10trblm {
        color: #333;
        font-weight: 500;
    }
    .result-container {
        background-color: #fafafa;
        padding: 0.5rem;
        border-radius: 2px;
        margin-top: 0.2rem;
    }
    .annotation {
        font-size: 0.7rem;
        color: #777;
        margin: 0;
    }
    .stSlider {
        padding-top: 0.1rem;
        padding-bottom: 0.4rem;
    }
    .control-section {
        margin-bottom: 0.5rem;
    }
    .output-panel {
        border-left: 1px solid #f0f0f0;
        padding-left: 0.5rem;
    }
    .image-slider-container {
        display: flex;
        flex-direction: row;
        align-items: center;
        width: 100%;
        margin-bottom: 10px;
        position: relative;
        gap: 10px;
    }
    .slider-column {
        width: 50px;
        height: 300px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        padding: 0 5px;
    }
    .image-column {
        flex-grow: 1;
        position: relative;
    }
    .left-slider-container {
        width: 50px;
        margin-right: 10px;
        height: 300px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .right-slider-container {
        width: 50px;
        margin-left: 10px;
        height: 300px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .main-image-container {
        flex-grow: 1;
        position: relative;
    }
    .slider-label {
        font-size: 0.8rem;
        text-align: center;
        margin-bottom: 0.2rem;
    }

    .output-container [class*="st-emotion-cache-"] {
    gap: 0 !important;
    }

    /* Target the specific container holding the image */
    .image-result [class*="st-emotion-cache-"] {
        gap: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Target the parent containers to remove vertical spacing */
    .output-results > div,
    .output-results > div > div {
        margin-top: 0 !important;
        margin-bottom: 0 !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }

    /* Target the image container specifically */
    .output-container .image-result img,
    .output-container .image-result div {
        margin-top: 0 !important;
        margin-bottom: 0 !important;
    }


    .result-box {
        margin-bottom: 1rem;
        flex: 1;
    }
    .upload-area {
        margin-bottom: 0.5rem;
        text-align: center;
    }
    .output-container {
        border-left: 1px solid #f0f0f0;
        height: 100%;
        padding-left: 1rem;
        margin-top: 0;
    }
    .math-solution {
        font-size: 1.5rem;
        text-align: center;
        padding: 15px;
        background-color: #333333;
        color: white;
        border-radius: 4px;
        margin: 10px 0;
    }
    .math-expression {
        font-size: 1.5rem;
        text-align: center;
        padding: 15px;
        background-color: #333333;
        color: white;
        border-radius: 4px;
        margin: 10px 0;
        white-space: pre-wrap; /* This preserves line breaks */
        font-family: monospace; /* For better alignment of fractions */
    }
    .result-header {
        color: white;
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 10px;
    }

    /* Enhanced checkbox styling - more direct and specific selectors */
    [data-testid="stCheckbox"] {
        margin: 0 !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        height: 100% !important;
    }

    /* Target the checkbox input directly */
    [data-testid="stCheckbox"] input[type="checkbox"] {
        width: 40px !important;
        height: 40px !important;
        min-width: 40px !important;
        min-height: 40px !important;
        transform: scale(2.5) !important;
        margin: 8px !important;
    }

    /* Target the label containing the emoji */
    [data-testid="stCheckbox"] label {
        font-size: 2.8rem !important; 
        padding-left: 20px !important;
        display: flex !important;
        align-items: center !important;
    }

    /* Target the container div */
    [data-testid="stCheckbox"] > div {
        display: flex !important;
        align-items: center !important;
        height: 60px !important;
    }

    /* Force SVG icons to be bigger */
    [data-testid="stCheckbox"] svg {
        width: 32px !important;
        height: 32px !important;
        min-width: 32px !important;
        min-height: 32px !important;
    }

    /* Vertical slider adjustments */
    [data-testid="stVerticalBlock"] > div:has(> [data-testid="stVerticalSlider"]) {
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }

    /* Remove gap from st-emotion-cache-gu2q4u that's affecting the layout */
    .output-container .st-emotion-cache-gu2q4u {
        gap: 0 !important; /* Remove gap in output container */
    }

    /* Preserve gaps in other containers that need them */
    .st-emotion-cache-gu2q4u:not(.output-container .st-emotion-cache-gu2q4u) {
        gap: 1rem; /* Keep gaps for other elements */
    }

    /* Additional checkbox container enhancement */
    .checkbox-container .stCheckbox {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
    }


    [class*="st-emotion-cache-jd5ja5"],
    [class*="st-emotion-cache-gu2q4u"],
    [class*="st-emotion-cache-1ehppsx"],
    [class*="st-emotion-cache-1hernd5"]{
        gap: 0 !important;
    }
    </style>
""", unsafe_allow_html=True)

CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', '+', '×', '/', '(', ')']
SUPERSCRIPT_MAP = {
    '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
    '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
    '+': 'ᐩ', '-': '⁻', '(': '⁽', ')': '⁾', '×': '*', '/': 'ᐟ'
}
colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Initialize session state variables
if 'image' not in st.session_state:
    st.session_state.image = None
if 'rotation' not in st.session_state:
    st.session_state.rotation = 0
if 'left' not in st.session_state:
    st.session_state.left = 0
if 'right' not in st.session_state:
    st.session_state.right = 0
if 'top' not in st.session_state:
    st.session_state.top = 0
if 'bottom' not in st.session_state:
    st.session_state.bottom = 0
if 'width' not in st.session_state:
    st.session_state.width = 0
if 'height' not in st.session_state:
    st.session_state.height = 0
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'processed' not in st.session_state:
    st.session_state.processed = False
if 'result_image' not in st.session_state:
    st.session_state.result_image = None
if 'expression' not in st.session_state:
    st.session_state.expression = None
if 'solution' not in st.session_state:
    st.session_state.solution = None
if 'cropped_image' not in st.session_state:
    st.session_state.cropped_image = None
if 'current_file' not in st.session_state:
    st.session_state.current_file = None
if 'vertical_top' not in st.session_state:
    st.session_state.vertical_top = 100
if 'vertical_bottom' not in st.session_state:
    st.session_state.vertical_bottom = 0
if 'horizontal_crop' not in st.session_state:
    st.session_state.horizontal_crop = [0, 100]
if 'display_image' not in st.session_state:
    st.session_state.display_image = None
if 'use_horizontal_vertical_crop' not in st.session_state:
    st.session_state.use_horizontal_vertical_crop = False


def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = f"{CLASSES[class_id]} ({confidence:.2f})"
    color = colors[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def process_single_line(detections):
    """Processes a single line of characters with properly handled grouped exponents"""
    sorted_detections = sorted(detections, key=lambda x: x['box'][0])
    original_expr = []
    display_expr = []
    exponent_buffer = []
    baseline_y = None
    baseline_height = None
    exponent_mode = False
    parenthesis_count = 0

    i = 0
    while i < len(sorted_detections):
        det = sorted_detections[i]
        x, y, w, h = det['box']
        char = det['class_name']
        scale = det['scale']

        # Convert normalized coordinates to pixel values
        y_center = (y + h / 2) * scale
        h_pixel = h * scale

        if not baseline_y:
            baseline_y = y_center
            baseline_height = h_pixel
            original_expr.append(char)
            display_expr.append(char)
            i += 1
            continue

        vertical_offset = baseline_y - y_center
        size_ratio = h_pixel / baseline_height
        is_operator = char in ['-', '+', '×', '/']

        # Horizontal proximity check
        prev_x = sorted_detections[i - 1]['box'][0] * scale if i > 0 else 0
        prev_w = sorted_detections[i - 1]['box'][2] * scale if i > 0 else 0
        horizontal_gap = (x * scale) - (prev_x + prev_w)

        # Exponent detection logic
        is_new_exponent = False
        if is_operator:
            is_new_exponent = (vertical_offset > baseline_height * 0.7 and
                               horizontal_gap > prev_w * 0.1)
        else:
            is_new_exponent = (vertical_offset > baseline_height * 0.7 and
                               size_ratio < 0.7)

        if is_new_exponent:
            # We have an exponent
            temp_exponents = []
            exponent_start_index = i

            # Collect all consecutive exponents
            while i < len(sorted_detections):
                current_det = sorted_detections[i]
                current_y = (current_det['box'][1] + current_det['box'][3] / 2) * current_det['scale']
                current_h = current_det['box'][3] * current_det['scale']
                current_offset = baseline_y - current_y
                current_size_ratio = current_h / baseline_height

                # Check if this is still an exponent
                is_still_exponent = False
                if current_det['class_name'] in ['-', '+', '×', '/']:
                    is_still_exponent = (current_offset > baseline_height * 0.7)
                else:
                    is_still_exponent = (current_offset > baseline_height * 0.7 and
                                         current_size_ratio < 0.7)

                if not is_still_exponent:
                    break

                temp_exponents.append(current_det['class_name'])
                i += 1

            # Add exponent to expression
            if temp_exponents:
                exponent_str = ''.join(temp_exponents)

                # Simple case: Single character exponent
                if len(temp_exponents) == 1:
                    original_expr.append(f"^{exponent_str}")
                    display_expr.append(SUPERSCRIPT_MAP.get(exponent_str, exponent_str))
                # Multiple characters: wrap in parentheses
                else:
                    original_expr.append(f"^({exponent_str})")
                    display_expr.append(''.join(SUPERSCRIPT_MAP.get(c, c) for c in temp_exponents))

            continue

        # Normal character
        original_expr.append(char)
        display_expr.append(char)

        # Update baseline with smoothing
        baseline_y = y_center * 0.3 + baseline_y * 0.7
        baseline_height = h_pixel * 0.4 + baseline_height * 0.6

        i += 1

    return ''.join(original_expr), ''.join(display_expr)


def detect_vertical_groups(detections, threshold_factor=1.6):
    if len(detections) < 2:
        return [detections]


    positions = []
    horizontal_spans = []

    for det in detections:
        x = det['box'][0] * det['scale']
        y = det['box'][1] * det['scale']
        w = det['box'][2] * det['scale']
        h = det['box'][3] * det['scale']
        positions.append((y, y + h))
        horizontal_spans.append((x, x + w))

    # Calculate character sizes and positions
    y_centers = [(y1 + y2) / 2 for y1, y2 in positions]
    heights = [y2 - y1 for y1, y2 in positions]
    avg_height = np.mean(heights)

    # Sort detections by vertical position
    sorted_indices = np.argsort(y_centers)
    sorted_y = [y_centers[i] for i in sorted_indices]

    # Find all significant vertical gaps
    gaps = []
    for i in range(len(sorted_y) - 1):
        gap = sorted_y[i + 1] - sorted_y[i]
        if gap > avg_height * threshold_factor:
            gaps.append((i, gap))

    if not gaps:
        return [detections]

    # Sort gaps by size (largest first)
    gaps.sort(key=lambda x: x[1], reverse=True)

    # Get horizontal overlap between potential groups to distinguish fractions from exponents
    def get_group_overlap(idx1, idx2):
        group1_xmin = min(horizontal_spans[i] for i in idx1)[0]
        group1_xmax = max(horizontal_spans[i] for i in idx1)[1]
        group2_xmin = min(horizontal_spans[i] for i in idx2)[0]
        group2_xmax = max(horizontal_spans[i] for i in idx2)[1]

        # Calculate overlap percentage
        overlap_length = max(0, min(group1_xmax, group2_xmax) - max(group1_xmin, group2_xmin))
        group1_width = group1_xmax - group1_xmin
        group2_width = group2_xmax - group2_xmin

        if min(group1_width, group2_width) == 0:
            return 0

        overlap_percentage = overlap_length / min(group1_width, group2_width)
        return overlap_percentage

    # Use gaps to split into multiple groups
    split_points = [0] + [g[0] + 1 for g in gaps] + [len(sorted_indices)]
    groups = []

    # Create initial groups based on vertical position
    for i in range(len(split_points) - 1):
        start, end = split_points[i], split_points[i + 1]
        group_indices = [sorted_indices[j] for j in range(start, end)]  # Convert to list
        if len(group_indices) > 0:  # Check length instead of using as boolean
            group = [detections[idx] for idx in group_indices]
            groups.append(group)

    # Check if likely to be exponents rather than fractions
    if len(groups) == 2:
        # Get character counts in each group
        group1_count = len(groups[0])
        group2_count = len(groups[1])

        # Check sizes - exponents are typically smaller
        group1_indices = [sorted_indices[j] for j in range(split_points[0], split_points[1])]
        group2_indices = [sorted_indices[j] for j in range(split_points[1], split_points[2])]

        group1_avg_height = np.mean([heights[i] for i in group1_indices])
        group2_avg_height = np.mean([heights[i] for i in group2_indices])
        size_ratio = min(group1_avg_height, group2_avg_height) / max(group1_avg_height, group2_avg_height)

        # Check horizontal overlap
        overlap = get_group_overlap(group1_indices, group2_indices)

        # Calculate the average y position for each group
        group1_avg_y = np.mean([y_centers[i] for i in group1_indices])
        group2_avg_y = np.mean([y_centers[i] for i in group2_indices])

        # Modified condition for better fraction detection
        is_likely_exponent = (
            # One group is much smaller than the other AND NOT a denominator at the bottom
                (min(group1_count, group2_count) <= 2 and
                 max(group1_count, group2_count) >= 3 and
                 not (group2_count < group1_count and group2_avg_y > group1_avg_y)) or
                # Significant size difference AND NOT a denominator at the bottom
                (size_ratio < 0.6 and
                 not (group2_avg_height < group1_avg_height and group2_avg_y > group1_avg_y)) or
                # Limited horizontal overlap but not for potential denominators
                (overlap < 0.4 and
                 not (group2_avg_y > group1_avg_y and overlap > 0.2))
        )

        if is_likely_exponent:
            # Probably an exponent, not a fraction - return as single group
            return [detections]

    return groups


def reconstruct_math_expression(detections):
    groups = detect_vertical_groups(detections)

    if len(groups) == 1:
        # Process as single line
        return process_single_line(groups[0])

    elif len(groups) >= 2:
        # Sort groups by vertical position (top to bottom)
        groups.sort(key=lambda g: np.mean([det['box'][1] * det['scale'] for det in g]))

        # For standard fractions (2 groups)
        if len(groups) == 2:
            numerator_orig, numerator_disp = process_single_line(groups[0])
            denominator_orig, denominator_disp = process_single_line(groups[1])

            # Calculate fraction line length
            max_length = max(len(numerator_disp), len(denominator_disp))
            frac_line = '─' * max_length

            # Create vertically aligned fraction display
            display_expr = f"{numerator_disp}\n{frac_line}\n{denominator_disp}"

            return (
                f"({numerator_orig})/({denominator_orig})",
                display_expr
            )

        else:
            # Process each group
            processed_groups = [process_single_line(g) for g in groups]

            # Build nested fraction expression
            orig_expr = processed_groups[0][0]
            for i in range(1, len(processed_groups)):
                orig_expr = f"({orig_expr})/({processed_groups[i][0]})"

            # Create multi-level fraction display
            display_lines = []
            frac_lines = []

            # Add each level with appropriate fraction line
            for i, (_, disp) in enumerate(processed_groups):
                display_lines.append(disp)
                if i < len(processed_groups) - 1:
                    max_length = max(len(display_lines[-1]),
                                     len(processed_groups[i + 1][1]))
                    frac_lines.append('─' * max_length)

            # Interleave the expression lines and fraction lines
            display_expr = ""
            for i in range(len(display_lines)):
                display_expr += display_lines[i]
                if i < len(frac_lines):
                    display_expr += f"\n{frac_lines[i]}\n"

            return orig_expr, display_expr


def solve_expression(expression):
    try:
        # Clean up the expression
        expr = expression.replace('×', '*')

        # Handle exponents properly
        expr = expr.replace('^', '**')

        # Check for empty parentheses in exponents
        if '**()' in expr:
            return "Error: Empty exponent detected"

        # Debug info
        print(f"Processing expression: {expr}")

        # Try to evaluate
        result = sp.sympify(expr).evalf()
        return str(result).rstrip('0').rstrip('.')
    except Exception as e:
        error_msg = str(e)
        print(f"Original expression: {expression}")
        print(f"Processed expression: {expr}")
        print(f"Error: {error_msg}")

        # Provide more user-friendly error messages
        if "unsupported operand type(s) for ** or pow()" in error_msg:
            return "Error: Invalid exponent format detected"
        elif "could not parse" in error_msg:
            return "Error: Expression could not be parsed correctly"
        else:
            return f"Error: {error_msg}"


def load_model():
    if st.session_state.model_loaded:
        return st.session_state.model

    try:
        model_path = os.path.join(os.path.dirname(__file__), "yoloL.onnx")
        st.session_state.model = cv2.dnn.readNetFromONNX(model_path)
        st.session_state.model_loaded = True
        return st.session_state.model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info(
            "This could be due to model compatibility issues with OpenCV. Try using a different ONNX model or check the version compatibility.")
        return None


def math_ocr_pipeline(image):
    model = load_model()
    if model is None:
        st.error("Model could not be loaded. OCR processing is unavailable.")
        return None, None, None

    if isinstance(image, np.ndarray):
        original_image = image
    else:
        original_image = np.array(image.convert('RGB'))
        original_image = original_image[:, :, ::-1].copy()

    if original_image is None:
        st.error("Error processing the image")
        return None, None, None

    height, width = original_image.shape[:2]
    length = max(height, width)
    scale = length / 640

    padded_image = original_image

    try:
        blob = cv2.dnn.blobFromImage(padded_image, 1 / 255.0, (640, 640), swapRB=True)
        model.setInput(blob)
        outputs = model.forward()
    except Exception as e:
        st.error(f"Error during inference: {str(e)}")
        return None, None, None

    try:
        outputs = np.array([cv2.transpose(outputs[0])])
        rows = outputs.shape[1]

        boxes, scores, class_ids = [], [], []
        for i in range(rows):
            classes_scores = outputs[0][i][4:]
            _, max_score, _, (_, max_class_id) = cv2.minMaxLoc(classes_scores)
            if max_score >= 0.25:
                x, y, w, h = outputs[0][i][:4]
                boxes.append([x, y, w, h])
                scores.append(max_score)
                class_ids.append(max_class_id)

        indices = cv2.dnn.NMSBoxes(boxes, scores, 0.5, 0.5, 0.5)
        detections = []

        visualized_image = original_image.copy()

        mask = np.all(original_image != [16, 20, 20], axis=-1)

        for i in indices:
            box = boxes[i]
            detections.append({
                'class_id': class_ids[i],
                'class_name': CLASSES[class_ids[i]],
                'confidence': scores[i],
                'box': box,
                'scale': scale
            })

            x, y, w, h = box
            x_center = int(x * scale)
            y_center = int(y * scale)
            w_scaled = int(w * scale)
            h_scaled = int(h * scale)

            x = x_center - w_scaled // 2
            y = y_center - h_scaled // 2
            x_plus_w = x + w_scaled
            y_plus_h = y + h_scaled

            draw_bounding_box(visualized_image, class_ids[i],
                              scores[i], x, y, x_plus_w, y_plus_h)

        original_expr, display_expr = reconstruct_math_expression(detections)
        solution = solve_expression(original_expr)

        return visualized_image, display_expr, solution
    except Exception as e:
        st.error(f"Error processing detection results: {str(e)}")
        return None, None, None


def rotate_image(img, angle):
    if angle == 0:
        return img

    if not isinstance(img, np.ndarray):
        img_array = np.array(img)
    else:
        img_array = img

    height, width = img_array.shape[:2]
    center = (width // 2, height // 2)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    abs_cos = abs(rotation_matrix[0, 0])
    abs_sin = abs(rotation_matrix[0, 1])

    new_width = int(height * abs_sin + width * abs_cos)
    new_height = int(height * abs_cos + width * abs_sin)

    
    rotation_matrix[0, 2] += new_width / 2 - center[0]
    rotation_matrix[1, 2] += new_height / 2 - center[1]

    rotated = cv2.warpAffine(img_array, rotation_matrix, (new_width, new_height))

    if not isinstance(img, np.ndarray):
        return Image.fromarray(rotated)
    return rotated


def pil_to_cv(pil_image):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def cv_to_pil(cv_image):
    return Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))


def rotate_left():
    st.session_state.rotation = (st.session_state.rotation + 90) % 360
    update_display_image()
    st.rerun()


def rotate_right():
    st.session_state.rotation = (st.session_state.rotation - 90) % 360
    update_display_image()
    st.rerun()


def update_display_image():
    if st.session_state.image is None:
        return

    rotated_image = rotate_image(st.session_state.image, st.session_state.rotation)

    width, height = rotated_image.size
    st.session_state.width = width
    st.session_state.height = height

    left = int(st.session_state.horizontal_crop[0] / 100 * width)
    right = int(st.session_state.horizontal_crop[1] / 100 * width)

    # Clamping to ensure we're within image bounds (0 to height-1)
    top = min(height - 1, max(0, int((100 - st.session_state.vertical_top) / 100 * height)))
    bottom = min(height - 1, max(0, int((100 - st.session_state.vertical_bottom) / 100 * height)))

    st.session_state.left = left
    st.session_state.right = right
    st.session_state.top = top
    st.session_state.bottom = bottom

    display_with_rect = rotated_image.copy()
    draw = ImageDraw.Draw(display_with_rect)

    if top > bottom:
        top, bottom = bottom, top

    draw.rectangle([left, top, right, bottom], outline="red", width=3)

    st.session_state.display_image = display_with_rect


def update_crop_from_ranges():
    if st.session_state.image is not None:
        update_display_image()


def handle_top_slider_change(value):
    st.session_state.vertical_top = value
    # Ensure minimum 10% gap
    if st.session_state.vertical_top < st.session_state.vertical_bottom + 10:
        st.session_state.vertical_top = st.session_state.vertical_bottom + 10
    update_display_image()


def handle_bottom_slider_change(value):
    st.session_state.vertical_bottom = value
    # Ensure minimum 10% gap
    if st.session_state.vertical_bottom > st.session_state.vertical_top - 10:
        st.session_state.vertical_bottom = st.session_state.vertical_top - 10
    update_display_image()


def handle_vertical_range_slider_change():
    values = st.session_state.v_range_slider
    bottom_value = values[0]
    top_value = values[1]

    # Enforce minimum 10% gap
    if top_value - bottom_value < 10:
        # Adjust the value that changed most recently
        if abs(bottom_value - st.session_state.vertical_bottom) > abs(top_value - st.session_state.vertical_top):
            # Bottom changed more
            bottom_value = top_value - 10
        else:
            # Top changed more
            top_value = bottom_value + 10

    st.session_state.vertical_bottom = bottom_value
    st.session_state.vertical_top = top_value
    update_display_image()


def handle_horizontal_slider_change():
    values = st.session_state.h_slider
    left_value = values[0]
    right_value = values[1]

    # Enforce minimum 10% gap
    if right_value - left_value < 10:
        # Adjust based on which one changed
        if abs(left_value - st.session_state.horizontal_crop[0]) > abs(
                right_value - st.session_state.horizontal_crop[1]):
            # Left changed more
            left_value = right_value - 10
        else:
            # Right changed more
            right_value = left_value + 10

    st.session_state.horizontal_crop = [left_value, right_value]
    update_display_image()


def toggle_slider_orientation():
    st.session_state.use_horizontal_vertical_crop = not st.session_state.use_horizontal_vertical_crop


def reset_image():
    # Reset all state variables without trying to directly modify widget values
    st.session_state.processed = False
    st.session_state.rotation = 0

    # Reset horizontal crop and vertical slider state variables
    st.session_state.horizontal_crop = [0, 100]
    st.session_state.vertical_top = 99
    st.session_state.vertical_bottom = 1
    update_display_image()
    # Remove the widget state keys to force them to reinitialize
    keys_to_remove = [
        "vertical_top_slider",
        "vertical_bottom_slider",
        "v_range_slider",
        "h_slider"
    ]

    for key in keys_to_remove:
        if key in st.session_state:
            del st.session_state[key]

    top_value = 100
    bottom_value = 0
    if top_value != st.session_state.vertical_top:
        if top_value < st.session_state.vertical_bottom + 10:
            top_value = st.session_state.vertical_bottom + 10
        st.session_state.vertical_top = top_value
        update_display_image()

    if bottom_value != st.session_state.vertical_bottom:
        if bottom_value > st.session_state.vertical_top - 10:
            bottom_value = st.session_state.vertical_top - 10
        st.session_state.vertical_bottom = bottom_value
        update_display_image()

    handle_top_slider_change(100)
    handle_bottom_slider_change(0)
    # Reset image-related state variables
    st.session_state.image = None
    st.session_state.current_file = None
    st.session_state.display_image = None
    st.session_state.result_image = None
    st.session_state.expression = None
    st.session_state.solution = None
    st.session_state.cropped_image = None

    # Clear file uploader by setting a flag to reset it

    st.rerun()


def resize_image(image, target_width=640):
    if image is None:
        return None

    width, height = image.size
    aspect_ratio = height / width
    new_height = int(target_width * aspect_ratio)

    resized_image = image.resize((target_width, new_height), Image.LANCZOS)
    return resized_image


def process_button_click():
    with st.spinner("Processing..."):
        rotated_image = rotate_image(st.session_state.image, st.session_state.rotation)

        left = int(st.session_state.horizontal_crop[0] / 100 * st.session_state.width)
        right = int(st.session_state.horizontal_crop[1] / 100 * st.session_state.width)

        top = int((100 - st.session_state.vertical_top) / 100 * st.session_state.height)
        bottom = int((100 - st.session_state.vertical_bottom) / 100 * st.session_state.height)

        if top > bottom:
            top, bottom = bottom, top

        cropped_image = rotated_image.crop((left, top, right, bottom))

        # Make the image square by padding with dark color (rgb(16,20,20))
        width, height = cropped_image.size
        max_dim = max(width, height)

        square_image = Image.new('RGB', (max_dim, max_dim), (16, 20, 20))

        paste_x = (max_dim - width) // 2
        paste_y = (max_dim - height) // 2
        square_image.paste(cropped_image, (paste_x, paste_y))

        st.session_state.cropped_image = cropped_image

        result_image, expression, solution = math_ocr_pipeline(square_image)

        st.session_state.result_image = result_image
        st.session_state.expression = expression
        st.session_state.solution = solution
        st.session_state.processed = True


def image_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


def main():
    # Create a row with two columns: small one for checkbox, large one for uploader
    checkbox_col, upload_col = st.columns([1, 9])  # 10% and 90% width allocation

    with checkbox_col:
        st.markdown('<div class="checkbox-container">', unsafe_allow_html=True)
        st.checkbox("📱", value=st.session_state.use_horizontal_vertical_crop,
                    key="toggle_slider_orientation",
                    on_change=toggle_slider_orientation,
                    label_visibility="visible")
        st.markdown('</div>', unsafe_allow_html=True)

    with upload_col:
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed",
                                         key="file_uploader")

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        current_file_hash = hash(file_bytes)

        if st.session_state.current_file != current_file_hash:
            st.session_state.current_file = current_file_hash
            image = Image.open(uploaded_file)

            image = resize_image(image, target_width=640)

            st.session_state.image = image

            st.session_state.processed = False
            st.session_state.rotation = 0
            st.session_state.result_image = None
            st.session_state.expression = None
            st.session_state.solution = None
            st.session_state.cropped_image = None

            width, height = image.size
            st.session_state.width = width
            st.session_state.height = height

            st.session_state.horizontal_crop = [0, 100]
            st.session_state.vertical_top = 100
            st.session_state.vertical_bottom = 0

            update_display_image()

        left_col, output_col = st.columns([1, 1], gap="small")

        with left_col:
            # Desktop layout
            if st.session_state.use_horizontal_vertical_crop:
                # First show the horizontal range slider for vertical crop
                st.markdown("Vertical Crop")
                v_slider = st.slider("Vertical Crop", 0, 100,
                                     [st.session_state.vertical_bottom, st.session_state.vertical_top],
                                     1,
                                     key="v_range_slider",
                                     label_visibility="collapsed",
                                     on_change=handle_vertical_range_slider_change)

                # Then show the image
                if st.session_state.display_image is not None:
                    st.image(st.session_state.display_image, use_container_width=True)
            else:
                # Desktop layout - place both sliders side-by-side at top, then image below
                slider_col1, slider_col2, image_col = st.columns([1, 1, 5], gap="small")

                with slider_col1:
                    st.markdown('<div class="slider-label">Top</div>', unsafe_allow_html=True)
                    top_value = vertical_slider(
                        key="vertical_top_slider",
                        height=300,
                        thumb_shape="circle",
                        step=1,
                        default_value=st.session_state.vertical_top,
                        min_value=0,
                        max_value=100,
                        track_color="#ffcece",
                        slider_color="#FF4B4B",
                        thumb_color="white",
                        value_always_visible=True
                    )
                    if top_value != st.session_state.vertical_top:
                        if top_value < st.session_state.vertical_bottom + 10:
                            top_value = st.session_state.vertical_bottom + 10
                        st.session_state.vertical_top = top_value
                        update_display_image()

                with slider_col2:
                    st.markdown('<div class="slider-label">Bottom</div>', unsafe_allow_html=True)
                    bottom_value = vertical_slider(
                        key="vertical_bottom_slider",
                        height=300,
                        thumb_shape="circle",
                        step=1,
                        default_value=st.session_state.vertical_bottom,
                        min_value=0,
                        max_value=100,
                        track_color="#ffcece",
                        slider_color="#FF4B4B",
                        thumb_color="white",
                        value_always_visible=True
                    )
                    if bottom_value != st.session_state.vertical_bottom:
                        if bottom_value > st.session_state.vertical_top - 10:
                            bottom_value = st.session_state.vertical_top - 10
                        st.session_state.vertical_bottom = bottom_value
                        update_display_image()

                with image_col:
                    # Image in the same row as the sliders
                    if st.session_state.display_image is not None:
                        st.image(st.session_state.display_image, use_container_width=True)

            # Horizontal slider below the three columns
            st.markdown("Horizontal Crop")
            h_slider = st.slider("Horizontal Crop", 0, 100,
                                 st.session_state.horizontal_crop,
                                 1,
                                 key="h_slider",
                                 on_change=handle_horizontal_slider_change,
                                 label_visibility="collapsed")

            # Rotation buttons
            btn1_col, btn2_col = st.columns(2)
            with btn1_col:
                if st.button("Rotate Left (⟲)", key="rotate_left", use_container_width=True):
                    rotate_left()
            with btn2_col:
                if st.button("Rotate Right (⟳)", key="rotate_right", use_container_width=True):
                    rotate_right()

            st.markdown(
                '<p class="annotation">Current rotation: {}°</p>'.format(st.session_state.rotation),
                unsafe_allow_html=True
            )

            # Process and reset buttons
            btn3_col, btn4_col = st.columns(2)
            with btn3_col:
                if st.button("Process Expression", type="primary", key="process", use_container_width=True):
                    process_button_click()
            with btn4_col:
                if st.button("Reset", key="reset", use_container_width=True):
                    reset_image()

        with output_col:
            st.markdown('<div class="output-container">', unsafe_allow_html=True)
            if st.session_state.processed and st.session_state.result_image is not None:
                st.markdown('<div class="output-results">', unsafe_allow_html=True)

                # Image Result with no gap
                with st.container():
                    st.markdown('<div class="image-result">', unsafe_allow_html=True)
                    if st.session_state.result_image is not None:
                        result_pil = cv_to_pil(st.session_state.result_image)

                        # Find the bounding box of non-padding pixels
                        # The padding color used is (16, 20, 20)
                        np_img = np.array(result_pil)
                        mask = np.any(np_img != [16, 20, 20], axis=2)  # Create a mask of non-padding pixels
                        rows = np.any(mask, axis=1)
                        cols = np.any(mask, axis=0)

                        # Find the bounding box
                        y_min, y_max = np.where(rows)[0][[0, -1]] if np.any(rows) else (0, np_img.shape[0] - 1)
                        x_min, x_max = np.where(cols)[0][[0, -1]] if np.any(cols) else (0, np_img.shape[1] - 1)

                        # Crop out the padding
                        cropped_result = result_pil.crop((x_min, y_min, x_max + 1, y_max + 1))

                        # Get the dimensions of the cropped image
                        img_width, img_height = cropped_result.size

                        # Define maximum constraints
                        max_width = 600
                        max_height = 400

                        # Calculate scaling factors
                        width_ratio = max_width / img_width
                        height_ratio = max_height / img_height

                        # Use the smaller ratio to maintain aspect ratio
                        scale_factor = min(width_ratio, height_ratio)

                        # Calculate new dimensions
                        new_width = int(img_width * scale_factor)
                        new_height = int(img_height * scale_factor)

                        # Center the image by adding CSS with no margin-top
                        st.markdown(f"""
                            <div style="display: flex; justify-content: center; margin: 0;">
                                <img src="data:image/png;base64,{image_to_base64(cropped_result.resize((new_width, new_height), Image.LANCZOS))}" 
                                     width="{new_width}" height="{new_height}">
                            </div>
                        """, unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)

                # Expression Result with specified gap
                with st.container():
                    st.markdown('<div class="expression-result">', unsafe_allow_html=True)

                    if st.session_state.expression:
                        st.markdown(f'<div class="math-expression">{st.session_state.expression}</div>',
                                    unsafe_allow_html=True)
                    else:
                        st.info("No expression detected")
                    st.markdown('</div>', unsafe_allow_html=True)

                # Solution Result with specified gap
                with st.container():
                    st.markdown('<div class="solution-result">', unsafe_allow_html=True)

                    if st.session_state.solution:
                        if st.session_state.solution.startswith("Error"):
                            # Check if it's a parsing error for an empty string
                            if "could not parse '''" in st.session_state.solution or "could not parse ''" in st.session_state.solution:
                                st.info("No expression detected")
                            else:
                                # Extract just the expression part from the error message
                                error_parts = st.session_state.solution.split("'")
                                if len(error_parts) >= 3:
                                    # The expression is usually between the second and third quote
                                    attempted_expression = error_parts[2]
                                    st.markdown(f'<div class="math-solution">wrong {attempted_expression}</div>',
                                                unsafe_allow_html=True)
                                else:
                                    # Fallback if we can't extract the expression cleanly
                                    st.info("Invalid expression")
                        else:
                            st.markdown(f'<div class="math-solution">{st.session_state.solution}</div>',
                                        unsafe_allow_html=True)
                    else:
                        st.info("No expression detected")
                    st.markdown('</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            else:
                st.info("Upload and process an image to see the results here.")

                st.markdown('<div class="result-header">Math Expression</div>', unsafe_allow_html=True)
                st.info("Math expression will appear here after processing")

                st.markdown('<div class="result-header">Solution</div>', unsafe_allow_html=True)
                st.info("Solution will appear here after processing")

            st.markdown('</div>', unsafe_allow_html=True)


    else:
        st.info("Upload an image of handwritten math to detect, parse, and solve expressions")



if __name__ == "__main__":
    
    main()