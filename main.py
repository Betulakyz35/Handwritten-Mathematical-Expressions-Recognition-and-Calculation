import cv2
import numpy as np
import sympy as sp


CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', '+', '×', '/', '(', ')']
SUPERSCRIPT_MAP = {
    '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
    '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
    '+': '⁺', '-': '⁻', '(': '⁽', ')': '⁾', '×': '*', '/': 'ᐟ'
}
colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))


def math_ocr_pipeline(onnx_model_path, image_path):
    # Load model and image
    model = cv2.dnn.readNetFromONNX(onnx_model_path)
    original_image = cv2.imread(image_path)

    if original_image is None:
        print(f"Error loading image: {image_path}")
        return None, None, None

    height, width = original_image.shape[:2]
    length = max(height, width)
    scale = length / 640

    # Create square image
    padded_image = np.zeros((length, length, 3), dtype=np.uint8)
    padded_image[:height, :width] = original_image

    # Prepare blob and run inference
    blob = cv2.dnn.blobFromImage(padded_image, 1 / 255.0, (640, 640), swapRB=True)
    model.setInput(blob)
    outputs = model.forward()

    # Process outputs
    outputs = np.array([cv2.transpose(outputs[0])])
    rows = outputs.shape[1]

    boxes, scores, class_ids = [], [], []
    for i in range(rows):
        classes_scores = outputs[0][i][4:]
        _, max_score, _, (_, max_class_id) = cv2.minMaxLoc(classes_scores)
        if max_score >= 0.5:
            x, y, w, h = outputs[0][i][:4]
            boxes.append([x, y, w, h])
            scores.append(max_score)
            class_ids.append(max_class_id)

    # Apply NMS and create detections
    indices = cv2.dnn.NMSBoxes(boxes, scores, 0.5, 0.5, 0.5)
    detections = []
    for i in indices:
        box = boxes[i]
        detections.append({
            'class_id': class_ids[i],
            'class_name': CLASSES[class_ids[i]],
            'confidence': scores[i],
            'box': box,
            'scale': scale
        })

    # Draw bounding boxes
    visualized_image = original_image.copy()
    for det in detections:
        x, y, w, h = det['box']
        x_center = int(x * scale)
        y_center = int(y * scale)
        w_scaled = int(w * scale)
        h_scaled = int(h * scale)

        # Adjust for top-left corner
        x = x_center - w_scaled // 2
        y = y_center - h_scaled // 2
        x_plus_w = x + w_scaled
        y_plus_h = y + h_scaled

        draw_bounding_box(visualized_image, det['class_id'],
                          det['confidence'], x, y, x_plus_w, y_plus_h)

    # Reconstruct and solve expression
    original_expr, display_expr = reconstruct_math_expression(detections)
    solution = solve_expression(original_expr)

    return visualized_image, display_expr, solution


def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = f"{CLASSES[class_id]} ({confidence:.2f})"
    color = colors[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
    cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)



def process_single_line(detections):

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

            continue  # We've already advanced i in the inner loop

        # Normal character
        original_expr.append(char)
        display_expr.append(char)

        # Update baseline with smoothing
        baseline_y = y_center * 0.3 + baseline_y * 0.7
        baseline_height = h_pixel * 0.4 + baseline_height * 0.6

        i += 1

    return ''.join(original_expr), ''.join(display_expr)


def detect_vertical_groups(detections, threshold_factor=1.5):

    if len(detections) < 2:
        return [detections]

    # Calculate vertical positions, heights, and horizontal spans
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
        """Calculate horizontal overlap between two groups of detections"""
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


        print(f"Processing expression: {expr}")

        # Try to evaluate
        result = sp.sympify(expr).evalf()
        return str(result).rstrip('0').rstrip('.')
    except Exception as e:
        error_msg = str(e)
        print(f"Original expression: {expression}")
        print(f"Processed expression: {expr}")
        print(f"Error: {error_msg}")


        if "unsupported operand type(s) for ** or pow()" in error_msg:
            return "Error: Invalid exponent format detected"
        elif "could not parse" in error_msg:
            return "Error: Expression could not be parsed correctly"
        else:
            return f"Error: {error_msg}"


if __name__ == "__main__":

    MODEL_PATH = "yoloL.onnx"
    IMAGE_PATH = r"D:\image.jpg"

    print(IMAGE_PATH)
    result_image, expression, solution = math_ocr_pipeline(MODEL_PATH, IMAGE_PATH)

    if result_image is not None:
        output_image_path = "output_image.jpg"
        cv2.imwrite(output_image_path, result_image)
        print(f"Processed image saved to: {output_image_path}")

        print(f"Expression: \n{expression}")
        print(f"Solution: {solution}")
    else:
        print("Processing failed")
