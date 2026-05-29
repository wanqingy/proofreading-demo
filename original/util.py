def find_bounding_box_3d(points):
    # Initialize bounding box coordinates to extreme values
    min_x, min_y, min_z = float('inf'), float('inf'), float('inf')
    max_x, max_y, max_z = float('-inf'), float('-inf'), float('-inf')
    
    # Iterate through each point
    for x, y, z in points:
        # Update the bounding box coordinates
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        min_z = min(min_z, z)
        max_x = max(max_x, x)
        max_y = max(max_y, y)
        max_z = max(max_z, z)
    
    # Return the bounding box
    return (min_x, min_y, min_z, max_x, max_y, max_z)