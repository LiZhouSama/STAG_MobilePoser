import math

def quaternion_to_axis_angle(x, y, z, w):
    # Ensure quaternion is normalized
    norm = math.sqrt(x*x + y*y + z*z + w*w)
    x, y, z, w = x/norm, y/norm, z/norm, w/norm

    # Compute rotation angle
    theta = 2 * math.acos(w)  # in radians

    # Compute denominator for axis
    sin_half_theta = math.sqrt(1 - w*w)
    if sin_half_theta < 1e-6:
        # Angle is close to 0: axis can be arbitrary unit vector
        return (1.0, 0.0, 0.0), 0.0

    # Compute rotation axis
    axis = (x / sin_half_theta,
            y / sin_half_theta,
            z / sin_half_theta)

    return axis, theta

if __name__ == "__main__":
    # Example usage:
    q = [-0.12436, 0.531968, 0.0273764, 0.837135]
    axis, angle = quaternion_to_axis_angle(*q)
    print(f"Rotation axis: ({axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f})")
    print(f"Rotation angle: {math.degrees(angle):.2f}°")
