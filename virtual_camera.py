import pygame
import numpy as np
import sys
import math
import os
import datetime

# Quaternion class (from book 4.8: q = s + i x + j y + k z)
class Quaternion:
    def __init__(self, s=1.0, v=np.zeros(3)):
        self.s = s
        self.v = v

    def __mul__(self, other):
        # Quaternion multiplication (book formula)
        s1, v1 = self.s, self.v
        s2, v2 = other.s, other.v
        s = s1 * s2 - np.dot(v1, v2)
        v = s1 * v2 + s2 * v1 + np.cross(v1, v2)
        return Quaternion(s, v)

    def conjugate(self):
        return Quaternion(self.s, -self.v)

    def normalize(self):
        norm = np.sqrt(self.s**2 + np.dot(self.v, self.v))
        if norm > 0:
            self.s /= norm
            self.v /= norm

    def to_rotation_matrix(self):
        # Convert to 3x3 rotation matrix (derived from book)
        s, x, y, z = self.s, self.v[0], self.v[1], self.v[2]
        return np.array([
            [1 - 2*y**2 - 2*z**2, 2*x*y - 2*s*z, 2*x*z + 2*s*y],
            [2*x*y + 2*s*z, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*s*x],
            [2*x*z - 2*s*y, 2*y*z + 2*s*x, 1 - 2*x**2 - 2*y**2]
        ])

# Vector3 for points (homogeneous as [x,y,z,1])
class Vector3:
    def __init__(self, x, y, z):
        self.data = np.array([x, y, z, 1.0])

# Cuboid class (book 4.3: 3D object with vertices)
class Cuboid:
    def __init__(self, pos, size):
        half = np.array(size) / 2
        offsets = [-1, 1]
        self.vertices = [Vector3(pos[0] + ox * half[0], pos[1] + oy * half[1], pos[2] + oz * half[2]).data
                         for ox in offsets for oy in offsets for oz in offsets]
        self.edges = [(0,1), (1,3), (3,2), (2,0), (4,5), (5,7), (7,6), (6,4), (0,4), (1,5), (2,6), (3,7)]

# Camera class (position + quaternion orientation + FOV)
class Camera:
    def __init__(self, pos=(0, 0, 10), fov=60.0):
        # ensure position uses float dtype so in-place additions from
        # float deltas don't fail with numpy casting rules
        self.pos = np.array(pos, dtype=float)
        self.quat = Quaternion()  # Identity (no rotation)
        self.fov = fov

    def rotate(self, axis, angle_deg):
        # Create delta quaternion for local rotation (book 4.8)
        angle_rad = math.radians(angle_deg / 2)
        s = math.cos(angle_rad)
        v = math.sin(angle_rad) * axis / np.linalg.norm(axis)
        delta = Quaternion(s, v)
        delta.normalize()
        # Local: Multiply on left for local space
        self.quat = delta * self.quat
        self.quat.normalize()

    def translate(self, delta_local):
        # Get local directions from rotation matrix (book 4.8)
        rot_mat = self.quat.to_rotation_matrix()
        delta_world = rot_mat @ delta_local
        self.pos += delta_world

    def get_view_matrix(self):
        # View = inverse(camera world) = inverse(R * T) = T^{-1} * R^{-1}
        # R^{-1} = transpose(R) since orthogonal
        rot_mat3 = self.quat.to_rotation_matrix().T  # Inverse rotation (3x3)
        # Build full 4x4 view matrix: [R^T  -R^T * pos; 0 1]
        view = np.eye(4, dtype=float)
        view[:3, :3] = rot_mat3
        view[:3, 3] = -rot_mat3 @ self.pos
        return view

# Perspective projection matrix (standard, not in Part 1; ref Hughes et al.)
def perspective_matrix(fov, aspect, near=0.1, far=100.0):
    f = 1.0 / math.tan(math.radians(fov / 2))
    return np.array([
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)],
        [0, 0, -1, 0]
    ])

# Main program
pygame.init()
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Virtual Camera - Zadanie 1")
clock = pygame.time.Clock()

# UI: font and help/menu toggle
font = pygame.font.SysFont(None, 20)
help_visible = True
# screenshot message state
screenshot_msg = ""
msg_timer = 0.0

camera = Camera(pos=(0, 0, 10), fov=60.0)

# Scene: Street with buildings (cuboids)
cuboids = [
    Cuboid((-4, -2, -5), (2, 4, 10)),  # Left building 1
    Cuboid((-4, -2, -20), (2, 6, 10)), # Left building 2
    Cuboid((4, -2, -5), (2, 5, 10)),   # Right building 1
    Cuboid((4, -2, -20), (2, 4, 10)),  # Right building 2
    Cuboid((0, -3, -10), (1, 1, 30))   # "Road" marker
]

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            if event.key == pygame.K_h:
                help_visible = not help_visible
            if event.key == pygame.K_r:
                # reset camera to initial state
                camera.pos = np.array((0.0, 0.0, 10.0), dtype=float)
                camera.quat = Quaternion()
                camera.fov = 60.0
            if event.key == pygame.K_SPACE:
                # take screenshot and save to app folder with timestamp
                try:
                    folder = os.path.dirname(os.path.abspath(__file__))
                    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                    filename = f"screenshot_{ts}.png"
                    path = os.path.join(folder, filename)
                    pygame.image.save(screen, path)
                    screenshot_msg = f"Saved: {filename}"
                    msg_timer = 2.5
                except Exception as e:
                    screenshot_msg = f"Save failed: {e}"
                    msg_timer = 3.0

    keys = pygame.key.get_pressed()
    move_speed = 0.2
    rot_speed = 2.0
    fov_speed = 5.0

    # Translation (local)
    forward = np.array([0, 0, -move_speed]) if keys[pygame.K_w] else np.array([0, 0, move_speed]) if keys[pygame.K_s] else np.zeros(3)
    right = np.array([move_speed, 0, 0]) if keys[pygame.K_d] else np.array([-move_speed, 0, 0]) if keys[pygame.K_a] else np.zeros(3)
    up = np.array([0, move_speed, 0]) if keys[pygame.K_e] else np.array([0, -move_speed, 0]) if keys[pygame.K_q] else np.zeros(3)
    camera.translate(forward + right + up)

    # Rotation (local axes)
    if keys[pygame.K_LEFT]: camera.rotate(np.array([0,1,0]), -rot_speed)  # Yaw left
    if keys[pygame.K_RIGHT]: camera.rotate(np.array([0,1,0]), rot_speed)  # Yaw right
    if keys[pygame.K_UP]: camera.rotate(np.array([1,0,0]), -rot_speed)    # Pitch up
    if keys[pygame.K_DOWN]: camera.rotate(np.array([1,0,0]), rot_speed)   # Pitch down
    if keys[pygame.K_z]: camera.rotate(np.array([0,0,1]), -rot_speed)     # Roll left
    if keys[pygame.K_x]: camera.rotate(np.array([0,0,1]), rot_speed)      # Roll right

    # Zoom (FOV)
    if keys[pygame.K_PLUS] or keys[pygame.K_KP_PLUS]: camera.fov = max(10, camera.fov - fov_speed)
    if keys[pygame.K_MINUS] or keys[pygame.K_KP_MINUS]: camera.fov = min(120, camera.fov + fov_speed)

    # Rendering
    screen.fill((0, 0, 0))
    proj = perspective_matrix(camera.fov, width / height)
    view = camera.get_view_matrix()

    # Clip edges against the near plane in camera space, then project
    near_plane = 0.1
    for cub in cuboids:
        for edge in cub.edges:
            # transform vertices to camera (view) space first
            cs1 = view @ cub.vertices[edge[0]]  # camera-space homogeneous [x,y,z,1]
            cs2 = view @ cub.vertices[edge[1]]
            z1 = float(cs1[2])
            z2 = float(cs2[2])
            # If both points are behind the near plane (z > -near), skip
            if z1 > -near_plane and z2 > -near_plane:
                continue
            # If the edge crosses the near plane, compute intersection and replace the behind point
            if (z1 > -near_plane) != (z2 > -near_plane):
                dz = (z2 - z1)
                if dz == 0.0:
                    continue
                t = (-near_plane - z1) / dz
                # clamp t just in case
                t = max(0.0, min(1.0, t))
                inter = cs1 + t * (cs2 - cs1)
                # replace the point that is behind the near plane with the intersection
                if z1 > -near_plane:
                    cs1 = inter
                    z1 = float(cs1[2])
                else:
                    cs2 = inter
                    z2 = float(cs2[2])
            # Now project the (possibly clipped) camera-space points
            pv1 = proj @ cs1
            pv2 = proj @ cs2
            w1 = float(pv1[3])
            w2 = float(pv2[3])
            if w1 == 0.0 or w2 == 0.0:
                continue
            x1 = float(pv1[0]) / w1
            y1 = float(pv1[1]) / w1
            x2 = float(pv2[0]) / w2
            y2 = float(pv2[1]) / w2
            # Skip if any coordinate is not finite
            if not (np.isfinite(x1) and np.isfinite(y1) and np.isfinite(x2) and np.isfinite(y2)):
                continue
            p1 = (int(width / 2 + x1 * (width / 2)), int(height / 2 - y1 * (height / 2)))
            p2 = (int(width / 2 + x2 * (width / 2)), int(height / 2 - y2 * (height / 2)))
            try:
                pygame.draw.line(screen, (255, 255, 255), p1, p2)
            except Exception:
                # ignore any remaining drawing errors
                continue

    # Draw HUD / help menu
    if help_visible:
        lines = [
            "Controls:",
            "W/S: forward/back",
            "A/D: left/right",
            "Q/E: down/up",
            "Arrow keys: pitch/yaw",
            "Z/X: roll",
            "+/-: zoom (change FOV)",
            "Space: take screenshot",
            "H: toggle this help",
            "R: reset camera",
            "Esc: quit"
        ]
        # background box
        padding = 8
        line_height = font.get_linesize()
        box_w = 240
        box_h = padding * 2 + line_height * len(lines)
        surf = pygame.Surface((box_w, box_h), flags=pygame.SRCALPHA)
        surf.fill((0, 0, 0, 150))
        # render lines
        for i, txt in enumerate(lines):
            color = (255, 255, 255) if i == 0 else (200, 200, 200)
            txt_surf = font.render(txt, True, color)
            surf.blit(txt_surf, (padding, padding + i * line_height))
        screen.blit(surf, (10, 10))

    # status (FOV and camera position)
    status = f"FOV: {camera.fov:.1f}  Pos: ({camera.pos[0]:.2f}, {camera.pos[1]:.2f}, {camera.pos[2]:.2f})"
    stat_surf = font.render(status, True, (255, 255, 0))
    screen.blit(stat_surf, (10, height - 40))

    # draw screenshot/save message if active
    if msg_timer > 0.0 and screenshot_msg:
        msg_surf = font.render(screenshot_msg, True, (0, 255, 0))
        # centered near top
        screen.blit(msg_surf, (width // 2 - msg_surf.get_width() // 2, 10))

    pygame.display.flip()
    ms = clock.tick(60)
    # update message timer
    if msg_timer > 0.0:
        msg_timer -= ms / 1000.0
        if msg_timer <= 0.0:
            screenshot_msg = ""
            msg_timer = 0.0

pygame.quit()
sys.exit()