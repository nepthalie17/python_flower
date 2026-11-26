"""
Flower Life Cycle - withered -> reviving -> bud -> bloom -> (wilt) -> repeat

A particle DAISY (pink petals, yellow centre, green stem) inside a faint starry
sphere, drawn with pygame + numpy.

Controls
    SPACE   pause / resume
    R       restart the cycle
    UP/DOWN speed up / slow down
    F       toggle fullscreen
    ESC     quit
"""
import argparse
import math

import numpy as np
import pygame

# --------------------------------------------------------------------------
# Timeline (seconds)
# --------------------------------------------------------------------------
HOLD_WITHERED = 2.0   # lifeless, drooping
GROW = 11.0           # withered -> bud -> full bloom
HOLD_BLOOM = 7.0      # enjoy the bloom
WILT = 7.0            # bloom -> withered again (the cycle repeats)

KEYS = (0.0, 0.4, 1.0)  # growth keyframes: withered, bud, bloom


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def phase(t):
    """Return (s, w, label). s = growth 0..1, w = wilt 0..1."""
    total = HOLD_WITHERED + GROW + HOLD_BLOOM + WILT
    t %= total
    if t < HOLD_WITHERED:
        return 0.0, 0.0, "Withered"
    t -= HOLD_WITHERED
    if t < GROW:
        s = ease(t / GROW)
        label = ("Reviving" if s < 0.15 else "Budding" if s < 0.45
                 else "Blooming" if s < 0.9 else "Full bloom")
        return s, 0.0, label
    t -= GROW
    if t < HOLD_BLOOM:
        return 1.0, 0.0, "Full bloom"
    t -= HOLD_BLOOM
    w = ease(t / WILT)
    return 1.0, w, ("Wilting" if w < 0.95 else "Withered")


def K(table, s, w):
    """Interpolate a keyframe table (last axis = withered, bud, bloom),
    then blend back toward 'withered' by the wilt amount w."""
    t = np.asarray(table, float)
    if s < KEYS[1]:
        a, b, f = t[..., 0], t[..., 1], s / KEYS[1]
    else:
        a, b, f = t[..., 1], t[..., 2], (s - KEYS[1]) / (1 - KEYS[1])
    v = a + (b - a) * f
    return v + (t[..., 0] - v) * w


# --------------------------------------------------------------------------
# Flower model
# --------------------------------------------------------------------------
# per petal-ring: count, length, rotation offset, half-width,
#   theta = opening angle from the flower axis [withered, bud, bloom]
#   bend  = how much the petal curves along its length [withered, bud, bloom]
LAYERS = [
    dict(n=18, L=0.44, rot=0.00, wid=0.043, th=(2.10, 0.12, 1.45), bend=(1.0, 0.25, 0.30)),
    dict(n=14, L=0.38, rot=0.17, wid=0.042, th=(1.90, 0.08, 1.25), bend=(0.9, 0.25, 0.30)),
]
PTS_PER_PETAL = 230
R0 = 0.12                       # petals start at the edge of the yellow disc

# colour tables: rows = R, G, B ; columns = withered, bud, bloom
PETAL_BASE = [[1.00, 1.00, 1.00], [0.62, 0.72, 0.82], [0.78, 0.85, 0.94]]
PETAL_TIP = [[1.00, 1.00, 1.00], [0.80, 0.86, 0.93], [0.88, 0.92, 0.98]]
DISC_COL = [[0.30, 0.60, 1.00], [0.22, 0.55, 0.78], [0.08, 0.15, 0.10]]
STEM_COL = [[0.40, 0.22, 0.20], [0.28, 0.62, 0.78], [0.12, 0.22, 0.25]]
LEAF_COL = [[0.38, 0.18, 0.15], [0.25, 0.55, 0.70], [0.10, 0.20, 0.22]]
BRIGHT = [0.55, 0.85, 1.00]

# stem bezier: top x / top y / control y
STEM_TOPX = [-0.38, -0.06, 0.0]
STEM_TOPY = [0.55, 0.85, 0.98]
STEM_CTRY = [0.38, 0.55, 0.60]
TILT = [1.00, 0.10, 0.0]        # head droop sideways (radians)
HEADX = [0.35, 0.15, 1.00]      # head turns toward the camera as it blooms

LEAVES = [  # attach t, azimuth, length
    (0.22, 0.25, 0.42),
    (0.38, math.pi - 0.30, 0.36),
    (0.52, -1.10, 0.26),
]
LEAF_ALPHA = [-0.95, 0.60, 0.50]
LEAF_CURL = [1.2, 0.60, 0.80]
LEAF_SCALE = [0.50, 0.80, 1.00]

SPHERE_C = np.array([0.0, 0.72, 0.0])
SPHERE_R = 1.08


class Flower:
    def __init__(self, W, H, seed=7):
        self.W, self.H = W, H
        self.unit = H * 0.30
        self.base_y = H * 0.80
        rng = np.random.default_rng(seed)
        self.rng = rng

        # ---- petals (static random attributes) ----
        P = {k: [] for k in ("phi", "u", "v", "L", "wid", "th", "bend",
                             "jth", "jk", "glow")}
        for li, lay in enumerate(LAYERS):
            for pi_ in range(lay["n"]):
                phi = lay["rot"] + 2 * math.pi * pi_ / lay["n"] + rng.normal(0, 0.05)
                jth, jk = rng.normal(0, 0.28), rng.normal(0, 0.25)
                n = PTS_PER_PETAL
                u = rng.random(n) ** 0.85
                v = rng.uniform(-1, 1, n)
                edge = rng.random(n) < 0.35          # outline points
                v[edge] = np.sign(v[edge])
                P["phi"].append(np.full(n, phi))
                P["u"].append(u)
                P["v"].append(v)
                P["L"].append(np.full(n, lay["L"] * rng.uniform(0.92, 1.08)))
                P["wid"].append(np.full(n, lay["wid"]))
                P["th"].append(np.tile(np.array(lay["th"], float), (n, 1)))
                P["bend"].append(np.tile(np.array(lay["bend"], float), (n, 1)))
                P["jth"].append(np.full(n, jth))
                P["jk"].append(np.full(n, jk))
                P["glow"].append(rng.uniform(0.55, 1.1, n))
        self.p = {k: np.concatenate(v) for k, v in P.items()}

        # ---- stem ----
        n = 650
        self.s_t = rng.random(n)
        self.s_off = rng.normal(0, 0.007, (n, 2))
        self.s_glow = rng.uniform(0.6, 1.1, n)

        # ---- yellow disc (sunflower-style spiral dome) ----
        n = 1100
        i = np.arange(n) + 0.5
        rr = 0.135 * np.sqrt(i / n)
        ga = i * math.pi * (3 - math.sqrt(5))
        self.disc = np.stack([rr * np.cos(ga),
                              0.05 * (1 - (rr / 0.135) ** 2) + 0.01,
                              rr * np.sin(ga)], 1)
        self.disc_g = rng.uniform(0.6, 1.2, n)
        self.disc_r = rr / 0.135

        # ---- leaves ----
        self.leaf = []
        for ta, psi, L in LEAVES:
            n = 450
            u = rng.random(n) ** 0.9
            v = rng.uniform(-1, 1, n)
            v[rng.random(n) < 0.15] = 0.0            # centre vein
            self.leaf.append(dict(ta=ta, psi=psi, L=L, u=u, v=v,
                                  glow=rng.uniform(0.55, 1.1, n)))

        # ---- gravel mound ----
        n = 2200
        r = 1.0 * np.sqrt(rng.random(n))
        a = rng.random(n) * 2 * math.pi
        y = 0.075 * (1 - r ** 2) - rng.random(n) * 0.06
        self.soil = np.stack([r * np.cos(a), y, r * np.sin(a)], 1)
        self.soil_g = rng.uniform(0.35, 1.0, n)

        # ---- star sphere + dust ----
        n = 1300
        i = np.arange(n) + 0.5
        yv = 1 - 2 * i / n
        rr = np.sqrt(1 - yv ** 2)
        th = i * math.pi * (3 - math.sqrt(5))
        self.shell = SPHERE_C + SPHERE_R * np.stack(
            [rr * np.cos(th), yv, rr * np.sin(th)], 1)
        self.shell_ph = rng.random(n) * 6.28
        m = 520
        d = rng.normal(size=(m, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        self.dust = SPHERE_C + d * SPHERE_R * 0.97 * rng.random((m, 1)) ** (1 / 3)
        self.dust_ph = rng.random(m) * 6.28

        # ---- fireflies ----
        n = 150
        self.ff_x = rng.uniform(-0.9, 0.9, n)
        self.ff_z = rng.uniform(-0.6, 0.6, n)
        self.ff_y = rng.random(n)
        self.ff_v = rng.uniform(0.04, 0.12, n)
        self.ff_ph = rng.random(n) * 6.28

        # ---- background gradient ----
        bg = pygame.Surface((W, H))
        for yy in range(H):
            f = yy / H
            pygame.draw.line(bg, (int(4 + 6 * f), int(4 + 4 * f), int(14 + 14 * f)),
                             (0, yy), (W, yy))
        self.bg = bg

    # ------------------------------------------------------------------
    def stem_point(self, t, top, ctrl):
        x = 2 * (1 - t) * t * ctrl[0] + t * t * top[0]
        y = 2 * (1 - t) * t * ctrl[1] + t * t * top[1]
        return x, y

    def build(self, t, s, w):
        """Return (positions Nx3, colours Nx3) for the current moment."""
        P = self.p
        life = float(K([0.0, 0.3, 1.0], s, w))
        bright = float(K(BRIGHT, s, w))
        pos, col = [], []

        top = (float(K(STEM_TOPX, s, w)), float(K(STEM_TOPY, s, w)))
        ctrl = (0.0, float(K(STEM_CTRY, s, w)))
        tilt = float(K(TILT, s, w))

        # head orientation: turn toward camera (X) then droop sideways (Z)
        hx = float(K(HEADX, s, w))
        cx_, sx_ = math.cos(hx), math.sin(hx)
        ct, st = math.cos(tilt), math.sin(tilt)

        def head(x, y, z):
            y1 = y * cx_ - z * sx_
            z1 = y * sx_ + z * cx_
            return (x * ct - y1 * st + top[0],
                    x * st + y1 * ct + top[1],
                    z1)

        # petals
        jf = float(K([1.0, 0.3, 0.15], s, w))
        th0 = K(P["th"], s, w) + P["jth"] * jf
        bend = np.maximum(K(P["bend"], s, w) + P["jk"] * jf * 0.5, 0.25)
        L = P["L"] * float(K(LEAF_SCALE, s, w))
        u, v, phi = P["u"], P["v"], P["phi"]
        ang = th0 + bend * u
        along = L * (np.cos(th0) - np.cos(ang)) / bend + R0
        up = L * (np.sin(ang) - np.sin(th0)) / bend
        width = (P["wid"] * float(K([0.7, 0.9, 1.0], s, w))
                 * np.sqrt(u) * np.sqrt(1 - u ** 4))
        side = v * width
        x = along * np.cos(phi) - side * np.sin(phi)
        z = along * np.sin(phi) + side * np.cos(phi)
        y = up + 0.8 * side ** 2
        # gentle breathing when alive
        y = y + 0.010 * life * np.sin(t * 1.6 + phi * 3)
        px, py, pz = head(x, y, z)
        pos.append(np.stack([px, py, pz], 1))
        base = K(PETAL_BASE, s, w)
        tip = K(PETAL_TIP, s, w)
        c = base + (tip - base) * (u ** 1.5)[:, None]
        shade_u = (0.65 + 0.35 * u)[:, None]          # cream base -> bright tip
        col.append(c * shade_u * (P["glow"] * bright)[:, None] * 0.60)

        # yellow centre
        dx, dy, dz = head(self.disc[:, 0], self.disc[:, 1], self.disc[:, 2])
        pos.append(np.stack([dx, dy, dz], 1))
        dc = K(DISC_COL, s, w)
        shade = (0.55 + 0.45 * self.disc_r)          # darker core, brighter rim
        col.append(np.tile(dc, (len(dx), 1))
                   * (self.disc_g * shade * bright * 1.3)[:, None])

        # stem
        tt = self.s_t
        sx, sy = self.stem_point(tt, top, ctrl)
        taper = (1 - 0.5 * tt)
        pos.append(np.stack([sx + self.s_off[:, 0] * taper, sy,
                             self.s_off[:, 1] * taper], 1))
        col.append(np.tile(K(STEM_COL, s, w), (len(tt), 1))
                   * (self.s_glow * bright * 0.8)[:, None])

        # leaves
        alpha0 = float(K(LEAF_ALPHA, s, w))
        curl = float(K(LEAF_CURL, s, w))
        lscale = float(K(LEAF_SCALE, s, w))
        lcol = K(LEAF_COL, s, w)
        for lf in self.leaf:
            ax, ay = self.stem_point(lf["ta"], top, ctrl)
            psi, uu, vv = lf["psi"], lf["u"], lf["v"]
            a = alpha0 - curl * uu
            Ll = lf["L"] * lscale
            horiz = Ll * (math.sin(alpha0) - np.sin(a)) / curl
            vert = Ll * (np.cos(a) - math.cos(alpha0)) / curl
            wd = 0.085 * lscale * np.sin(math.pi * uu ** 0.7) * vv
            lx = ax + horiz * math.cos(psi) - wd * math.sin(psi)
            lz = horiz * math.sin(psi) + wd * math.cos(psi)
            pos.append(np.stack([lx, ay + vert, lz], 1))
            col.append(np.tile(lcol, (len(uu), 1)) * (lf["glow"] * bright * 0.8)[:, None])

        # gravel
        sb = float(K([0.35, 0.55, 0.70], s, w))
        pos.append(self.soil)
        col.append(np.tile([0.60, 0.45, 0.30], (len(self.soil), 1))
                   * (self.soil_g * sb * 0.50)[:, None])

        # star sphere
        tw = 0.6 + 0.4 * np.sin(t * 1.8 + self.shell_ph)
        pos.append(self.shell)
        col.append(np.tile([0.45, 0.55, 1.0], (len(self.shell), 1))
                   * (tw * (0.10 + 0.25 * life))[:, None])
        tw = 0.5 + 0.5 * np.sin(t * 1.3 + self.dust_ph)
        pos.append(self.dust)
        col.append(np.tile([0.7, 0.75, 1.0], (len(self.dust), 1))
                   * (tw * (0.04 + 0.18 * life))[:, None])

        # fireflies
        if life > 0.02:
            fy = -0.1 + ((self.ff_y + t * self.ff_v) % 1.0) * 1.7
            fx = self.ff_x + 0.05 * np.sin(t * self.ff_v * 12 + self.ff_ph)
            pos.append(np.stack([fx, fy, self.ff_z], 1))
            tw = 0.5 + 0.5 * np.sin(t * 3 + self.ff_ph)
            col.append(np.tile([1.0, 0.95, 0.6], (len(fx), 1))
                       * (tw * life * 0.55)[:, None])

        return np.concatenate(pos), np.concatenate(col)

    # ------------------------------------------------------------------
    def render(self, t):
        W, H = self.W, self.H
        s, w, label = phase(t)
        pos, col = self.build(t, s, w)

        # camera: slow sway around Y + slight downward pitch
        a = 0.30 * math.sin(t * 0.35)
        ca, sa = math.cos(a), math.sin(a)
        pitch = 0.22
        cp, sp = math.cos(pitch), math.sin(pitch)
        x = pos[:, 0] * ca + pos[:, 2] * sa
        z = -pos[:, 0] * sa + pos[:, 2] * ca
        y = pos[:, 1] * cp - z * sp
        zd = z * cp + pos[:, 1] * sp
        persp = 1.0 / (1.0 - 0.12 * zd)
        sx = W / 2 + x * self.unit * persp
        sy = self.base_y - y * self.unit * persp
        col = col * np.clip(0.8 + 0.25 * zd, 0.5, 1.1)[:, None]

        ix, iy = sx.astype(np.int32), sy.astype(np.int32)
        ok = (ix >= 0) & (ix < W - 1) & (iy >= 0) & (iy < H - 1)
        flat = ix[ok] * H + iy[ok]
        col = col[ok]
        buf = np.empty((W, H, 3), np.float32)
        for ch in range(3):
            buf[..., ch] = np.bincount(flat, weights=col[:, ch],
                                       minlength=W * H).reshape(W, H)
        # 2x2 splat so particles read as soft dots
        buf[1:] = buf[1:] + buf[:-1]
        buf[:, 1:] = buf[:, 1:] + buf[:, :-1]

        img = np.clip(buf * 255.0, 0, 255).astype(np.uint8)
        layer = pygame.surfarray.make_surface(img)

        frame = self.bg.copy()
        frame.blit(layer, (0, 0), special_flags=pygame.BLEND_ADD)
        # cheap bloom: blur by down/up-scaling and add back
        for div, reps in ((4, 1), (12, 1)):
            small = pygame.transform.smoothscale(layer, (W // div, H // div))
            big = pygame.transform.smoothscale(small, (W, H))
            for _ in range(reps):
                frame.blit(big, (0, 0), special_flags=pygame.BLEND_ADD)
        return frame, label, s, w


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Flower life cycle (withered -> bloom)")
    ap.add_argument("--size", default="720x860", help="window size, e.g. 720x860")
    ap.add_argument("--fullscreen", action="store_true")
    args = ap.parse_args()
    W, H = (int(v) for v in args.size.lower().split("x"))

    pygame.init()
    pygame.display.set_caption("Life cycle of a daisy - Python")
    flags = pygame.SCALED | pygame.RESIZABLE | (pygame.FULLSCREEN if args.fullscreen else 0)
    screen = pygame.display.set_mode((W, H), flags)
    font = pygame.font.SysFont("segoeui,arial", 26)
    small = pygame.font.SysFont("segoeui,arial", 15)

    flower = Flower(W, H)
    clock = pygame.time.Clock()
    t, speed, paused = 0.0, 1.0, False

    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key == pygame.K_SPACE:
                    paused = not paused
                elif e.key == pygame.K_r:
                    t = 0.0
                elif e.key == pygame.K_UP:
                    speed = min(speed * 1.5, 8.0)
                elif e.key == pygame.K_DOWN:
                    speed = max(speed / 1.5, 0.15)
                elif e.key == pygame.K_f:
                    pygame.display.toggle_fullscreen()
        if not paused:
            t += dt * speed

        frame, label, s, w = flower.render(t)
        screen.blit(frame, (0, 0))

        txt = font.render(label, True, (190, 200, 255))
        screen.blit(txt, (W // 2 - txt.get_width() // 2, 28))
        hint = small.render("SPACE pause   R restart   UP/DOWN speed   F fullscreen   ESC quit",
                            True, (90, 100, 150))
        screen.blit(hint, (W // 2 - hint.get_width() // 2, H - 26))
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
