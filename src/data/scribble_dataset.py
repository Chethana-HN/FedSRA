import numpy as np
import torch

from pathlib import Path
from PIL import Image
from scipy.ndimage import binary_dilation
from torch.utils.data import Dataset


def resize_image(arr, size, mask=False):
    image = Image.fromarray(arr)
    resample = Image.NEAREST if mask else Image.BILINEAR
    return np.array(image.resize((size, size), resample=resample))


def load_image(path, size):
    image = np.array(Image.open(path).convert("RGB"))
    image = resize_image(image, size).astype(np.float32) / 255.0
    return image.transpose(2, 0, 1)


def load_mask(path, size):
    mask = np.array(Image.open(path).convert("L"))
    mask = resize_image(mask, size, mask=True)
    return (mask > 127).astype(np.float32)[None]


def normalize(v, eps=1e-8):
    v = np.asarray(v, dtype=np.float64)
    norm = np.linalg.norm(v)

    if norm < eps:
        return np.array([1.0, 0.0], dtype=np.float64)

    return v / norm


def active_coords(mask):
    return np.argwhere(mask > 0).astype(np.int32)


def nearest_active(mask, point, coords=None):
    h, w = mask.shape
    py = int(round(float(point[0])))
    px = int(round(float(point[1])))

    if 0 <= py < h and 0 <= px < w and mask[py, px]:
        return py, px

    for radius in (1, 2, 3, 4, 6):
        y0 = max(0, py - radius)
        y1 = min(h, py + radius + 1)
        x0 = max(0, px - radius)
        x1 = min(w, px + radius + 1)

        ys, xs = np.where(mask[y0:y1, x0:x1])

        if len(ys):
            ys += y0
            xs += x0
            d2 = (ys - point[0]) ** 2 + (xs - point[1]) ** 2
            i = int(np.argmin(d2))
            return int(ys[i]), int(xs[i])

    if coords is None:
        coords = active_coords(mask)

    if not len(coords):
        return py, px

    d2 = (
        (coords[:, 0] - point[0]) ** 2
        + (coords[:, 1] - point[1]) ** 2
    )
    i = int(np.argmin(d2))

    return int(coords[i, 0]), int(coords[i, 1])


def bezier_point(t, p0, p1, p2, p3):
    u = 1.0 - t

    return (
        u**3 * p0
        + 3.0 * u**2 * t * p1
        + 3.0 * u * t**2 * p2
        + t**3 * p3
    )


def draw_bezier(mask, start, end, ctrl1, ctrl2):
    coords = active_coords(mask)

    if not len(coords):
        return np.zeros_like(mask, dtype=np.float32)

    p0 = np.asarray(start, dtype=np.float64)
    p1 = np.asarray(ctrl1, dtype=np.float64)
    p2 = np.asarray(ctrl2, dtype=np.float64)
    p3 = np.asarray(end, dtype=np.float64)

    steps = max(16, int(np.ceil(np.linalg.norm(p3 - p0) * 2.5)))
    out = np.zeros_like(mask, dtype=bool)

    for i in range(steps + 1):
        point = bezier_point(i / steps, p0, p1, p2, p3)
        y, x = nearest_active(mask, point, coords)
        out[y, x] = True

    return out.astype(np.float32)


def principal_axes(mask):
    coords = active_coords(mask)

    if len(coords) < 2:
        return (
            coords.mean(axis=0),
            np.array([1.0, 0.0]),
            np.array([0.0, 1.0]),
            coords,
        )

    center = coords.mean(axis=0).astype(np.float64)
    centered = coords.astype(np.float64) - center
    cov = centered.T @ centered / max(len(coords) - 1, 1)

    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1]

    major = normalize(vectors[:, order[0]])
    minor = normalize(np.array([-major[1], major[0]]))

    return center, major, minor, coords


def foreground_scribble(mask, rng, width=0):
    mask = mask.astype(bool)

    if not mask.any():
        return np.zeros_like(mask, dtype=np.float32)

    center, major, minor, coords = principal_axes(mask)
    projection = (coords.astype(np.float64) - center) @ major

    low = float(np.quantile(projection, 0.08))
    high = float(np.quantile(projection, 0.92))

    if high <= low:
        low = float(projection.min())
        high = float(projection.max())

    p0 = center + low * major
    p3 = center + high * major
    span = float(np.linalg.norm(p3 - p0))

    bend1 = float(rng.uniform(-0.16, 0.16) * span)
    bend2 = float(rng.uniform(-0.18, 0.18) * span)

    if abs(bend1 - bend2) < 0.04 * span:
        bend2 *= -1.0

    p1 = p0 + 0.32 * (p3 - p0) + bend1 * minor
    p2 = p0 + 0.68 * (p3 - p0) + bend2 * minor

    scribble = draw_bezier(mask, p0, p3, p1, p2).astype(bool)
    branch_length = float(np.clip(0.24 * span, 14.0, 27.0))

    for t, side in ((0.34, 1.0), (0.68, -1.0)):
        start = bezier_point(t, p0, p1, p2, p3)
        y, x = nearest_active(mask, start, coords)
        start = np.array([float(y), float(x)])

        direction = normalize(
            side * minor
            + float(rng.uniform(-0.45, 0.45)) * major
        )

        end = start + branch_length * direction

        c1 = (
            start
            + 0.35 * branch_length * direction
            + float(rng.uniform(-0.12, 0.12) * branch_length) * major
        )

        c2 = (
            start
            + 0.72 * branch_length * direction
            + float(rng.uniform(-0.18, 0.18) * branch_length) * major
        )

        scribble |= draw_bezier(
            mask,
            start,
            end,
            c1,
            c2,
        ).astype(bool)

    if width > 0:
        scribble = binary_dilation(scribble, iterations=int(width))

    scribble &= mask

    return scribble.astype(np.float32)


def background_branch(
    mask,
    rng,
    width=0,
    inner_margin=3,
    outer_margin=18,
    side=1,
):
    mask = mask.astype(bool)

    if not mask.any():
        return np.zeros_like(mask, dtype=np.float32)

    center, major, minor, _ = principal_axes(mask)

    inner = binary_dilation(
        mask,
        iterations=max(1, int(inner_margin)),
    )

    outer = binary_dilation(
        mask,
        iterations=max(int(outer_margin), int(inner_margin) + 2),
    )

    band = outer & ~inner & ~mask

    if band.sum() < 20:
        band = outer & ~binary_dilation(mask, iterations=1) & ~mask

    if not band.any():
        band = ~mask

    coords = active_coords(band)

    if not len(coords):
        return np.zeros_like(mask, dtype=np.float32)

    radius = 0.55 * (float(inner_margin) + float(outer_margin))
    start = center + float(side) * radius * minor

    y, x = nearest_active(band, start, coords)
    p0 = np.array([float(y), float(x)])

    length = float(
        np.clip(
            np.sqrt(max(int(mask.sum()), 1)) * 0.52,
            22.0,
            36.0,
        )
    )

    sign = 1.0 if rng.random() < 0.5 else -1.0

    direction = normalize(
        sign * major
        + float(rng.uniform(-0.30, 0.30)) * minor
    )

    p3 = p0 + length * direction

    p1 = (
        p0
        + 0.33 * length * direction
        + side * float(rng.uniform(0.08, 0.18)) * length * minor
    )

    p2 = (
        p0
        + 0.70 * length * direction
        - side * float(rng.uniform(0.05, 0.14)) * length * minor
    )

    branch = draw_bezier(
        band,
        p0,
        p3,
        p1,
        p2,
    ).astype(bool)

    if width > 0:
        branch = binary_dilation(branch, iterations=int(width))

    branch &= ~mask
    branch &= outer

    return branch.astype(np.float32)


def background_scribble(
    mask,
    rng,
    width=0,
    inner_margin=3,
    outer_margin=18,
    num_branches=2,
):
    mask = mask.astype(bool)
    out = np.zeros_like(mask, dtype=np.float32)

    if not mask.any():
        return out

    sides = (1, -1)

    for i in range(min(int(num_branches), 2)):
        out = np.maximum(
            out,
            background_branch(
                mask,
                rng,
                width,
                inner_margin,
                outer_margin,
                sides[i],
            ),
        )

    for i in range(max(0, int(num_branches) - 2)):
        side = 1 if i % 2 == 0 else -1

        out = np.maximum(
            out,
            background_branch(
                mask,
                rng,
                width,
                inner_margin,
                outer_margin,
                side,
            ),
        )

    return out * (~mask).astype(np.float32)


def make_sparse(
    mask,
    mode="scribble",
    points_per_class=8,
    scribble_width=0,
    rng=None,
    bg_inner_margin=3,
    bg_outer_margin=18,
    bg_num_branches=2,
):
    if rng is None:
        rng = np.random.default_rng(42)

    pos = np.zeros_like(mask, dtype=np.float32)
    neg = np.zeros_like(mask, dtype=np.float32)

    for channel in range(mask.shape[0]):
        fg = mask[channel] > 0.5

        if mode == "point":
            fg_idx = np.argwhere(fg)

            if len(fg_idx):
                n = min(points_per_class, len(fg_idx))
                selected = fg_idx[
                    rng.choice(len(fg_idx), n, replace=False)
                ]
                pos[channel, selected[:, 0], selected[:, 1]] = 1.0

            bg_idx = np.argwhere(~fg)

            if len(bg_idx):
                n = min(points_per_class, len(bg_idx))
                selected = bg_idx[
                    rng.choice(len(bg_idx), n, replace=False)
                ]
                neg[channel, selected[:, 0], selected[:, 1]] = 1.0

        elif mode == "scribble":
            pos[channel] = foreground_scribble(
                fg,
                rng,
                scribble_width,
            )

            neg[channel] = background_scribble(
                fg,
                rng,
                scribble_width,
                bg_inner_margin,
                bg_outer_margin,
                bg_num_branches,
            )

        else:
            raise ValueError(f"Unknown sparse mode: {mode}")

        pos[channel] *= fg.astype(np.float32)
        neg[channel] *= (~fg).astype(np.float32)

    return np.concatenate((pos, neg), axis=0).astype(np.float32)


def augment(image, mask, rng):
    if rng.random() < 0.5:
        image = image[:, :, ::-1].copy()
        mask = mask[:, :, ::-1].copy()

    if rng.random() < 0.5:
        image = image[:, ::-1, :].copy()
        mask = mask[:, ::-1, :].copy()

    k = int(rng.integers(4))

    if k:
        image = np.rot90(image, k, axes=(1, 2)).copy()
        mask = np.rot90(mask, k, axes=(1, 2)).copy()

    return image, mask


class ScribbleDataset(Dataset):
    def __init__(
        self,
        image_dir,
        mask_dir,
        image_size=256,
        num_classes=1,
        mask_values=None,
        sparse_mode="scribble",
        points_per_class=8,
        mask_suffix="",
        augment=False,
        sparse_seed=42,
        scribble_width=0,
        bg_inner_margin=3,
        bg_outer_margin=18,
        bg_num_branches=2,
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_size = int(image_size)
        self.num_classes = int(num_classes)
        self.mask_values = mask_values
        self.sparse_mode = sparse_mode
        self.points_per_class = int(points_per_class)
        self.mask_suffix = str(mask_suffix)
        self.augment = bool(augment)
        self.sparse_seed = int(sparse_seed)
        self.scribble_width = int(scribble_width)
        self.bg_inner_margin = int(bg_inner_margin)
        self.bg_outer_margin = int(bg_outer_margin)
        self.bg_num_branches = int(bg_num_branches)

        extensions = {
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".tif",
            ".tiff",
        }

        self.images = sorted(
            p
            for p in self.image_dir.iterdir()
            if p.suffix.lower() in extensions
        )

        if not self.images:
            raise RuntimeError(f"No images found in {self.image_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        mask_path = self.mask_dir / (
            f"{image_path.stem}{self.mask_suffix}{image_path.suffix}"
        )

        if not mask_path.exists():
            raise FileNotFoundError(
                f"Mask not found for {image_path.name}"
            )

        image = load_image(image_path, self.image_size)
        mask = load_mask(mask_path, self.image_size)

        rng = np.random.default_rng(self.sparse_seed + index)

        if self.augment:
            image, mask = augment(image, mask, rng)

        sparse = make_sparse(
            mask,
            mode=self.sparse_mode,
            points_per_class=self.points_per_class,
            scribble_width=self.scribble_width,
            rng=rng,
            bg_inner_margin=self.bg_inner_margin,
            bg_outer_margin=self.bg_outer_margin,
            bg_num_branches=self.bg_num_branches,
        )

        return {
            "image": torch.from_numpy(image).float(),
            "mask": torch.from_numpy(mask).float(),
            "sparse": torch.from_numpy(sparse).float(),
            "name": image_path.name,
        }