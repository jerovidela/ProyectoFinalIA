#!/usr/bin/env python3
import cv2
import json
import numpy as np
from pathlib import Path
from skimage import morphology


class ImageFeatureExtractor:
    def __init__(self,
                 input_dir="out",
                 output_json_dir="features_json",
                 eps_frac=0.02):
        self.input_dir = Path(input_dir)
        self.output_json_dir = Path(output_json_dir)
        self.output_json_dir.mkdir(parents=True, exist_ok=True)
        self.eps_frac = eps_frac

    # ---------- helpers ----------

    def _binarize_and_get_main_mask(self, img_gray):
        _, bin_img = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
        if cv2.countNonZero(bin_img) > bin_img.size // 2:
            bin_img = cv2.bitwise_not(bin_img)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_img, 8)
        if num_labels <= 1:
            return None

        largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = (labels == largest_idx).astype(np.uint8) * 255
        return mask

    def _get_main_contour(self, mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        return max(contours, key=cv2.contourArea)

    # ---------- feature groups ----------

    def extract_shape_descriptors(self, contour, mask):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        x, y, w, h = cv2.boundingRect(contour)
        bbox_area = w * h

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)

        rect = cv2.minAreaRect(contour)
        rect_area = rect[1][0] * rect[1][1]

        (cx_circ, cy_circ), circle_radius = cv2.minEnclosingCircle(contour)

        if len(contour) >= 5:
            (ell_cx, ell_cy), (ell_MA, ell_mA), ell_angle = cv2.fitEllipse(contour)
        else:
            ell_cx = ell_cy = ell_MA = ell_mA = ell_angle = 0.0

        # --- FIX: clamp para que no haga sqrt de negativo ---
        if ell_MA > 0:
            ratio = ell_mA / ell_MA
            ratio = np.clip(ratio, 0.0, 1.0)
            ellipse_ecc = float(np.sqrt(1.0 - ratio ** 2))
        else:
            ellipse_ecc = 0.0

        return {
            "area": float(area),
            "perimeter": float(perimeter),
            "bbox_x": int(x),
            "bbox_y": int(y),
            "bbox_width": int(w),
            "bbox_height": int(h),
            "bbox_area": float(bbox_area),
            "aspect_ratio": float(w / h) if h > 0 else 0.0,
            "extent": float(area / bbox_area) if bbox_area > 0 else 0.0,
            "solidity": float(area / hull_area) if hull_area > 0 else 0.0,
            "compactness": float((perimeter ** 2) / area) if area > 0 else 0.0,
            "circularity": float((4 * np.pi * area) / (perimeter ** 2)) if perimeter > 0 else 0.0,
            "convex_area": float(hull_area),
            "convexity": float(hull_area / area) if area > 0 else 0.0,
            "min_rect_area_ratio": float(area / rect_area) if rect_area > 0 else 0.0,
            "circle_radius": float(circle_radius),
            "circle_area_ratio": float(area / (np.pi * circle_radius ** 2)) if circle_radius > 0 else 0.0,
            "ellipse_center_x": float(ell_cx),
            "ellipse_center_y": float(ell_cy),
            "ellipse_major_axis": float(ell_MA),
            "ellipse_minor_axis": float(ell_mA),
            "ellipse_eccentricity": ellipse_ecc,
            "ellipse_angle": float(ell_angle),
        }

    def extract_geometric_features(self, contour):
        eps = self.eps_frac * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, eps, True)
        pts = approx.reshape(-1, 2).astype(float)

        side_lengths = []
        angles = []
        n = len(pts)

        if n >= 3:
            for i in range(n):
                p_prev = pts[i - 1]
                p = pts[i]
                p_next = pts[(i + 1) % n]

                side = np.linalg.norm(p_next - p)
                side_lengths.append(side)

                v1 = p_prev - p
                v2 = p_next - p
                n1 = np.linalg.norm(v1)
                n2 = np.linalg.norm(v2)
                if n1 > 0 and n2 > 0:
                    cos_a = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
                    angle = np.degrees(np.arccos(cos_a))
                else:
                    angle = 0.0
                angles.append(angle)

        return {
            "polygon_vertices": int(n),
            "sides_mean": float(np.mean(side_lengths)) if side_lengths else 0.0,
            "sides_std": float(np.std(side_lengths)) if side_lengths else 0.0,
            "sides_min": float(np.min(side_lengths)) if side_lengths else 0.0,
            "sides_max": float(np.max(side_lengths)) if side_lengths else 0.0,
            "sides_ratio_max_min": float(np.max(side_lengths) / np.min(side_lengths)) if side_lengths and np.min(side_lengths) > 0 else 0.0,
            "angles_mean": float(np.mean(angles)) if angles else 0.0,
            "angles_std": float(np.std(angles)) if angles else 0.0,
            "angles_min": float(np.min(angles)) if angles else 0.0,
            "angles_max": float(np.max(angles)) if angles else 0.0,
            "angles_range": float((np.max(angles) - np.min(angles))) if angles else 0.0,
        }

    def extract_hu_moments(self, contour):
        M = cv2.moments(contour)
        hu = cv2.HuMoments(M).flatten()
        feats = {}
        for i, v in enumerate(hu, 1):
            if v == 0:
                feats[f"hu_moment_{i}"] = 0.0
            else:
                feats[f"hu_moment_{i}"] = float(-np.sign(v) * np.log10(abs(v)))
        return feats

    def extract_distance_features(self, contour):
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return {}
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        dists = []
        for p in contour:
            x, y = p[0]
            d = np.hypot(x - cx, y - cy)
            dists.append(d)
        dists = np.array(dists, dtype=float)

        if dists.size:
            hist, _ = np.histogram(dists, bins=10, range=(dists.min(), dists.max()))
        else:
            hist = []

        feat = {
            "centroid_x": float(cx),
            "centroid_y": float(cy),
            "distance_mean": float(dists.mean()) if dists.size else 0.0,
            "distance_std": float(dists.std()) if dists.size else 0.0,
            "distance_min": float(dists.min()) if dists.size else 0.0,
            "distance_max": float(dists.max()) if dists.size else 0.0,
            "distance_range": float(dists.max() - dists.min()) if dists.size else 0.0,
            "distance_cv": float(dists.std() / dists.mean()) if dists.size and dists.mean() > 0 else 0.0,
        }
        for i, h in enumerate(hist):
            feat[f"distance_hist_bin_{i}"] = int(h)
        return feat

    def extract_curvature_features(self, contour, k=5):
        pts = contour.reshape(-1, 2)
        n = len(pts)
        if n < 2 * k + 1:
            return {}
        curv = []
        for i in range(n):
            p1 = pts[(i - k) % n]
            p2 = pts[i]
            p3 = pts[(i + k) % n]
            v1 = p2 - p1  # 2D
            v2 = p3 - p2  # 2D
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 > 0 and n2 > 0:
                # --- FIX: cross 2D a mano: (x1, y1) x (x2, y2) = x1*y2 - y1*x2 ---
                cross = v1[0] * v2[1] - v1[1] * v2[0]
                c = cross / (n1 * n2)
                curv.append(abs(c))
        if not curv:
            return {}
        curv = np.array(curv, dtype=float)
        return {
            "curvature_mean": float(curv.mean()),
            "curvature_std": float(curv.std()),
            "curvature_max": float(curv.max()),
            "curvature_sum": float(curv.sum()),
        }

    def extract_skeleton_features(self, mask):
        skeleton = morphology.skeletonize(mask > 0)
        coords = np.column_stack(np.where(skeleton))
        if coords.size == 0:
            return {}
        skeleton_len = int(skeleton.sum())
        endpoints = 0
        branches = 0
        for (y, x) in coords:
            neigh = skeleton[max(0, y - 1):y + 2, max(0, x - 1):x + 2]
            count = int(neigh.sum()) - 1
            if count == 1:
                endpoints += 1
            elif count > 2:
                branches += 1
        return {
            "skeleton_length": skeleton_len,
            "skeleton_endpoints": int(endpoints),
            "skeleton_branches": int(branches),
            "skeleton_complexity": float((endpoints + branches) / skeleton_len) if skeleton_len > 0 else 0.0,
        }

    def extract_hole_features(self, mask):
        """
        FIX: usar connectedComponentsWithStats que sí devuelve 4 valores.
        """
        inv = cv2.bitwise_not(mask)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inv, 8)

        if num_labels <= 1:
            return {
                "holes_count": 0,
                "holes_mean_area": 0.0,
                "holes_max_area": 0.0,
            }

        areas = stats[1:, cv2.CC_STAT_AREA]  # skip label 0
        if len(areas) > 0:
            biggest = np.argmax(areas)
            hole_areas = [a for i, a in enumerate(areas) if i != biggest]
        else:
            hole_areas = []

        return {
            "holes_count": int(len(hole_areas)),
            "holes_mean_area": float(np.mean(hole_areas)) if hole_areas else 0.0,
            "holes_max_area": float(np.max(hole_areas)) if hole_areas else 0.0,
        }

    def extract_edge_features(self, mask):
        edges = cv2.Canny(mask, 50, 150)
        edge_count = int(np.count_nonzero(edges))
        mask_area = int(np.count_nonzero(mask))
        return {
            "edges_pixel_count": edge_count,
            "edges_density": float(edge_count / mask_area) if mask_area > 0 else 0.0,
        }

    # ---------- NEW: contour frequency analysis ----------

    def extract_contour_frequency_features(self, contour):
        """
        Turn contour into radial-distance signal and analyze its spectrum.
        Screws with thread should show stronger high-frequency content.
        """
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return {}

        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]

        # radial signal
        pts = contour.reshape(-1, 2)
        r = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)

        # remove DC (mean)
        r_centered = r - np.mean(r)

        # FFT
        fft_vals = np.fft.rfft(r_centered)
        mag = np.abs(fft_vals)

        if mag.size == 0:
            return {}

        # skip frequency 0 (DC)
        mag_no_dc = mag[1:]

        # basic stats
        dom_freq_idx = int(np.argmax(mag_no_dc)) + 1  # +1 because we skipped DC
        dom_freq_mag = float(mag[dom_freq_idx])

        # high frequency energy = top 20% of spectrum
        n = mag_no_dc.size
        if n > 0:
            hf_start = int(0.8 * n)
            hf_energy = float(np.sum(mag_no_dc[hf_start:]))
            total_energy = float(np.sum(mag_no_dc)) if np.sum(mag_no_dc) > 0 else 1.0
            hf_ratio = hf_energy / total_energy
        else:
            hf_energy = 0.0
            hf_ratio = 0.0

        return {
            "contour_fft_len": int(mag_no_dc.size),
            "contour_fft_dom_freq_idx": dom_freq_idx,
            "contour_fft_dom_freq_mag": dom_freq_mag,
            "contour_fft_highfreq_energy": hf_energy,
            "contour_fft_highfreq_ratio": hf_ratio,
        }

    # ---------- main per-image ----------

    def analyze_single_image(self, image_path: Path):
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"❌ Cannot read {image_path}")
            return None

        mask = self._binarize_and_get_main_mask(img)
        if mask is None:
            print(f"❌ No object in {image_path}")
            return None

        contour = self._get_main_contour(mask)
        if contour is None:
            print(f"❌ No contour in {image_path}")
            return None

        features = {
            "filename": image_path.name,
            "orig_width": int(img.shape[1]),
            "orig_height": int(img.shape[0]),
        }

        features.update(self.extract_shape_descriptors(contour, mask))
        features.update(self.extract_geometric_features(contour))
        features.update(self.extract_hu_moments(contour))
        features.update(self.extract_distance_features(contour))
        features.update(self.extract_curvature_features(contour))
        features.update(self.extract_skeleton_features(mask))
        features.update(self.extract_hole_features(mask))
        features.update(self.extract_edge_features(mask))
        # NEW
        features.update(self.extract_contour_frequency_features(contour))

        # quick label by name
        fname = image_path.name.lower()
        if "arandela" in fname:
            features["label_hint"] = "arandela"
        elif "tuerca" in fname:
            features["label_hint"] = "tuerca"
        elif "tornillo" in fname:
            features["label_hint"] = "tornillo"
        elif "clavo" in fname:
            features["label_hint"] = "clavo"
        else:
            features["label_hint"] = "unknown"

        return features

    # ---------- process all ----------

    def run(self):
        exts = [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]
        images = []
        for e in exts:
            images += list(self.input_dir.glob(f"*{e}"))
            images += list(self.input_dir.glob(f"*{e.upper()}"))

        if not images:
            print(f"❌ No images found in {self.input_dir}")
            return

        print(f"📁 Found {len(images)} images in {self.input_dir}")

        for img_path in images:
            feats = self.analyze_single_image(img_path)
            if feats is None:
                continue
            out_path = self.output_json_dir / f"{img_path.stem}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(feats, f, indent=2, ensure_ascii=False)
            print(f"✅ Saved features for {img_path.name} → {out_path}")


def main():
    extractor = ImageFeatureExtractor(
        input_dir="out",
        output_json_dir="outjson",
        eps_frac=0.02
    )
    extractor.run()


if __name__ == "__main__":
    main()
