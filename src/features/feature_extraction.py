import numpy as np
import cv2
from skimage.morphology import skeletonize, thin
from skimage.measure import label, regionprops
from scipy.ndimage import distance_transform_edt

class VesselFeatureExtractor:
    def __init__(self, optic_disc_mask_ratio=0.15):
        self.od_ratio = optic_disc_mask_ratio

    def exclude_optic_disc(self, mask):
        """Mask central 15% of image to exclude optic disc features."""
        h, w = mask.shape
        cy, cx = h // 2, w // 2
        # Calculate radius for roughly 15% area exclusion
        r = int(min(h, w) * np.sqrt(self.od_ratio / np.pi))
        
        output_mask = mask.copy()
        cv2.circle(output_mask, (cx, cy), r, 0, -1)
        return output_mask

    def get_fractal_dimension(self, mask):
        """Box-counting method: ~1.7 healthy, ~1.5 diseased."""
        # Only count non-zero pixels
        pixels = np.where(mask > 0)
        if len(pixels[0]) == 0: return 0
        
        p = np.stack(pixels, axis=1)
        
        # Binary box counting
        def count_boxes(radius):
            counts = np.unique(p // radius, axis=0).shape[0]
            return counts

        scales = np.logspace(0.5, 4, num=10, endpoint=False, base=2).astype(int)
        counts = [count_boxes(s) for s in scales]
        
        # Linear fit in log-log space
        coeffs = np.polyfit(np.log(1/scales), np.log(counts), 1)
        return coeffs[0]

    def get_tortuosity(self, mask):
        """Arc/chord length ratio per connected segment."""
        skel = skeletonize(mask > 0).astype(np.uint8)
        labeled_skel = label(skel)
        props = regionprops(labeled_skel)
        
        tortuosities = []
        for prop in props:
            # Arc length is the number of pixels in the segment
            arc_len = prop.area 
            if arc_len < 10: continue # Skip noise
            
            # Chord length: Euclidean distance between endpoints
            coords = prop.coords
            # Simple heuristic for endpoints: furthest points in the segment
            dist_matrix = np.linalg.norm(coords[:, None] - coords, axis=2)
            chord_len = np.max(dist_matrix)
            
            if chord_len > 0:
                tortuosities.append(arc_len / chord_len)
        
        return np.mean(tortuosities) if tortuosities else 1.0

    def get_av_ratio_proxy(self, mask):
        """Thin vs thick vessel pixel ratio (arteriolar narrowing)."""
        dist_map = distance_transform_edt(mask > 0)
        # Using distance transform: 
        # Large values = thick vessels (veins proxy)
        # Small values = thin vessels (arterioles proxy)
        thick_threshold = np.percentile(dist_map[dist_map > 0], 70)
        
        thin_vessels = np.sum((dist_map > 0) & (dist_map < thick_threshold))
        thick_vessels = np.sum(dist_map >= thick_threshold)
        
        return thin_vessels / (thick_vessels + 1e-6)

    def extract_all_features(self, binary_mask):
        """Extracts 7 vessel features (4 basic + 3 new)."""
        # Pre-process: Exclude Optic Disc
        clean_mask = self.exclude_optic_disc(binary_mask)
        
        # 1-3. New features
        fractal_dim = self.get_fractal_dimension(clean_mask)
        tortuosity = self.get_tortuosity(clean_mask)
        av_ratio = self.get_av_ratio_proxy(clean_mask)
        
        # 4-7. Structural features
        density = np.mean(clean_mask > 0)
        
        dist_map = distance_transform_edt(clean_mask > 0)
        avg_width = np.mean(dist_map[dist_map > 0]) * 2 if np.any(dist_map > 0) else 0
        
        skel = skeletonize(clean_mask > 0)
        branch_points = self._count_branch_points(skel)
        
        # Complexity (Perimeter-to-Area)
        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = sum(cv2.arcLength(c, True) for c in contours)
        complexity = perimeter / (np.sum(clean_mask > 0) + 1e-6)

        feature_dict = {
            "fractal_dim": fractal_dim,
            "tortuosity": tortuosity,
            "av_ratio": av_ratio,
            "vessel_density": density,
            "avg_width": avg_width,
            "branch_points": branch_points,
            "vessel_complexity": complexity
        }
        
        # Calculate cvd_score: weighted sum (Page 6)
        feature_dict["cvd_score"] = (
            (fractal_dim * 0.3) + 
            (av_ratio * 0.2) + 
            (density * 0.15) + 
            (avg_width * 0.1) + 
            (tortuosity * 0.1) +
            (branch_points * 0.1) +
            (complexity * 0.05)
        )
        
        return feature_dict

    def _count_branch_points(self, skeleton):
        """Counts junctions in the skeletonized vessel map."""
        skel_img = skeleton.astype(np.uint8)
        # Kernel to find pixels with > 2 neighbors
        kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 11, 1]], dtype=np.uint8)
        # Simplified: check neighborhood sum
        filtered = cv2.filter2D(skel_img, -1, np.ones((3,3))) * skel_img
        return np.sum(filtered > 3)