import numpy as np
import networkx as nx
import logging

logger = logging.getLogger(__name__)

class QualityGate:
    def __init__(self, min_confidence=0.45, max_loss=2.5, min_angle_deg=5.0, max_angle_deg=120.0):
        # Thiết lập các hằng số làm ngưỡng đánh giá
        self.min_confidence = min_confidence
        self.max_loss = max_loss
        self.min_angle = min_angle_deg
        self.max_angle = max_angle_deg

    def build_covisibility_graph(self, poses):
        """
        Xây dựng đồ thị quan sát chung (Co-visibility Graph) giữa các camera bằng NetworkX.
        Các đỉnh (nodes): từng camera góc nhìn i in [0, N-1].
        Các cạnh (edges): kết nối giữa 2 camera nếu góc chênh lệch nằm trong ngưỡng an toàn [min_angle, max_angle].
        """
        n = len(poses)
        G = nx.Graph()
        G.add_nodes_from(range(n))

        if n < 2:
            return G

        # Trích xuất vector hướng nhìn (forward vector) từ ma trận pose
        forward_vectors = []
        for p in poses:
            z_dir = p[:3, 2]
            norm = np.linalg.norm(z_dir)
            forward_vectors.append(z_dir / max(norm, 1e-6))

        for i in range(n):
            for j in range(i + 1, n):
                cos_theta = np.dot(forward_vectors[i], forward_vectors[j])
                cos_theta = np.clip(cos_theta, -1.0, 1.0)
                angle = np.degrees(np.arccos(cos_theta))
                # Kết nối nếu góc chênh lệch đủ để có overlap nhưng không quá lớn
                if self.min_angle <= angle <= self.max_angle:
                    G.add_edge(i, j, weight=float(angle))

        return G

    def evaluate(self, poses, confidence_map, ba_loss):
        """
        Đánh giá chất lượng tái tạo 3D theo 3 tầng kiểm soát:
        1. Kiểm tra Confidence (Độ tự tin matching điểm)
        2. Kiểm tra Bundle Adjustment Loss (Sai số tối ưu toàn cục)
        3. Phân tích đồ thị liên thông NetworkX (Camera Co-visibility Graph Connectivity)
        """
        # 1. Kiểm tra Confidence
        mean_conf = np.mean(confidence_map)
        if mean_conf < self.min_confidence:
            return False, f"Low confidence: {mean_conf:.2f} < {self.min_confidence}"

        # 2. Kiểm tra Bundle Adjustment Loss
        if ba_loss > self.max_loss:
            return False, f"High loss: {ba_loss:.2f} > {self.max_loss}"

        # 3. Kiểm tra tính liên thông của đồ thị quan sát qua NetworkX
        if poses is not None and len(poses) >= 2:
            G = self.build_covisibility_graph(poses)
            if not nx.is_connected(G):
                num_components = nx.number_connected_components(G)
                return False, f"Disconnected camera graph ({num_components} components) — Góc chụp bị đứt đoạn"

        # Vượt qua toàn bộ bài test
        return True, "Quality passed"   