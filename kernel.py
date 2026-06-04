import numpy as np
import pandas as pd
import os


class VectorizedGeoAttributionEngine:
    def __init__(self, decay_constant: float):
        """
        Initializes the engine using NumPy arrays for high-performance matrix execution.
        """
        self.lamda = decay_constant

        # Channel weight mapping: 0 = awareness, 1 = conversion, 2 = whatsapp
        self.channel_weights = np.array([0.5, 1.5, 2.5])
        print("[System] Vectorized Matrix Engine Active.")

    def calculate_matrix_attribution(self, data_matrix: np.ndarray) -> np.ndarray:
        """
        Processes an entire multi-dimensional core_logic array simultaneously using linear algebra.
        Returns a vector containing individual scores for every single row.
        """
        channel_indices = data_matrix[:, 0].astype(int)
        interactions = data_matrix[:, 1]
        distances = data_matrix[:, 2]

        # Vectorized array indexing and mathematical transformations
        w_c = self.channel_weights[channel_indices]
        normalized_interactions = np.log(interactions + 1)
        spatial_decay = np.exp(-self.lamda * distances)

        # Compute element-wise attribution vector
        touchpoint_scores = w_c * normalized_interactions * spatial_decay
        return touchpoint_scores


# --- BATCH PIPELINE RUN ---
if __name__ == "__main__":
    # 1. Initialize the computing engine
    matrix_engine = VectorizedGeoAttributionEngine(decay_constant=0.04)

    # 2. Define the path to our synthetic dataset
    csv_path = os.path.join("core_logic", "simulated_touchpoints.csv")

    if not os.path.exists(csv_path):
        print(f"[Error] Target dataset not found at {csv_path}. Run generate_data.py first.")
    else:
        print(f"[Pipeline] Ingesting source core_logic from: {csv_path}")

        # 3. Read CSV using Pandas
        raw_df = pd.read_csv(csv_path)

        # 4. Extract underlying values directly into a raw NumPy multi-dimensional array
        # This converts a friendly table into a high-speed mathematical matrix
        computation_matrix = raw_df.to_numpy()

        print(f"[Pipeline] Matrix dimensions mapped: {computation_matrix.shape} (Rows, Columns)")

        # 5. Execute matrix calculations across all 10,000 rows simultaneously
        calculated_scores = matrix_engine.calculate_matrix_attribution(computation_matrix)

        # 6. Append calculated analytics back to the dataframe for macro reporting
        raw_df["attribution_probability_score"] = calculated_scores

        # 7. Calculate aggregate macro statistics to prove system stability
        total_global_score = calculated_scores.sum()
        mean_pipeline_score = calculated_scores.mean()
        max_outlier_score = calculated_scores.max()

        print("\n" + "=" * 45)
        print("📊 PIPELINE METRICS & MASS-SCALE ANALYSIS")
        print("=" * 45)
        print(f"Total Rows Processed       : {len(raw_df)}")
        print(f"Cumulative System Score   : {total_global_score:.4f}")
        print(f"Mean Interaction Score     : {mean_pipeline_score:.4f}")
        print(f"Max Outlier Intent Score   : {max_outlier_score:.4f}")
        print("=" * 45)

        # Save enriched analytical core_logic back out for visualization phases
        output_analyzed_path = os.path.join("core_logic", "analyzed_metrics.csv")
        raw_df.to_csv(output_analyzed_path, index=False)
        print(f"[Success] Enriched analytics exported to: {output_analyzed_path}\n")
