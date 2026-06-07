import pandas as pd
import numpy as np
import os


def generate_simulation_dataset(num_records: int = 10000):
    """
    Generates a synthetic geo-attribution dataset compatible
    with the Streamlit Geo-Attribution Matrix Kernel.
    """

    print(f"[Simulation] Generating {num_records:,} mock user touchpoints...")

    # Ensure reproducibility
    np.random.seed(42)

    # --------------------------------------------------
    # 1. Marketing Channel Distribution
    # --------------------------------------------------
    channels = np.random.choice(
        [0, 1, 2],
        size=num_records,
        p=[0.5, 0.3, 0.2]
    )

    channel_map = {
        0: "meta_awareness_ad",
        1: "meta_conversion_ad",
        2: "whatsapp_lead_trigger"
    }

    utm_sources = [channel_map[channel] for channel in channels]

    # --------------------------------------------------
    # 2. Interaction Counts
    # --------------------------------------------------
    interactions = np.random.geometric(
        p=0.4,
        size=num_records
    )

    # --------------------------------------------------
    # 3. Distance Generation (km)
    # --------------------------------------------------
    distances = np.random.normal(
        loc=15.0,
        scale=8.0,
        size=num_records
    )

    distances = np.clip(
        distances,
        0.1,
        50.0
    )

    distances = np.round(distances, 2)

    # --------------------------------------------------
    # 4. Generate Geographic Coordinates
    # Simulated around Lagos, Nigeria
    # --------------------------------------------------
    center_lat = 6.5244
    center_lon = 3.3792

    latitudes = center_lat + np.random.normal(
        loc=0,
        scale=0.15,
        size=num_records
    )

    longitudes = center_lon + np.random.normal(
        loc=0,
        scale=0.15,
        size=num_records
    )

    # --------------------------------------------------
    # 5. Build DataFrame
    # --------------------------------------------------
    df = pd.DataFrame({
        "latitude": np.round(latitudes, 6),
        "longitude": np.round(longitudes, 6),
        "distance_km": distances,
        "utm_source": utm_sources,
        "interaction_count": interactions
    })

    # --------------------------------------------------
    # 6. Create Output Directory
    # --------------------------------------------------
    os.makedirs("core_logic", exist_ok=True)

    output_path = os.path.join(
        "core_logic",
        "simulated_touchpoints.csv"
    )

    # --------------------------------------------------
    # 7. Save Dataset
    # --------------------------------------------------
    df.to_csv(output_path, index=False)

    print("\n[Success] Dataset generated successfully.")
    print(f"[Location] {output_path}")

    print("\nDataset Preview:")
    print(df.head(10))

    print("\nColumns:")
    print(df.columns.tolist())

    print(f"\nTotal Records: {len(df):,}")


if __name__ == "__main__":
    generate_simulation_dataset(num_records=10000)

