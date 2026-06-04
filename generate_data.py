import pandas as pd
import numpy as np
import os


def generate_simulation_dataset(num_records: int = 10000):
    """
    Generates a large-scale, mathematically structured synthetic dataset
    to stress-test the geo-attribution matrix engine.
    """
    print(f"[Simulation] Generating {num_records} mock user touchpoints...")

    # 1. Randomly assign channel indices: 0 (Awareness), 1 (Conversion), 2 (WhatsApp)
    # Using a specific probability distribution favoring awareness ads
    channels = np.random.choice([0, 1, 2], size=num_records, p=[0.5, 0.3, 0.2])

    # 2. Generate random interaction counts using a geometric distribution
    # (Most users click 1-2 times, very few click many times)
    interactions = np.random.geometric(p=0.4, size=num_records)

    # 3. Generate spatial distances using a normal distribution centered around 15km
    # Clip at 0.1 to ensure no negative distances occur
    distances = np.random.normal(loc=15.0, scale=8.0, size=num_records)
    distances = np.clip(distances, 0.1, 50.0)

    # 4. Construct a Pandas DataFrame
    df = pd.DataFrame({
        "channel_index": channels,
        "interaction_count": interactions,
        "distance_km": np.round(distances, 2)
    })

    # 5. Ensure a dedicated core_logic output directory exists
    os.makedirs("core_logic", exist_ok=True)

    # Save to a clean CSV file
    output_path = os.path.join("core_logic", "simulated_touchpoints.csv")
    df.to_csv(output_path, index=False)

    print(f"[Success] Dataset successfully written to: {output_path}")
    print(df.head(10))  # Display the first 10 rows of the generated matrix


if __name__ == "__main__":
    # Seed the random generator for academic reproducibility
    np.random.seed(42)
    generate_simulation_dataset(num_records=10000)
