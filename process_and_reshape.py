import pandas as pd
import numpy as np
import os

def process_lightcurve(raw_csv_path, output_dir="data/processed", target_length= 3197):
    os.makedirs(output_dir, exist_ok=True)
    tic_id = os.path.splitext(os.path.basename(raw_csv_path))[0]
    output_csv_filename = os.path.join(output_dir, f"{tic_id}_processed.csv")

    try:
        df= pd.read_csv(raw_csv_path)
        flux= df["pdcsap_flux"].values
        flux= flux[~np.isnan(flux)]
        flux= (flux - np.mean(flux)) / np.std(flux)

        if len(flux) > target_length:
            flux= flux[:target_length]
        else:
            flux= np.pad(flux, (0, target_length - len(flux)), 'constant', constant_values=0)

        processed_df= pd.DataFrame([np.concatenate(([0], flux))])
        processed_df.columns= ["LABEL"] + [f"FLUX_{i}" for i in range(target_length)]

        processed_df.to_csv(output_csv_filename, index=False)
        print(f"Processed light curve saved to {output_csv_filename}")

        return output_csv_filename
    
    except Exception as e:
        print(f"Error while processing {raw_csv_path}: {e}")
        return None