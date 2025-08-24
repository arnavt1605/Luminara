import lightkurve as lk
import numpy as np
import os

def fetch_lightcurve(tic_id, output_dir="data/raw"):
    os.makedirs(output_dir, exist_ok=True)
    output_csv_filename=os.path.join(output_dir, f"{tic_id.replace(' ','_')}.csv")

    try:
        print(f"Searching for light curve")
        search_result= lk.search_lightcurve(tic_id)
        if len(search_result) == 0:
            raise ValueError(f"No light curve found for {tic_id}")
        
        spoc_result= search_result[search_result.author == "SPOC"]
        if len(spoc_result) == 0:
            raise ValueError(f"No sspoc pipeline data found for {tic_id}")
        
        sorted_idx= np.argsort(spoc_result.table['exptime'])
        best_result= spoc_result[sorted_idx[0]]

        row= best_result.table
        best_sector= row['mission'][0] if 'mission' in row.colnames else "Unknown"
        best_exposure= row['exptime'][0] if 'exptime' in row.colnames else "Unknown"

        lc= best_result.download()
        lc[["time", "pdcsap_flux"]].write(output_csv_filename, format="csv", overwrite=True)
        print(f"Light curve saved to {output_csv_filename}")

        return output_csv_filename
    
    except Exception as e:
        print(f"Error fetching light curve for {tic_id}: {e}")
        return None