from astroquery.simbad import Simbad

def get_star_info(tic_id: str) -> dict:
    try:
        # Add extra fields: all known identifiers
        Simbad.add_votable_fields("ids")

        result_table = Simbad.query_object(tic_id)
        if result_table is None:
            return {"error": f"No SIMBAD results for {tic_id}"}

        # Main name SIMBAD prefers
        star_name = result_table['main_id'][0]

        # Coordinates
        ra = result_table['ra'][0]
        dec = result_table['dec'][0]

        # Spectral type if available
        spectral_type = (
            result_table['sp_type'][0]
            if 'sp_type' in result_table.colnames and result_table['sp_type'][0]
            else "Not Available"
        )

        other_ids = []
        if 'IDS' in result_table.colnames:
            other_ids = result_table['IDS'][0].split('|')

        return {
            "star_name": star_name,
            "ra": ra,
            "dec": dec,
            "spectral_type": spectral_type,
            "other_ids": other_ids
        }

    except Exception as e:
        return {"error": str(e)}
