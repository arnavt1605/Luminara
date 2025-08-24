from astroquery.simbad import Simbad

def get_star_info(tic_id: str) -> dict:
    try:
        result_table = Simbad.query_object(tic_id)
        if result_table is None:
            return {"error": f"No SIMBAD results for {tic_id}"}

        star_name = result_table['main_id'][0]
        ra = result_table['ra'][0]
        dec = result_table['dec'][0]
        spectral_type = (
            result_table['sp_type'][0]
            if 'sp_type' in result_table.colnames and result_table['sp_type'][0]
            else "Not Available"
        )

        return {
            "star_name": star_name,
            "ra": ra,
            "dec": dec,
            "spectral_type": spectral_type
        }
    except Exception as e:
        return {"error": str(e)}