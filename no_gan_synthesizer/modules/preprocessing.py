import pandas as pd
from typing import List
def wrap_category_columns(data:pd.DataFrame,
                          cat_cols:List[str])->pd.DataFrame:
    
    df = data.copy()

    cat_data = data[cat_cols]

    flag_vector = [list(row) for row in cat_data.drop_duplicates().to_records(index=False)]

    key_to_idx = {str(v).replace("[","").replace("]",""):i for i, v in
                  enumerate(flag_vector,1)}
    idx_to_key = {i:tuple(v) for i, v in enumerate(flag_vector,1)}


    # key_to_idx[str(idx_to_key[0]).replace("(","").replace(")","")]
    
    df["cat_label"] = [key_to_idx[str(row).replace("(","").replace(")","")] 
                       for row in cat_data.to_records(index=False)]

    return df, idx_to_key, key_to_idx