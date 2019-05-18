from scipy import sparse
import pandas as pd

def to_sparse(t, m, n):  # create sparse matrix from multi-index
    i, j = list(zip(*t.index))
    data = t.values
    return sparse.coo_matrix((data, (i, j)), shape=(m, n))


def map_index(df, col_list, dict_list):  # col is string array of column names, dict is a dictionary array
    df_mapped = df.copy()
    for i in range(len(col_list)):
        col = col_list[i]
        d = dict_list[i]
        # only apply if x exists as key , else return None

        df_mapped[col] = df_mapped[col].apply(lambda x: d.get(x))
        df_mapped = df_mapped.dropna(subset=[col, ])
    return df_mapped.set_index(col_list)


def create_user_dict(diffusion):
    # create a uid dict
    user_dict = {}
    uid = diffusion[['user_id']].drop_duplicates().reset_index(drop=True)
    for row in uid.itertuples():
        user_dict[row.user_id] = row.Index
    return user_dict


def create_message_dict(diffusion):
    message_dict = {}
    mid = diffusion[['original_status_id']].drop_duplicates().reset_index(drop=True)

    for row in mid.itertuples():
        message_dict[row.original_status_id] = row.Index

    return message_dict
