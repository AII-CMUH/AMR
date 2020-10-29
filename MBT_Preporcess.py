def data_proc(i):
    df_idx = MRSA_index
    dpath = df_idx.loc[i]
    _, data = ng.bruker.read_pdata(filepath)
    # baseline corraction
    temp_data = baseline_corr(data)[1999:20000]
    # normalized
    pdata = norm_scaler(temp_data)
    pdata[pdata < 0] = 0
    return pdata

def data_prep_batch(df_idx):
    db = []
    for i in range(len(df_idx)):
        # read and filtering data 
        db.append(data_proc.remote(i))
    return db
