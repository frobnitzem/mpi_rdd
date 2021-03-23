import pandas as pd
for i in range(20):
    pd.DataFrame([(4*i+j,j*j) for j in range(4)], columns=['A', 'B']).to_parquet("data.%d.pq"%i)

