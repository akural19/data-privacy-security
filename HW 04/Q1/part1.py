import pandas as pd
import hashlib

df1 = pd.read_csv("digitalcorp.txt")
df2 = pd.read_csv("rockyou.txt", header = None)

def hashing_func(input_string):
    hash_object = hashlib.sha512()
    hash_object.update(input_string.encode('utf-8'))
    return hash_object.hexdigest()

df2.columns = ["password"]
df2["hash_of_password"] = None
for ii in range(len(df2)):
    df2.iloc[ii,1] = hashing_func(df2.iloc[ii,0])
    
result_df = pd.merge(df1, df2, on = "hash_of_password", how = "left")

print(result_df[["username", "password"]])