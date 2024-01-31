import pandas as pd
import hashlib

df1 = pd.read_csv("salty-digitalcorp.txt")
df2 = pd.read_csv("rockyou.txt")

def hashing_func(string, salt, opt):
    hash_object = hashlib.sha512()
    input_string = ""
    if opt == 0:
        input_string = salt + string
    elif opt == 1:
        input_String = string + salt
    hash_object.update(input_string.encode('utf-8'))
    return hash_object.hexdigest()

df1["password"] = None
for ii in range(len(df1)):
    for jj in range(len(df2)):
        if hashing_func(df2.iloc[jj,0], df1.loc[ii, "salt"], 0) == df1.loc[ii, "hash_outcome"]:
            df1.loc[ii, "password"] = df2.iloc[jj, 0]
            break
        elif hashing_func(df2.iloc[jj,0], df1.loc[ii, "salt"], 1) == df1.loc[ii, "hash_outcome"]:
            df1.loc[ii, "password"] = df2.iloc[jj, 0]
            break
        
print(df1[["username", "password"]])