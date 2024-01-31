import pandas as pd
import hashlib
from itertools import permutations

df1 = pd.read_csv("keystreching-digitalcorp.txt")
df2 = pd.read_csv("rockyou.txt")

def hashing_func(input_string):
    hash_object = hashlib.sha512()
    hash_object.update(input_string.encode('utf-8'))
    return hash_object.hexdigest()

def indexer(temp_arr, index, permutations_arr):
    return list(map(temp_arr.__getitem__, permutations_arr[index]))

permutations_arr = list(permutations({0, 1, 2}, 3))

df1["password"] = None
for ii in range(len(df1)):
    for jj in range(len(df2)):
        found = False
        salt = df1.loc[ii, "salt"]
        password = df2.iloc[jj, 0]
        for kk in range(6):
            hash_result = ""
            for zz in range(2000):
                input_string = "".join(indexer([salt, password, hash_result], kk, permutations_arr))
                hash_result = hashing_func(input_string)
                if hash_result == df1.loc[ii, "hash_outcome"]:                    
                    df1.loc[ii, "password"] = password
                    found = True
                    break
            if found:
                break
        if found:
            break
        
print(df1[["username", "password"]])