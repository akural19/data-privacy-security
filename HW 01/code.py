import warnings 
import pandas as pd
import numpy as np
import os
import math
import sys
import time

warnings.filterwarnings('ignore')

if sys.version_info[0] < 3 or sys.version_info[1] < 5:
    sys.stdout.write("Requires Python 3.5+.\n")
    sys.exit(1)

def construct_dgh(file_path) :
    with open(file_path, "r") as file:
        string_array = file.readlines()
        dgh = []
        for ii in range(len(string_array)):
            item = string_array[ii].rstrip()
            level = item.count("\t") + 1
            pair = (item.lstrip(), level)
            dgh.append(pair)
        return dgh

def node_index(node_name, dgh):
    for ii in range(len(dgh)):
        node = dgh[ii]
        if node[0] == node_name:
            return ii
    raise Exception("node_index -> Node not found in dgh!") 

def num_descendant_leaves(node_name, dgh):
    num_descendant = 0
    index = node_index(node_name, dgh)
    node_level = dgh[index][1]
    if index == len(dgh) - 1:
        num_descendant = 1
    else :
        for ii in range(index, len(dgh) - 1):
            if index != ii and node_level >= dgh[ii][1]:
                break
            elif dgh[ii][1] < dgh[ii+1][1]:
                if ii == (len(dgh) - 2):
                    num_descendant += 1
                else:
                    continue
            else:
                if (ii == len(dgh) - 2) and (dgh[len(dgh) - 1][1] > node_level):
                    num_descendant += 2
                else:
                    num_descendant += 1
    return num_descendant

def return_level(node_name, dgh):
    for ii in range (len(dgh)):
        node = dgh[ii]
        if node[0] == node_name:
            return node[1]
    raise Exception("return_level -> Node not found in dgh!") 
    
def df_level(df, DGHs):
    df_level_score = 0
    for ii in range(len(df.columns) - 1):
        attr = df.columns[ii]
        for jj in range(len(df)):
            value = return_level(df.loc[jj, attr], DGHs[attr])
            df_level_score += value
    return df_level_score

def cost_MD(raw_dataset_file, anonymized_dataset_file, DGH_folder):
    raw_df = pd.read_csv(raw_dataset_file, dtype = str)
    anonymized_df = pd.read_csv(anonymized_dataset_file, dtype = str)
    DGHs = {}
    for filename in os.listdir(DGH_folder):
        key = filename.strip(".txt")
        value = construct_dgh(DGH_folder + "/" + filename)
        DGHs[key] = value 
    raw_df_node_level_score = df_level(raw_df, DGHs)
    anonymized_df_level_score = df_level(anonymized_df, DGHs)
    return (raw_df_node_level_score - anonymized_df_level_score)

def df_lm_cost(df, DGHs):
    df_lm_cost = 0
    for ii in range(len(df.columns) - 1):
        attr = df.columns[ii]
        dgh = DGHs[attr]
        total_num_leaves = num_descendant_leaves(dgh[0][0],dgh)
        for jj in range(len(df)): 
            entry_lm_cost = (num_descendant_leaves(df.loc[jj, attr], dgh) - 1) / (total_num_leaves - 1)
            df_lm_cost += entry_lm_cost
    return (df_lm_cost / (len(df.columns) - 1))

def cost_LM(raw_dataset_file, anonymized_dataset_file, DGH_folder):
    raw_df = pd.read_csv(raw_dataset_file, dtype = str)
    anonymized_df = pd.read_csv(anonymized_dataset_file, dtype = str)
    DGHs = {}
    for filename in os.listdir(DGH_folder):
        key = filename.strip(".txt")
        value = construct_dgh(DGH_folder + "/" + filename)
        DGHs[key] = value
    return (df_lm_cost(anonymized_df, DGHs) - df_lm_cost(raw_df, DGHs))

def list_childs(node_name, dgh):
    childs = []
    index = node_index(node_name, dgh)
    level = dgh[index][1]
    for ii in range(index + 1, len(dgh)):
        if dgh[ii][1] == level:
            break
        elif dgh[ii][1] == level + 1:
            childs.append(dgh[ii][0])
    return childs

def find_parent(node_name, dgh):
    index = node_index(node_name, dgh)
    level = dgh[index][1]
    while index >= 0:
        current_node = dgh[index]
        if current_node[1] < level:
            return current_node[0]
        index -= 1
    return node_name

def find_common_ancestor(node_name1, node_name2, dgh):
    node_level1 = return_level(node_name1, dgh) 
    node_level2 = return_level(node_name2, dgh)
    
    if node_level1 > node_level2:
        while node_level1 != node_level2:
            node_name1 = find_parent(node_name1, dgh)
            node_level1 -= 1
    elif node_level2 > node_level1:
        while node_level2 != node_level1:
            node_name2 = find_parent(node_name2, dgh)
            node_level2 -= 1
    if node_name1 == node_name2:
        return node_name1 
    else:
        parent1 = find_parent(node_name1, dgh)
        parent2 = find_parent(node_name2, dgh)
        return find_common_ancestor(parent1, parent2, dgh)
    
def create_equivalence_class(df, DGHs):
    for ii in range(len(df.columns) - 2):
        attr = df.columns[ii]
        unique_elements = list(dict(df[attr].value_counts()).keys())
        class_qi = unique_elements[0]
        qi_level = return_level(class_qi, DGHs[attr])
        for jj in range(len(unique_elements) - 1):
            for kk in range(jj + 1, len(unique_elements)):
                common_ancestor = find_common_ancestor(unique_elements[jj], unique_elements[kk], DGHs[attr])
                ancestor_level = return_level(common_ancestor, DGHs[attr])
                if ancestor_level < qi_level:
                    class_qi = common_ancestor
                    qi_level = ancestor_level
        df.loc[:, attr] = class_qi
    return df

def random_anonymizer(raw_dataset_file, DGH_folder, k, output_file, s):
    raw_df = pd.read_csv(raw_dataset_file, dtype = str)
    DGHs = {}
    for filename in os.listdir(DGH_folder):
        key = filename.strip(".txt")
        value = construct_dgh(DGH_folder + "/" + filename)
        DGHs[key] = value
    raw_df["row-number"] = raw_df.index
    np.random.seed(s)
    shuffle_arr = np.array(raw_df)
    np.random.shuffle(shuffle_arr)
    suffled_raw_df = pd.DataFrame(shuffle_arr)
    suffled_raw_df.columns = raw_df.columns
    
    num_classes = math.floor(len(suffled_raw_df) / k)
    result_df = pd.DataFrame()
    for ii in range(num_classes):
        if ii != (num_classes - 1):
            copy_df = create_equivalence_class(suffled_raw_df[ii*k:(ii + 1)*k].copy(), DGHs)
        else :
            copy_df = create_equivalence_class(suffled_raw_df[ii*k:].copy(), DGHs)
        result_df = pd.concat([result_df, copy_df], axis = 0)
    result_df.sort_values(by = ["row-number"], ascending = True, inplace = True)
    result_df.drop("row-number", axis = 1, inplace = True)
    result_df.to_csv(output_file, index = False)
    
def records_distance(r1, r2, DGHs):
    records_lm_cost = 0
    for ii in range(len(r1) - 2):
        attr = r1.index[ii]
        dgh = DGHs[attr]
        total_num_leaves = num_descendant_leaves(dgh[0][0],dgh)
        ancestor = find_common_ancestor(r1[ii], r2[ii], dgh)
        r1_lm_cost = (num_descendant_leaves(r1[ii], dgh) - 1) / (total_num_leaves - 1)
        r2_lm_cost = (num_descendant_leaves(r2[ii], dgh) - 1) / (total_num_leaves - 1)   
        ancestor_lm_cost = (num_descendant_leaves(ancestor, dgh) - 1) / (total_num_leaves - 1)   
        records_lm_cost += ancestor_lm_cost * 2 - (r1_lm_cost + r2_lm_cost)
    return records_lm_cost / (len(r1) - 2)

def clustering_anonymizer(raw_dataset_file, DGH_folder, k, output_file):
    raw_df = pd.read_csv(raw_dataset_file, dtype = str)
    DGHs = {}
    for filename in os.listdir(DGH_folder):
        key = filename.strip(".txt")
        value = construct_dgh(DGH_folder + "/" + filename)
        DGHs[key] = value
    result_df = pd.DataFrame()
    
    raw_df["row-number"] = raw_df.index
    unused_indices = list(raw_df.index)
    
    num_classes = math.floor(len(raw_df) / k)
   
    for ii in range(num_classes):
        distances = []
        current_index = unused_indices[0]
        unused_indices.remove(current_index)
        rec = raw_df.loc[current_index]
       
        for index in unused_indices:
            distance_pair = (records_distance(rec, raw_df.loc[index], DGHs), index)
            distances.append(distance_pair)
        distances.sort()
        if ii != (num_classes - 1):
            distances = distances[:(k-1)]
            
        indices = [current_index] 
        for tup in distances:
            indices.append(tup[1])
            unused_indices.remove(tup[1])
        
        copy_df = create_equivalence_class(raw_df.loc[indices].copy(), DGHs)
        result_df = pd.concat([result_df, copy_df], axis = 0)

    result_df.sort_values(by = ["row-number"], ascending = True, inplace = True)
    result_df.drop("row-number", axis = 1, inplace = True)
    result_df.to_csv(output_file, index = False)
    
def fully_generalize(record, DGHs):
    for attr in record.index:
        dgh = DGHs[attr]
        root_name = dgh[0][0]
        record[attr] = root_name
    return record

def is_compatible(r1, g1, DGHs):
    for ii in range(len(r1) - 1):
        record_name = r1[ii]
        attr = r1.index[ii]
        dgh = DGHs[attr]
        record_level = return_level(record_name, dgh)
        generalization_level = return_level(g1[ii], dgh)
        while record_level > generalization_level:
            record_name = find_parent(record_name, dgh)
            record_level -= 1
        if record_name != g1[ii]:
            return False
    return True

def child_num_in_df(df, attr, current_node, childs, DGHs):
    present_childs = []
    for child in childs:
        compatible_indices = []
        temp_node = current_node.copy()
        temp_node[attr] = child
        for ii in df.index:
            if is_compatible(df.loc[ii], temp_node, DGHs):
                compatible_indices.append(ii)
        num_compatible = len(compatible_indices)
        if num_compatible != 0:
            present_childs.append((child, num_compatible, compatible_indices))
    return present_childs

def is_k_anonymous(present_childs, k):
    for tup in present_childs:
        if tup[1] < k:
            return False
    return True

def l1_distance(num_childs, temp_present_childs):
    total_occurences = 0
    distance_cost = 0
    distribution = 1 / num_childs
    for tup in temp_present_childs:
        total_occurences += tup[1]
    for tup in temp_present_childs:
        marginal_cost = abs((tup[1] / total_occurences) - distribution) 
        distance_cost += marginal_cost
    return distance_cost

def min_child_attribute(df, current_node, DGHs, k):
    min_child_num = float("inf")
    min_child_attr = None
    present_childs = None
    distance = None
    for ii in range(len(current_node)):
        attr = current_node.index[ii]
        dgh = DGHs[attr]
        childs = list_childs(current_node[ii], dgh)
        temp_present_childs = child_num_in_df(df, attr, current_node, childs, DGHs)
        num_childs = len(temp_present_childs)
        if (num_childs < min_child_num and num_childs != 0):
            if is_k_anonymous(temp_present_childs, k):
                min_child_num = num_childs
                min_child_attr = attr
                present_childs = temp_present_childs
                distance = l1_distance(min_child_num, present_childs)
        elif (num_childs == min_child_num):
            if is_k_anonymous(temp_present_childs, k):
                temp_distance = l1_distance(num_childs, temp_present_childs)
                if distance > temp_distance: 
                    min_child_num = num_childs
                    min_child_attr = attr
                    present_childs = temp_present_childs
                    distance = temp_distance
    return min_child_attr, present_childs

def construct_nodes(current_node, min_child_attr, present_childs):
    nodes_list = []
    for tup in present_childs:
        temp_node = current_node.copy()
        temp_node[min_child_attr] = tup[0]
        nodes_list.append((temp_node, tup[2]))
    return nodes_list

def find_end_states(df, nodes_list, DGHs, k):
    end_nodes_list = []
    for node in nodes_list:
        current_node = node[0]
        min_child_attr, present_childs = min_child_attribute(df.loc[node[1]], current_node, DGHs, k)
        if present_childs == None:
            end_nodes_list.append(node)
        else:
            new_nodes = construct_nodes(current_node, min_child_attr, present_childs)
            end_nodes_list.extend(find_end_states(df, new_nodes, DGHs, k))
    return end_nodes_list

def topdown_anonymizer(raw_dataset_file, DGH_folder, k, output_file):
    raw_df = pd.read_csv(raw_dataset_file, dtype = str)
    DGHs = {}
    for filename in os.listdir(DGH_folder):
        key = filename.strip(".txt")
        value = construct_dgh(DGH_folder + "/" + filename)
        DGHs[key] = value

    root_node = fully_generalize(raw_df.iloc[0, :(len(raw_df.columns) - 1)].copy(), DGHs)
    end_nodes_list = find_end_states(raw_df, [(root_node, raw_df.index)], DGHs, k)
    for end_node in end_nodes_list:
        node = end_node[0]
        indices = end_node[1]
        
        for ii in range(len(raw_df.columns) - 1):
            attr = raw_df.columns[ii]
            raw_df.loc[indices, attr] = node[attr]
        
    raw_df.to_csv(output_file, index = False)
    
# Command line argument handling and calling of respective anonymizer:
if len(sys.argv) < 6:
    print(f"Usage: python3 {sys.argv[0]} algorithm DGH-folder raw-dataset.csv anonymized.csv k")
    print(f"\tWhere algorithm is one of [clustering, random, topdown]")
    sys.exit(1)

algorithm = sys.argv[1]
if algorithm not in ['clustering', 'random', 'topdown']:
    print("Invalid algorithm.")
    sys.exit(2)

dgh_path = sys.argv[2]
raw_file = sys.argv[3]
anonymized_file = sys.argv[4]
k = int(sys.argv[5])

function = eval(f"{algorithm}_anonymizer")

if function == random_anonymizer:
    if len(sys.argv) < 7:
        print(f"Usage: python3 {sys.argv[0]} algorithm DGH-folder raw-dataset.csv anonymized.csv k seed(for random only)")
        print(f"\tWhere algorithm is one of [clustering, random, topdown]")
        sys.exit(1)
        
    seed = int(sys.argv[6])
    start_time = time.time()
    function(raw_file, dgh_path, k, anonymized_file, seed)
else:
    start_time = time.time()    
    function(raw_file, dgh_path, k, anonymized_file)

end_time = time.time()
run_time = end_time - start_time

cost_md = cost_MD(raw_file, anonymized_file, dgh_path)
cost_lm = cost_LM(raw_file, anonymized_file, dgh_path)
print (f"Results of {k}-anonimity:\n\tCost_MD: {cost_md}\n\tCost_LM: {cost_lm}\n\tCost_Runtime: {run_time: .3f}s\n")