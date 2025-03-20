def extract_tree_info(xgb_model, tree_index, feature_names):
    """
    Extracts tree structure from XGBoost model in a usable format.
    :param xgb_model: Trained XGBoost model
    :param tree_index: Index of the tree to extract
    :param feature_names: List of feature names in the dataset
    :return: Dictionary representing tree structure
    """
    booster = xgb_model.get_booster()
    dump = booster.get_dump(fmap="", with_stats=True)
    tree_str = dump[tree_index]
    
    tree = {'v': {}, 'a': {}, 'b': {}, 't': {}, 'r': {}, 'd': {}}
    
    for line in tree_str.split("\n"):
        if line.strip():
            parts = line.split("[")
            node_id = int(parts[0].split(":")[0])
            
            if len(parts) > 1 and "<" in parts[1]:  # Internal node
                condition = parts[1].split("]")[0]
                feature, threshold = condition.split("<")
                left_right = parts[1].split(",")
                left_child = int(left_right[0].split("=")[1])
                right_child = int(left_right[1].split("=")[1].split(" ")[0])
                cover = float(parts[1].split("cover=")[1])
                
                tree['v'][node_id] = "internal"
                
                # Convert feature name to index if needed
                if feature in feature_names:
                    tree['d'][node_id] = feature_names.index(feature)  # Find index in feature list
                elif feature.startswith("f") and feature[1:].isdigit():
                    tree['d'][node_id] = int(feature[1:])  # Convert "f0" to 0, "f1" to 1, etc.
                else:
                    raise ValueError(f"Unexpected feature format: {feature}")
                
                tree['t'][node_id] = float(threshold)
                tree['a'][node_id] = left_child
                tree['b'][node_id] = right_child
                tree['r'][node_id] = cover
                
                # Ensure children are initialized in the dictionary
                tree['v'].setdefault(left_child, "internal")
                tree['v'].setdefault(right_child, "internal")
                tree['r'].setdefault(left_child, 0.0)
                tree['r'].setdefault(right_child, 0.0)
            elif "leaf=" in parts[0]:  # Leaf node
                leaf_value = float(parts[0].split("leaf=")[1].split(",")[0])
                cover = float(parts[0].split("cover=")[1])
                
                tree['v'][node_id] = leaf_value
                tree['r'][node_id] = cover
    
    return tree

