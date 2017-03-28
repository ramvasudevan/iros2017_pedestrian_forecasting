import os
import json
def read_json(fname):
    with open(fname) as f:
        st = f.read()
    json_acceptable_string = st.replace("'", "\"")
    dic = json.loads(json_acceptable_string)
    return dic

root = os.path.dirname(os.path.abspath(__file__))

config = read_json(os.path.join(root, "config.json"))
