import streamlit as st 
import pandas as pd 
import json 

PATH_TESTDATA = '../vqa_dataset/phong2004/final-vqa-dataset/versions/4/test.csv'

def main(df, image_id, image_path):
    st.image(image_path)
  
    btn1, btn2, *_ = st.columns(6, border=0)
    with btn1:
        if st.button('Wrong', icon='❌'):
            st.session_state["button_pressed"] = "wrong"
    
    with btn2:
        if st.button('Accept', icon='✅'):
            st.session_state['button_pressed'] = 'accept'


    df = df[df['image_id'] == image_id]
    st.dataframe(df)

def open_json(name_file):
    image_id_unique = 0
    with open(name_file, "r") as fread:
        if 'image_id_unique' in st.session_state:
            image_id_unique = st.session_state['image_id_unique'] 
            image_id_unique += 1
            json_data = json.loads(fread.read())
        else: 
            json_data = {"image_id": {}}
    
    return json_data, image_id_unique

def write_json(name_file):
    if 'button_pressed' in st.session_state:
        prev_image_id = st.session_state['prev_image_id']
        status = st.session_state['button_pressed']
        
        json_data['image_id'][str(prev_image_id)][status] = st.session_state['button_pressed']

    json_data
    st.session_state["prev_image_id"] = image_id
    json_data["image_id"][str(image_id)] = {"path": image_path}

    with open(name_file, "w") as fout:
        json.dump(json_data, fout)





df = pd.read_csv(PATH_TESTDATA)

json_data, image_id_unique = open_json("status.json")

image_id = df['image_id'].unique()[image_id_unique]

while  image_id in json_data and image_id in json_data['image_id'].keys():
    image_id = df['image_id'].unique()[image_id_unique]
    image_id_unique += 1

# json_data
image_path = r"D:\HCMUS\learn-gui-python\streamlit\bear.jpg"
image_path_tmp = df.loc[df['image_id'] == image_id, 'image_path'].iloc[0]

image_id, image_path_tmp

st.session_state['image_id_unique'] = image_id_unique
main(df, image_id, image_path)

write_json("status.json")

    