import streamlit as st 
import pandas as pd 
import json, os

PATH_SRC = os.path.dirname(os.path.dirname(__file__))

PATH_TESTDATA = PATH_SRC + '/vqa_dataset/phong2004/final-vqa-dataset/versions/4/test.csv'
PATH_IMAGE = PATH_SRC + '/vqa_dataset/images/' 
FILE_STATUS_JSON = PATH_SRC + "/gui/status.json"

# ex: ../vqa_dataset/images/airplane/aircraft-holiday-sun-tourism-99567.jpeg

def open_json(file_name):
    with open(file_name, "r") as f:
        content = f.read()
        if content.strip():
            json_data = json.loads(content)
        else:
            json_data = {"image_id": {}}
    
    # nhung image id da kiem duyet
    list_image_id = list(json_data["image_id"].keys())

    return json_data, list_image_id

def write_json(file_name, json_data, image_path, image_id, status):
     
    json_data['image_id'][str(image_id)] = {
        "path": image_path,
        "status": status
    }

    with open(file_name, "w") as fout:
        
        json.dump(json_data, fout)


def run():
    df = pd.read_csv(PATH_TESTDATA)

    json_data, list_image_id= open_json(FILE_STATUS_JSON)

    for val in df['image_id'].unique():
        image_id = val
        if str(image_id) not in list_image_id or json_data['image_id'][str(image_id)]['status'] == 'None': 
            break
    
    image_id 
    image_path = PATH_IMAGE + df.loc[df['image_id'] == image_id, 'image_path'].iloc[0]
    
    if len(list_image_id) == 0:
        write_json(FILE_STATUS_JSON,  json_data, image_path, image_id, "None")
    
    image_path
    if os.path.exists(image_path): 
        st.image(image_path)
    
    btn1, btn2, *_ = st.columns(6, border=0)
    with btn1:
        if st.button('Wrong', icon='❌'):
            write_json(FILE_STATUS_JSON, json_data, image_path, list_image_id[-1], "wrong")

            for val in df['image_id'].unique():
                image_id = val
                if str(image_id) not in list_image_id: 
                    break
            
            write_json(FILE_STATUS_JSON,  json_data, image_path, image_id, "None")

            st.rerun()
            

    with btn2:
        if st.button('Accept', icon='✅'):
            write_json(FILE_STATUS_JSON,  json_data, image_path, list_image_id[-1], "accept")

            for val in df['image_id'].unique():
                image_id = val
                if str(image_id) not in list_image_id:
                    break
            
            write_json(FILE_STATUS_JSON,  json_data, image_path, image_id, "None")

            st.rerun()

    
    df = df[df['image_id'] == image_id]

    data = {
        "image_id": df['image_id'],
        "question": df['question'],
        "answer": df['answer'],
        "image_path": df['image_path']
    }

    df = pd.DataFrame(data)

    edited_df = st.data_editor(df, num_rows="dynamic")
    if st.button("Lưu thay đổi"):
        edited_df.to_csv(PATH_SRC + "/gui/updated_data.csv", mode='a', header=False, index=False)
        st.success("Dữ liệu đã được lưu!")


    json_data

run()