import cv2
from sentence_transformers import SentenceTransformer
from scipy import spatial
from PIL import Image
import pandas as pd
import numpy as np
import copy
import os

model_name = 'clip-ViT-B-16'
st_model = SentenceTransformer(model_name)

def vectorize_img(img_path: str, model: SentenceTransformer=st_model) -> np.array:
    img = Image.open(img_path)
    return st_model.encode(img)

def create_images_db(images_folder: str, model: SentenceTransformer=st_model) -> pd.DataFrame:
    data_dict = dict()
    for file_name in os.listdir(images_folder):
        if os.path.isfile(images_folder + file_name):
            image_path = images_folder + file_name
            emb = vectorize_img(image_path)
            data_dict[file_name] = emb
    return pd.DataFrame(data_dict.items(), columns=['Image', 'Embedding'])

def get_df(df_path: str) -> pd.DataFrame:
    try:
        data_df = pd.read_json(df_path)
        data_df['Embedding'] = data_df['Embedding'].apply(lambda x: np.array(x))
        return data_df
    except Exception:
        return None

def calculate_cos_dist(emb_a: np.array, emb_b: np.array) -> float:
    result_distance = spatial.distance.cosine(emb_a, emb_b)
    return result_distance

def found_similar_images(input_img_path: str, images_db: pd.DataFrame, n: int=1) -> pd.DataFrame:
    input_vec = vectorize_img(input_img_path)
    result_df = copy.deepcopy(images_db)
    result_df['Distance_with_input'] = result_df.apply(lambda x: calculate_cos_dist(input_vec, x['Embedding']), axis=1)
    result_df_sorted = result_df.sort_values('Distance_with_input').reset_index()
    result_df_sorted = result_df_sorted[['Image', 'Distance_with_input']]
    return result_df_sorted.head(n)

def set_df(df_path: str, data_df: pd.DataFrame):
    data_df.to_json(df_path)
    data_df['Embedding'] = data_df['Embedding'].apply(lambda x: np.array(x))
    return data_df

images_folder = 'image_folder/'
images_db = get_df("df.json")
if images_db is None:
    images_db = create_images_db(images_folder)
    set_df("df.json", images_db)
input_img_path = 'img_1.png'
result_df = found_similar_images(input_img_path, images_db)
print(result_df.iloc[0])
