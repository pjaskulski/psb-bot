""" import danych do chromadb """
import os
import time
import json
from pathlib import Path
import ollama
import chromadb


# pomiar czasu wykonania
start_time = time.time()

tom_psb = "2"

dbfiles = []

# dane z json
print('Wczytanie danych z pliku json...')
postacie = {}
file_json = 'postacie_gpt.json'
with open(file_json, "r", encoding='utf-8') as f:
    json_data = json.load(f)
    for i, person in enumerate(json_data['persons']):
        volume = person["volume"]
        if volume != tom_psb:
            continue

        name = person['name']
        publ_year = person["publ_year"]
        page = person["page"]
        file_name = person["file_new"]
        incipit = person["incipit"]
        p_key = file_name

        postacie[p_key] = (incipit, page, publ_year)


# wczytanie danych z loga
print('Wczytanie danych z loga...')
if os.path.exists('import.log'):
    with open('import.log', 'r', encoding='utf-8') as fdb:
        dbfiles = fdb.readlines()
        dbfiles = [x.strip() for x in dbfiles]

EMBEDDINGS_MODEL = "aroxima/gte-qwen2-1.5b-instruct:latest"

chroma = chromadb.HttpClient(host="localhost", port=8000)
collection = chroma.get_or_create_collection(name="psb", metadata={"hnsw:space": "cosine"})

knowledge_path = Path('.') / 'RAG' / f'tom_{tom_psb.zfill(2)}'

file_count = 0
for path in os.listdir(knowledge_path):
    if os.path.isfile(os.path.join(knowledge_path, path)):
        file_count += 1

file_list = knowledge_path.glob('*.txt')

licznik = 0
for data_file in file_list:
    licznik += 1
    data_file_name = os.path.basename(data_file)
    print(f'{licznik}/{file_count}: {data_file_name}')

    if data_file_name in dbfiles:
        continue

    with open(data_file, "r", encoding='utf-8') as f:
        chunk = f.read()

    embed = ollama.embeddings(model=EMBEDDINGS_MODEL, prompt=chunk)['embedding']
    
    incipit = page = publication_year = ""
    if data_file_name in postacie:
        incipit, page, publ_year = postacie[data_file_name]
        
    collection.add(ids=[f'tom_{tom_psb.zfill(2)}-{data_file_name}'],
                   embeddings=[embed],
                   documents=[chunk],
                   metadatas={"source": f"{data_file_name}", 
                              "volume": tom_psb,
                              "biogram": "",
                              "biogram": incipit,
                              "page": page,
                              "publication_year": publ_year,
                              "book": "Polski Słownik Biograficzny"
                             }
                   )

    with open('import.log', 'a', encoding='utf-8') as fl:
        fl.write(data_file_name + '\n')

# czas wykonania programu
end_time = time.time()
elapsed_time = end_time - start_time
print(f'Czas wykonania programu: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))} s.')
