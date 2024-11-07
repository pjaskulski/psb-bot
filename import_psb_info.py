""" import danych do chromadb """
import os
import time
import ollama
import chromadb


# pomiar czasu wykonania
start_time = time.time()


EMBEDDINGS_MODEL = "aroxima/gte-qwen2-1.5b-instruct:latest"

chroma = chromadb.HttpClient(host="localhost", port=8000)
collection = chroma.get_or_create_collection(name="psb", metadata={"hnsw:space": "cosine"})

psb = []

psb.append("""Polski Słownik Biograficzny

Polski Słownik Biograficzny jest monumentalną historią Polski widzianą przez pryzmat indywidualnych losów ludzkich. Ukazuje się w Krakowie od r. 1935 z przerwą w latach okupacji hitlerowskiej i okresu stalinowskiego. W 54 tomach, na 35 tysiącach stron, zapisanych jest ponad 28 500 życiorysów, począwszy od Popiela i Piasta, Mieszka I i Bolesława Chrobrego, skończywszy na osobach zmarłych w r. 2000. Oprócz biogramów królów, książąt, polityków i wodzów znajdują się w PSB życiorysy pisarzy, malarzy, rzeźbiarzy, architektów, ludzi teatru i filmu, uczonych, sportowców, duchownych i świętych, postaci pierwszoplanowych, mniej znanych, a czasem zupełnie zapomnianych. Wszystkie te artykuły wchodzą ze sobą w rozmaite związki, wzajemnie się uzupełniają i oświetlają, ukazując przeszłość z coraz to nowej perspektywy – tak z punktu widzenia osób działających w dawnych latach i dawnych wiekach, jak z punktu widzenia najnowszych ustaleń nauki.

PSB to wydawnictwo na najwyższym poziomie naukowym. Biogramy, opracowane przez najwybitniejszych specjalistów z kraju i zagranicy (od r. 1935 współpracowało z redakcją ok. 4 tysiące autorów), oparte są na wynikach wieloletnich badań i zawierają bogatą dokumentację źródłową. Życiorysy są pisane w sposób skondensowany, obiektywny i maksymalnie konkretny, układają się jednak w tak fascynujące historie losów ludzkich, że nieraz przybierają charakter pasjonującej opowieści – obyczajowej, przygodowej, bądź awanturniczej.""")


psb.append( """Z dziejów Polskiego Słownika Biograficznego

Inicjatorem Polskiego Słownika Biograficznego i pierwszym jego redaktorem głównym w l. 1931-49 był WŁADYSŁAW KONOPCZYŃSKI (1880-1952) – historyk, profesor Uniwersytetu Jagiellońskiego, znawca dziejów Polski XVI-XVIII wieku, w czasie drugiej wojny światowej więzień obozu w Sachsenhausen, w r. 1948 zmuszony pod naciskiem ówczesnych władz do rezygnacji z zajmowanych stanowisk, a w r. 1949 – z funkcji redaktora głównego PSB. Zasady wydawnictwa ukształtowały się w l. 1931-4; 10 stycznia 1935 ukazał się nakładem Polskiej Akademii Umiejętności pierwszy zeszyt Słownika. Przed wybuchem wojny wydano cztery tomy oraz większość tomu piątego, ogółem do hasła Drohojowski Jan. Po wojnie, w l. 1946-9, pod redakcją Konopczyńskiego, ukazały się jeszcze dwa tomy (do hasła Firlej Henryk) oraz przygotowane zostały 4 zeszyty kolejnego tomu, po czym w okresie stalinowskim wydawnictwo zostało zawieszone. Wznowiono je w r.1958, wydając tom VII, złożony w większości z materiałów przygotowanych pod redakcją nieżyjącego już Konopczyńskiego.""")

licznik = 0
for chunk in psb:
    licznik += 1
    if licznik == 1:
        title = "O PSB"
        source = "https://www.psb.pan.krakow.pl/o-psb/"
    else:
        title = "Z dziejów PSB"
        source = "https://www.psb.pan.krakow.pl/z-dziejow-psb/"

    embed = ollama.embeddings(model=EMBEDDINGS_MODEL, prompt=chunk)['embedding']

    collection.add(ids=[f'psb_info_{licznik}'],
                   embeddings=[embed],
                   documents=[chunk],
                   metadatas={"source": f"{source}",
                              "volume": "",
                              "biogram": title,
                              "page": "",
                              "publication_year": "2024",
                              "book": "Strona internetowa PSB"
                             }
                   )

# czas wykonania programu
end_time = time.time()
elapsed_time = end_time - start_time
print(f'Czas wykonania programu: {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))} s.')
