import pandas as pd
from pathlib import Path

media = "results/"  # Или укажите другой путь, например, Path("./data")
file_name="meta-llama_Llama-3.3-70B-Instruct_2025-03-31_book_0_log"
file_name_json=media + file_name+".json"
file_name_xlsx=media + file_name+ ".xlsx"


# Проверка и исправление структуры JSON файла
def ensure_json_structure(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        content = file.read().strip()
        if content.endswith(','):
            content = content[:-1]    
    if not (content.startswith('[') and content.endswith(']')):
        content = '[' + content + ']'
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(content)

# Проверка структуры JSON файла
ensure_json_structure(file_name_json)

# Чтение JSON файла
with open(file_name_json, 'r', encoding='utf-8') as file:
    data = pd.read_json(file)

# Преобразование данных в табличный вид (DataFrame)
df = pd.DataFrame(data)

# Заполнение поля question для role=finalizer значением question из вышестоящей строки, в которой role=reviewer
for i in range(1, len(df)):
    if df.at[i, 'role'] == 'finalizer' and df.at[i-1, 'role'] == 'reviewer':
        df.at[i, 'question'] = df.at[i-1, 'question']
        df.at[i, 'feedback'] = df.at[i-1, 'feedback']


df_finalizer = df[df['role'] == 'finalizer']
# Запись данных в Excel файл
df_finalizer.to_excel(file_name_xlsx, index=False)

print("Данные успешно записаны в файл "+file_name_xlsx)