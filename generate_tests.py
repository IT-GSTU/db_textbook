import json
import os
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from my_llms import get_llm # доступные llm

import time

# Установка ключей API
HF_API_KEY = os.environ.get("HF_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

if not HF_API_KEY or not GOOGLE_API_KEY or not DEEPSEEK_API_KEY or not OPENROUTER_API_KEY:
    raise ValueError("Необходимо установить все переменные окружения: HF_API_KEY, GOOGLE_API_KEY, DEEPSEEK_API_KEY, OPENROUTER_API_KEY")

# Список LLM

model_list = [
    {"model_name": "meta-llama/Llama-3.3-70B-Instruct", "api_key": HF_API_KEY },
    {"model_name": "deepseek-ai/DeepSeek-R1", "api_key": HF_API_KEY},
    {"model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", "api_key": HF_API_KEY},
    {"model_name": "gemini-2.0-flash", "api_key": GOOGLE_API_KEY},
    {"model_name": "deepseek-chat", "api_key": DEEPSEEK_API_KEY},
    {"model_name": "deepseek-reasoner", "api_key": DEEPSEEK_API_KEY},
    {"model_name": "deepseek/deepseek-chat", "api_key": OPENROUTER_API_KEY},
    {"model_name": "deepseek/deepseek-r1-distill-llama-70b:free", "api_key": OPENROUTER_API_KEY},
    {"model_name": "deepseek/deepseek-r1:free", "api_key": OPENROUTER_API_KEY}
]

# Выбор LLM

number=3 # номер позиции в списке, отсчет с 0
model_name = model_list[number].get("model_name")  # Выбор модели
apy_key=model_list[number].get("api_key")
llm = get_llm(model_name, apy_key)


# считывание теоретических данных из json файла
json_filename="results/gemini-2.0-flash-exp_log_full.json"
with open(json_filename, 'r', encoding='utf-8') as file:
    # Загружаем данные из JSON файла
    data = json.load(file)
    list_answers =[record['final_answer'] 
                   for record in data if record.get('role') == 'finalizer'
                   ]
    


# Задание каталогов и имен файлов для записи созданных LLM тестов

result_filename = "results/"+ model_name.replace("/", "_").replace(":", "_") + f"2_tests.txt" # тесты
json_filename = "results/"+ model_name.replace("/", "_").replace(":", "_") + f"2_tests_log.json" # журнал

# Память для записи взаимодействий с LLM
json_log = []

# Шаблон промпта для LLM
expert_prompt = PromptTemplate(
    template="""
    Вы преподаватель в университете и эксперт по базам данных и системам управления базами данных, информационным системам и разработке программных приложений для них.
    Вы выдали студенту теоретический материал для изучения вопроса по учебной дисциплине "Базы данных".
    Теоретический материал: {question}
    Для проверки его усвоения создай 7 тестов в формате GIFT, включающих:
     3 вопроса одинакового уровня сложности с множественным выбором ответов по пять вариантов ответа в каждом;
     2 вопроса одинакового уровня сложности с коротким ответом;
     2 вопроса одинакового уровня сложности на истинность/ложность.
    
    Вопросы тестов должны быть разнообразны и охватывать все затронутые в материале тематические вопросы.

    Тесты:

    """,
    input_variables=["question"],
)

# Создание цепочек
expert_chain = expert_prompt | llm | StrOutputParser()

# Граф
class GraphState(dict):
    """
    Represents the state of the process.

    Attributes:
        question: Current question being processed.
        generated_test: Feedback from the reviewer.
    """
    question: str
    generated_test: str

# Узлы графа
def expert_node(state):
    question = state["question"]
    max_attempts = 5  # Максимальное количество попыток при плохой работе сервиса
    for attempt in range(max_attempts):
        try:
            generated_test = expert_chain.invoke({"question": question})
            # Если выполнение успешно, прерываем цикл
            if generated_test=="":
                raise(f"Попытка запроса номер {attempt + 1} из {max_attempts} вернула пустой ответ")
            break
        except Exception as e:
            print(f"Попытка запроса номер {attempt + 1} из {max_attempts} завершилась ошибкой: {e}")
        if attempt == max_attempts - 1:  # Если это последняя попытка
            raise  # Повторно выбрасываем исключение
        else:
            time.sleep(40)
            continue  # Продолжаем следующую попытку

    record = {"question": question, "generated_test": generated_test}
    json_log.append(record)
    save_to_json(record, json_filename)
    return {"question": question, "generated_test": generated_test}

def finalize_answer(state):
    question = state["question"]
    generated_test = state["generated_test"]
    return {"question": question, "generated_test": generated_test}

# Построение графа
workflow = StateGraph(GraphState)
workflow.add_node("expert", expert_node)
workflow.add_node("finish", finalize_answer)
workflow.set_entry_point("expert")
workflow.add_edge("expert", "finish")
workflow.add_edge("finish", END)

# Компиляция графа
agent_workflow = workflow.compile()

def run_agent(question):
    state = {"question": question}
    output = agent_workflow.invoke(state)
    with open(result_filename, "a", encoding="utf-8") as f:
        f.write(f"Тесты: {output['generated_test']}\n\n")
    return output

def load_from_file(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            text = file.read()
        return text
    except FileNotFoundError:
        print(f"Файл {filename} не найден.")
        return []

def save_to_json(json_records, filename):
    with open(filename, 'a', encoding='utf-8') as file:
        json.dump(json_records, file, ensure_ascii=False, indent=4)


# Запуск процесса генерации тестов
for a in list_answers:
    print(run_agent(a))

