import json
import os
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from matplotlib import pyplot as plt
import networkx as nx
from my_llms import get_llm # доступные настроенные llm
import re
import time
import datetime

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
    {"model_name": "gemini-2.5-pro-exp-03-25", "api_key": GOOGLE_API_KEY},
    {"model_name": "gemma3:1b"}, #локальная модель
    {"model_name": "deepseek-r1:1.5b"}, #локальная модель
    {"model_name": "deepseek-r1:8b"}, #локальная модель
    {"model_name": "qwen2.5:3b"}, #локальная модель        
    {"model_name": "deepseek-chat", "api_key": DEEPSEEK_API_KEY},
    {"model_name": "deepseek-reasoner", "api_key": DEEPSEEK_API_KEY},
    {"model_name": "deepseek/deepseek-chat", "api_key": OPENROUTER_API_KEY},
    {"model_name": "deepseek/deepseek-chat-v3-0324:free", "api_key": OPENROUTER_API_KEY},
    {"model_name": "deepseek/deepseek-r1-distill-llama-70b:free", "api_key": OPENROUTER_API_KEY},
    {"model_name": "deepseek/deepseek-r1:free", "api_key": OPENROUTER_API_KEY},
    {"model_name": "google/gemini-2.5-pro-exp-03-25:free", "api_key": OPENROUTER_API_KEY},
    {"model_name": "google/gemini-2.0-pro-exp-02-05:free", "api_key": OPENROUTER_API_KEY},
    {"model_name": "qwen/qwq-32b:free", "api_key": OPENROUTER_API_KEY},
    {"model_name": "google/gemma-3-27b-it:free", "api_key": OPENROUTER_API_KEY},
]

# Выбор LLM для ответов на вопросы
number=0 # номер позиции в списке, отсчет с 0
model_name = model_list[number].get("model_name")  # Выбор модели
apy_key=model_list[number].get("api_key")
llm = get_llm(model_name, apy_key)

# Выбор LLM для проверки ответов
number=10 # номер позиции в списке, отсчет с 0
double_check = True # использовать ли вторую проверку
review_llm_model_name = model_list[number].get("model_name")  # Выбор модели
apy_key=model_list[number].get("api_key")
review_llm = get_llm(review_llm_model_name, apy_key)


# Задание каталогов и имен файлов 
answers_folder = "results/" # папка
current_date = datetime.date.today().isoformat()
result_filename = answers_folder + model_name.replace("/", "_").replace(":", "_") +f"_{current_date}"+ "_book.txt" # оценки и отзывами
json_filename = answers_folder + model_name.replace("/", "_").replace(":", "_") +f"_{current_date}" + "_book_log.json" # журнал

# Журнал для записи взаимодействий
json_log = []



# Шаблоны промптов
manager_prompt = PromptTemplate(
    template="""
    Вы специалист и преподаватель по базам данных и системам управления базами данных, информационным системам и разработке программных приложений баз данных,
    создающий учебное пособие для студентов университета, изучающих информационные технологии, по дисциплине "Базы данных". 
    Вы исключительно хорошо владеете английским и русским языками, знаете содержание документации связанной с дисциплиной, знакомы с содержимым большого количества признанных учебных пособий.
    Ответьте достаточно ли тех источников, на которых ты обучен, и достаточно ли они компетентные, чтобы дать развернутые, корректные и правильные ответы на каждый из прилагаемых вопросов в списке по дисциплине.
    Вопросы: {questions}
    Ответ:
    """,
)

expert_prompt = PromptTemplate(
    template="""
    Вы эксперт по базам данных и системам управления базами данных, информационным системам и разработке программных приложений баз данных.
    Вы исключительно хорошо владеете русским и английским языками. 
    Дайте ответ на поставленный вопрос, основываясь на информации из признанных компетентных источников и учебных пособий.
    Вопрос: {question}
    
    Ответ на вопрос должен быть подробным, точным и полезным, содержать примеры, позволяющие глубже студенту понять изложенное.

    Ответ:
    """,
    input_variables=["question"],
)

reviewer_prompt = PromptTemplate(
    template="""
    Вы опытный эксперт по базам данных и системам управления базами данных, информационным системам и разработке программных приложений для них.
    Проверь, насколько ответ соответствует вопросу.
    Перечисли основные подтемы вопроса. Насколько ответ раскрывает каждую подтему.
    Насколько информация и терминология в ответе соответствует фактам и терминологии в компетентных источниках.
    Если в ответе присутствует программный код - проверь его правильность.

    Оцените предоставленный ответ на вопрос на: 
    правильность утверждений и фактов, корректность использования специальных терминов; 
    полноту ответа; 
    наличие примеров; 
    качество и корректность кода, в случае если программный код приведен.

    Выставьте оценку ответу от 1 до 10 (Оценка: ...).
    Напишите рекомендации, содержащие предложения по уточнениям, правкам или дополнениям, если это требуется.

    Вопрос: {question}
    Ответ: {answer}
    Оценка:
    Рекомендации:
    """,
    input_variables=["question", "answer"],
)

finalizer_prompt = PromptTemplate(
    template="""
    Вы исключительно опытный эксперт по базам данных, системам управления базами данных и автоматизированным информационным системам, а также по разработке программных приложений. Вы обладаете большими познаниями в смежных областях. 
    Ваша задача - на основе вопроса, ответа на него эксперта и замечаний оценщика сформировать совершенный и полный ответ.
    Ответ должен быть на русском языке.
    Вопрос: {question}   
    Ответ: {answer}
    Замечания: {feedback}
    Ответ:
    """,
    input_variables=["question", "answer", "feedback"],
)

# Создание цепочек
manager_chain = manager_prompt | llm | StrOutputParser()
expert_chain = expert_prompt | llm | StrOutputParser()
reviewer_chain = reviewer_prompt | review_llm | StrOutputParser()
finalizer_chain = finalizer_prompt | llm | StrOutputParser()


# Процесс работы
class GraphState(dict):
    """
    Represents the state of the process.

    Attributes:
        question: Current question being processed.
        expert_answer: Answer provided by the teacher.
        assessment: Score from the reviewer
        reviewer_feedback: Feedback from the reviewer.
        final_answer: Finalized answer after feedback.
    """
    question: str
    expert_answer: str
    assessment: str
    reviewer_feedback: str
    final_answer: str


# Функция для запуска повторного запроса при возникновении ошибки

def get_answer_with_retry(chain,  json_arg, max_attempts=6, time_sleep=40):
    """
    Получает обратную связь с механизмом повторных попыток при ошибках    
    Параметры:
    chain - цепочка 
    max_attempts - максимальное количество попыток (по умолчанию 6) 
    time_sleep - временная задержка между попытками в секундах (по умолчанию 40)     
    Возвращает:
    answer - текст ответа от LLM
    
    Выбрасывает:
    Исключение после исчерпания всех попыток
    """
    for attempt in range(max_attempts):
        try:
            answer = chain.invoke(json_arg)            
            if not answer:
                raise ValueError(f"Пустой ответ от системы (попытка {attempt + 1})")                
            return answer            
        except Exception as e:
            error_message = (
                f"Попытка {attempt + 1}/{max_attempts} завершилась ошибкой: {str(e)}"
            )
            print(error_message)
            
            if attempt == max_attempts - 1:
                raise  # Пробрасываем исключение наружу после последней попытки
                
            time.sleep(time_sleep)  # Задержка перед следующей попыткой

    raise Exception("Все попытки подключения к API исчерпаны")


# Узлы графа

def manager_node(state):
    question = state["question"]
    answer=""
    # # возможна дополнительная обработка вопроса, например, перевод на другой язык
    if qustion_number==init_number:
        answer=manager_chain.invoke({"questions": selected_questions})
    record = {"role": "manager", "question": question, "answer": answer}
    json_log.append(record)
    if answer!="": 
        save_to_json(record, json_filename)
    return {"question": question, "manager_answer": answer}

def expert_node(state):
    question = state["question"]
    answer = get_answer_with_retry(expert_chain,  {"question": question})
    record = {"role": "expert", "question": question, "answer": answer}
    json_log.append(record)
    #save_to_json(record, json_filename)
    return {"question": question,"expert_answer": answer}

def reviewer_node(state):
    question = state["question"]
    answer = state["expert_answer"]
    feedback = get_answer_with_retry(reviewer_chain,  {"question": question, "answer": answer})
    # шаблон поиска числовой оценки в тексте ответа
    pattern = r"(?:Оценка(?: ответа)?:\s*\*{0,2}\s*)(\d+(?:\.\d+)?)(?:\s*[/\s]|\s*из\s*)"
    match = re.search(pattern, feedback)
    if match:
        assessment= float(match.group(1))
    else:
        assessment=0
    record={"role": "reviewer", "question": question, "assessment":assessment, "feedback": feedback}
    json_log.append(record)
    #save_to_json(record, json_filename)
    return {"question": question, "expert_answer": answer, "assessment": assessment, "reviewer_feedback": feedback}


def finalizer_node(state):
    question = state["question"]
    answer = state["expert_answer"]
    feedback = state["reviewer_feedback"]
    assessment=state["assessment"]
    final_answer = get_answer_with_retry(finalizer_chain,  {"question": question, "answer": answer, "feedback":feedback})

    record={"role": "finalizer",  "question": question, "answer": answer, "assessment":assessment, "feedback": feedback, "final_answer":  final_answer}
    json_log.append(record)
    #save_to_json(record, json_filename)
    return {"question": question, "expert_answer": answer, "assessment": assessment, "reviewer_feedback": feedback, "final_answer": final_answer}

def finalize_answer(state):
    question = state["question"]
    assessment = state["assessment"]
    feedback = state["reviewer_feedback"]
    expert_answer = state["expert_answer"]
    final_answer = state["final_answer"]
    if double_check:
        feedback2 = get_answer_with_retry(reviewer_chain,  {"question": question, "answer": final_answer})
        # шаблон поиска числовой оценки в тексте ответа
        pattern = r"(?:Оценка(?: ответа)?:\s*\*{0,2}\s*)(\d+(?:\.\d+)?)(?:\s*[/\s]|\s*из\s*)"
        match = re.search(pattern, feedback2)
        if match:
            assessment2= float(match.group(1))
        else:
            assessment2=0
    # возможна дополнительная обработка ответов, например, перевод на другой язык
    record={"role": "finalizer",  "question": question, "answer": expert_answer, "assessment":assessment, "feedback": feedback, "final_answer":  final_answer, "assessment2":assessment2, "feedback2": feedback2}
    json_log.append(record)
    save_to_json(record, json_filename)
    return {"question": question, "expert_answer": expert_answer, "assessment": assessment, "reviewer_feedback": feedback, "final_answer": final_answer, "assessment2":assessment2, "feedback2": feedback2}

# Построение графа
workflow = StateGraph(GraphState)
workflow.add_node("manager", manager_node)
workflow.add_node("expert", expert_node)
workflow.add_node("reviewer", reviewer_node)
workflow.add_node("finalizer", finalizer_node)
workflow.add_node("finish", finalize_answer)
workflow.set_entry_point("manager")

workflow.add_edge("manager", "expert")
workflow.add_edge("expert", "reviewer")
workflow.add_edge("reviewer", "finalizer")
workflow.add_edge("finalizer", "finish")
workflow.add_edge("finish", END)

""" # Визуализация через NetworkX
nx_graph = nx.DiGraph()

# Добавляем узлы и ребра из LangGraph в NetworkX
for node in workflow.nodes:
    nx_graph.add_node(node)
for from_node, to_node in workflow.edges:
    nx_graph.add_edge(from_node, to_node)
# Рисуем граф
pos = nx.spring_layout(nx_graph)  # Раскладка узлов
nx.draw(nx_graph, pos, with_labels=True, node_color="skyblue", node_size=1000, arrows=True)
plt.title("LangGraph Visualization")
plt.show() """



# Компиляция графа
agent_workflow = workflow.compile()

def run_agent(question):
    state = {"question": question}
    output = agent_workflow.invoke(state)
    with open(result_filename, "a", encoding="utf-8") as f:
        f.write(f"Вопрос: {state['question']}\nОтвет: {output['final_answer']}\n\n")
    return output

def load_questions_from_file(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            questions = [line.strip() for line in file.readlines()]
        return questions
    except FileNotFoundError:
        print(f"Файл {filename} не найден.")
        return []

def save_to_json(interaction_memory, filename):
    with open(filename, "a", encoding='utf-8') as file:
        json.dump(interaction_memory, file, ensure_ascii=False, indent=4)
        file.write(",")  # Добавление ,  после каждой записи



# Загрузка вопросов 
questions = load_questions_from_file('questions.txt')
init_number=13 # номер вопроса, начиная с которого будут даваться ответы.
qustion_number=init_number
selected_questions=questions[init_number:]
# Запуск процесса формирования ответов на каждый из вопросов выбранного списка
for q in selected_questions:
    print(run_agent(q))
    qustion_number+=1    



