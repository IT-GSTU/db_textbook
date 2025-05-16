import json
import os
import fitz  # PyMuPDF для работы с PDF
from pathlib import Path
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from my_llms import get_llm # доступные llm
import re
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
    {"model_name": "models/gemini-2.5-flash-preview-04-17", "api_key": GOOGLE_API_KEY},
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

# Выбор LLM

number=9 # номер позиции в списке, отсчет с 0
model_name = model_list[number].get("model_name")  # Выбор модели
apy_key=model_list[number].get("api_key")
llm = get_llm(model_name, apy_key)

# Выбор LLM для rag
number=9 # номер позиции в списке, отсчет с 0
rag_model_name = model_list[number].get("model_name")  # Выбор модели
apy_key=model_list[number].get("api_key")
rag_llm = get_llm(rag_model_name, apy_key)

# Выбор LLM для проверки ответов
number=10 # номер позиции в списке, отсчет с 0
review_llm_model_name = model_list[number].get("model_name")  # Выбор модели
apy_key=model_list[number].get("api_key")
review_llm = get_llm(review_llm_model_name, apy_key)



# Задание каталогов и имен файлов 
answers_folder = "results/" # папка 
result_filename = answers_folder + model_name.replace("/", "_").replace(":", "_") + "_book.txt" # оценки и отзывами
json_filename = answers_folder + model_name.replace("/", "_").replace(":", "_") + "_book_log.json" # журнал

# Память для записи взаимодействий
json_log = []

# Шаблоны промптов

start_prompt = PromptTemplate(
    template="""
    Вы исключительно опытный эксперт по базам данных, системам управления базами данных и автоматизированным информационным системам, а также по разработке программных приложений. Вы обладаете большими познаниями в смежных областях.
    Переформулируй и укрупни прилагаемые вопросы по учебной дисциплине "Базы данных", так чтобы осталось 14 емких и не пересекающихся между собой вопросов, отражающих все содержание учебной дисциплины.
    Сделай выборку ключевой информации из учебных пособий для ответов на укрупненные вопросы, информация не должна дублироваться. 
    Выборка должна быть достаточно подробной и содержательной.
    Вопросы по учебной дисциплине "Базы данных": {questions}
    Информация из учебных пособий: {context}

    Сформированный вами ответ будет учитываться экспертами при формировании ответов на каждый из вопросов.

    Ответ:
    """,
    input_variables=["questions", "context"],

)


manager_prompt = PromptTemplate(
    template="""
    Вы специалист и менеджер (manager), создающий учебное пособие для студентов университета для изучения дисциплины "Базы данных". 
    Вы исключительно хорошо владеете русским и английским языками.
    Настрой среду для дальнейшего взаимодействия между тремя экспертами для ответа на каждый вопрос из списка.
    Первый эксперт (expert) получает на вход один из вопросов из списка и отвечает на него со всей полнотой, правильностью и ясностью.
    Второй эксперт (reviewer) анализирует ответ первого эксперта на этот вопрос и дает рекомендации по улучшению качества и полноты ответа.
    Третий эксперт (finalizer) на основании вопроса, ответа первого эксперта и рекомендаций второго эксперта, формирует итоговый ответ и записывает его в файл.
    После этого идет переход к новому вопросу из списка и все повторяется до тех пор, пока не будут получены полные, правильные и исчерпывающие ответы на все вопросы.
    
    Дополнительная информация для ответа, полученная из обзора рекомендуемых источников: {history}

    Ответ:
    
    """,
    input_variables=["history"],

)

expert_prompt = PromptTemplate(
    template="""
    Вы эксперт по базам данных и системам управления базами данных, информационным системам и разработке программных приложений баз данных.
    Вы исключительно хорошо владеете русским и английским языками. 
    Дайте ответ на поставленный вопрос, основываясь на информации из признанных компетентных источников и учебных пособий и из прилагаемого контекста
    Вопрос: {question}
    Дополнительная информация для ответа, полученная из обзора рекомендуемых источников: {history}    
    Ответ на вопрос должен быть подробным, точным и полезным, содержать примеры, позволяющие глубже студенту понять изложенное.
    
    Ответ:
    """,
    input_variables=["question", "history"],
)

reviewer_prompt = PromptTemplate(
    template="""
    Вы опытный эксперт по базам данных и системам управления базами данных, информационным системам и разработке программных приложений для них.
    Проверь, насколько ответ соответствует вопросу.
    Перечисли основные подтемы вопроса. Насколько ответ раскрывает каждую подтему.
    Насколько информация и терминология в ответе соответствует фактам и терминологии в компетентных источниках.
    Если в ответе присутствует программный код - проверь его.

    Оцените предоставленный ответ на вопрос на: правильность утверждений и корректность использования специальных терминов; полноту; наличие примеров; качество кода, в случае если программный код приведен. 
    Выставьте оценку ответу от 1 до 10 (Оценка: ...).
    Напишите рекомендации, содержащие предложения по уточнениям, правкам или дополнениям, если это требуется.
    Вопрос: {question}
    Ответ: {answer}

    Дополнительная информация для ответа, полученная из обзора рекомендуемых источников: {history}
    
    Оценка:
    Рекомендации:
    """,
    input_variables=["question", "answer", "history"],
)

finalizer_prompt = PromptTemplate(
    template="""
    Вы исключительно опытный эксперт по базам данных, системам управления базами данных и автоматизированным информационным системам, а также по разработке программных приложений. Вы обладаете большими познаниями в смежных областях. 
    Ваша задача - на основе вопроса, ответа на него эксперта и замечаний оценщика сформировать наиболее полный и совершенный ответ с объемом не меньше первоначального объема.
    Вопрос: {question}
    Ответ: {answer}
    Замечания: {feedback}

    Ответ:
    """,
    input_variables=["question", "answer", "feedback"],
)


# Фабрика для создания цепочек
def create_chain(prompt_template, llm, output_parser, input_data):
    chain = prompt_template | llm | output_parser
    output = chain.invoke(input_data)
    return output

# Создание цепочек с памятью
start_chain = start_prompt | rag_llm | StrOutputParser() # Начальная цепочка
manager_chain = manager_prompt | llm | StrOutputParser()
expert_chain = expert_prompt| llm | StrOutputParser()
reviewer_chain = reviewer_prompt| llm | StrOutputParser()
finalizer_chain = finalizer_prompt| llm | StrOutputParser()



# Процесс работы
class GraphState(dict):
    """
    Represents the state of the process.

    Attributes:
        history:
        question: Current question being processed.
        expert_answer: Answer provided by the teacher.
        assessment: Score from the reviewer
        reviewer_feedback: Feedback from the reviewer.
        final_answer: Finalized answer after feedback.
    """
    history: str=""
    question: str
    expert_answer: str
    assessment: str
    reviewer_feedback: str
    final_answer: str


# Функция повторно запроса при возникновении ошибки

def get_answer_with_retry(chain,  json_arg, max_attempts=6, time_sleep=40):
    """
    Получает обратную связь с механизмом повторных попыток при ошибках    
    Параметры:
    chain - цепочка 
    max_attempts - максимальное количество попыток (по умолчанию 6) 
    time_sleep - временная задержка между попытками в секндах (по умолчанию 40)     
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
    history= state["history"]
    # возможна дополнительная обработка вопросов, например, перевод на другой язык
    answer=get_answer_with_retry(manager_chain,  {"question": question, "history": history})
    json_log.append({"role": "manager", "answer": answer})
    save_to_json(record, json_filename)
    return {"question": question, "manager_answer": answer, "history": history}

def expert_node(state):
    question = state["question"]
    history= state["history"]
    answer=get_answer_with_retry(expert_chain,  {"question": question, "history":history})
    record={"role": "expert", "question": question, "answer": answer, "history": history}
    json_log.append(record)
    save_to_json(record, json_filename)
    return {"question": question,"expert_answer": answer}

def reviewer_node(state):
    question = state["question"]
    answer = state["expert_answer"]
    history= state["history"]
    feedback=get_answer_with_retry(reviewer_chain,  {"question": question, "answer": answer, "history": history})
    # шаблон поиска числовой оценки в тексте ответа
    pattern = r"(?:Оценка(?: ответа)?:\s*\*{0,2}\s*)(\d+(?:\.\d+)?)(?:\s*[/\s]|\s*из\s*)"
    match = re.search(pattern, feedback)
    if match:
        assessment= float(match.group(1))
    else:
        assessment=0

    record = {"role": "reviewer", "question": question, "assessment":assessment, "feedback": feedback}
    json_log.append(record)
    return {"question": question, "expert_answer": answer, "assessment": assessment, "reviewer_feedback": feedback}


def finalizer_node(state):
    question = state["question"]
    answer = state["expert_answer"]
    feedback = state["reviewer_feedback"]
    assessment=state["assessment"]
    final_answer=get_answer_with_retry(finalizer_chain,  {"question": question, "answer": answer, "feedback":feedback})
    record={"role": "finalizer",  "question": question, "answer": answer, "assessment":assessment, "feedback": feedback, "final_answer":  final_answer}
    json_log.append(record)
    save_to_json(record, json_filename)
    return {"question": question, "expert_answer": answer, "assessment": assessment, "reviewer_feedback": feedback, "final_answer": final_answer}

def finalize_answer(state):
    question = state["question"]
    assessment = state["assessment"]
    feedback = state["reviewer_feedback"]
    expert_answer = state["expert_answer"]
    # возможна дополнительная обработка ответов, например, перевод на другой язык
    final_answer = state["final_answer"]
    return {"question": question, "expert_answer": expert_answer, "assessment": assessment, "reviewer_feedback": feedback, "final_answer": final_answer}

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

# Компиляция графа
agent_workflow = workflow.compile()

def run_agent(question):
    state = {"question": question,"history": history}
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

def save_to_json(json_records, filename):
    with open(filename, 'a', encoding='utf-8') as file:
        json.dump(json_records, file, ensure_ascii=False, indent=4)


def save_to_text_file(context, filename):
    try:
        with open(filename, "a", encoding='utf-8') as file:  # Открываем файл для записи
            file.write(context)  # Записываем текст в файл
    except Exception as e:
        print(f"Произошла ошибка при записи в файл {filename}: {e}")


# Извлекает текст из PDF-файла
def extract_text_from_pdf(pdf_path):
    """Извлекает текст из PDF-файла."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except fitz.FileDataError:
        print(f"Ошибка: Не удалось открыть файл {pdf_path}. Возможно, файл поврежден или не существует.")
        return None  # Важно возвращать None при ошибке
    except Exception as e:
        print(f"Произошла ошибка при обработке PDF: {e}")
        return None


# Извлечение текста из всех PDF-файлов в папке
pdf_texts = []
pdf_folder = Path("./books")  # Укажите путь к папке с PDF-файлами
if not pdf_folder.exists():
    print(f"Папка {pdf_folder} не найдена. Создайте папку 'pdfs' и поместите туда PDF файлы")
    exit()

for pdf_file in pdf_folder.glob("*.pdf"):  # Ищем все файлы с расширением .pdf
    pdf_text = extract_text_from_pdf(pdf_file)
    if pdf_text:
        pdf_texts.append(pdf_text)
    else:
        print(f"Файл {pdf_file.name} будет пропущен")
    
if not pdf_texts: #проверка, что текст из pdf был загружен
    print("Не удалось загрузить текст ни из одного PDF файла. Работа будет продолжена без контекста")
    context = ""
else:
    # Объединение текста из PDF в общий контекст
    context = "\n".join(pdf_texts)

save_to_text_file(context, answers_folder +"pdfs_context.txt")



# Загрузка вопросов
questions = load_questions_from_file('questions.txt')
init_number=0 # номер вопроса, начиная с которого будут даваться ответы.
selected_questions=questions[init_number:]

# Начало процесса ответов
history=start_chain.invoke({"questions":questions,"context":context})
save_to_text_file(history, answers_folder +"context_from_llm.txt")
record={"role": "start", "answer": history}
json_log.append(record)

for q in selected_questions:
    print(run_agent(q))

