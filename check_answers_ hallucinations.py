import os
import pandas as pd
from google import genai
import re
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from my_llms import get_llm # доступные настроенные llm
import time

# Установка ключей API
HF_API_KEY = os.environ.get("HF_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

if not HF_API_KEY or not GOOGLE_API_KEY or not DEEPSEEK_API_KEY or not OPENROUTER_API_KEY:
    raise ValueError("Необходимо установить все переменные окружения: HF_API_KEY, GOOGLE_API_KEY, DEEPSEEK_API_KEY, OPENROUTER_API_KEY")

media = "results/"  # Или укажите другой путь, например, Path("./data")
file_name="qwen_qwq-32b_free_book_0_log"
file_name_json=media + file_name+".json"

# сравнение с эталонным файлом
check=False

# загрузка эталонного файла и настройки клиента
if check:
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
    llm_model = "gemini-2.5-pro-exp-03-25"
    media_for_etalon = "books/"  # Или укажите другой путь, например, Path("./data")
    # Эталонный файл для проверки 
    etalon_pdf = client.files.upload(file=media_for_etalon + "test.pdf")



# Выбор LLM для проверки ответов
# Список LLM
model_list = [
    {"model_name": "models/gemini-2.5-flash-preview-04-17", "api_key": GOOGLE_API_KEY},
    {"model_name": "gemini-2.5-pro-exp-03-25", "api_key": GOOGLE_API_KEY},   
    {"model_name": "deepseek-chat", "api_key": DEEPSEEK_API_KEY},
    {"model_name": "deepseek-reasoner", "api_key": DEEPSEEK_API_KEY},
    {"model_name": "deepseek/deepseek-chat", "api_key": OPENROUTER_API_KEY},
    {"model_name": "deepseek/deepseek-chat-v3-0324:free", "api_key": OPENROUTER_API_KEY},
    {"model_name": "deepseek/deepseek-r1-distill-llama-70b:free", "api_key": OPENROUTER_API_KEY},
    {"model_name": "deepseek/deepseek-r1:free", "api_key": OPENROUTER_API_KEY},
    {"model_name": "google/gemini-2.5-pro-exp-03-25:free", "api_key": OPENROUTER_API_KEY},
    {"model_name": "google/gemini-2.0-pro-exp-02-05:free", "api_key": OPENROUTER_API_KEY},
    {"model_name": "qwen/qwq-32b:free", "api_key": OPENROUTER_API_KEY},
]


number=7 # номер позиции в списке, отсчет с 0
double_check = True # использовать ли вторую проверку
review_llm_model_name = model_list[number].get("model_name")  # Выбор модели
apy_key=model_list[number].get("api_key")
review_llm = get_llm(review_llm_model_name, apy_key)

# имя файла для сохранения результатов
if check:
    model_name=llm_model
else:
    model_name=review_llm_model_name
file_name_xlsx=media + file_name+ "_"+model_name.replace("/", "_").replace(":", "_") +"_check.xlsx"
def load_questions_from_file(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            questions = [line.strip() for line in file.readlines()]
        return questions
    except FileNotFoundError:
        print(f"Файл {filename} не найден.")
        return []

# Загрузка вопросов 
questions = load_questions_from_file('questions.txt')
# Промпт для проверки каждого вопроса и ответа
initial_prompt = """           
    Ты опытный специалист по базам данных, системам управления базами данных и ты преподаешь курс "Базы данных".
    Подтверди,что ты успешно прочитал содержание прилагаемого pdf файла.
    Проанализируй прилагаемый список вопрос и содержимое файла и перечисли номера вопросов из приведенного ниже списка вопросов на которые:
    можно получить полные ответы;
    нельзя получить полные ответы из-за отсутсвия инфоомации;
    можно лишь частично получить из-за неполноты информации.

    Список вопросов: {questions}

    Список номеров вопросов, на которые можно получить полные ответы из содержания прилагаемого pdf файла:
    Список номеров вопросов, на которые нельзя получить полные ответы из содержания прилагаемого pdf файла:
    Список номеров вопросов, на которые можно частично получить ответы из содержания прилагаемого pdf файла:
  """

# Промпт для проверки каждого вопроса и ответа на соответсвие инфоомации в эалонном файле
check_prompt_pdf = """           
    Ты опытный специалист по базам данных, который выступает в качестве судьи.
    Есть вопрос и ответ на него, которые приводятся ниже.
    Оцени содержание ответа по шкале от 1 до 5.
    Критерии оценки: 
        правильность утверждений и фактов, 
        корректность использования терминов. 
    Оценку проведи исходя из информации в контексте прилагаемого pdf файла компетентного источника.
    Вопрос: {question}
    Ответ: {answer}
    Оценка:
    Отзыв:
     """

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
reviewer_chain = reviewer_prompt | review_llm | StrOutputParser()
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
                raise  
                
            time.sleep(time_sleep)  # Задержка перед следующей попыткой

    raise Exception("Все попытки подключения к API исчерпаны")



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

# Оценка вопросов
if (check and double_check):
    formatted_prompt = initial_prompt.format(questions=questions)
    feedback = client.models.generate_content(
            model=llm_model,
            contents=[formatted_prompt, etalon_pdf],
        )
    check_feedback=feedback.text
    print(check_feedback)

# Проверка структуры JSON файла
ensure_json_structure(file_name_json)

# Чтение JSON файла
with open(file_name_json, 'r', encoding='utf-8') as file:
    data = pd.read_json(file)

# Преобразование данных в табличный вид (DataFrame)
df = pd.DataFrame(data)
if check:
    df['check_grade'] = ""  
    df['check_feedback'] = ""  
if double_check:
    df["assessment2"]=""
    df["feedback2"]=""
# Заполнение поля question для role=finalizer значением question из вышестоящей строки, в которой role=reviewer
#indices=[11,32]
indices=range(1, len(df))
for i in indices:
    if df.at[i, 'role'] == 'finalizer':
        question=df.at[i, 'question']
        answer=df.at[i,"final_answer"]
        if check:
            formatted_prompt = check_prompt_pdf.format(question=question, answer=answer)
            feedback = client.models.generate_content(
                model=llm_model,
                contents=[formatted_prompt, etalon_pdf],
                )
            check_feedback = feedback.text
            print(question)
            print(check_feedback)
            check_grade = 5
            pattern = r"(?:Оценка(?: ответа)?:\s*\*{0,2}\s*)(\d+(?:\.\d+)?)(?:\s*[/\s]|\s*из\s*)"
            match = re.search(pattern, check_feedback)
            if match:
                check_grade = float(match.group(1))
            else:
                check_grade = 0
            df.at[i, "check_feedback"] = check_feedback
            df.at[i, "check_grade"] = check_grade
        if double_check:
            feedback2 = get_answer_with_retry(reviewer_chain,  {"question": question, "answer": answer})
            print(question)
            print(feedback2)
            # шаблон поиска числовой оценки в тексте ответа
            pattern = r"(?:Оценка(?: ответа)?:\s*\*{0,2}\s*)(\d+(?:\.\d+)?)(?:\s*[/\s]|\s*из\s*)"
            match = re.search(pattern, feedback2)
            if match:
                assessment2= float(match.group(1))
            else:
                assessment2=0
            df.at[i,"assessment2"]=assessment2
            df.at[i,"feedback2"]=feedback2

df_finalizer = df[df['role'] == 'finalizer']
# Запись данных в Excel файл
df_finalizer.to_excel(file_name_xlsx, index=False)

print("Данные успешно записаны в файл "+file_name_xlsx)