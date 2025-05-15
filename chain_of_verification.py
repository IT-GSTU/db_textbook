import re
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from my_llms import get_llm  # Предполагается, что это ваш кастомный модуль
from langchain.agents import ConversationalChatAgent, AgentExecutor
from langchain.tools import Tool
import os
import requests
import pandas as pd

# Функция для выполнения поиска через Google Custom Search API
def google_search(query: str, api_key: str, cx: str, num_results: int = 10) -> str:
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": api_key,
        "cx": cx,
        "q": query,
        "num": num_results,
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    results = response.json()
    items = results.get("items", [])
    snippets = [item.get("snippet", "No snippet available") for item in items]
    return "\n".join(snippets)

# Настройка Google Search API
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
GOOGLE_CX = os.environ.get("GOOGLE_CX")

# Создание инструмента для Google Search API
google_search_tool = Tool(
    name="Google Search",
    func=lambda query: google_search(query, GOOGLE_API_KEY, GOOGLE_CX),
    description="Google Custom Search API для выполнения поиска.",
)

# Инициализация модели
api_key = os.environ.get("DEEPSEEK_API_KEY")
llm = get_llm("deepseek-chat", api_key)
check_with_llm=False
check_with_search=True
# Шаблон и цепочка для базового ответа
base_response_template = """
Вы эксперт по базам данных и системам управления базами данных, информационным системам и разработке программных приложений баз данных.
Дайте ответ на поставленный вопрос, основываясь на информации из признанных компетентных источников.
Ответ на вопрос должен быть подробным, точным и полезным, содержать примеры, позволяющие глубже студенту понять изложенное.
Question: {query} 
Answer:"""

base_response_prompt_template = PromptTemplate(
    input_variables=["query"], 
    template=base_response_template
)


# Цепочка для базового ответа
base_response_chain = (
    RunnablePassthrough.assign(query=lambda x: x)
    | base_response_prompt_template
    | llm
    | RunnableLambda(lambda x: x.content)  # Добавляем преобразование в чистый текст
)
# Модель и парсер для проверочных вопросов
class PlanVerificationsOutput(BaseModel):
    query: str = Field(description="The user's query")
    base_response: str = Field(description="The cleaned response content")
    facts_and_verification_questions: dict[str, str] = Field(
        description="Facts and verification questions"
    )

plan_verifications_output_parser = PydanticOutputParser(
    pydantic_object=PlanVerificationsOutput
)

# Цепочка для генерации проверочных вопросов
plan_verifications_template = """
Given the below Question and Answer, generate a series of verification questions in Russian for Internet search, that test the factual claims in the original baseline response.
For example if part of a longform model response contains the statement “The Mexican–American War was an armed conflict between the United States and Mexico from 1846 to 1848”, then one possible
verification question to check those dates could be “When did the Mexican American war start and end?”

Question: {query}
Answer: {base_response}

{format_instructions}"""

plan_verifications_prompt_template = PromptTemplate(
    template=plan_verifications_template,
    input_variables=["query", "base_response"],
    partial_variables={
        "format_instructions": plan_verifications_output_parser.get_format_instructions()
    },
)

plan_verifications_chain = (
    plan_verifications_prompt_template
    | llm
    | plan_verifications_output_parser
)

# Комбинированная цепочка
answer_and_plan_verification = (
    RunnablePassthrough.assign(
        base_response=base_response_chain,
        query=lambda x: x["query"]
    )
    | plan_verifications_chain
)



# Итоговая оценка
final_evaluation_template = """
Вы эксперт по базам данных и системам управления базами данных, информационным системам и разработке программных приложений баз данных.
Даны ВОПРОС, ОТВЕТ и ПРОВЕРЕННЫЙ_ИСТОЧНИК.
Оцени по шкале от от 1 до 10 точность ответа (ОТВЕТ) на заданный ВОПРОС, используя информацию из проверенных источников (ПРОВЕРЕННЫЙ_ИСТОЧНИК).
При выставлении оценки учитывай насколько информация в ответе на вопрос соответствует информации из проверенного источника.
<ВОПРОС>{query}
<ОТВЕТ>{base_response}
<ПРОВЕРЕННЫЙ_ИСТОЧНИК>{verify_results}

Формат ответа:
Оценка:
Отзыв:
"""


final_response_chain = (
    {
        "query": RunnablePassthrough(),
        "base_response": RunnablePassthrough(),
        "verify_results": RunnablePassthrough()
    }
    | PromptTemplate.from_template(final_evaluation_template)
    | llm
    )

def extract_assessment_and_feedback(final_score):
    feedback = final_score
    pattern = r"(?:Оценка(?: точности)?(?: ответа)?:\s*\*{0,2}\s*)(\d+(?:\.\d+)?)(?:\s*[/\s]|\s*из\s*)"
    match = re.search(pattern, feedback)
    if match:
        assessment = float(match.group(1))
    else:
        assessment = 0
    return assessment, feedback

# Основной процесс выполнения



# Открытие xlsx файла
file_path = "results/deepseek-chat_for_search_testing.xlsx"  # путь к файлу
df = pd.read_excel(file_path)

# Фильтрация строк, где checked=True
filtered_df = df[(df['checked'] == True) & (pd.isna(df['grade']))]

# Обработка строк
for index, row in filtered_df.iterrows():
    query = row['question']  # Считываем значение из столбца 'question'
    base_response = row['final_answer']  # Считываем значение из столбца 'final_answer'
    try:
        questions = plan_verifications_chain.invoke({"query": query, "base_response": base_response}).facts_and_verification_questions.values()  # Получаем проверочные вопросы
        print(f"Проверочные вопросы:\n{list(questions)}")  # Выводим на печать questions

    except Exception as e:
        print(f"Ошибка при выполнении цепочки: {e}")
        continue # Пропускаем итерацию в случае ошибки

    if check_with_search:
        # Проверка через Google Search
        tools = [google_search_tool]
        chat_agent = ConversationalChatAgent.from_llm_and_tools(
            llm=llm,
            tools=tools,
            system_message="Assistant assumes no knowledge and relies on internet search to answer user's queries."
        )
        search_executor = AgentExecutor.from_agent_and_tools(
            agent=chat_agent,
            tools=tools,
            max_iterations=5,
        )

        print("\nПроверка через Google Search:")
        verify_results_str = ""
        for question in questions:
            answer = search_executor.invoke({"input": question, "chat_history": []}).get("output")
            result_str = f"Вопрос: {question}\nОтвет: {answer}\n\n"
            print(result_str, flush=True)
            verify_results_str += result_str

        final_score = final_response_chain.invoke({
            "query": query,
            "base_response": base_response,
            "verify_results": verify_results_str
        })
        assessment, feedback = extract_assessment_and_feedback(final_score.content)
        print(f"Оценка: {assessment}, Отзыв: {feedback}")

    # Устанавливаем значение поля 'grade' assessment
    df.at[index, 'grade'] = assessment
    df.at[index,'grade_decription']=feedback
    # Выводим информацию для проверки
    print(f"Обработана строка: Question={query}, Answer={base_response}, Grade={df.at[index, 'grade']}")

    # Сохраняем изменения в файл после обработки каждой строки
    df.to_excel(file_path, index=False)

    # Выводим информацию для проверки
    print(f"Обработана строка: Question={query}, Answer={base_response}, Grade={df.at[index, 'grade']}")




# Список запросов
queries = []    

# Выполнение для каждого запроса
for query in queries:
    print(f"\nОбработка запроса: {query}\n")

    # Шаг 1: Генерация ответа и проверочных вопросов
    try:
        #intermediate_result = answer_and_plan_verification.invoke({"query": query})
        #base_response=intermediate_result.base_response # Получаем ответ
        #questions=intermediate_result.facts_and_verification_questions.values() # Получаем проверочные вопросы
        base_response=base_response_chain.invoke({"query": query}) # Получаем ответ
        questions=plan_verifications_chain.invoke({"query": query, "base_response": base_response}).facts_and_verification_questions.values() # Получаем проверочные вопросы
        
        print(f"Ответ на вопрос:\n{base_response}\n")
    except Exception as e:
        print(f"Ошибка при выполнении цепочки: {e}")
        continue # Пропускаем итерацию в случае ошибки

    if check_with_llm:
        # Проверка фактов через LLM
        verify_chain = PromptTemplate.from_template("Вы эксперт по базам данных и системам управления базами данных, информационным системам и разработке программных приложений баз данных. Кратко ответь на вопрос: {question}") | llm

        verify_results_str = ""
        for question in questions:
            answer = verify_chain.invoke({"question": question}).content
            result_str = f"Вопрос: {question}\nОтвет: {answer}\n\n"
            print(result_str, flush=True)
            verify_results_str += result_str

        final_score = final_response_chain.invoke({
            "query": query,
            "base_response": base_response,
            "verify_results": verify_results_str
        })
        assessment, feedback = extract_assessment_and_feedback(final_score)
        print(f"Оценка: {assessment}, Отзыв: {feedback}")



