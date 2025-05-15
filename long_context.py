import google.generativeai as genai
import fitz  # PyMuPDF для работы с PDF
import os
from pathlib import Path

# Ключ API Gemini (замените на свой или установите как переменную окружения)
GEMINI_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Необходимо установить переменную окружения GOOGLE_API_KEY")


model_name = "gemini-2.0-flash"

# Параметры модели (теперь настраиваются)
model_params = {
    "temperature": 0,
    "max_output_tokens": 6000,
    "top_p": 0.7,
    "top_k": 40,
}


# Инициализация клиента Gemini
client =genai.GenerativeModel(model_name)

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
def save_to_text_file(context, filename):
    try:
        with open(filename, 'w', encoding='utf-8') as file:  # Открываем файл для записи
            file.write(context)  # Записываем текст в файл
    except Exception as e:
        print(f"Произошла ошибка при записи в файл {filename}: {e}")

def answer_question(question, context):
    """Отправляет запрос в Gemini API с вопросом и контекстом."""
    prompt = f"Контекст:\n{context}\n\nВопрос:\n{question}\n\nОтвет:"
    try:
        response = client.generate_content(model_name, prompt)
        return response.text
    except genai.APIError as e:
        print(f"Ошибка Gemini API: {e}")
        return f"Ошибка Gemini API: {e}" # Возвращаем текст ошибки
    except Exception as e:
        print(f"Произошла ошибка при запросе: {e}")
        return f"Произошла ошибка при запросе: {e}" # Возвращаем текст ошибки

if __name__ == "__main__":
    # Загрузка вопросов из файла
    try:
        with open("questions.txt", "r", encoding="utf-8") as f:
            questions = [line.strip() for line in f]
    except FileNotFoundError:
        print("Ошибка: Файл questions.txt не найден.")
        exit()

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
    save_to_text_file(context, "pdfs_context.txt")
    # Обработка вопросов
    for question in questions:
        print(f"\nВопрос: {question}")
        answer = answer_question(question, context)
        print(f"Ответ: {answer}")