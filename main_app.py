import json
from google import genai
from google.genai import types

client = genai.Client(api_key="api-key")

def send_gemini_request(prompt, temperature=0.7, top_p=1.0, json_output=False):
    try:
        config_args = {
            "temperature": temperature,
            "top_p": top_p,
        }
        # Якщо потрібен суворий вивід у форматі JSON
        if json_output:
            config_args["response_mime_type"] = "application/json"

        response = client.models.generate_content(
            model='gemini-3.5-flash', 
            contents=prompt,
            config=types.GenerateContentConfig(**config_args)
        )
        return response.text
    except Exception as e:
        return f"Помилка: {e}"

# Текстовий зразок для аналізу в завданнях 1-4
sample_text = """
Штучний інтелект (ШІ) стрімко змінює сучасний світ, проникаючи в усі сфери нашого життя: від медицини до фінансів. 
Машинне навчання, як підрозділ ШІ, дозволяє комп'ютерам навчатися на основі даних без явного програмування. 
Однак із цим пов'язані й серйозні виклики, зокрема питання етики, безпеки даних та можливого скорочення робочих місць через автоматизацію. 
Тому державам і корпораціям необхідно розробляти чіткі правила регулювання цієї галузі для забезпечення балансу між інноваціями та захистом прав громадян.
"""


print("=" * 50)
print("ЗАВДАННЯ 1: ПРОСУНУТИЙ ПРОМПТІНГ (GEMINI)")
print("=" * 50)

# Варіант 1: Звичайний prompt[cite: 10]
simple_prompt = f"Проаналізуй наведений текст та визнач його основні ідеї:\n{sample_text}"
print("\n--- Варіант 1 (Звичайний prompt) ---")
print(send_gemini_request(simple_prompt))

# Варіант 2: Просунутий prompt[cite: 10]
advanced_prompt = f"""
Ти викладач університету[cite: 10]. Проаналізуй наведений нижче текст.
Виконай завдання поетапно[cite: 10]:
1. Визнач основну тему[cite: 10].
2. Виділи 3-5 ключових ідей[cite: 10].
3. Визнач основні поняття[cite: 10].
4. Сформулюй короткий висновок[cite: 10].

Текст для аналізу:
{sample_text}

Результат подай структуровано[cite: 10].
"""
print("\n--- Варіант 2 (Просунутий prompt) ---")
print(send_gemini_request(advanced_prompt))


print("\n" + "=" * 50)
print("ЗАВДАННЯ 2: ДЕКОМПОЗИЦІЯ СКЛАДНОЇ ЗАДАЧІ")
print("=" * 50)

print("\n[Етап 1] Визначення основних понять:")
step1 = send_gemini_request(f"Випиши 3 головні поняття з цього тексту:\n{sample_text}")
print(step1)

print("\n[Етап 2] Визначення ключових фактів:")
step2 = send_gemini_request(f"Опиши 2 ключові факти з цього тексту:\n{sample_text}")
print(step2)

print("\n[Етап 3-5] Формування тестового запитання:")
step3 = send_gemini_request("На основі попередніх завдань створи одне тестове запитання з 4 варіантами (A, B, C, D) та вкажи правильну відповідь.")
print(step3)


print("\n" + "=" * 50)
print("ЗАВДАННЯ 3: ІМІТАЦІЯ ReAct")
print("=" * 50)

react_prompt = f"""
Проаналізуй текст та визнач, які твердження є фактами, а які висновками автора. 
Дотримуйся такого алгоритму:
1. Завдача (зрозуміти мету).
2. План дій (як виконати аналіз).
3. Виконання (поетапний розбір тексту).
4. Перевірка (чи логічний результат).
5. Результат (фінальна відповідь).

Текст:
{sample_text}
"""
print(send_gemini_request(react_prompt))


print("\n" + "=" * 50)
print("ЗАВДАННЯ 4: СТРУКТУРОВАНИЙ РЕЗУЛЬТАТ (JSON)")
print("=" * 50)

json_prompt = f"""
Проаналізуй текст і поверни результат у форматі JSON з такими полями:
- "topic": рядок (основна тема),
- "key_ideas": масив рядків (3 ключові ідеї),
- "keywords": масив рядків (ключові поняття),
- "conclusion": рядок (короткий висновок).

Текст:
{sample_text}
"""
# Gemini дозволяє гарантовано отримати JSON через параметр response_mime_type
json_response = send_gemini_request(json_prompt, temperature=0, json_output=True)
print("Сирий вивід моделі:\n", json_response)

try:
    parsed_data = json.loads(json_response)
    print("\n[Успішно] JSON успішно розібрано Python[cite: 11]:")
    print(json.dumps(parsed_data, ensure_ascii=False, indent=4))
except json.JSONDecodeError as e:
    print("\n[Помилка] Не вдалося розпарсити JSON:", e)


print("\n" + "=" * 50)
print("ЗАВДАННЯ 5: РОБОТА З ВЕЛИКИМ КОНТЕКСТОМ (CHUNKING)")
print("=" * 50)

large_document = [
    "Частина 1: Вступ до екологічних проблем. Сучасний світ стикається зі зміною клімату, викликаною викидами парникових газів від спалювання викопного палива.",
    "Частина 2: Відновлювані джерела енергії. Сонячна та вітрова енергетика стають дешевшими й ефективнішими, що дозволяє поступово відмовлятися від вугільних електростанцій.",
    "Частина 3: Роль кожного громадянина. Енергоефективність у побуті, сортування сміття та зменшення використання пластику відіграють важливу роль у збереженні довкілля."
]

# 5.1 Без chunking[cite: 13]
full_text_combined = "\n".join(large_document)
print("\n--- 5.1 Результат без chunking ---")
print(send_gemini_request(f"Зроби загальне узагальнення документа:\n{full_text_combined}"))

# 5.2 Реалізація chunking[cite: 13]
print("\n--- 5.2 Реалізація chunking ---")
chunk_summaries = []
for i, chunk in enumerate(large_document, 1):
    c_summary = send_gemini_request(f"Коротко узагальни цей фрагмент:\n{chunk}")
    chunk_summaries.append(f"Фрагмент {i}: {c_summary}")

aggregated_input = "\n".join(chunk_summaries)
final_chunk_summary = send_gemini_request(f"На основі наступних проміжних узагальнень сформуй єдине загальне узагальнення документа:\n{aggregated_input}")
print("\nЗагальне узагальнення через Chunking:")
print(final_chunk_summary)


print("\n" + "=" * 50)
print("ЗАВДАННЯ 6: ДОСЛІДЖЕННЯ ПАРАМЕТРІВ ГЕНЕРАЦІЇ")
print("=" * 50)

task_gen = "Створи коротку ідею для мобільного додатку, що допомагає вивчати іноземні мови."

print("\n--- Експеримент з Temperature ---")
for temp in [0.0, 0.5, 1.0]:
    print(f"Temperature = {temp}:")
    print(send_gemini_request(task_gen, temperature=temp))
    print("-" * 30)

print("\n--- Експеримент з Top-p ---")
for tp in [0.5, 0.8, 1.0]:
    print(f"Top-p = {tp}:")
    print(send_gemini_request(task_gen, temperature=0.7, top_p=tp))
    print("-" * 30)