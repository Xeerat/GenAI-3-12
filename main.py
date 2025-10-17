from argparse import ArgumentParser
from transformers import pipeline
from json import load


def get_args():
    """
    Функция для получения опций.

    Входные данные:
    Нет

    Возвращает:
    args.theme(str): тема текста для генерации
    args.style(str): стиль текста для генерации
    args.number(int): количество необходимых промтов
    args.max_num(int): максимальное количество попыток генерации
    """
    parser = ArgumentParser()
    parser.add_argument("--theme", type=str, help="Тема для текста", default="Космос")
    parser.add_argument("--style", type=str, help="Стиль в котором нужно создать текст", default="Киберпанк")
    parser.add_argument("--number", type=int, help="Необходимое количество промтов (чем больше, тем дольше генерация)", default=1)
    parser.add_argument("--max_num", type=int, help="Максимальное количество попыток генерации(должно быть больше, чем --number)", default=2)
    args = parser.parse_args()

    if (args.number <= 0):
        raise ValueError("Количество необходимых промтов должно быть больше 0")
    
    if (args.max_num <= 0):
        raise ValueError("Максимальное количество попыток генерации должно быть больше 0")
    
    if (args.max_num <= args.number):
        raise ValueError("Максимальное количество попыток генерации должно быть больше необходимого количества")
    
    return args.theme, args.style, args.number, args.max_num


def create_parametrs(number_sent, max_number_sent):
    """
    Функция для создания параметров

    Входные данные:
    number_sent(int): необходимое количество промтов
    max_number_sent(int): максимальное количество попыток генерации

    Возвращает:
    parametrs(dict()): словарь содержащий все основные параметры
    """
    if (not isinstance(number_sent, int)):
        raise TypeError("number_sent должно быть целым числом")

    if (not isinstance(max_number_sent, int)):
        raise TypeError("max_number_sent должно быть целым числом")
    
    with open("config.json", "r", encoding="utf-8") as file:
        parametrs = load(file)
    parametrs["max_number_sent"] = max_number_sent

    return parametrs



def init_generator(model):
    """
    Функция для создания генератора текста

    Входные данные:
    model(str): модель для генерации текста

    Возвращает:
    generator(transformers.pipelines.text_generation.TextGenerationPipeline): генератор текста
    """
    if (not isinstance(model, str)):
        raise TypeError("Модель должна быть в виде строки")

    generator = pipeline(
        "text-generation",
        model=model,
        device_map="auto",
        dtype="auto"
    )

    return generator


def create_theme_promt(theme):
    """
    Входные данные:
    theme(str): тема для генерации текста

    Возвращает:
    promt(str): промт для генерации текста по теме
    """
    if (not isinstance(theme, str)):
        raise TypeError("Тема должна быть в виде строки")

    promt = f'Создай художественный текст по теме: "{theme}". Не используй в тексте точные данные.\nОтвет:'
    return promt


def correct_string(string):
    """
    Входные данные:
    string(list(str)): сгенерированный текст разделенный на слова

    Возвращает:
    result(str): откорректированный текст 
    """
    if (not isinstance(string, list)):
        raise TypeError("string должно быть списком")
    
    for word in string:
        if (not isinstance(word, str)):
            raise TypeError("Внутри списка string должны быть строки")
    
    # Удаляем промт, если он сгенерировался в ответе
    if "Ответ:" in string:
        idx = string.index("Ответ:")
        string = string[idx + 1:]

    # Удаляем не до конца сгенерированное предложение
    # А также ограничиваем 50 слов
    for idx in reversed(range(len(string))):
        if string[idx][-1] in ".!?" and idx < 50:
            string = string[:idx+1]
            break

    result = " ".join(string)

    return result


def generate_text(generator, promt, parametrs):
    """
    Входные данные:
    generator(transformers.pipelines.text_generation.TextGenerationPipeline): генератор текста
    promt(str): промт для генерации текста
    parametrs(dict()): параметры генерации

    Возвращает:
    result(str): сгенерированный и откорректированный текст
    """
    if (not isinstance(promt, str)):
        raise TypeError("promt должно быть строкой")
    
    if (not isinstance(parametrs["max_new_tokens"], int)):
        raise TypeError("max_new_tokens должно быть целым числом")
    
    if (not isinstance(parametrs["do_sample"], bool)):
        raise TypeError("do_sample должно быть типа bool")
    
    if (not isinstance(parametrs["temperature"], float)):
        raise TypeError("temperature должно быть типом float")
    
    if (not isinstance(parametrs["top_p"], float)):
        raise TypeError("top_p должно быть типом float")
    
    if (not isinstance(parametrs["top_k"], int)):
        raise TypeError("top_k должно быть целым числом")
    
    if (not isinstance(parametrs["repetition_penalty"], float)):
        raise TypeError("repetition_penalty должно быть типом float")

    result = generator(
        promt,
        max_new_tokens=parametrs["max_new_tokens"],
        do_sample=parametrs["do_sample"],
        temperature=parametrs["temperature"],
        top_p=parametrs["top_p"],
        top_k=parametrs["top_k"],
        repetition_penalty=parametrs["repetition_penalty"]
    )

    if not result or 'generated_text' not in result[0]:
        raise RuntimeError("Ошибка генерации текста моделью")

    result = correct_string(result[0]['generated_text'].split())

    return result


def create_style_words(style):
    """
    Входные данные:
    style(str): стиль в котором нужно создать текст

    Возвращает:
    promt(str): промт для генерации списка слов по стилю
    """
    if (not isinstance(style, str)):
        raise TypeError("Стиль должен быть в виде строки")
    
    promt = f'Создай по стилю "{style}" список слов. Эти слова обязательно должны быть частью этого стиля. Слова перечисли через запятую.\nОтвет:'
    return promt


def create_style_promt(style, text):
    """
    Входные данные:
    style(str): стиль в котором нужно создать текст
    text(str): текст, который нужно стилизовать

    Возвращает:
    promt(str): промт для генерации текста по заданому стилю
    """
    if (not isinstance(style, str)):
        raise TypeError("Стиль должен быть в виде строки")
    
    if (not isinstance(text, str)):
        raise TypeError("Текст должен быть в виде строки")

    promt = f'Вот текст: "{text}". Перепиши его в стиле "{style}", сохранив тему.\nОтвет:'
    return promt


def cycle_generation(generator, promt, parametrs, number_sent, style_words):
    """
    Функция для цикличной генерации текстов

    Входные данные:
    generator(transformers.pipelines.text_generation.TextGenerationPipeline): генератор текста
    promt(str): промт для генерации
    parametrs(dict()): параметры генерации
    number_sent(int): необходимое количество текстов
    style_words(list(str)): список слов соответствующих стилю

    Возвращает:
    answer(list(str)): сгенерированные тексты
    """
    if (not isinstance(number_sent, int)):
        raise TypeError("number_sent должно быть целым числом")
    
    if (not isinstance(style_words, list)):
        raise TypeError("style_words должно быть списком")
    
    for word in style_words:
        if (not isinstance(word, str)):
            raise TypeError("Внутри списка style_words должны быть строки")
        
    answer = []
    for i in range(0, parametrs["max_number_sent"]):
        # Если сгенерировано нужно количество текстов
        if len(answer) == number_sent:
            break
        
        text = generate_text(generator, promt, parametrs)

        # Проверяем корректность генерации
        text_words = [w.strip(",.!?").lower() for w in text.split()]
        for word in style_words:
            if word in text_words:
                answer.append(text)
                break

    # Не было сгенерировано нужное количество текстов
    if (len(answer) < number_sent):
        print("Превышено максимальное количество генераций. Будет сгенерировано меньше промтов")

    return answer


def save_answer(answer):
    """
    Функция для записи ответа в файл

    Входные данные:
    answer(list(str)): сгенерированные тексты

    Возвращает:
    Ничего
    """
    if (not isinstance(answer, list)):
        raise TypeError("answer должно быть списком")
    
    for word in answer:
        if (not isinstance(word, str)):
            raise TypeError("Внутри списка answer должны быть строки")

    with open("output.txt", "w", encoding="utf-8") as output:
        for string in answer:
            output.write(string + '\n')


def main():
    # Получаем опции
    theme, style, number_sent, max_number_sent = get_args()
    # Получаем параметры генерации
    parametrs = create_parametrs(number_sent, max_number_sent)

    # Создаем генератор
    generator = init_generator(parametrs["model"])

    # Создаем промт для генерации текста по теме
    promt = create_theme_promt(theme)
    # Генерируем текст по теме
    text = generate_text(generator, promt, parametrs)
    
    # Создаем промт для генерации слов соответсвующих стилю
    promt = create_style_words(style)
    # Генерируем слова
    style_words = generate_text(generator, promt, parametrs)
    # Получаем корректный список слов
    style_words = [w.strip(",.!?").lower() for w in style_words.split()]

    # Создаем промт для генерации стилизованного текста
    promt = create_style_promt(style, text)
    # Циклично генерируем тексты
    answer = cycle_generation(generator, promt, parametrs, number_sent, style_words)

    # Сохраняем ответ
    save_answer(answer)


if __name__ == "__main__":
    main()