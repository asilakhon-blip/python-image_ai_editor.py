"""
AI-powered Image Editor (LM Studio) — Версия с интерактивным меню
=================================================================
Требования: pip install opencv-python requests numpy
"""

import cv2
import numpy as np
import os
import sys
import json
import urllib.request
import requests

LMSTUDIO_URL = "http://localhost:1234/v1/chat/completions"

# ─────────────────────────────────────────────
# АВТОМАТИЧЕСКИ БЕРЁМ МОДЕЛЬ ИЗ LM STUDIO
# ─────────────────────────────────────────────

def get_model():
    print("\nПроверяю подключение к LM Studio...")
    try:
        r = requests.get("http://localhost:1234/v1/models", timeout=5)
        models = [m["id"] for m in r.json().get("data", [])]
        if models:
            model = models[0]
            print(f"[✓] LM Studio подключён!")
            print(f"[✓] Модель: {model}")
            return model
        else:
            print("[ОШИБКА] Модель не загружена в LM Studio!")
            print("  -> Открой LM Studio, во вкладке Local Server выбери модель и нажми Start Server")
            sys.exit(1)
    except Exception:
        print("[ОШИБКА] LM Studio не отвечает по адресу http://localhost:1234")
        print("  -> Открой LM Studio, во вкладке Local Server нажми Start Server")
        sys.exit(1)

# ─────────────────────────────────────────────
# ЗАГРУЗКА ИЗОБРАЖЕНИЯ
# ─────────────────────────────────────────────

def load_image():
    print("\n" + "═" * 52)
    print("  🖼️  ЗАГРУЗКА ИЗОБРАЖЕНИЯ")
    print("═" * 52)
    print("  [1] Указать путь к файлу на компьютере")
    print("  [2] Скачать по URL")
    print("  [3] Использовать встроенное изображение (Gen z girl)")
    print("═" * 52)
    while True:
        choice = input("Ваш выбор (1/2/3): ").strip()
        if choice == "1":
            path = input("Путь к файлу: ").strip().strip('"').strip("'")
            if not os.path.isfile(path):
                print(f"  Файл не найден — попробуй ещё раз")
                continue
            img = cv2.imread(path)
            if img is None:
                print("  Не удалось прочитать файл — попробуй другой")
                continue
            print(f"[✓] Загружено: {os.path.basename(path)}")
            return img, os.path.basename(path)
        elif choice == "2":
            url = input("URL изображения: ").strip()
            tmp = "_tmp_dl.jpg"
            print("  Скачиваю...")
            try:
                urllib.request.urlretrieve(url, tmp)
                img = cv2.imread(tmp)
                os.remove(tmp)
                if img is None:
                    print("  Не является изображением — попробуй другой URL")
                    continue
                print("[✓] Изображение скачано!")
                return img, "downloaded_image.jpg"
            except Exception as e:
                print(f"  Ошибка: {e}")
                continue
        elif choice == "3":
            # Используем твой фиксированный путь к картинке
            path = r"C:\Users\user\Downloads\Gen z girl.jpg"
            print(f"  Загружаю встроенное изображение из: {path}...")
            
            if not os.path.isfile(path):
                print("[ОШИБКА] Не удалось найти файл по этому пути!")
                print("  Убедись, что файл лежит именно в папке Загрузок и называется точно 'Gen z girl.jpg'")
                print("  Пока переключаюсь на выбор источника...")
                continue
                
            img = cv2.imread(path)
            if img is None:
                print("[ОШИБКА] OpenCV не смог прочитать файл (возможно, повреждён формат) — выбери другой вариант")
                continue
                
            print("[✓] Прекрасное изображение успешно загружено!")
            return img, "Gen_z_girl.jpg"
        else:
            print("  Введи 1, 2 или 3")

# ─────────────────────────────────────────────
# ВЫЗОВ LM STUDIO (для интеллектуальных команд)
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """\
Ты — переводчик команд для OpenCV. Отвечай ТОЛЬКО валидным JSON без пояснений и без markdown-блоков.
Действия:
  grayscale    -> {"action": "grayscale", "params": {}}
  edges        -> {"action": "edges", "params": {"threshold1": 100, "threshold2": 200}}
  red_channel  -> {"action": "red_channel", "params": {}}
Любое другое -> {"action": "unknown", "params": {}}
"""

def ask_ai(user_text, model):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_text},
        ],
        "temperature": 0,
        "max_tokens": 100
    }
    r = requests.post(LMSTUDIO_URL, json=payload, timeout=60)
    r.raise_for_status()
    raw = r.json()["choices"][0]["message"]["content"].strip()
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        data = json.loads(raw)
        return data.get("action", "unknown"), data.get("params", {})
    except:
        return "unknown", {}

# ─────────────────────────────────────────────
# ПРИМЕНЕНИЕ КОМАНД OPENCV
# ─────────────────────────────────────────────

def apply_command(img, action, params):
    if action == "rotate":
        angle = int(params.get("angle", 90))
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), -angle, 1.0)
        cos, sin = abs(M[0,0]), abs(M[0,1])
        nW, nH = int(h*sin + w*cos), int(h*cos + w*sin)
        M[0,2] += (nW-w)/2;  M[1,2] += (nH-h)/2
        return cv2.warpAffine(img, M, (nW, nH))

    elif action == "resize":
        return cv2.resize(img,
            (int(params.get("width", img.shape[1])),
             int(params.get("height", img.shape[0]))),
            interpolation=cv2.INTER_AREA)

    elif action == "red_channel":
        r = np.zeros_like(img)
        r[:,:,2] = img[:,:,2]
        return r

    elif action == "grayscale":
        return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

    elif action == "blur":
        k = int(params.get("kernel", 15))
        k = k if k % 2 == 1 else k+1
        return cv2.GaussianBlur(img, (k, k), 0)

    elif action == "flip":
        return cv2.flip(img, int(params.get("flipcode", 1)))

    elif action == "brightness":
        return cv2.convertScaleAbs(img,
            alpha=float(params.get("contrast", 1.0)),
            beta=int(params.get("brightness", 0)))

    elif action == "edges":
        gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray,
            int(params.get("threshold1", 100)),
            int(params.get("threshold2", 200)))
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    else:
        print(f"  Неизвестное действие: '{action}'")
        return img

# ─────────────────────────────────────────────
# ПОДМЕНЮ С ВАРИАНТАМИ (ИНТЕРАКТИВНЫЕ)
# ─────────────────────────────────────────────

def menu_rotate():
    print("\n  📐 ВЫБОР УГЛА ПОВОРОТА:")
    print("  [1]  30 градусов")
    print("  [2]  45 градусов")
    print("  [3]  60 градусов")
    print("  [4]  90 градусов")
    print("  [5] Ввести свой точный угол")
    print("  [0] Отмена")
    c = input("  Ваш выбор: ").strip()
    angles = {"1": 30, "2": 45, "3": 60, "4": 90}
    if c in angles:
        return "rotate", {"angle": angles[c]}
    elif c == "5":
        val = input("  Введи угол в градусах (например, -45 или 120): ").strip()
        if val.lstrip("-").isdigit():
            return "rotate", {"angle": int(val)}
        print("  [Ошибка] Некорректный угол.")
    return None, None

def menu_resize(img):
    h, w = img.shape[:2]
    print(f"\n  🖼️ ИЗМЕНЕНИЕ РАЗРЕШЕНИЯ (Текущий размер: {w}x{h}):")
    print("  [1] 1280x720  (HD)")
    print("  [2] 800x600   (Средний)")
    print("  [3] 400x300   (Маленький)")
    print("  [4] Ввести свои размеры вручную")
    print("  [0] Отмена")
    c = input("  Ваш выбор: ").strip()
    sizes = {"1": (1280, 720), "2": (800, 600), "3": (400, 300)}
    if c in sizes:
        nw, nh = sizes[c]
        return "resize", {"width": nw, "height": nh}
    elif c == "4":
        try:
            nw = int(input("  Введите новую ширину (px): ").strip())
            nh = int(input("  Введите новую высоту (px): ").strip())
            return "resize", {"width": nw, "height": nh}
        except:
            print("  [Ошибка] Неверный ввод размеров.")
    return None, None

def menu_flip():
    print("\n  ↔️ ОТРАЖЕНИЕ ИЗОБРАЖЕНИЯ:")
    print("  [1] По горизонтали (влево-вправо)")
    print("  [2] По вертикали   (вверх-вниз)")
    print("  [3] В обоих направлениях сразу")
    print("  [0] Отмена")
    c = input("  Ваш выбор: ").strip()
    codes = {"1": 1, "2": 0, "3": -1}
    if c in codes:
        return "flip", {"flipcode": codes[c]}
    return None, None

def menu_blur():
    print("\n  💧 СИЛА РАЗМЫТИЯ (BLUR):")
    print("  [1] Слабое  (размытие 5)")
    print("  [2] Среднее (размытие 15)")
    print("  [3] Сильное (размытие 31)")
    print("  [4] Указать точную силу вручную")
    print("  [0] Отмена")
    c = input("  Ваш выбор: ").strip()
    kernels = {"1": 5, "2": 15, "3": 31}
    if c in kernels:
        return "blur", {"kernel": kernels[c]}
    elif c == "4":
        val = input("  Введи нечетное число для силы размытия (например, 21): ").strip()
        if val.isdigit():
            k = int(val)
            return "blur", {"kernel": k}
        print("  [Ошибка] Должно быть целое число.")
    return None, None

def menu_brightness():
    print("\n  ☀️ ЯРКОСТЬ И КОНТРАСТ:")
    print("  [1] Увеличить яркость (+50)")
    print("  [2] Уменьшить яркость (-50)")
    print("  [3] Повысить контраст (х1.5)")
    print("  [4] Понизить контраст (х0.6)")
    print("  [5] Настроить вручную")
    print("  [0] Отмена")
    c = input("  Ваш выбор: ").strip()
    presets = {
        "1": {"brightness": 50,  "contrast": 1.0},
        "2": {"brightness": -50, "contrast": 1.0},
        "3": {"brightness": 0,   "contrast": 1.5},
        "4": {"brightness": 0,   "contrast": 0.6},
    }
    if c in presets:
        return "brightness", presets[c]
    elif c == "5":
        try:
            b = int(input("  Введи яркость (от -100 до 100, где 0 — без изменений): ").strip())
            a = float(input("  Введи контраст (например, 0.5 — блекло, 1.0 — норма, 2.0 — сочно): ").strip())
            return "brightness", {"brightness": b, "contrast": a}
        except:
            print("  [Ошибка] Некорректный ввод коэффициентов.")
    return None, None

# ─────────────────────────────────────────────
# ГЛАВНОЕ МЕНЮ
# ─────────────────────────────────────────────

MENU = """
╔══════════════════════════════════════════════╗
║   🤖  AI IMAGE EDITOR  (OpenCV + LM Studio) ║
╠══════════════════════════════════════════════╣
║  Выберите номер команды для применения:      ║
║                                              ║
║  [1]  Поворот (30°, 45°, 90° или свой угол)  ║
║  [2]  Изменить разрешение (выбор из списков) ║
║  [3]  Выделить красный канал                 ║
║  [4]  Чёрно-белое (Запрос к AI)              ║
║  [5]  Размытие (Выбор силы или ручной ввод)  ║
║  [6]  Отражение (Горизонтально / Вертикально)║
║  [7]  Яркость / Контраст (Больше / Меньше)   ║
║  [8]  Найти края (Запрос к AI)               ║
║                                              ║
║  Служебные:                                  ║
║   [s] Сохранить   [r] Сброс                  ║
║   [i] Инфо        [m] Меню       [q] Выход   ║
╚══════════════════════════════════════════════╝
"""

def save_image(img, name, step):
    base, ext = os.path.splitext(name)
    out = f"{base}_step{step}{ext or '.jpg'}"
    cv2.imwrite(out, img)
    print(f"  [✓] Сохранено: {out}")
    return out

def show_info(img):
    h, w = img.shape[:2]
    print(f"  Размер: {w}x{h} px  |  ~{img.nbytes/1024:.1f} KB в памяти")

def main():
    print(MENU)
    model = get_model()
    img_orig, img_name = load_image()
    img_cur = img_orig.copy()
    step = 0
    show_info(img_cur)
    print(MENU)

    while True:
        try:
            user_input = input("\n▶  Команда (1-8 или s/r/i/m/q): ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\nВыход.")
            break

        if not user_input:
            continue

        action, params = None, None

        # ── Команды с подменю ──
        if user_input == "1":
            action, params = menu_rotate()
        elif user_input == "2":
            action, params = menu_resize(img_cur)
        elif user_input == "5":
            action, params = menu_blur()
        elif user_input == "6":
            action, params = menu_flip()
        elif user_input == "7":
            action, params = menu_brightness()

        # ── Прямые команды без ввода текста ──
        elif user_input == "3":
            action, params = "red_channel", {}
        elif user_input == "4":
            print("  🤔 Спрашиваю нейросеть...")
            action, params = ask_ai("сделай чёрно-белым", model)
        elif user_input == "8":
            print("  🤔 Спрашиваю нейросеть...")
            action, params = ask_ai("найди края на изображении", model)

        # ── Служебные клавиши ──
        elif user_input in ["s", "ы", "сохранить"]:
            save_image(img_cur, img_name, step)
            continue
        elif user_input in ["r", "к", "сброс"]:
            img_cur = img_orig.copy()
            step = 0
            print("[✓] Изображение сброшено до оригинала.")
            show_info(img_cur)
            continue
        elif user_input in ["i", "ш", "инфо"]:
            show_info(img_cur)
            continue
        elif user_input in ["m", "ь", "меню", "help"]:
            print(MENU)
            continue
        elif user_input in ["q", "й", "выход"]:
            print("До свидания!")
            break
        else:
            print("  [Ошибка] Неверный пункт меню. Введите цифру 1-8 или служебную букву (s, r, i, m, q)")
            continue

        # Если в подменю была нажата "Отмена" (вернулись None, None)
        if action is None:
            print("  Действие отменено.")
            continue

        if action == "unknown":
            print("  ❓ Нейросеть не смогла обработать этот тип запроса.")
            continue

        print(f"  ⚙️  Выполняю действие: {action} с параметрами {params}")
        try:
            img_cur = apply_command(img_cur, action, params)
            step += 1
        except Exception as e:
            print(f"  [ОШИБКА OpenCV] {e}")
            continue

        save_image(img_cur, img_name, step)
        show_info(img_cur)
        print("  ✅ Шаг успешно применён!")
        
        # Обновляем графическое окно просмотра
        cv2.imshow("AI Image Editor", img_cur)
        cv2.waitKey(1)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()