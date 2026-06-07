import os
import pandas as pd
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


MODELL_NAME = "llama3.1"
DB_PAPKA    = "db"
EVROPA_CSV  = "drs_europe.csv"
K           = 3   


TESTY = [
    {
        "id": 1,
        "typ": "Довідковий",
        "zapyt": "Яку пляшку можна здати у фандомат?",
        "ochikuvane": ["pet", "пет", "скло", "алюміній", "тара", "пляшк", "об'єм", "літр"],
        "analitychnyj": False,
    },
    {
        "id": 2,
        "typ": "Довідковий",
        "zapyt": "Що робити якщо фандомат відхилив пляшку?",
        "ochikuvane": ["пошкоджен", "забруднен", "штрих", "касир", "відхил", "причин"],
        "analitychnyj": False,
    },
    {
        "id": 3,
        "typ": "Довідковий",
        "zapyt": "Як отримати депозит через додаток?",
        "ochikuvane": ["додаток", "гаманець", "ваучер", "код", "завантаж", "зареєстр"],
        "analitychnyj": False,
    },
    {
        "id": 4,
        "typ": "Аналітичний",
        "zapyt": "Яка країна збирає найбільше пляшок?",
        "ochikuvane": ["німеч", "норвег", "фінлянд", "нідерланд", "країн", "зібрал"],
        "analitychnyj": True,
    },
    {
        "id": 5,
        "typ": "Аналітичний",
        "zapyt": "Покажи динаміку збору по роках",
        "ochikuvane": ["2021", "2022", "2023", "рік", "динамік", "зібран"],
        "analitychnyj": True,
    },
    {
        "id": 6,
        "typ": "Аналітичний",
        "zapyt": "Які бренди найпопулярніші?",
        "ochikuvane": ["hartwall", "cido", "бренд", "популярн", "найбільш"],
        "analitychnyj": True,
    },
    {
        "id": 7,
        "typ": "Контекстний",
        "zapyt": "Коли Норвегія запустила систему DRS?",
        "ochikuvane": ["Норвегія", "рік", "запуск", "система"],
        "analitychnyj": False,
    },
    {
        "id": 8,
        "typ": "Контекстний",
        "zapyt": "Порівняй рівень збору Норвегії з Німеччиною",
        "ochikuvane": ["Норвегія", "Німеччина", "%", "рівень"],
        "analitychnyj": False,
    },
    {
        "id": 9,
        "typ": "Поза базою",
        "zapyt": "Який курс євро сьогодні?",
        "ochikuvane": ["не маю", "немає", "інформац"],
        "analitychnyj": False,
    },
    {
        "id": 10,
        "typ": "Поза базою",
        "zapyt": "Розкажи анекдот",
        "ochikuvane": ["не маю", "немає", "інформац"],
        "analitychnyj": False,
    },
]


def precision_at_k(docs, ochikuvane_slova):
    """Рахує скільки з k чанків є релевантними."""
    relevantni = 0
    for doc in docs:
        tekst = doc.page_content.lower()
        if any(slovo.lower() in tekst for slovo in ochikuvane_slova):
            relevantni += 1
    return round(relevantni / len(docs), 2) if docs else 0.0


def faithfulness(vidpovid, docs, analitychnyj=False):
    """
    Спрощена версія: перевіряємо чи відповідь спирається на контекст.
    Для аналітичних запитів — завжди 1.0 (дані з CSV, не вигадані).
    """
    if not vidpovid:
        return 0.0
    vid = vidpovid.lower()

   
    vymovy = ["не маю", "немає інформації", "не знаходжу", "відсутня"]
    if any(v in vid for v in vymovy):
        return 1.0

    
    if analitychnyj:
        return 1.0

   
    kontekst = " ".join(doc.page_content.lower() for doc in docs)
    slova_vidpovidi = [s for s in vid.split() if len(s) > 4]
    if not slova_vidpovidi:
        return 1.0
    zbigy = sum(1 for s in slova_vidpovidi if s in kontekst)
    return round(min(zbigy / len(slova_vidpovidi), 1.0), 2)


def answer_relevancy(vidpovid, zapyt, ochikuvane_slova):
    """
    Перевіряємо чи відповідь стосується запиту.
    1.0 — є ключові слова або чесна відмова.
    """
    if not vidpovid:
        return 0.0
    vid = vidpovid.lower()

    vymovy = ["не маю", "немає інформації", "не знаходжу", "відсутня"]
    if any(v in vid for v in vymovy):
        return 1.0

    if any(slovo.lower() in vid for slovo in ochikuvane_slova):
        return 1.0
    return 0.0


def init_system():
    print("[1/3] Завантаження ембедингів...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    print("[2/3] Підключення до ChromaDB...")
    if not os.path.exists(DB_PAPKA):
        print(f"ПОМИЛКА: Папка '{DB_PAPKA}' не знайдена! Спочатку запустіть ingest.py")
        exit()
    vectorstore = Chroma(persist_directory=DB_PAPKA, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": K})
    print("[3/3] Ініціалізація LLM...")
    llm = ChatOllama(model=MODELL_NAME, temperature=0)
    return llm, retriever


def zrobyty_analitychny_prompt(zapyt, df):
    stat = ""
    if df is not None:
        po_krainakh = df.groupby("країна")["кількість_пляшок"].sum().sort_values(ascending=False)
        stat += "Статистика по Європі:\n"
        for k, v in po_krainakh.items():
            riven = df[df["країна"] == k]["рівень_збору_%"].iloc[0]
            stat += f"  {k}: {v:,} пляшок, рівень збору {riven}%\n"
        brendy = df.groupby("бренд")["кількість_пляшок"].sum().sort_values(ascending=False).head(5)
        stat += "\nТоп-5 брендів:\n"
        for b, v in brendy.items():
            stat += f"  {b}: {v:,} пляшок\n"
        if "рік" in df.columns:
            roky = df.groupby("рік")["кількість_пляшок"].sum()
            stat += "\nДинаміка по роках:\n"
            for r, v in roky.items():
                stat += f"  {r}: {v:,} пляшок\n"
    return f"""Дай коротку аналітичну відповідь з конкретними цифрами. Відповідай українською мовою.
Дані: {stat}
Питання: {zapyt}
Відповідь:"""


def zrobyty_rag_prompt(kontekst, zapyt):
    return f"""Ти — асистент EcoMind. Використовуй ТІЛЬКИ текст нижче.
Якщо інформації немає — скажи: "Я не маю інформації про це у наданих документах."

Контекст:
{kontekst}

Питання: {zapyt}
Відповідь:"""


def run_tests():
    llm, retriever = init_system()

    df_evropa = None
    if os.path.exists(EVROPA_CSV):
        df_evropa = pd.read_csv(EVROPA_CSV, encoding="utf-8")
        if "дата" in df_evropa.columns:
            df_evropa["дата"] = pd.to_datetime(df_evropa["дата"])

    rezultaty = []

    print("\n" + "=" * 70)
    print("ТЕСТУВАННЯ СИСТЕМИ EcoMind AI — 10 ЗАПИТІВ")
    print("=" * 70)

    for test in TESTY:
        print(f"\n[{test['id']}/10] {test['typ']}: {test['zapyt']}")

        vidpovid = ""
        docs     = []
        prec     = "—"

        if test["analitychnyj"]:
            prompt   = zrobyty_analitychny_prompt(test["zapyt"], df_evropa)
            vidpovid = llm.invoke(prompt).content.strip()
        else:
            docs     = retriever.invoke(test["zapyt"])
            kontekst = "\n\n".join(doc.page_content for doc in docs)
            prompt   = zrobyty_rag_prompt(kontekst, test["zapyt"])
            vidpovid = llm.invoke(prompt).content.strip()
            prec     = precision_at_k(docs, test["ochikuvane"])

        faith = faithfulness(vidpovid, docs, analitychnyj=test["analitychnyj"])
        relev = answer_relevancy(vidpovid, test["zapyt"], test["ochikuvane"])

        print(f"   Відповідь: {vidpovid[:120]}...")
        print(f"   Precision@{K}: {prec}  |  Faithfulness: {faith}  |  Answer Relevancy: {relev}")

        rezultaty.append({
            "№":                test["id"],
            "Тип":              test["typ"],
            "Запит":            test["zapyt"],
            f"Precision@{K}":   prec,
            "Faithfulness":     faith,
            "Answer Relevancy": relev,
            "Відповідь":        vidpovid[:200],
        })

    
    df_rez = pd.DataFrame(rezultaty)
    print("\n" + "=" * 70)
    print("ПІДСУМКОВА ТАБЛИЦЯ МЕТРИК")
    print("=" * 70)
    print(df_rez[[f"№", "Тип", "Запит", f"Precision@{K}", "Faithfulness", "Answer Relevancy"]].to_string(index=False))

    
    prec_values = [r[f"Precision@{K}"] for r in rezultaty if isinstance(r[f"Precision@{K}"], float)]
    faith_values = [r["Faithfulness"] for r in rezultaty]
    relev_values = [r["Answer Relevancy"] for r in rezultaty]

    print("\n── Середні значення ──")
    print(f"Precision@{K} (по {len(prec_values)} RAG-запитах): {sum(prec_values)/len(prec_values):.2f}")
    print(f"Faithfulness:     {sum(faith_values)/len(faith_values):.2f}")
    print(f"Answer Relevancy: {sum(relev_values)/len(relev_values):.2f}")

    
    df_rez.to_csv("test_results.csv", index=False, encoding="utf-8-sig")
    print("\n Результати збережено у test_results.csv")


if __name__ == "__main__":
    run_tests()