# 📘 Розробка засобів генеративного штучного інтелекту для реалізації задач аналітики даних

> Локальний AI-асистент EcoMind на архітектурі Advanced RAG — обробляє довідкові запити по текстовій базі знань та аналізує структуровані дані з автоматичною візуалізацією результатів. Без хмарних API, без галюцинацій.

---

## 👤 Автор

- **ПІБ**: Красовська Соломія Володимирівна
- **Група**: ФЕП-43
- **Керівник**: Ляшкевич Василь Яремович, кандидат технічних наук, доцент
- **Дата виконання**: 30.05.2026

---

## 📌 Загальна інформація

- **Тип проєкту**: Локальний веб-застосунок (AI-асистент)
- **Мова програмування**: Python 3.11
- **Фреймворки / Бібліотеки**: LangChain, Streamlit, ChromaDB, Pandas, Plotly, HuggingFace Embeddings, Ollama

---

## 🧠 Опис функціоналу

- Чат-інтерфейс з підтримкою україномовних запитів
- Семантичний пошук по текстовій базі знань (правила депозитної системи DRS)
- Аналітична обробка CSV-даних про збір тари в країнах Європи
- Автоматична генерація інтерактивних графіків Plotly за аналітичними запитами
- Логічний роутер — автоматично визначає тип запиту (довідковий / аналітичний)
- Пам'ять розмови — збереження контексту останніх 6 повідомлень
- Коректна відмова на запити поза базою знань (без галюцинацій)

---

## 🧱 Опис основних файлів

| Файл / Директорія | Призначення |
|---|---|
| `streamlit_app.py` | Головний файл застосунку — веб-інтерфейс, роутинг, обробка запитів |
| `ingest.py` | Скрипт завантаження документів, чанкінгу та індексації в ChromaDB |
| `main.py` | Консольна версія RAG-чату для тестування без інтерфейсу |
| `data_for_rag.txt` | Текстова база знань про правила депозитної системи (43 блоки) |
| `drs_europe.csv` | Аналітичні дані про збір тари в 10 країнах Європи (2021–2026) |
| `db/` | Папка з векторною базою ChromaDB (створюється автоматично) |
| `requirements.txt` | Перелік залежностей проєкту |
| `test_metrics.py` | Скрипт автоматизованого тестування. |
| `test_results.csv` | Результати автоматизованого тестування з метриками Precision@3, Faithfulness, Answer Relevancy  | 

---

## ▶️ Як запустити проєкт "з нуля"

### 1. Встановлення інструментів

- Python 3.11+
- [Ollama](https://ollama.com) — платформа для локального запуску LLM

### 2. Клонування репозиторію

```bash
git clone https://github.com/SolomiyaKrasovska/EcoMind_AI
cd EcoMind_AI
```

### 3. Встановлення залежностей

```bash
pip install -r requirements.txt
```

### 4. Завантаження моделі

```bash
ollama pull llama3.1
```

### 5. Індексація бази знань

```bash
python ingest.py
```

> Виконується один раз перед першим запуском. Створює папку `db/` з векторною базою.

### 6. Запуск застосунку

```bash
streamlit run streamlit_app.py
```

Застосунок буде доступний за адресою: `http://localhost:8501`

---

## 🔌 Приклади запитів

### 💬 Довідковий запит

**Вхід**: `Яку пляшку можна здати у фандомат?`

**Відповідь**: Система шукає у текстовій базі знань та повертає точну відповідь з посиланням на джерело.

---

### 📊 Аналітичний запит

**Вхід**: `Яка країна збирає найбільше пляшок?`

**Відповідь**: Система обробляє CSV-дані через Pandas та повертає текстову відповідь + графік Plotly.

---

## 🖱️ Інструкція для користувача

1. **Введіть запит** у поле внизу чату
2. **Довідкові запити** — про правила DRS, типи тари, умови повернення депозиту
3. **Аналітичні запити** — про статистику збору по країнах, брендах, роках
4. **Контекстні запити** — система пам'ятає попередні 6 повідомлень розмови
5. Кнопка **🗑️ Очистити чат** — скидає історію розмови

---

## 🧪 Проблеми і рішення

| Проблема | Рішення |
|---|---|
| Ollama не запускається | Перевірити чи встановлено Ollama та чи завантажено модель (`ollama list`) |
| Папка `db/` порожня | Запустити `python ingest.py` перед стартом застосунку |
| Повільна генерація відповіді | Нормально для локальної моделі (~30–35 сек). Потребує GPU для пришвидшення |
| Відповідь англійською | Переформулюй запит українською або додай в промпт вимогу мови |

---

## 🧾 Використані джерела / література

- [LangChain Documentation](https://python.langchain.com/docs/concepts/rag/)
- [Ollama](https://ollama.com)
- [ChromaDB](https://www.trychroma.com)
- [HuggingFace Embeddings](https://huggingface.co)
- [Streamlit](https://streamlit.io)

- [What is the difference between LLMs and traditional NLP models?](https://www.educative.io/blog/nlp-vs-llm?utm_source=chatgpt.com)
- [Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., Polosukhin, I. (2017). Attention Is All You Need. Advances in Neural Information Processing Systems, 30. DOI: 10.48550/arXiv.1706.03762.](https://arxiv.org/abs/1706.03762)
- [Maslej, N., Fattorini, L., Perrault, R., et al. (2024). Artificial Intelligence Index Report 2024. Stanford Institute for Human-Centered Artificial Intelligence (HAI).](https://aiindex.stanford.edu/report/)
- [Bubeck, S., Chandrasekaran, V., Eldan, R., et al. (2023). Sparks of Artificial General Intelligence: Early experiments with GPT-4. Microsoft Research. DOI: 10.48550/arXiv.2303.12712.](https://arxiv.org/abs/2303.12712)
- [Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., Liang, P. (2023). Lost in the Middle: How Language Models Use Long Contexts. Transactions of the Association for Computational Linguistics, 12, 157–173. DOI: 10.48550/arXiv.2307.03172.](https://arxiv.org/abs/2307.03172) 
- Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-augmented generation for knowledge-intensive NLP tasks. In Proceedings of the 34th International Conference on Neural Information Processing Systems (NIPS ʼ20). Curran Associates Inc., Red Hook, NY, USA, Article 793, 9459–9474. DOI: 10.5555/3495724.3496517  
- Hu L., Wu Z., Xiong B., Chen Y., Li H., Wang R. Retrieval-Augmented Generation for Natural Language Processing: A Survey. arXiv preprint. 2024. DOI:10.48550/arXiv.2407.13193
- [Shuster, K., Poff, S., Chen, M., et al. (2021). Retrieval Augmentation Reduces Hallucination in Conversation. arXiv preprint arXiv:2104.07567. DOI: 10.48550/arXiv.2104.07567.](https://arxiv.org/abs/2104.07567).
- Jeff Johnson, Matthijs Douze, and Hervé Jégou. 2021. Billion-scale similarity search with GPUs. IEEE Transactions on Big Data, Vol. 7, No. 3. IEEE Computer Society, Los Alamitos, CA, USA, 535–547. DOI: 10.1109/TBDATA.2019.2921572.
- [Question answering using embeddings-based search](https://github.com/openai/openai-cookbook/blob/main/examples/Question_answering_using_embeddings.ipynb).
- Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Ye Jin Bang, Andrea Madotto, and Pascale Fung. 2023. Survey of Hallucination in Natural Language Generation. ACM Comput. Surv. 55, 12, Article 248 (December 2023). DOI: 10.1145/3571730.
- Panchenko, D., Maksymenko, D., Turuta, O., Yerokhin, A., Daniiel, Y., Turuta, O. (2022). Evaluation and Analysis of the NLP Model Zoo for Ukrainian Text Classification. In: Ermolayev, V., et al. Information and Communication Technologies in Education, Research, and Industrial Applications. ICTERI 2021. Communications in Computer and Information Science, vol 1698. Springer, Cham. DOI: 10.1007/978-3-031-20834-8_6.
- [Es, S., James, J., Espinosa-Anke, L., Schockaert, S. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation. arXiv preprint. DOI: 10.48550/arXiv.2309.15217.](https://arxiv.org/abs/2309.15217)
