import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="XAI: кредитный скоринг", page_icon="🔍", layout="wide")


@st.cache_resource(show_spinner="Загружаю данные и обучаю модель…")
def load():
    d = fetch_openml("credit-g", version=1, as_frame=True)
    X = pd.get_dummies(d.data, drop_first=True).astype(float)
    y = (d.target == "good").astype(int)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=200, random_state=42).fit(Xtr, ytr)
    acc = accuracy_score(yte, model.predict(Xte))
    return model, Xte, shap.TreeExplainer(model), acc


@st.cache_resource(show_spinner="Считаю важность признаков…")
def global_values(_explainer, _X):
    return _explainer(_X)


model, Xte, explainer, acc = load()

st.title("Кредитный скоринг с объяснением решения")
st.caption("Объяснимый ИИ в системе поддержки принятия решений: модель Random Forest и метод SHAP")

st.sidebar.header("Клиент")
idx = st.sidebar.slider("Номер клиента", 0, len(Xte) - 1, 0)
x = Xte.iloc[[idx]].copy()
st.sidebar.write("Измените данные клиента:")
x["duration"] = st.sidebar.number_input("Срок кредита (мес.)", 1, 72, int(x["duration"].iloc[0]))
x["credit_amount"] = st.sidebar.number_input("Сумма кредита", 250, 20000, int(x["credit_amount"].iloc[0]))
x["age"] = st.sidebar.number_input("Возраст", 18, 80, int(x["age"].iloc[0]))
st.sidebar.caption(f"Точность модели на тестовых данных: {acc:.0%}")

tab1, tab2, tab3 = st.tabs(["Решение по клиенту", "Что важно для модели в целом", "О проекте"])

with tab1:
    p = float(model.predict_proba(x)[0, 1])
    c1, c2 = st.columns(2)
    c1.metric("Шанс одобрения", f"{p:.0%}")
    with c2:
        if p >= 0.5:
            st.success("Рекомендация: одобрить")
        else:
            st.error("Рекомендация: отказать")

    sv = explainer(x)[0, :, 1]
    contrib = pd.Series(sv.values, index=x.columns)
    top = contrib.reindex(contrib.abs().sort_values(ascending=False).index).head(3)
    parts = [f"{name} ({'повышает' if v > 0 else 'понижает'} шанс)" for name, v in top.items()]
    st.write("Сильнее всего на решение повлияли: " + "; ".join(parts) + ".")

    st.subheader("Почему такое решение (SHAP, локальное объяснение)")
    shap.plots.waterfall(sv, max_display=10, show=False)
    st.pyplot(plt.gcf(), clear_figure=True)
    st.caption("Красное повышает шанс одобрения, синее понижает. Объяснение помогает эксперту, но не заменяет его.")

with tab2:
    st.subheader("Важность признаков по всей выборке (SHAP, глобальное объяснение)")
    sv_all = global_values(explainer, Xte)
    shap.summary_plot(sv_all[:, :, 1], Xte, plot_type="bar", max_display=12, show=False)
    st.pyplot(plt.gcf(), clear_figure=True)
    st.caption("Чем длиннее столбец, тем сильнее признак в среднем влияет на решения модели.")

with tab3:
    st.markdown(
        """
**Задача.** Показать, что точная модель машинного обучения может быть понятной для человека.

**Данные.** German Credit: 1000 клиентов, 20 признаков, метка «хороший» или «плохой» заёмщик.

**Модель.** Random Forest из 200 деревьев. Это «чёрный ящик»: решение складывается из ответов сотен деревьев.

**Объяснение.** SHAP делит вклад между признаками клиента. Сумма вкладов и среднего значения даёт итоговый прогноз.

**Важно.** Объяснение является приближением и помогает эксперту проверить решение, но не заменяет его.
"""
    )
