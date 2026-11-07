import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import minimize_scalar

# Настройка страницы
st.set_page_config(page_title="Оптимизатор скважины Pro", layout="wide")
st.title("📊 Well optimization tool")

# --- БОКОВАЯ ПАНЕЛЬ ---
st.sidebar.header("🛠️ 1. Технологические параметры")
pi = st.sidebar.number_input("Коэффициент продуктивности (PI), м3/сут/атм", min_value=0.1, max_value=50.0, value=2.5, step=0.1)
oil_density = st.sidebar.number_input("Плотность нефти, т/м3", min_value=0.7, max_value=1.0, value=0.86, step=0.01)

st.sidebar.header("💰 2. Экономика и Инвестиции")
capex = st.sidebar.number_input("Инвестиции в ГТМ / Ремонт (CAPEX), руб", min_value=0, value=1500000, step=100000)
oil_price = st.sidebar.number_input("Цена чистой нефти, руб/тонна", min_value=1000, value=45000, step=500)
lifting_cost = st.sidebar.number_input("Затраты на подъем 1 м3 жидкости, руб", min_value=0, value=150, step=10)
water_cost = st.sidebar.number_input("Утилизация 1 м3 воды, руб", min_value=0, value=80, step=5)
fixed_opex = st.sidebar.number_input("Постоянные суточные затраты, руб/сут", min_value=0, value=5000, step=500)

st.sidebar.header("📈 3. Данные исследований")
st.sidebar.write("Замеры (Депрессия vs Обводненность):")

# Интерактивная таблица вместо текстовой строки
default_data = [
    {"Депрессия, атм": 15.0, "Обводненность, %": 20.0},
    {"Депрессия, атм": 25.0, "Обводненность, %": 25.0},
    {"Депрессия, атм": 35.0, "Обводненность, %": 35.0},
    {"Депрессия, атм": 45.0, "Обводненность, %": 55.0},
    {"Депрессия, атм": 55.0, "Обводненность, %": 85.0},
]
edited_df = st.sidebar.data_editor(
    default_data, 
    num_rows="dynamic", 
    key="research_table"
)

# Валидация введенных данных
try:
    research_depression = np.array([row["Депрессия, атм"] for row in edited_df if row.get("Депрессия, атм") is not None])
    research_watercut = np.array([row["Обводненность, %"] for row in edited_df if row.get("Обводненность, %") is not None])
    
    if len(research_depression) < 3:
        st.error("📋 Ошибка: введите минимум 3 точки замеров для построения тренда.")
        st.stop()
except Exception as e:
    st.error("⚠️ Ошибка формата данных в таблице.")
    st.stop()

# Построение математической модели обводнения
wc_model = np.polyfit(research_depression, research_watercut / 100.0, 2)

def get_watercut(depression):
    wc = np.polyval(wc_model, depression)
    return np.clip(wc, 0.0, 1.0)

def calculate_economics(depression):
    q_liq = pi * depression
    wc = get_watercut(depression)
    q_wat = q_liq * wc
    q_oil_m3 = q_liq * (1 - wc)
    q_oil_tons = q_oil_m3 * oil_density
    
    revenue = q_oil_tons * oil_price
    opex_var = (q_liq * lifting_cost) + (q_wat * water_cost)
    total_opex = opex_var + fixed_opex
    ebitda = revenue - total_opex
    
    payback_days = capex / ebitda if ebitda > 0 else float('inf')
    
    return {
        'q_liq': q_liq,
        'q_oil': q_oil_tons,
        'watercut': wc * 100,
        'revenue': revenue,
        'opex': total_opex,
        'ebitda': ebitda,
        'payback': payback_days
    }

# Оптимизация
bounds = (research_depression.min(), research_depression.max())
res = minimize_scalar(lambda d: -calculate_economics(d)['ebitda'], bounds=bounds, method='bounded')
opt_dep = res.x
opt = calculate_economics(opt_dep)

# --- ВЫВОД РЕЗУЛЬТАТОВ (KPI) ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Оптимальный дебит жидкости", f"{opt['q_liq']:.1f} м³/сут", f"при {opt_dep:.1f} атм")
col2.metric("Дебит нефти", f"{opt['q_oil']:.1f} т/сут", f"Обводн: {opt['watercut']:.1f}%")
col3.metric("Прибыль (EBITDA)", f"{int(opt['ebitda']):,} руб/сут".replace(",", " "))

if opt['payback'] != float('inf'):
    col4.metric("Срок окупаемости", f"{opt['payback']:.1f} дней", f"~{opt['payback']/30.4:.1f} мес.")
else:
    col4.metric("Срок окупаемости", "Не окупается", "EBITDA <= 0")

st.markdown("---")

# --- ИНТЕРАКТИВНЫЕ ГРАФИКИ (PLOTLY) ---
dep_range = np.linspace(bounds[0], bounds[1], 100)
results_range = [calculate_economics(d) for d in dep_range]

ebitda_plot = [r['ebitda'] for r in results_range]
q_liq_plot = [r['q_liq'] for r in results_range]
wc_plot = [r['watercut'] for r in results_range]

fig = make_subplots(rows=1, cols=2, subplot_titles=("Экономический оптимум", "Зависимость обводненности"))

# График 1: EBITDA vs Дебит жидкости
fig.add_trace(
    go.Scatter(x=q_liq_plot, y=ebitda_plot, mode='lines', name='Суточный EBITDA', line=dict(color='#1f77b4', width=3)),
    row=1, col=1
)
fig.add_shape(
    type="line", x0=opt['q_liq'], y0=min(ebitda_plot), x1=opt['q_liq'], y1=max(ebitda_plot),
    line=dict(color="Red", dash="dash"), row=1, col=1
)

# График 2: Модель обводнения
fig.add_trace(
    go.Scatter(x=research_depression, y=research_watercut, mode='markers', name='Замеры', marker=dict(color='red', size=10)),
    row=1, col=2
)
fig.add_trace(
    go.Scatter(x=dep_range, y=wc_plot, mode='lines', name='Модель тренда', line=dict(color='#2ca02c', width=2.5)),
    row=1, col=2
)
fig.add_shape(
    type="line", x0=opt_dep, y0=0, x1=opt_dep, y1=100,
    line=dict(color="Red", dash="dash"), row=1, col=2
)

# Настройка осей и темы
fig.update_layout(height=500, template="plotly_white", showlegend=True)
fig.update_xaxes(title_text="Дебит жидкости, м³/сут", row=1, col=1)
fig.update_yaxes(title_text="Прибыль (EBITDA), руб/сут", row=1, col=1)
fig.update_xaxes(title_text="Депрессия, атм", row=1, col=2)
fig.update_yaxes(title_text="Обводненность, %", row=1, col=2)

st.plotly_chart(fig, use_container_width=True)
