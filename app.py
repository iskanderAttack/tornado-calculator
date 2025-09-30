import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import io

# ================== НАСТРОЙКА СТРАНИЦЫ ==================
st.set_page_config(
    page_title="Калькулятор тепловентилятора Торнадо",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================== БАЗА ДАННЫХ ТЕПЛОВЕНТИЛЯТОРОВ ==================
@st.cache_data
def load_heat_exchangers():
    return [
        {'model': 'Торнадо 3', 'power': 20, 'air_flow': 1330, 'height': 300, 'width': 280, 'rows': 4, 'price': 65000},
        {'model': 'Торнадо 4', 'power': 33, 'air_flow': 2670, 'height': 400, 'width': 400, 'rows': 3, 'price': 85000},
        {'model': 'Торнадо 5', 'power': 55, 'air_flow': 4500, 'height': 500, 'width': 500, 'rows': 3, 'price': 120000},
        {'model': 'Торнадо 10', 'power': 110, 'air_flow': 9000, 'height': 500, 'width': 1000, 'rows': 4, 'price': 280000}
    ]

# ================== МАТЕРИАЛЫ ==================
MATERIALS = {
    'кирпич': 0.7,
    'газоблок': 0.18,
    'пеноблок': 0.16,
    'керамзитоблок': 0.4,
    'сэндвич панель': 0.05,
    'брус': 0.15
}

U_VALUES = {
    'окно_одинарное': 5.0,
    'окно_двойное': 2.9,
    'окно_тройное': 1.5,
    'окно_евро': 1.3,
    'дверь_деревянная': 2.0,
    'дверь_металлическая': 1.5,
    'дверь_утепленная': 0.8,
    'пол_неутепленный': 0.5,
    'пол_утепленный': 0.2,
    'потолок_неутепленный': 0.6,
    'потолок_утепленный': 0.25
}

RADIATOR_POWER = {
    ('алюминиевые', 350): 140,
    ('алюминиевые', 500): 180,
    ('чугунные', 350): 120,
    ('чугунные', 500): 160,
}

# ================== РАСЧЁТ ТЕПЛОПОТЕРЬ ==================
def calculate_heat_loss(room_params):
    total_loss = 0
    temp_diff = room_params['temp_in'] - room_params['temp_out']

    # Стены
    wall_loss = (room_params['wall_area'] *
                MATERIALS[room_params['wall_material']] /
                max(room_params['wall_thickness'], 0.01) * temp_diff)
    total_loss += wall_loss

    # Окна
    window_loss = room_params['window_area'] * U_VALUES[room_params['window_type']] * temp_diff
    total_loss += window_loss

    # Двери
    door_loss = room_params['door_area'] * U_VALUES[room_params['door_type']] * temp_diff
    total_loss += door_loss

    # Пол
    floor_type = 'пол_утепленный' if room_params['floor_insulated'] else 'пол_неутепленный'
    floor_loss = room_params['floor_area'] * U_VALUES[floor_type] * temp_diff
    total_loss += floor_loss

    # Потолок
    ceiling_type = 'потолок_утепленный' if room_params['ceiling_insulated'] else 'потолок_неутепленный'
    ceiling_loss = room_params['ceiling_area'] * U_VALUES[ceiling_type] * temp_diff
    total_loss += ceiling_loss

    # Инфильтрация
    infiltration_loss = room_params['room_volume'] * 0.3 * 1.2 * 1005 * temp_diff / 3600
    total_loss += infiltration_loss

    # Радиаторы
    if room_params.get('has_radiators', False):
        power_per_section = RADIATOR_POWER[(room_params['radiator_type'], room_params['radiator_height'])]
        radiator_heat = power_per_section * room_params['radiator_sections']
        total_loss -= radiator_heat

    return max(total_loss, 0)

# ================== ПОДБОР ОБОРУДОВАНИЯ ==================
def select_heat_exchangers(required_power, room_volume):
    exchangers = load_heat_exchangers()
    suitable = []

    for unit in exchangers:
        count = max(1, int(np.ceil(required_power / unit['power'])))
        total_power = unit['power'] * count
        air_exchange = (unit['air_flow'] * count) / room_volume if room_volume > 0 else 0

        if air_exchange >= 2.5:
            suitable.append({
                'model': unit['model'],
                'count': count,
                'total_power': total_power,
                'air_flow': unit['air_flow'] * count,
                'air_exchange': round(air_exchange, 1),
                'price': unit['price'] * count
            })

    return suitable

# ================== ВИЗУАЛИЗАЦИЯ ПОТОКА ==================
def create_airflow_visualization(units, room_length, room_height):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(0, room_length)
    ax.set_ylim(0, room_height)
    ax.set_title("Распределение теплого воздуха")

    for i, unit in enumerate(units):
        x = 0.5 + i * 1.5
        y = 0.5

        # корпус
        ax.add_patch(plt.Rectangle((x-0.2, y-0.2), 0.4, 0.4, color='red'))

        # поток (полупрозрачный градиент)
        for alpha, width in zip([0.3, 0.2, 0.1], [2, 3.5, 5]):
            ax.add_patch(plt.Rectangle((x, y-0.5), width, room_height, color='orange', alpha=alpha))

        ax.text(x, y+0.5, f"{unit['model']} x{unit['count']}", fontsize=8, color='black')

    return fig

# ================== ОСНОВНАЯ ПРОГРАММА ==================
def main():
    st.title("🔥 Калькулятор тепловентилятора 'Торнадо'")

    # ========== Сайдбар ==========
    with st.sidebar:
        st.header("📐 Параметры помещения")
        area = st.number_input("Площадь помещения (м²)", min_value=10.0, max_value=1000.0, value=100.0, step=1.0)
        height = st.number_input("Высота помещения (м)", min_value=2.0, max_value=10.0, value=4.0, step=0.1)
        room_volume = area * height

        wall_material = st.selectbox("Материал стен", list(MATERIALS.keys()))
        wall_thickness = st.number_input("Толщина стен (м)", min_value=0.1, max_value=1.0, value=0.4, step=0.05)

        window_area = st.number_input("Площадь окон (м²)", min_value=0.0, value=10.0, step=0.5)
        window_type = st.selectbox("Тип окон", list([k for k in U_VALUES if k.startswith("окно")]))

        door_area = st.number_input("Площадь дверей (м²)", min_value=0.0, value=2.0, step=0.1)
        door_type = st.selectbox("Тип дверей", list([k for k in U_VALUES if k.startswith("дверь")]))

        floor_insulated = st.checkbox("Утепленный пол", value=True)
        ceiling_insulated = st.checkbox("Утепленный потолок", value=True)

        st.subheader("🌡️ Климат")
        temp_out = st.number_input("Температура наружного воздуха (°C)", value=-15)
        temp_in = st.number_input("Температура внутри помещения (°C)", value=20)
        temp_water = st.number_input("Температура теплоносителя (°C)", value=70)

        st.subheader("🚿 Радиаторы")
        has_radiators = st.checkbox("Есть радиаторы отопления", value=False)
        radiator_type, radiator_height, radiator_sections = None, None, 0
        if has_radiators:
            radiator_type = st.selectbox("Тип радиатора", ["алюминиевые", "чугунные"])
            radiator_height = st.selectbox("Высота радиатора (мм)", [350, 500])
            radiator_sections = st.number_input("Количество секций", min_value=1, max_value=100, value=10)

        st.subheader("📍 Расположение тепловентилятора")
        unit_position = st.radio("Расположение", ["У стены", "В углу", "По центру"], index=0)

    # ========== Расчёты ==========
    wall_area = area * height * 0.4
    floor_area = area
    ceiling_area = area

    params = {
        'wall_area': wall_area,
        'floor_area': floor_area,
        'ceiling_area': ceiling_area,
        'room_volume': room_volume,
        'wall_material': wall_material,
        'wall_thickness': wall_thickness,
        'window_area': window_area,
        'window_type': window_type,
        'door_area': door_area,
        'door_type': door_type,
        'floor_insulated': floor_insulated,
        'ceiling_insulated': ceiling_insulated,
        'temp_in': temp_in,
        'temp_out': temp_out,
        'has_radiators': has_radiators,
        'radiator_type': radiator_type,
        'radiator_height': radiator_height,
        'radiator_sections': radiator_sections
    }

    heat_loss = calculate_heat_loss(params) / 1000  # кВт

    st.header("📊 Результаты расчёта")
    st.metric("Объем помещения", f"{room_volume:.1f} м³")
    st.metric("Теплопотери", f"{heat_loss:.2f} кВт")

    st.subheader("🔥 Подбор тепловентиляторов")
    units = select_heat_exchangers(heat_loss, room_volume)
    if units:
        df = pd.DataFrame(units)
        st.dataframe(df, use_container_width=True, hide_index=True)

        fig = create_airflow_visualization(units, room_length=10, room_height=height)
        st.pyplot(fig)
    else:
        st.warning("Нет подходящих моделей. Попробуйте изменить параметры.")

if __name__ == "__main__":
    main()
