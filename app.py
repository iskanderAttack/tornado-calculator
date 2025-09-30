import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io

# Настройка страницы
st.set_page_config(
    page_title="Калькулятор тепловентилятора 'Торнадо'",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# База данных тепловентиляторов "Торнадо"
@st.cache_data
def load_heat_exchangers():
    return [
        {'model': 'Торнадо 3', 'power': 20, 'air_flow': 1330, 'price': 65000, 'type': 'торнадо'},
        {'model': 'Торнадо 4', 'power': 33, 'air_flow': 2670, 'price': 85000, 'type': 'торнадо'},
        {'model': 'Торнадо 5', 'power': 55, 'air_flow': 4500, 'price': 120000, 'type': 'торнадо'},
        {'model': 'Торнадо 10', 'power': 240, 'air_flow': 9000, 'price': 280000, 'type': 'торнадо'}
    ]

# Теплопроводность материалов (Вт/м·°C)
MATERIALS = {
    'кирпич': 0.7,
    'газоблок': 0.18,
    'пеноблок': 0.16,
    'керамзитоблок': 0.4,
    'сэндвич панель': 0.05,
    'брус': 0.15
}

# Коэффициенты теплопотерь
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

def calculate_heat_loss(params):
    """Расчет теплопотерь помещения"""
    total_loss = 0
    temp_diff = params['inside_temp'] - params['outside_temp']

    # Стены
    wall_loss = (params['wall_area'] *
                MATERIALS[params['wall_material']] /
                max(params['wall_thickness'], 0.01) * temp_diff)
    total_loss += wall_loss

    # Окна
    window_loss = params['window_area'] * U_VALUES[params['window_type']] * temp_diff
    total_loss += window_loss

    # Двери
    door_loss = params['door_area'] * U_VALUES[params['door_type']] * temp_diff
    total_loss += door_loss

    # Пол
    floor_type = 'пол_утепленный' if params['floor_insulated'] else 'пол_неутепленный'
    floor_loss = params['floor_area'] * U_VALUES[floor_type] * temp_diff
    total_loss += floor_loss

    # Потолок
    ceiling_type = 'потолок_утепленный' if params['ceiling_insulated'] else 'потолок_неутепленный'
    ceiling_loss = params['ceiling_area'] * U_VALUES[ceiling_type] * temp_diff
    total_loss += ceiling_loss

    # Инфильтрация
    infiltration_loss = params['room_volume'] * 0.3 * 1.2 * 1005 * temp_diff / 3600
    total_loss += infiltration_loss

    return max(total_loss, 0)

def select_heat_exchangers(required_power, room_volume, preferred_type="торнадо", max_units=3):
    """Подбор одного или нескольких тепловентиляторов"""
    heat_exchangers = load_heat_exchangers()
    suitable_models = []

    for n_units in range(1, max_units + 1):
        for unit in heat_exchangers:
            if preferred_type != "все" and unit['type'] != preferred_type:
                continue

            total_power = unit['power'] * n_units
            total_air_flow = unit['air_flow'] * n_units
            power_margin = total_power / required_power if required_power > 0 else 0
            air_exchange = total_air_flow / room_volume if room_volume > 0 else 0

            if power_margin >= 1.15 and 2.5 <= air_exchange <= 7:
                suitable_models.append({
                    'model': f"{n_units} × {unit['model']}",
                    'power': total_power,
                    'air_flow': total_air_flow,
                    'air_exchange': round(air_exchange, 1),
                    'power_reserve': round((total_power - required_power) / required_power * 100, 1),
                    'price': unit['price'] * n_units,
                    'units': n_units,
                    'base_model': unit['model']
                })

    return sorted(suitable_models, key=lambda x: (x['power_reserve'], x['price']))

def create_airflow_visualization(room_length, room_width, units_count=1):
    """Схема распределения теплого воздуха"""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_title("Схема распределения теплого воздуха")
    ax.set_xlim(0, room_length)
    ax.set_ylim(0, room_width)

    for i in range(units_count):
        x = room_length / (units_count + 1) * (i + 1)
        y = 0.5 * room_width
        ax.scatter(x, y, color="red", s=200, marker="^", label="Тепловентилятор" if i == 0 else "")
        ax.arrow(x, y, room_length * 0.3, 0, head_width=0.3, head_length=0.5, fc='orange', ec='orange')

    ax.set_xlabel("Длина помещения (м)")
    ax.set_ylabel("Ширина помещения (м)")
    ax.legend()
    plt.tight_layout()
    return fig

def main():
    st.title("🔥 Калькулятор тепловентилятора 'Торнадо'")
    st.markdown("Рассчитайте теплопотери и подберите оптимальный тепловентилятор или их комбинацию.")

    # 📍 Местоположение
    location = st.selectbox(
        "📍 Выберите помещение:",
        ["Цех №1", "Цех №2", "Склад", "Ангар / гараж", "Подсобное помещение", "Офис внутри цеха"],
        index=0
    )
    st.info(f"Тепловентилятор будет установлен в: **{location}**")

    # Параметры помещения
    st.sidebar.header("📐 Параметры помещения")
    area = st.number_input("Площадь (м²)", min_value=10.0, max_value=1000.0, value=100.0, step=10.0)
    height = st.number_input("Высота (м)", min_value=2.0, max_value=15.0, value=5.0, step=0.1)
    volume = area * height

    wall_material = st.sidebar.selectbox("Материал стен", list(MATERIALS.keys()))
    wall_thickness = st.sidebar.number_input("Толщина стен (м)", min_value=0.1, max_value=1.0, value=0.4, step=0.05)

    window_area = st.sidebar.number_input("Площадь окон (м²)", min_value=0.0, value=5.0, step=0.5)
    window_type = st.sidebar.selectbox("Тип окон", ["окно_евро", "окно_тройное", "окно_двойное", "окно_одинарное"])

    door_area = st.sidebar.number_input("Площадь дверей (м²)", min_value=0.0, value=2.0, step=0.1)
    door_type = st.sidebar.selectbox("Тип дверей", ["дверь_утепленная", "дверь_деревянная", "дверь_металлическая"])

    floor_insulated = st.sidebar.checkbox("Утепленный пол", value=True)
    ceiling_insulated = st.sidebar.checkbox("Утепленный потолок", value=True)

    # Климат
    st.sidebar.header("🌡️ Климатические параметры")
    outside_temp = st.sidebar.slider("Температура снаружи (°C)", -40, 15, -15)
    inside_temp = st.sidebar.slider("Температура внутри (°C)", 10, 30, 20)
    coolant_temp = st.sidebar.slider("Температура теплоносителя (°C)", 40, 90, 70)

    # Расчет теплопотерь
    room_params = {
        'wall_material': wall_material,
        'wall_thickness': wall_thickness,
        'wall_area': height * (area / height) * 2 + area,  # упрощённо
        'window_area': window_area,
        'window_type': window_type,
        'door_area': door_area,
        'door_type': door_type,
        'floor_area': area,
        'ceiling_area': area,
        'floor_insulated': floor_insulated,
        'ceiling_insulated': ceiling_insulated,
        'room_volume': volume,
        'inside_temp': inside_temp,
        'outside_temp': outside_temp
    }

    heat_loss = calculate_heat_loss(room_params)

    # Результаты
    st.header("📊 Результаты расчета")
    st.metric("Объем помещения", f"{volume:.1f} м³")
    st.metric("Теплопотери", f"{heat_loss/1000:.2f} кВт")

    # Подбор вентиляторов
    suitable_units = select_heat_exchangers(heat_loss / 1000, volume, "торнадо")

    if suitable_units:
        df = pd.DataFrame(suitable_units)
        df_display = df[['model', 'power', 'air_flow', 'air_exchange', 'power_reserve', 'price']].copy()
        df_display.columns = ['Модель', 'Мощность, кВт', 'Расход воздуха, м³/ч', 'Кратность возд.', 'Запас, %', 'Цена, руб.']
        st.dataframe(df_display, use_container_width=True, hide_index=True)

        best_option = suitable_units[0]
        st.success(f"🎯 Рекомендуемая модель: {best_option['model']}")

        # Визуализация потока воздуха
        st.subheader("🌬️ Визуализация распределения воздуха")
        fig = create_airflow_visualization(room_length=10, room_width=area/10, units_count=best_option['units'])
        st.pyplot(fig)
    else:
        st.warning("⚠️ Не найдено подходящих моделей. Попробуйте изменить параметры помещения или увеличить количество агрегатов.")

if __name__ == "__main__":
    main()
