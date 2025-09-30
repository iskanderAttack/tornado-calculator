# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import math

# -----------------------------
# Конфигурация страницы
# -----------------------------
st.set_page_config(
    page_title="Калькулятор тепловентиляторов Торнадо",
    page_icon="🌪️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Данные и константы
# -----------------------------

@st.cache_data
def load_heat_exchangers():
    """
    Загружает данные о моделях.
    throw_length — примерная дальность воздушной струи в метрах.
    """
    return [
        {'model': 'Торнадо 3',  'power_nominal': 20.0,  'air_flow': 1330, 'price': 65000,  't_water_nom': 90, 't_air_nom': 15, 'throw_length': 14},
        {'model': 'Торнадо 4',  'power_nominal': 33.0,  'air_flow': 2670, 'price': 85000,  't_water_nom': 90, 't_air_nom': 15, 'throw_length': 22},
        {'model': 'Торнадо 5',  'power_nominal': 55.0,  'air_flow': 4500, 'price': 120000, 't_water_nom': 90, 't_air_nom': 15, 'throw_length': 28},
        {'model': 'Торнадо 10', 'power_nominal': 106.0, 'air_flow': 9000, 'price': 280000, 't_water_nom': 90, 't_air_nom': 15, 'throw_length': 35},
    ]

MATERIALS = { 'Кирпич': 0.7, 'Газоблок': 0.18, 'Пеноблок': 0.16, 'Керамзитоблок': 0.4, 'Сэндвич-панель': 0.05, 'Брус': 0.15 }
U_VALUES = { 'Окно одинарное': 5.0, 'Окно двойное': 2.9, 'Окно тройное': 1.5, 'Окно европакет': 1.3, 'Дверь деревянная': 2.0, 'Дверь металлическая': 1.5, 'Дверь утепленная': 0.8, 'Пол неутепленный': 0.5, 'Пол утепленный': 0.2, 'Потолок неутепленный': 0.6, 'Потолок утепленный': 0.25 }
INFILTRATION_RATES = { "Высокая (цех, частые ворота)": 1.5, "Средняя (склад, редкие ворота)": 0.7, "Низкая (герметичное здание)": 0.3 }
SECTION_COEFF = { 'Алюминиевые': {350: 2.4, 500: 3.2}, 'Чугунные': {350: 1.8, 500: 2.6} }

# -----------------------------
# Вспомогательные функции
# -----------------------------

def infer_room_sides_from_area(area_m2, ratio=1.0):
    if area_m2 <= 0: return 1.0, 1.0
    width = math.sqrt(area_m2 / ratio)
    length = area_m2 / width
    return float(length), float(width)

def calculate_radiator_heat(sections, rad_type, height, t_fluid_in, t_in):
    if not sections or sections <= 0: return 0.0
    coeff = SECTION_COEFF.get(rad_type, {}).get(height, 2.2)
    delta_t = max(t_fluid_in - t_in, 0.0)
    return sections * coeff * delta_t

def calculate_heat_loss(params):
    area, height = params['area'], params['height']
    t_in, t_out = params['t_in'], params['t_out']
    delta_t = max(t_in - t_out, 0.0)

    length, width = infer_room_sides_from_area(area, params.get('shape_ratio', 1.0))
    wall_area = 2 * (length + width) * height
    room_volume = area * height

    wall_loss = wall_area * (MATERIALS.get(params['wall_material'], 0.4) / max(params['wall_thickness'], 0.01)) * delta_t
    window_loss = params['window_area'] * U_VALUES[params['window_type']] * delta_t
    door_loss = params['door_area'] * U_VALUES[params['door_type']] * delta_t
    floor_loss = area * U_VALUES['Пол утепленный' if params['floor_insulated'] else 'Пол неутепленный'] * delta_t
    ceiling_loss = area * U_VALUES['Потолок утепленный' if params['ceiling_insulated'] else 'Потолок неутепленный'] * delta_t
    infiltration_loss = room_volume * params['infiltration_rate'] * 1.2 * 1005 * delta_t / 3600

    components = {'Стены': wall_loss, 'Окна': window_loss, 'Двери': door_loss, 'Пол': floor_loss, 'Потолок': ceiling_loss, 'Инфильтрация': infiltration_loss}
    total_loss = sum(components.values())
    return total_loss, components, room_volume, (length, width)

def correct_fan_power(fan, t_water_actual, t_air_actual):
    POWER_EXPONENT = 1.2
    delta_t_nominal = fan['t_water_nom'] - fan['t_air_nom']
    delta_t_actual = t_water_actual - t_air_actual
    if delta_t_nominal <= 0 or delta_t_actual <= 0: return 0.0
    return fan['power_nominal'] * ((delta_t_actual / delta_t_nominal) ** POWER_EXPONENT)

def select_heat_exchangers(required_kw, room_volume, t_water_in, t_air_in, max_units=4):
    all_units = load_heat_exchangers()
    candidates = []
    for n in range(1, max_units + 1):
        for unit_base in all_units:
            corrected_power_per_unit = correct_fan_power(unit_base, t_water_in, t_air_in)
            total_power = corrected_power_per_unit * n
            total_air_flow = unit_base['air_flow'] * n
            if required_kw <= 0: continue
            
            power_margin_ratio = total_power / required_kw
            air_exchange_rate = total_air_flow / max(room_volume, 1.0)
            
            if power_margin_ratio >= 1.15 and 2.5 <= air_exchange_rate <= 7.0:
                candidates.append({
                    'model': f"{n} × {unit_base['model']}", 'base_model': unit_base['model'],
                    'base_data': unit_base, 'units': n, 'power_kW': total_power,
                    'air_flow': total_air_flow, 'air_exchange': round(air_exchange_rate, 2),
                    'power_reserve_%': round((total_power - required_kw) / required_kw * 100, 1),
                    'price': unit_base['price'] * n
                })
    return sorted(candidates, key=lambda x: (x['price'], x['power_reserve_%']))

def calculate_recommended_flow(power_kw, delta_t_target=20.0):
    """Рассчитывает рекомендуемый расход воды (м³/ч) для целевого охлаждения."""
    if power_kw <= 0: return 0.0
    WATER_DENSITY_KG_M3 = 970
    WATER_SPECIFIC_HEAT_J_KG_C = 4190
    power_w = power_kw * 1000
    mass_flow_kg_s = power_w / (WATER_SPECIFIC_HEAT_J_KG_C * delta_t_target)
    volume_flow_m3h = mass_flow_kg_s * 3600 / WATER_DENSITY_KG_M3
    return volume_flow_m3h

def create_room_visual(length, width, fan_config):
    """Создает схему помещения с визуализацией воздушных потоков."""
    fig, ax = plt.subplots(figsize=(8, 8 * (width / max(length, 1e-6))))
    ax.set_xlim(-1, length + 1)
    ax.set_ylim(-1, width + 1)
    ax.set_aspect('equal')
    ax.set_title("Схема размещения и воздушных потоков")
    ax.add_patch(patches.Rectangle((0, 0), length, width, fill=False, linewidth=2, edgecolor='gray'))
    ax.grid(True, linestyle=':', alpha=0.6)

    num_fans = fan_config.get('units', 0)
    if num_fans > 0:
        fan_data = fan_config['base_data']
        throw_length = min(fan_data['throw_length'], width * 0.95)
        cone_width = throw_length * 0.4

        for i in range(num_fans):
            x_fan = length * (i + 1) / (num_fans + 1)
            y_fan = width - 0.2

            # Вентилятор
            ax.scatter(x_fan, y_fan, s=150, marker='s', color='black', zorder=10, ec='white')
            ax.text(x_fan, y_fan + 0.5, fan_config['base_model'], ha='center', va='bottom', fontsize=9, weight='bold')

            # Поток воздуха (градиент)
            gradient = np.linspace(1.0, 0.0, 256).reshape(-1, 1)
            gradient = np.tile(gradient, (1, 64))
            
            x_start = x_fan - cone_width / 2
            y_start = y_fan - throw_length
            
            ax.imshow(gradient, cmap='Oranges', extent=[x_start, x_start + cone_width, y_start, y_fan], 
                      aspect='auto', zorder=5, alpha=0.7)

    ax.set_xlabel("Длина (м)")
    ax.set_ylabel("Ширина (м)")
    plt.tight_layout()
    return fig

# -----------------------------
# Интерфейс Streamlit
# -----------------------------
def main():
    st.title("🌪️ Калькулятор подбора тепловентиляторов Торнадо")
    st.markdown("Заполните параметры, и калькулятор подберет оптимальную модель, рассчитает теплопотери и предложит схему размещения.")

    with st.sidebar:
        st.header("📐 Параметры помещения")
        area = st.number_input("Площадь (м²)", 10.0, 10000.0, 200.0, 10.0)
        height = st.number_input("Высота потолков (м)", 2.5, 25.0, 6.0, 0.1)
        ratio_map = {"1:1 (квадрат)": 1.0, "2:1": 2.0, "3:1": 3.0}
        shape_ratio = ratio_map[st.selectbox("Соотношение Длина:Ширина", list(ratio_map.keys()))]

        st.subheader("Ограждающие конструкции")
        wall_material = st.selectbox("Материал стен", list(MATERIALS.keys()), 0)
        wall_thickness_cm = st.number_input("Толщина стен (см)", 5, 200, 38, 1)

        window_area = st.number_input("Общая площадь окон (м²)", 0.0, value=20.0, step=1.0)
        window_type = st.selectbox("Тип окон", list(U_VALUES.keys())[0:4], 3)
        door_area = st.number_input("Общая площадь дверей/ворот (м²)", 0.0, value=4.0, step=0.5)
        door_type = st.selectbox("Тип дверей/ворот", list(U_VALUES.keys())[4:7], 2)

        st.subheader("Утепление и герметичность")
        floor_insulated = st.checkbox("Пол утеплен", value=True)
        ceiling_insulated = st.checkbox("Потолок утеплен", value=True)
        infiltration_rate = INFILTRATION_RATES[st.selectbox("Герметичность помещения", list(INFILTRATION_RATES.keys()), 1)]
        
        st.header("🌡️ Климат и теплоноситель")
        t_out = st.number_input("Температура снаружи (°C)", value=-25.0, step=1.0)
        t_in = st.number_input("Требуемая температура внутри (°C)", value=18.0, step=1.0)
        t_fluid_in = st.number_input("Температура теплоносителя на входе (°C)", value=80.0, step=1.0)

        st.header("♨️ Существующие радиаторы")
        rad_present = st.checkbox("Учесть имеющиеся радиаторы")
        rad_sections_total, rad_type, rad_height = 0, None, None
        if rad_present:
            rad_type = st.selectbox("Тип радиаторов", list(SECTION_COEFF.keys()))
            rad_height = st.selectbox("Высота секции (мм)", [350, 500])
            input_mode = st.radio("Как указать количество секций?", ("Общее количество", "По связкам"))
            if input_mode == "Общее количество":
                rad_sections_total = st.number_input("Общее количество секций", 0, value=50, step=1)
            else:
                banks = st.number_input("Количество связок", 1, value=5)
                per_bank = st.number_input("Секций в одной связке", 1, value=10)
                rad_sections_total = banks * per_bank

    params = {
        'area': area, 'height': height, 'shape_ratio': shape_ratio,
        'wall_material': wall_material, 'wall_thickness': wall_thickness_cm / 100.0,
        'window_area': window_area, 'window_type': window_type,
        'door_area': door_area, 'door_type': door_type,
        'floor_insulated': floor_insulated, 'ceiling_insulated': ceiling_insulated,
        'infiltration_rate': infiltration_rate, 't_out': t_out, 't_in': t_in
    }

    total_loss_w, breakdown, volume, (length, width) = calculate_heat_loss(params)
    radiator_heat_w = calculate_radiator_heat(rad_sections_total, rad_type, rad_height, t_fluid_in, t_in)
    net_need_kw = max(total_loss_w - radiator_heat_w, 0.0) / 1000.0
    suitable_options = select_heat_exchangers(net_need_kw, volume, t_fluid_in, t_in)
    
    st.header("📊 Результаты расчета")
    col1, col2, col3 = st.columns(3)
    col1.metric("Общие теплопотери", f"{total_loss_w/1000:.2f} кВт")
    col2.metric("Теплоотдача радиаторов", f"{radiator_heat_w/1000:.2f} кВт")
    col3.metric("Требуемая мощность", f"{net_need_kw:.2f} кВт", delta_color="inverse")

    st.subheader("🔥 Рекомендованные конфигурации")
    if net_need_kw < 0.1:
        st.success("Текущей системы отопления достаточно. Дополнительные тепловентиляторы не требуются.")
    elif suitable_options:
        df = pd.DataFrame(suitable_options)
        df_display = df[['model', 'power_kW', 'air_flow', 'air_exchange', 'power_reserve_%', 'price']].copy()
        df_display.columns = ['Конфигурация', 'Мощность, кВт', 'Воздух, м³/ч', 'Кратность, 1/ч', 'Запас, %', 'Цена, руб']
        st.dataframe(df_display, use_container_width=True, hide_index=True)
        
        best_option = suitable_options[0]
        st.success(f"**Оптимальный вариант:** {best_option['model']} (Общая цена: {best_option['price']:,} руб.)")
        
        c1, c2, c3 = st.columns(3)
        power_per_unit_kw = best_option['power_kW'] / best_option['units']
        rec_flow = calculate_recommended_flow(power_per_unit_kw)
        c1.metric("Скоррект. мощность", f"{best_option['power_kW']:.2f} кВт")
        c2.metric("Запас мощности", f"{best_option['power_reserve_%']:.1f} %")
        c3.metric("Рекомендуемый расход воды", f"{rec_flow:.2f} м³/ч", help="Расход на 1 аппарат для охлаждения воды на 20°C (например, с 80°C до 60°C).")
    else:
        st.warning("Подходящих моделей не найдено. Попробуйте изменить параметры.")

    main_col, empty_col = st.columns([2, 1])
    with main_col:
        tab1, tab2 = st.tabs(["Схема помещения", "Детализация теплопотерь"])
        with tab1:
            if suitable_options:
                fig = create_room_visual(length, width, suitable_options[0])
                st.pyplot(fig)
            else:
                st.info("Визуализация будет доступна после подбора оборудования.")
        with tab2:
            if total_loss_w > 0:
                df_breakdown = pd.DataFrame(list(breakdown.items()), columns=['Компонент', 'Теплопотери, Вт'])
                df_breakdown['Доля, %'] = (df_breakdown['Теплопотери, Вт'] / total_loss_w * 100).round(1)
                st.dataframe(df_breakdown, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()

