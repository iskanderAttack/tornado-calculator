# tornado_calculator_full.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import math

st.set_page_config(
    page_title="Калькулятор тепловентилятора Торнадо — полный",
    page_icon="🌪️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Данные / константы
# -----------------------------
@st.cache_data
def load_heat_exchangers():
    # power — кВт, air_flow — м³/ч
    return [
        {'model': 'Торнадо 3',  'power': 20.0,  'air_flow': 1330, 'price': 65000,  'type': 'торнадо'},
        {'model': 'Торнадо 4',  'power': 33.0,  'air_flow': 2670, 'price': 85000,  'type': 'торнадо'},
        {'model': 'Торнадо 5',  'power': 55.0,  'air_flow': 4500, 'price': 120000, 'type': 'торнадо'},
        {'model': 'Торнадо 10', 'power': 106.0, 'air_flow': 9000, 'price': 280000, 'type': 'торнадо'},
        # Можно добавить ещё модели
    ]

# Материалы (бетон удалён)
MATERIALS = {
    'кирпич': 0.7,
    'газоблок': 0.18,
    'пеноблок': 0.16,
    'керамзитоблок': 0.4,
    'сэндвич панель': 0.05,
    'брус': 0.15
}

# U-значения (Вт/м²·°C)
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

# Эмпирические коэффициенты тепловой отдачи секции (Вт/°C на секцию)
SECTION_COEFF = {
    'алюминиевые': {350: 2.4, 500: 3.2},
    'чугунные':    {350: 1.8, 500: 2.6}
}

# -----------------------------
# Вспомогательные функции
# -----------------------------
def infer_room_sides_from_area(area_m2, ratio=1.0):
    """ Получаем длину и ширину по площади и соотношению сторон """
    if area_m2 <= 0:
        return 1.0, 1.0
    width = math.sqrt(area_m2 / ratio)
    length = area_m2 / width
    return float(length), float(width)

def radiator_total_heat(sections_total, rad_type, height_mm, t_fluid_in, t_in):
    """Q = sections_total * coeff * (t_fluid_in - t_in) (Вт)"""
    if sections_total <= 0:
        return 0.0
    coeff = SECTION_COEFF.get(rad_type, {}).get(height_mm, None)
    if coeff is None:
        coeff = 2.2
    delta = max(t_fluid_in - t_in, 0.0)
    return sections_total * coeff * delta

def calculate_heat_loss_by_components(params):
    """ Расчёт всех компонент теплопотерь в ваттах """
    area = params['area']
    height = params['height']
    t_out = params['t_out']
    t_in = params['t_in']
    delta = max(t_in - t_out, 0.0)

    length_est, width_est = infer_room_sides_from_area(area, params.get('shape_ratio', 1.0))
    perimeter = 2 * (length_est + width_est)
    wall_area = perimeter * height

    wall_loss = wall_area * (MATERIALS.get(params['wall_material'], 0.4) / max(params['wall_thickness'], 0.01)) * delta
    window_loss = params['window_area'] * U_VALUES[params['window_type']] * delta
    door_loss = params['door_area'] * U_VALUES[params['door_type']] * delta
    floor_type = 'пол_утепленный' if params['floor_insulated'] else 'пол_неутепленный'
    floor_loss = area * U_VALUES[floor_type] * delta
    ceiling_type = 'потолок_утепленный' if params['ceiling_insulated'] else 'потолок_неутепленный'
    ceiling_loss = area * U_VALUES[ceiling_type] * delta

    room_volume = area * height
    infiltration_loss = room_volume * 0.3 * 1.2 * 1005 * delta / 3600  # Вт

    components = {
        'Стены': wall_loss,
        'Окна': window_loss,
        'Двери': door_loss,
        'Пол': floor_loss,
        'Потолок': ceiling_loss,
        'Инфильтрация': infiltration_loss
    }
    total_loss = sum(components.values())
    return total_loss, components, room_volume, (length_est, width_est), wall_area

def select_heat_exchangers(required_kw, room_volume, prefer_type="торнадо", max_units=4):
    """ Подбор 1..max_units агрегатов """
    units = load_heat_exchangers()
    candidates = []
    for n in range(1, max_units+1):
        for u in units:
            if prefer_type != "все" and u['type'] != prefer_type:
                continue
            total_power = u['power'] * n  # кВт
            total_air = u['air_flow'] * n  # м3/ч
            if required_kw <= 0:
                continue
            power_margin_ratio = total_power / required_kw
            air_exchange = total_air / max(room_volume, 1.0)
            if power_margin_ratio >= 1.15 and 2.5 <= air_exchange <= 7.0:
                candidates.append({
                    'model': f"{n} × {u['model']}",
                    'base_model': u['model'],
                    'units': n,
                    'power_kW': total_power,
                    'air_flow': total_air,
                    'air_exchange': round(air_exchange, 2),
                    'power_reserve_%': round((total_power - required_kw) / required_kw * 100, 1),
                    'price': u['price'] * n
                })
    candidates_sorted = sorted(candidates, key=lambda x: (x['price'], abs(x['power_reserve_%'])))
    return candidates_sorted

def create_room_visual(length_m, width_m, fan_positions, fan_directions, fan_models, show_grid=False):
    """ Рисует комнату (length x width), отображает вентиляторы с подписями моделей """
    fig, ax = plt.subplots(figsize=(8, max(4, 6 * (width_m / max(length_m,1e-6)))))
    ax.set_xlim(0, length_m)
    ax.set_ylim(0, width_m)
    ax.set_aspect('equal')
    ax.set_title("Схема помещения и размещение тепловентиляторов")
    ax.add_patch(plt.Rectangle((0, 0), length_m, width_m, fill=False, linewidth=2))

    if show_grid:
        ax.grid(True, linestyle=':', alpha=0.5)

    for idx, pos in enumerate(fan_positions):
        x, y = pos
        ax.scatter(x, y, s=160, marker='^', color='tab:orange', zorder=10)
        label = fan_models[idx] if idx < len(fan_models) else f"Торнадо {idx+1}"
        ax.text(x, y - 0.3, label, ha='center', va='top', fontsize=9, weight='bold')
        angle = fan_directions[idx] if idx < len(fan_directions) else 0.0
        dx = math.cos(angle) * max(length_m, width_m) * 0.35
        dy = math.sin(angle) * max(length_m, width_m) * 0.35
        ax.arrow(x, y, dx, dy,
                 head_width=0.2*max(1, width_m/10),
                 head_length=0.25*max(1,length_m/10),
                 color='tab:orange', alpha=0.8)

    ax.set_xlabel("Длина (м)")
    ax.set_ylabel("Ширина (м)")
    plt.tight_layout()
    return fig

# -----------------------------
# Интерфейс Streamlit
# -----------------------------
def main():
    st.title("🌪️ Калькулятор тепловентилятора Торнадо — продвинутый")
    st.markdown("Заполните параметры помещения и радиаторов (установленные). Калькулятор онлайн подберёт тепловентиляторы (1..4) и предложит их расположение.")

    # Левая панель — параметры помещения
    with st.sidebar:
        st.header("📐 Параметры помещения")

        area = st.number_input("Площадь помещения (м²)", min_value=4.0, max_value=100000.0, value=100.0, step=1.0)
        height = st.number_input("Высота помещения (м)", min_value=2.0, max_value=30.0, value=6.0, step=0.1)

        st.markdown("Форма помещения (для визуализации)")
        shape_ratio = st.selectbox("Соотношение длины к ширине (L:W)", ["1:1 (квадрат)", "2:1", "3:1", "пользовательское"], index=0)
        if shape_ratio == "1:1 (квадрат)":
            ratio = 1.0
        elif shape_ratio == "2:1":
            ratio = 2.0
        elif shape_ratio == "3:1":
            ratio = 3.0
        else:
            ratio = st.number_input("Введите L/W (например 1.5)", min_value=0.2, max_value=10.0, value=1.0, step=0.1)

        st.subheader("Ограждающие конструкции")
        wall_material = st.selectbox("Материал стен", list(MATERIALS.keys()))
        wall_thickness_cm = st.number_input("Толщина стен (см)", min_value=5, max_value=200, value=30, step=1)
        wall_thickness = wall_thickness_cm / 100.0

        st.markdown("Окна и двери")
        window_area = st.number_input("Площадь окон (м²)", min_value=0.0, value=5.0, step=0.1)
        window_type = st.selectbox("Тип окон", ["окно_евро", "окно_тройное", "окно_двойное", "окно_одинарное"])
        door_area = st.number_input("Площадь дверей (м²)", min_value=0.0, value=2.0, step=0.1)
        door_type = st.selectbox("Тип дверей", ["дверь_утепленная", "дверь_деревянная", "дверь_металлическая"])

        st.subheader("Утепление")
        floor_insulated = st.checkbox("Утеплённый пол", value=True)
        ceiling_insulated = st.checkbox("Утеплённый потолок", value=True)

        st.subheader("Климат")
        t_out = st.number_input("Температура снаружи °C", value=-20.0, step=0.5)
        t_in = st.number_input("Температура внутри (целевая) °C", value=18.0, step=0.5)
        t_fluid_in = st.number_input("Температура теплоносителя на входе °C", value=70.0, step=0.5)

        st.markdown("---")
        st.header("♨️ Радиаторы (те, что уже установлены)")
        rad_present = st.checkbox("У меня есть радиаторы (учесть в расчёте)", value=False)
        rad_sections_total = 0
        rad_type = None
        rad_height = None
        if rad_present:
            rad_type = st.selectbox("Тип радиаторов", ['алюминиевые', 'чугунные'])
            rad_height = st.selectbox("Высота секции (мм)", [350, 500])
            sections_mode = st.radio("Ввод количества секций:", ["общее количество секций", "секций в связке + число связок"], index=0)
            if sections_mode == "общее количество секций":
                rad_sections_total = st.number_input("Общее количество секций", min_value=0, value=0, step=1)
            else:
                per_bank = st.number_input("Секций в одной связке", min_value=1, value=4, step=1)
                banks = st.number_input("Количество связок", min_value=1, value=1, step=1)
                rad_sections_total = per_bank * banks

        st.markdown("---")
        st.markdown("Максимальное число агрегатов для подбора (каскад)")
        max_units = st.slider("Макс. агрегатов", min_value=1, max_value=4, value=3, step=1)

    # -----------------------------
    # Основная часть: расчёты
    # -----------------------------
    params = {
        'area': area,
        'height': height,
        'shape_ratio': ratio,
        'wall_material': wall_material,
        'wall_thickness': wall_thickness,
        'window_area': window_area,
        'window_type': window_type,
        'door_area': door_area,
        'door_type': door_type,
        'floor_insulated': floor_insulated,
        'ceiling_insulated': ceiling_insulated,
        't_out': t_out,
        't_in': t_in
    }

    total_loss_w, breakdown, room_volume, (room_length, room_width), wall_area = calculate_heat_loss_by_components(params)

    # Радиаторы
    radiator_heat_w = 0.0
    if rad_present and rad_sections_total > 0:
        radiator_heat_w = radiator_total_heat(rad_sections_total, rad_type, rad_height, t_fluid_in, t_in)

    net_need_w = max(total_loss_w - radiator_heat_w, 0.0)
    net_need_kw = net_need_w / 1000.0

    suitable = select_heat_exchangers(net_need_kw if net_need_kw > 0 else 0.001, room_volume, prefer_type="торнадо", max_units=max_units)

    # -----------------------------
    # Отображение результатов
    # -----------------------------
    st.header("📊 Результаты расчёта (онлайн)")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Площадь", f"{area:.1f} м²")
        st.metric("Объём", f"{room_volume:.1f} м³")
        st.metric("Площадь стен (прибл.)", f"{wall_area:.1f} м²")
    with c2:
        st.metric("Т наружн.", f"{t_out:.1f} °C")
        st.metric("Т внутр.", f"{t_in:.1f} °C")
        st.metric("Т вх. теплоносителя", f"{t_fluid_in:.1f} °C")
    with c3:
        st.metric("Теплопотери (итого)", f"{total_loss_w/1000:.3f} кВт")
        st.metric("Радиаторы (отдача)", f"{radiator_heat_w/1000:.3f} кВт")
        st.metric("Остаток (нужна мощность)", f"{net_need_kw:.3f} кВт")

    st.subheader("🔎 Детализация по компонентам (Вт)")
    df_comp = pd.DataFrame(list(breakdown.items()), columns=['Компонент','Вт'])
    st.dataframe(df_comp, use_container_width=True)

    # -----------------------------
    # Подбор и вывод вариантов
    # -----------------------------
    st.subheader("🔥 Подбор тепловентиляторов (включая каскадные решения)")
    if net_need_kw <= 0:
        st.success("Радиаторов достаточно — дополнительный тепловентилятор не требуется.")
        suitable = []
    else:
        if suitable:
            df_suit = pd.DataFrame(suitable)
            df_suit_display = df_suit[['model','power_kW','air_flow','air_exchange','power_reserve_%','price']].copy()
            df_suit_display.columns = ['Конфигурация','Мощность, кВт','Расход воздуха, м³/ч','Кратность, 1/ч','Запас, %','Цена, руб.']
            st.dataframe(df_suit_display, use_container_width=True)
            best = suitable[0]
            st.success(f"Рекомендуется: {best['model']} (мощность {best['power_kW']} кВт, запас {best['power_reserve_%']}%)")
        else:
            st.warning("Нет подходящих моделей под текущие параметры.")

    # -----------------------------
    # Визуализация
    # -----------------------------
    st.subheader("📈 Визуализация помещения и воздушных потоков")
    if suitable:
        best = suitable[0]
        units_to_place = best['units']
        fan_models = [best['base_model']] * units_to_place
    else:
        units_to_place = 1
        fan_models = ["Торнадо"]

    default_positions = []
    default_directions = []
    for i in range(units_to_place):
        x = room_length * (i+1) / (units_to_place+1)
        y = 0.5
        if x < room_length/2 and x < room_width/2:
            angle = 0
        elif x >= room_length/2 and x < room_width/2:
            angle = math.pi
        elif y < room_width/2:
            angle = math.pi/2
        else:
            angle = -math.pi/2
        default_positions.append((x,y))
        default_directions.append(angle)

    fig = create_room_visual(room_length, room_width, default_positions, default_directions, fan_models, show_grid=True)
    st.pyplot(fig)

    st.markdown("---")
    st.markdown("Разработано для подбора тепловентиляторов 🌪️ Торнадо. Профессиональный инструмент инженеров-теплотехников.")

if __name__ == "__main__":
    main()
