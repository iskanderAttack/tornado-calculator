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
        {'model': 'Торнадо 3', 'power': 20.0,  'air_flow': 1330, 'price': 65000, 'type': 'торнадо'},
        {'model': 'Торнадо 4', 'power': 33.0,  'air_flow': 2670, 'price': 85000, 'type': 'торнадо'},
        {'model': 'Торнадо 5', 'power': 55.0,  'air_flow': 4500, 'price': 120000,'type': 'торнадо'},
        {'model': 'Торнадо 10','power': 240.0, 'air_flow': 9000, 'price': 280000,'type': 'торнадо'},
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
    """
    Получаем длину и ширину по площади и соотношению сторон (length/width = ratio).
    Если ratio=1 -> квадрат.
    """
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
    """
    Расчёт всех компонент теплопотерь в ваттах.
    params: dict с ключами (area, height, wall_material, wall_thickness, window_area, window_type,
    door_area, door_type, floor_insulated, ceiling_insulated, t_out, t_in)
    """
    area = params['area']
    height = params['height']
    t_out = params['t_out']
    t_in = params['t_in']
    delta = max(t_in - t_out, 0.0)

    # аппроксимация периметра: предполагаем квадратное основание (при необходимости пользователь может задать ratio)
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
    """
    Подбор 1..max_units агрегатов. Возвращает список подходящих конфигураций.
    Требование: суммарная мощность >= required * 1.15 и кратность воздухообмена в диапазоне 2.5..7
    """
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
    # сортировка: сначала по меньшей цене, затем по меньшей избыточности (предпочитаем экономичное)
    candidates_sorted = sorted(candidates, key=lambda x: (x['price'], abs(x['power_reserve_%'])))
    return candidates_sorted

def create_room_visual(length_m, width_m, fan_positions, fan_directions, show_grid=False):
    """
    Рисует комнату (length x width), отображает вентиляторы в указанных позициях.
    fan_positions: list of (x, y) in meters (0..length, 0..width).
    fan_directions: list of angles (radians) направления потока.
    """
    fig, ax = plt.subplots(figsize=(8, max(4, 6 * (width_m / max(length_m,1e-6)))))
    ax.set_xlim(0, length_m)
    ax.set_ylim(0, width_m)
    ax.set_aspect('equal')
    ax.set_title("Схема помещения и размещение тепловентиляторов")
    ax.add_patch(plt.Rectangle((0, 0), length_m, width_m, fill=False, linewidth=2))

    if show_grid:
        ax.set_xticks([round(x,1) for x in list(range(0, int(math.ceil(length_m))+1))])
        ax.set_yticks([round(y,1) for y in list(range(0, int(math.ceil(width_m))+1))])
        ax.grid(True, linestyle=':', alpha=0.5)

    # рисуем вентиляторы и стрелки потока
    for idx, pos in enumerate(fan_positions):
        x, y = pos
        ax.scatter(x, y, s=160, marker='^', color='tab:orange', zorder=10)
        ax.text(x, y - 0.3, f"FV{idx+1}", ha='center', va='top', fontsize=9, weight='bold')
        # направление: стрелка вперёд
        angle = fan_directions[idx] if idx < len(fan_directions) else 0.0
        dx = math.cos(angle) * max(length_m, width_m) * 0.35
        dy = math.sin(angle) * max(length_m, width_m) * 0.35
        ax.arrow(x, y, dx, dy, head_width=0.2*max(1, width_m/10), head_length=0.25*max(1,length_m/10), color='tab:orange', alpha=0.8)

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
        wall_thickness = st.number_input("Толщина стен (м)", min_value=0.05, max_value=2.0, value=0.3, step=0.01)

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
        st.header("📍 Местоположение (зона)")
        location = st.selectbox("Выберите зону установки тепловентилятора:", [
            "Цех №1", "Цех №2", "Склад", "Ангар / гараж", "Подсобное помещение", "Офис внутри цеха"
        ], index=0)

        st.markdown("Максимальное число агрегатов для подбора (каскад)")
        max_units = st.slider("Макс. агрегатов", min_value=1, max_value=4, value=3, step=1)

    # -----------------------------
    # Основная часть: расчёты (онлайн)
    # -----------------------------
    # Собираем параметры
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

    # Тепло от радиаторов (если заданы)
    radiator_heat_w = 0.0
    if rad_present and rad_sections_total > 0:
        radiator_heat_w = radiator_total_heat(rad_sections_total, rad_type, rad_height, t_fluid_in, t_in)

    net_need_w = max(total_loss_w - radiator_heat_w, 0.0)
    net_need_kw = net_need_w / 1000.0

    # Подбор теплообменников (каскад)
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
            st.warning("Нет подходящих моделей под текущие параметры (попробуйте увеличить max агрегатов или изменить параметры помещения).")

    # -----------------------------
    # Выбор расположения и ручная настройка координат
    # -----------------------------
    st.subheader("📍 Расположение тепловентилятора(ов) — автомат или вручную")
    placement_mode = st.radio("Режим размещения", ["Автоматическое размещение", "Ручное размещение (в метрах)"], index=0)

    # Предустановленные позиции в промзонах (авто)
    default_positions = []
    default_directions = []
    if placement_mode == "Автоматическое размещение":
        # Если есть рекомендация с units — расставим равномерно вдоль одной стороны
        units_to_place = best['units'] if (net_need_kw > 0 and len(suitable)>0) else 1
        for i in range(units_to_place):
            x = room_length * (i + 1) / (units_to_place + 1)
            y = room_width * 0.08  # вдоль короткой стены, 8% от ширины
            default_positions.append((x, y))
            default_directions.append(0.0)  # направлено вдоль оси X
        st.info(f"Авто-размещение: размещено {len(default_positions)} агрегата(ов) вдоль стены (зона: {location})")
    else:
        # ручная: создаём контролы для каждого агрегата (если есть рекомендованный набор) либо для 1..max_units
        units_to_place = best['units'] if (net_need_kw > 0 and len(suitable)>0) else 1
        st.info(f"Укажите координаты для {units_to_place} агрегата(ов). Координаты в пределах: x ∈ [0, {room_length:.2f}], y ∈ [0, {room_width:.2f}]")
        for i in range(units_to_place):
            st.markdown(f"**Агрегат {i+1}**")
            x = st.number_input(f"X_{i+1} (м)", min_value=0.0, max_value=room_length, value=room_length*(i+1)/(units_to_place+1), step=0.1, key=f"x_{i}")
            y = st.number_input(f"Y_{i+1} (м)", min_value=0.0, max_value=room_width, value=room_width*0.1, step=0.1, key=f"y_{i}")
            angle_deg = st.slider(f"Угол направления тёплого потока для агрегата {i+1} (град)", min_value=-180, max_value=180, value=0, key=f"ang_{i}")
            default_positions.append((x,y))
            default_directions.append(math.radians(angle_deg))

    # Если автомат — directions = 0 (вдоль X), если ручной — user-provided angles used above
    if placement_mode == "Автоматическое размещение":
        # даём пользователю возможность выбрать тип авторазмещения
        auto_choice = st.selectbox("Тип авторазмещения:", ["Вдоль длинной стены", "По центру", "В углах"], index=0)
        if auto_choice == "Вдоль длинной стены":
            # разместим по центру вдоль длинной стены
            units_to_place = best['units'] if (net_need_kw > 0 and len(suitable)>0) else 1
            default_positions = []
            default_directions = []
            for i in range(units_to_place):
                x = room_length * (i + 1) / (units_to_place + 1)
                # выбираем стену: если length>=width — вдоль нижней стенки (y small), иначе по боковой
                if room_length >= room_width:
                    y = room_width * 0.06
                    dir_angle = 0.0
                else:
                    y = room_width * (i+1)/(units_to_place+1)
                    x = room_length * 0.06
                    dir_angle = math.pi/2
                default_positions.append((x,y))
                default_directions.append(dir_angle)
        elif auto_choice == "По центру":
            default_positions = [(room_length/2, room_width/2)]
            default_directions = [0.0]
        else:  # углы
            default_positions = [(room_length*0.08, room_width*0.08), (room_length*0.92, room_width*0.92)]
            default_directions = [0.0, math.pi]

    # Визуализация
    st.subheader("📈 Визуализация помещения и воздушных потоков")
    fig = create_room_visual(room_length, room_width, default_positions, default_directions, show_grid=True)
    st.pyplot(fig)

    # -----------------------------
    # Экспорт результатов
    # -----------------------------
    st.subheader("📥 Экспорт результатов")
    out = {
        'area_m2': area,
        'height_m': height,
        'volume_m3': room_volume,
        't_out_C': t_out,
        't_in_C': t_in,
        't_fluid_in_C': t_fluid_in,
        'total_loss_kW': round(total_loss_w/1000.0, 3),
        'radiator_heat_kW': round(radiator_heat_w/1000.0, 3),
        'need_kW': round(net_need_kw, 3),
        'recommended_configuration': (best['model'] if (net_need_kw>0 and len(suitable)>0) else '—'),
        'recommended_price': (best['price'] if (net_need_kw>0 and len(suitable)>0) else 0),
        'location_zone': location
    }
    df_out = pd.DataFrame([out])
    csv_buf = io.StringIO()
    df_out.to_csv(csv_buf, index=False)
    st.download_button("Скачать CSV результатов", csv_buf.getvalue(), file_name=f"toronado_calc_{int(area)}m2.csv", mime="text/csv")

    report = f"""ОТЧЕТ — Калькулятор тепловентилятора Торнадо
Площадь: {area:.1f} м²
Высота: {height:.2f} м
Объём: {room_volume:.1f} м³
Температуры: наружн. {t_out:.1f}°C, внутренн. {t_in:.1f}°C
Температура теплоносителя: {t_fluid_in:.1f}°C

Теплопотери (итого): {total_loss_w/1000.0:.3f} кВт
Отдача радиаторов: {radiator_heat_w/1000.0:.3f} кВт
Остаток (нужна мощность): {net_need_kw:.3f} кВт

Рекомендуемая конфигурация: {(best['model'] if (net_need_kw>0 and len(suitable)>0) else 'Не требуется / не найдена')}
"""
    st.download_button("Скачать текстовый отчёт", report, file_name=f"toronado_report_{int(area)}m2.txt", mime="text/plain")

    st.markdown("---")
    st.caption("Примечание: расчёты приближённые и служат для быстрой предварительной оценки. Для точного инженерного подбора рекомендуется привлекать специалиста и использовать паспортные данные оборудования.")

if __name__ == "__main__":
    main()
