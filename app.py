import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import math

st.set_page_config(
    page_title="Калькулятор тепловентилятора Торнадо",
    page_icon="🌪️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------
# Данные и константы
# -------------------------
@st.cache_data
def load_heat_exchangers():
    # Модельные данные (мощность в кВт, расход воздуха м³/ч)
    return [
        {'model': 'Торнадо 3', 'power': 20, 'air_flow': 1330, 'height': 300, 'width': 280, 'rows': 4, 'price': 65000, 'type': 'торнадо'},
        {'model': 'Торнадо 4', 'power': 33, 'air_flow': 2670, 'height': 400, 'width': 400, 'rows': 3, 'price': 85000, 'type': 'торнадо'},
        {'model': 'Торнадо 5', 'power': 55, 'air_flow': 4500, 'height': 500, 'width': 500, 'rows': 3, 'price': 120000, 'type': 'торнадо'},
        {'model': 'Торнадо 10', 'power': 240, 'air_flow': 9000, 'height': 500, 'width': 1000, 'rows': 4, 'price': 280000, 'type': 'торнадо'},
        # Базовые модели
        {'model': 'TF 400.200.2', 'power': 13, 'air_flow': 850, 'height': 200, 'width': 400, 'rows': 2, 'price': 45000, 'type': 'базовая'},
        {'model': 'TF 500.300.3', 'power': 34, 'air_flow': 1600, 'height': 300, 'width': 500, 'rows': 3, 'price': 78000, 'type': 'базовая'},
    ]

# Теплопроводности (в условных относительных единицах для приближённого расчёта)
MATERIALS = {
    'кирпич': 0.7,
    'газоблок': 0.18,
    'пеноблок': 0.16,
    'керамзитоблок': 0.4,
    'сэндвич панель': 0.05,
    'брус': 0.15
    # 'бетон' удалён по запросу
}

# Коэффициенты U (Вт/м²·°C) приблизительно
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

# Эмпирические коэффициенты теплоотдачи на секцию радиатора (Вт/°C на секцию)
# section_coeff[type][height_mm] = Вт/°C/секция
SECTION_COEFF = {
    'алюминиевые':    {350: 2.4, 500: 3.2},
    'чугунные':       {350: 1.8, 500: 2.6},
    'стальные':       {350: 2.0, 500: 2.8}
}

# -------------------------
# Функции расчёта
# -------------------------
def calculate_heat_loss(params):
    """
    Возвращает суммарные теплопотери (Вт) и разбиение по компонентам.
    Вход: params — словарь с параметрами:
      - area (м²), height (м)
      - wall_material, wall_thickness (м)
      - window_area, window_type
      - door_area, door_type
      - floor_insulated (bool), ceiling_insulated (bool)
      - t_out (°C), t_in (°C)
      - room_volume (м³) (можно посчитать)
      - radiator_heat (Вт) — суммарная отдача радиаторов (отрицательное значение уменьшающее потери)
    """
    t_out = params['t_out']
    t_in = params['t_in']
    delta = max(t_in - t_out, 0.0)  # положительная дельта
    area = params['area']
    height = params['height']

    # Оценка периметра при неизвестных формах: принимаем приближённо квадрат: side = sqrt(area)
    side = math.sqrt(max(area, 0.01))
    perimeter = 4 * side
    wall_area = perimeter * height

    # Стены
    wall_loss = wall_area * (MATERIALS.get(params['wall_material'], 0.4) / max(params['wall_thickness'], 0.01)) * delta

    # Окна и двери
    window_loss = params['window_area'] * U_VALUES[params['window_type']] * delta
    door_loss = params['door_area'] * U_VALUES[params['door_type']] * delta

    # Пол и потолок
    floor_type = 'пол_утепленный' if params['floor_insulated'] else 'пол_неутепленный'
    floor_loss = area * U_VALUES[floor_type] * delta
    ceiling_type = 'потолок_утепленный' if params['ceiling_insulated'] else 'потолок_неутепленный'
    ceiling_loss = area * U_VALUES[ceiling_type] * delta

    # Инфильтрация — приближённо: объем * кратность воздухообмена (0.3) * плотность воздуха * cp * dT / 3600
    room_volume = area * height
    infiltration_loss = room_volume * 0.3 * 1.2 * 1005 * delta / 3600

    # Суммируем
    total = wall_loss + window_loss + door_loss + floor_loss + ceiling_loss + infiltration_loss

    # Учитываем тепло от радиаторов (если есть)
    radiator_heat = params.get('radiator_heat', 0.0)  # Вт (положительное = тепло от радиатора)
    # поскольку радиаторы дают тепло, они уменьшают потребность в источнике отопления
    net = max(total - radiator_heat, 0.0)

    breakdown = {
        'Стены': wall_loss,
        'Окна': window_loss,
        'Двери': door_loss,
        'Пол': floor_loss,
        'Потолок': ceiling_loss,
        'Инфильтрация': infiltration_loss
    }
    if radiator_heat > 0:
        breakdown['Радиаторы (отдача)'] = -radiator_heat  # отрицательное значение как вклад в покрытие потерь

    return net, breakdown, room_volume, wall_area

def radiator_total_heat(sections_total, rad_type, height_mm, t_fluid_in, t_in):
    """
    Примерная выдача тепла радиатора:
      Q = sections_total * coeff * (t_fluid_in - t_in)  (Вт)
    coeff из SECTION_COEFF (Вт/°C на секцию).
    """
    if sections_total <= 0:
        return 0.0
    coeff = SECTION_COEFF.get(rad_type, {}).get(height_mm, None)
    if coeff is None:
        # запасной коэффициент
        coeff = 2.2
    delta = max(t_fluid_in - t_in, 0.0)
    return sections_total * coeff * delta

def select_heat_exchanger(required_kw, room_volume, prefer_type="торнадо"):
    """
    Подбор теплообменников.
    required_kw — требуемая мощность, кВт
    room_volume — м³
    """
    units = load_heat_exchangers()
    suitable = []
    for u in units:
        # минимальный запас 15%
        if u['power'] < required_kw * 1.15:
            continue
        # проверим кратность воздухообмена
        air_exchange = u['air_flow'] / max(room_volume, 1.0)
        if 2.5 <= air_exchange <= 7:
            power_margin = (u['power'] - required_kw) / required_kw * 100 if required_kw > 0 else 0
            efficiency = u['power'] / (1 + abs(3.5 - air_exchange))  # простая эвристика
            suitable.append({
                'model': u['model'],
                'power': u['power'],
                'air_flow': u['air_flow'],
                'air_exchange': round(air_exchange, 2),
                'power_reserve_%': round(power_margin, 1),
                'price': u['price'],
                'efficiency': round(efficiency, 2)
            })
    return sorted(suitable, key=lambda x: (-x['efficiency'], x['price']))

# -------------------------
# Визуализация размещения тепловентилятора
# -------------------------
def create_placement_visualization(area_m2, height_m, recommended_location='along_long_wall'):
    """
    Рисуем упрощённую схему: прямоугольник (помещение), позиция вентилятора и стрелки потока воздуха.
    recommended_location: 'center', 'along_long_wall', 'corner'
    """
    # для визуализации надо задать соотношение сторон. Предположим квадратную форму.
    side = math.sqrt(max(area_m2, 1.0))
    fig, ax = plt.subplots(figsize=(6, 6 * (side/ max(side,1))))
    ax.set_xlim(0, side)
    ax.set_ylim(0, side)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Эффективное расположение тепловентилятора и направление тёплого потока воздуха")

    # Рисуем контур помещения
    rect = plt.Rectangle((0,0), side, side, fill=False, linewidth=2)
    ax.add_patch(rect)

    # Вычислим точку для тепловентилятора
    if recommended_location == 'center':
        fx, fy = side/2, side/2
    elif recommended_location == 'corner':
        fx, fy = side*0.15, side*0.85
    else:  # along_long_wall — по центру одной стены (рекомендуется для равномерной дистрибуции)
        fx, fy = side*0.1, side/2

    # Рисуем вентилятор (треугольник + круг)
    fan_circle = plt.Circle((fx, fy), side*0.03, color='orange', alpha=0.9)
    ax.add_patch(fan_circle)
    # направление потока — стрелки
    # создадим несколько радиальных линий-стрелок от фанки
    n_arrows = 10
    max_len = side * 0.9
    for i in range(n_arrows):
        ang = -math.pi/6 + (i/(n_arrows-1))*(math.pi/3)  # веер ±30°
        dx = math.cos(ang) * max_len
        dy = math.sin(ang) * max_len
        ax.arrow(fx, fy, dx*0.25, dy*0.25, head_width=side*0.02, head_length=side*0.03, length_includes_head=True, alpha=0.6)
        # более дальние линии (имитация рассеивания)
        ax.arrow(fx + dx*0.25, fy + dy*0.25, dx*0.25, dy*0.25, head_width=side*0.015, head_length=side*0.02, length_includes_head=True, alpha=0.35)

    # подписи
    ax.text(fx, fy - side*0.06, "Тепловентилятор\n(рекомендованное место)", ha='center')
    ax.text(side*0.95, side*0.05, f"Площадь: {area_m2:.1f} м²\nВысота: {height_m:.2f} м", ha='right', fontsize=9)
    plt.tight_layout()
    return fig

# -------------------------
# Интерфейс Streamlit
# -------------------------
def main():
    st.title("🌪️ Калькулятор тепловентилятора Торнадо")
    st.markdown("Этот инструмент онлайн сразу рассчитывает потребность в отоплении и подбирает подходящие теплообменники 'Торнадо'. Изменения входных параметров мгновенно пересчитываются.")

    st.sidebar.header("📐 Параметры помещения")

    # Площадь вместо длины/ширины
    area = st.sidebar.number_input("Площадь помещения, м²", min_value=4.0, max_value=10000.0, value=20.0, step=1.0)
    height = st.sidebar.number_input("Высота помещения, м", min_value=2.0, max_value=12.0, value=3.0, step=0.1)

    st.sidebar.subheader("🏠 Ограждающие конструкции")
    wall_material = st.sidebar.selectbox("Материал стен", list(MATERIALS.keys()))
    wall_thickness = st.sidebar.number_input("Толщина стен (м)", min_value=0.05, max_value=1.0, value=0.3, step=0.01)

    st.sidebar.markdown("**Окна и двери**")
    window_area = st.sidebar.number_input("Площадь окон (м²)", min_value=0.0, value=2.0, step=0.1)
    window_type = st.sidebar.selectbox("Тип окон", ["окно_евро", "окно_тройное", "окно_двойное", "окно_одинарное"])
    door_area = st.sidebar.number_input("Площадь дверей (м²)", min_value=0.0, value=1.8, step=0.1)
    door_type = st.sidebar.selectbox("Тип дверей", ["дверь_утепленная", "дверь_деревянная", "дверь_металлическая"])

    st.sidebar.subheader("🔧 Утепление")
    floor_ins = st.sidebar.checkbox("Утеплённый пол", value=True)
    ceiling_ins = st.sidebar.checkbox("Утеплённый потолок", value=True)

    st.sidebar.subheader("🔥 Радиаторы (если есть)")
    has_radiators = st.sidebar.checkbox("Есть радиаторы отопления", value=False)
    radiator_heat_total = 0.0
    rad_type = None
    if has_radiators:
        rad_type = st.sidebar.selectbox("Тип радиатора", list(SECTION_COEFF.keys()))
        rad_height = st.sidebar.selectbox("Высота секции (мм)", [350, 500], index=0)
        # Ввод количества секций: либо связки, либо общее
        sections_input_mode = st.sidebar.radio("Ввод секций", ["общее количество секций", "секций в связке + число связок"], index=0)
        if sections_input_mode == "общее количество секций":
            sections_total = st.sidebar.number_input("Общее количество секций", min_value=0, value=0, step=1)
        else:
            per_bank = st.sidebar.number_input("Секций в одной связке", min_value=1, value=4, step=1)
            banks = st.sidebar.number_input("Количество связок", min_value=1, value=1, step=1)
            sections_total = per_bank * banks

        # Климатические параметры и температура теплоносителя
    st.sidebar.subheader("🌡️ Климат")
    t_out = st.sidebar.number_input("Температура снаружи, °C", value=-20.0, step=0.5)
    t_in = st.sidebar.number_input("Температура внутри (целевая), °C", value=18.0, step=0.5)
    t_fluid_in = st.sidebar.number_input("Температура теплоносителя на входе в теплообменник, °C", value=70.0, step=0.5)

    # Параметры подбора оборудования: убран 'настройка подбора'
    st.sidebar.markdown("_(параметры подбора выполнены автоматически)_")

    # -------------------------
    # Быстрый расчёт (онлайн)
    # -------------------------
    # Рассчитаем теплоотдачу радиаторов (если есть)
    if has_radiators:
        radiator_heat_total = radiator_total_heat(sections_total, rad_type, rad_height, t_fluid_in, t_in)
    else:
        sections_total = 0

    params = {
        'area': area,
        'height': height,
        'wall_material': wall_material,
        'wall_thickness': wall_thickness,
        'window_area': window_area,
        'window_type': window_type,
        'door_area': door_area,
        'door_type': door_type,
        'floor_insulated': floor_ins,
        'ceiling_insulated': ceiling_ins,
        't_out': t_out,
        't_in': t_in,
        'radiator_heat': radiator_heat_total
    }

    net_loss_w, breakdown, room_volume, wall_area = calculate_heat_loss(params)
    net_loss_kw = net_loss_w / 1000.0

    # Подбор теплообменников (используем требуемую мощность в кВт)
    suitable = select_heat_exchanger(net_loss_kw, room_volume)

    # -------------------------
    # Вывод результатов
    # -------------------------
    st.header("📊 Результаты расчёта (онлайн)")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Площадь", f"{area:.1f} м²")
        st.metric("Объём помещения", f"{room_volume:.1f} м³")
    with col2:
        st.metric("Расчётная потребность (теплопотери)", f"{net_loss_kw:.2f} кВт")
        st.metric("Температурная разница (внутр.-наруж.)", f"{(t_in - t_out):.1f} °C")
    with col3:
        st.metric("Тепло от радиаторов", f"{radiator_heat_total/1000:.2f} кВт")
        st.metric("Секции (всего)", f"{sections_total}")

    st.subheader("🔎 Детализация (Вт)")
    df_break = pd.DataFrame.from_dict(breakdown, orient='index', columns=['Вт'])
    df_break['Вт_abs'] = df_break['Вт'].abs()
    st.dataframe(df_break[['Вт']], use_container_width=True)

    # -------------------------
    # Визуализация размещения тепловентилятора
    # -------------------------
    st.subheader("📍 Визуализация эффективного расположения тепловентилятора")
    # Рекомендация: если узкое помещение (side_ratio), другое расположение — но для простоты дадим три варианта и выберем
    # Выберем along_long_wall как стандартную рекомендацию
    fig_place = create_placement_visualization(area, height, recommended_location='along_long_wall')
    st.pyplot(fig_place)

    # -------------------------
    # Подбор оборудования
    # -------------------------
    st.header("🔥 Подходящие теплообменники 'Торнадо'")
    if suitable:
        df = pd.DataFrame(suitable)
        df_display = df[['model', 'power', 'air_flow', 'air_exchange', 'power_reserve_%', 'price']].copy()
        df_display.columns = ['Модель', 'Мощность, кВт', 'Расход воздуха, м³/ч', 'Кратность возд./ч', 'Запас, %', 'Цена, руб.']
        st.dataframe(df_display, use_container_width=True, hide_index=True)

        best = suitable[0]
        st.success(f"🎯 Рекомендуемая модель: {best['model']} — {best['power']} кВт")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"Кратность воздухообмена: {best['air_exchange']} 1/ч")
        with col2:
            st.info(f"Запас мощности: {best['power_reserve_%']} %")
        with col3:
            st.info(f"Цена: {best['price']:,} руб.")
    else:
        st.warning("⚠️ Подходящие модели не найдены по текущим параметрам. Рассмотрите увеличение мощности (t_fluid_in), изменение конфигурации радиаторов или каскадное решение.")

    # -------------------------
    # Экспорт
    # -------------------------
    st.subheader("📥 Экспорт результатов")
    out_df = pd.DataFrame([{
        'area_m2': area,
        'height_m': height,
        'volume_m3': room_volume,
        't_out_C': t_out,
        't_in_C': t_in,
        't_fluid_in_C': t_fluid_in,
        'heat_need_kW': round(net_loss_kw, 3),
        'radiator_heat_kW': round(radiator_heat_total/1000.0, 3),
        'sections_total': sections_total
    }])
    csv_buffer = io.StringIO()
    out_df.to_csv(csv_buffer, index=False)
    csv_data = csv_buffer.getvalue()

    st.download_button("Скачать CSV с результатами", csv_data, file_name=f"расчет_торнадо_area_{int(area)}m2.csv", mime="text/csv", use_container_width=True)

    report_txt = f"""ОТЧЕТ — Калькулятор тепловентилятора Торнадо
Площадь: {area:.1f} м²
Высота: {height:.2f} м
Объём: {room_volume:.1f} м³
Т наружн.: {t_out:.1f} °C
Т внутр.: {t_in:.1f} °C
Т вх. теплоносителя: {t_fluid_in:.1f} °C
Расчётная потребность: {net_loss_kw:.3f} кВт
Отдача радиаторов: {radiator_heat_total/1000.0:.3f} кВт
Рекомендуемая модель: {best['model'] if suitable else 'не найдена'}
"""
    st.download_button("Скачать текстовый отчёт", report_txt, file_name=f"отчет_торнадо_area_{int(area)}m2.txt", mime="text/plain", use_container_width=True)

if __name__ == "__main__":
    main()
