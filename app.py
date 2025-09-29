import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io

# Настройка страницы
st.set_page_config(
    page_title="Калькулятор теплопотерь 'Торнадо'",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# База данных теплообменников "Торнадо"
@st.cache_data
def load_heat_exchangers():
    return [
        # Существующие модели из таблицы (выборочно)
        {'model': 'TF 400.200.2', 'power': 13, 'air_flow': 850, 'height': 200, 'width': 400, 'rows': 2, 'price': 45000, 'type': 'базовая'},
        {'model': 'TF 500.300.3', 'power': 34, 'air_flow': 1600, 'height': 300, 'width': 500, 'rows': 3, 'price': 78000, 'type': 'базовая'},
        {'model': 'TF 700.400.4', 'power': 80, 'air_flow': 3000, 'height': 400, 'width': 700, 'rows': 4, 'price': 145000, 'type': 'базовая'},
        
        # Новые модели "Торнадо"
        {'model': 'Торнадо 3', 'power': 20, 'air_flow': 1330, 'height': 300, 'width': 280, 'rows': 4, 'price': 65000, 'type': 'торнадо'},
        {'model': 'Торнадо 4', 'power': 33, 'air_flow': 2670, 'height': 400, 'width': 400, 'rows': 3, 'price': 85000, 'type': 'торнадо'},
        {'model': 'Торнадо 5', 'power': 55, 'air_flow': 4500, 'height': 500, 'width': 500, 'rows': 3, 'price': 120000, 'type': 'торнадо'},
        {'model': 'Торнадо 10', 'power': 240, 'air_flow': 9000, 'height': 500, 'width': 1000, 'rows': 4, 'price': 280000, 'type': 'торнадо'}
    ]

# Теплопроводность материалов (Вт/м·°C)
MATERIALS = {
    'кирпич': 0.7, 
    'газоблок': 0.18, 
    'пеноблок': 0.16,
    'керамзитоблок': 0.4, 
    'сэндвич панель': 0.05, 
    'брус': 0.15,
    'бетон': 1.7
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
    'потолок_утепленный': 0.25,
    'радиатор': -80  # минус - теплоотдача!
}

def calculate_heat_loss(room_params):
    """Расчет теплопотерь помещения"""
    total_loss = 0
    temp_diff = room_params['temp_difference']
    
    # Теплопотери через стены
    wall_loss = (room_params['wall_area'] * 
                MATERIALS[room_params['wall_material']] / 
                max(room_params['wall_thickness'], 0.01) * temp_diff)
    total_loss += wall_loss
    
    # Теплопотери через окна
    window_loss = room_params['window_area'] * U_VALUES[room_params['window_type']] * temp_diff
    total_loss += window_loss
    
    # Теплопотери через двери
    door_loss = room_params['door_area'] * U_VALUES[room_params['door_type']] * temp_diff
    total_loss += door_loss
    
    # Теплопотери через пол
    floor_type = 'пол_утепленный' if room_params['floor_insulated'] else 'пол_неутепленный'
    floor_loss = room_params['floor_area'] * U_VALUES[floor_type] * temp_diff
    total_loss += floor_loss
    
    # Теплопотери через потолок
    ceiling_type = 'потолок_утепленный' if room_params['ceiling_insulated'] else 'потолок_неутепленный'
    ceiling_loss = room_params['ceiling_area'] * U_VALUES[ceiling_type] * temp_diff
    total_loss += ceiling_loss
    
    # Инфильтрация (приток холодного воздуха)
    infiltration_loss = room_params['room_volume'] * 0.3 * 1.2 * 1005 * temp_diff / 3600
    total_loss += infiltration_loss
    
    # Тепловыделения от радиаторов (минус!)
    if room_params.get('has_radiators', False):
        radiator_heat = room_params.get('radiator_count', 0) * U_VALUES['радиатор']
        total_loss += radiator_heat
    
    return max(total_loss, 0)

def select_heat_exchanger(required_power, room_volume, preferred_type="торнадо"):
    """Подбор подходящих теплообменников"""
    suitable_models = []
    heat_exchangers = load_heat_exchangers()
    
    for unit in heat_exchangers:
        # Фильтр по типу (торнадо/базовая)
        if preferred_type != "все" and unit['type'] != preferred_type:
            continue
            
        # Запас мощности 15-30%
        power_margin = unit['power'] / required_power if required_power > 0 else 0
        
        if power_margin >= 1.15:  # Минимальный запас 15%
            # Проверяем кратность воздухообмена (оптимально 3-6 раз в час)
            air_exchange = unit['air_flow'] / room_volume if room_volume > 0 else 0
            
            if 2.5 <= air_exchange <= 7:  # Допустимый диапазон
                suitable_models.append({
                    'model': unit['model'],
                    'power': unit['power'],
                    'air_flow': unit['air_flow'],
                    'air_exchange': round(air_exchange, 1),
                    'power_reserve': round((unit['power'] - required_power) / required_power * 100, 1),
                    'dimensions': f"{unit['height']}x{unit['width']}",
                    'rows': unit['rows'],
                    'price': unit['price'],
                    'type': unit['type'],
                    'efficiency': round(power_margin * (1 / max(air_exchange - 2, 1)), 2)
                })
    
    # Сортируем по эффективности
    return sorted(suitable_models, key=lambda x: (-x['efficiency'], x['price']))

def create_visualization(heat_loss_breakdown):
    """Создание визуализации теплопотерь"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Круговая диаграмма теплопотерь
    labels = list(heat_loss_breakdown.keys())
    values = [abs(x) for x in heat_loss_breakdown.values()]
    
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc', '#c2c2f0', '#ffb3e6']
    ax1.pie(values, labels=labels, colors=colors[:len(values)], autopct='%1.1f%%', startangle=90)
    ax1.set_title('Распределение теплопотерь')
    
    # Столбчатая диаграмма
    components = list(heat_loss_breakdown.keys())
    values = list(heat_loss_breakdown.values())
    bars = ax2.bar(components, values, color=colors[:len(components)])
    ax2.set_title('Теплопотери по компонентам (Вт)')
    ax2.set_ylabel('Мощность, Вт')
    plt.xticks(rotation=45)
    
    # Добавляем значения на столбцы
    for bar, value in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{abs(value):.0f}', 
                ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

def main():
    st.title("❄️ Калькулятор теплопотерь и подбор теплообменника 'Торнадо'")
    st.markdown("""
    Этот калькулятор поможет рассчитать теплопотери вашего помещения и подобрать оптимальный теплообменник из серии "Торнадо".
    """)
    st.markdown("---")
    
    # Сайдбар для ввода параметров
    with st.sidebar:
        st.header("📐 Параметры помещения")
        
        # Основные размеры
        col1, col2 = st.columns(2)
        with col1:
            length = st.number_input("Длина (м)", min_value=1.0, max_value=50.0, value=5.0, step=0.1)
        with col2:
            width = st.number_input("Ширина (м)", min_value=1.0, max_value=50.0, value=4.0, step=0.1)
        
        height = st.number_input("Высота (м)", min_value=2.0, max_value=10.0, value=3.0, step=0.1)
        
        # Характеристики ограждающих конструкций
        st.subheader("🏠 Ограждающие конструкции")
        wall_material = st.selectbox("Материал стен", list(MATERIALS.keys()))
        wall_thickness = st.number_input("Толщина стен (м)", min_value=0.1, max_value=1.0, value=0.4, step=0.05)
        
        col1, col2 = st.columns(2)
        with col1:
            window_area = st.number_input("Площадь окон (м²)", min_value=0.0, value=2.0, step=0.5)
            window_type = st.selectbox("Тип окон", ["окно_евро", "окно_тройное", "окно_двойное", "окно_одинарное"])
        with col2:
            door_area = st.number_input("Площадь дверей (м²)", min_value=0.0, value=1.8, step=0.1)
            door_type = st.selectbox("Тип дверей", ["дверь_утепленная", "дверь_деревянная", "дверь_металлическая"])
        
        # Утепление
        st.subheader("🔧 Дополнительные параметры")
        col1, col2 = st.columns(2)
        with col1:
            floor_insulated = st.checkbox("Утепленный пол", value=True)
        with col2:
            ceiling_insulated = st.checkbox("Утепленный потолок", value=True)
        
        # Отопление
        has_radiators = st.checkbox("Есть радиаторы отопления", value=False)
        radiator_count = 0
        if has_radiators:
            radiator_count = st.number_input("Количество радиаторов", min_value=1, max_value=20, value=3)
        
        # Климатические параметры
        st.subheader("🌡️ Климатические параметры")
        temp_difference = st.slider("Разница температур (улица-помещение, °C)", 
                                  min_value=10, max_value=60, value=35)
        
        # Настройки подбора
        st.subheader("⚙️ Настройки подбора")
        preferred_type = st.radio("Предпочтительный тип:", ["торнадо", "базовая", "все"], index=0)
        
        calculate_btn = st.button("🎯 Рассчитать и подобрать", type="primary", use_container_width=True)

    # Основная область результатов
    if calculate_btn:
        # Подготовка параметров
        room_volume = length * width * height
        wall_area = 2 * (length + width) * height
        floor_area = ceiling_area = length * width
        
        room_params = {
            'length': length, 'width': width, 'height': height,
            'room_volume': room_volume, 'wall_area': wall_area,
            'floor_area': floor_area, 'ceiling_area': ceiling_area,
            'wall_material': wall_material, 'wall_thickness': wall_thickness,
            'window_area': window_area, 'window_type': window_type,
            'door_area': door_area, 'door_type': door_type,
            'floor_insulated': floor_insulated, 'ceiling_insulated': ceiling_insulated,
            'temp_difference': temp_difference, 'has_radiators': has_radiators,
            'radiator_count': radiator_count
        }
        
        # Расчет теплопотерь
        heat_loss = calculate_heat_loss(room_params)
        
        # Детализированный расчет для визуализации
        heat_loss_breakdown = {
            'Стены': (wall_area * MATERIALS[wall_material] / max(wall_thickness, 0.01) * temp_difference),
            'Окна': (window_area * U_VALUES[window_type] * temp_difference),
            'Двери': (door_area * U_VALUES[door_type] * temp_difference),
            'Пол': (floor_area * U_VALUES['пол_утепленный' if floor_insulated else 'пол_неутепленный'] * temp_difference),
            'Потолок': (ceiling_area * U_VALUES['потолок_утепленный' if ceiling_insulated else 'потолок_неутепленный'] * temp_difference),
            'Инфильтрация': (room_volume * 0.3 * 1.2 * 1005 * temp_difference / 3600)
        }
        
        if has_radiators:
            heat_loss_breakdown['Радиаторы'] = radiator_count * U_VALUES['радиатор']
        
        # Вывод результатов
        st.header("📊 Результаты расчета")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Объем помещения", f"{room_volume:.1f} м³")
        with col2:
            st.metric("Теплопотери", f"{heat_loss/1000:.2f} кВт")
        with col3:
            st.metric("Рекомендуемый запас", "15-30%")
        
        # Визуализация
        st.subheader("📈 Визуализация теплопотерь")
        fig = create_visualization(heat_loss_breakdown)
        st.pyplot(fig)
        
        # Подбор теплообменников
        st.header("🔥 Подходящие теплообменники")
        suitable_units = select_heat_exchanger(heat_loss/1000, room_volume, preferred_type)
        
        if suitable_units:
            # Создаем DataFrame для красивого отображения
            df = pd.DataFrame(suitable_units)
            df_display = df[['model', 'power', 'air_flow', 'air_exchange', 'power_reserve', 'price']].copy()
            df_display.columns = ['Модель', 'Мощность, кВт', 'Расход воздуха, м³/ч', 'Кратность возд.', 'Запас, %', 'Цена, руб.']
            
            # Форматирование
            st.dataframe(df_display, use_container_width=True, hide_index=True)
            
            # Рекомендация
            best_option = suitable_units[0]
            st.success(f"🎯 **Рекомендуемая модель: {best_option['model']}**")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info(f"**Мощность:** {best_option['power']} кВт")
            with col2:
                st.info(f"**Воздухообмен:** {best_option['air_exchange']} раз/час")
            with col3:
                st.info(f"**Цена:** {best_option['price']:,} руб.")
                
            # Экспорт результатов
            st.subheader("📥 Экспорт результатов")
            
            # Создаем CSV для скачивания
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_str = csv_buffer.getvalue()
            
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="Скачать расчет в CSV",
                    data=csv_str,
                    file_name=f'расчет_торнадо_{length}x{width}x{height}.csv',
                    mime='text/csv',
                    use_container_width=True
                )
            with col2:
                # Создаем текстовый отчет
                report = f"""
                ОТЧЕТ ПО РАСЧЕТУ ТЕПЛОПОТЕРЬ
                Помещение: {length}x{width}x{height} м
                Объем: {room_volume:.1f} м³
                Материал стен: {wall_material}
                Расчетные теплопотери: {heat_loss/1000:.2f} кВт
                Рекомендуемая модель: {best_option['model']}
                Мощность: {best_option['power']} кВт
                Запас мощности: {best_option['power_reserve']}%
                """
                
                st.download_button(
                    label="Скачать отчет в TXT",
                    data=report,
                    file_name=f'отчет_торнадо_{length}x{width}x{height}.txt',
                    mime='text/plain',
                    use_container_width=True
                )
            
        else:
            st.warning("""
            ⚠️ Не найдено подходящих моделей. 
            
            **Возможные решения:**
            - Увеличьте разницу температур
            - Измените тип оборудования на 'все'
            - Рассмотрите каскадное решение из нескольких теплообменников
            - Уточните параметры утепления помещения
            """)

    else:
        # Инструкция при первом запуске
        st.info("""
        🚀 **Как пользоваться калькулятором:**
        1. Заполните параметры помещения в левой панели
        2. Укажите характеристики стен, окон, дверей
        3. Отметьте наличие утепления и радиаторов
        4. Установите климатические параметры
        5. Нажмите кнопку 'Рассчитать и подобрать'
        
        💡 **Совет:** Для точного расчета укажите реальные параметры вашего помещения.
        """)

if __name__ == "__main__":
    main()
