code = """
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(
    page_title="Калькулятор тепловентилятора Торнадо",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Материалы и коэффициенты
MATERIALS = {'кирпич': 0.7, 'газоблок': 0.18, 'пеноблок': 0.16, 'керамзитоблок': 0.4, 'сэндвич панель': 0.05, 'брус': 0.15}
U_VALUES = {
    'окно_одинарное': 5.0, 'окно_двойное': 2.9, 'окно_тройное': 1.5, 'окно_евро': 1.3,
    'дверь_деревянная': 2.0, 'дверь_металлическая': 1.5, 'дверь_утепленная': 0.8,
    'пол_неутепленный': 0.5, 'пол_утепленный': 0.2, 'потолок_неутепленный': 0.6, 'потолок_утепленный': 0.25,
    'радиатор': -80
}

# Загрузка теплообменников
def load_heat_exchangers():
    return [
        {'model': 'Торнадо 3', 'power': 20, 'air_flow': 1330, 'height': 300, 'width': 280, 'rows': 4, 'price': 65000, 'type': 'торнадо'},
        {'model': 'Торнадо 4', 'power': 33, 'air_flow': 2670, 'height': 400, 'width': 400, 'rows': 3, 'price': 85000, 'type': 'торнадо'},
        {'model': 'Торнадо 5', 'power': 55, 'air_flow': 4500, 'height': 500, 'width': 500, 'rows': 3, 'price': 120000, 'type': 'торнадо'},
        {'model': 'Торнадо 10', 'power': 106, 'air_flow': 9000, 'height': 500, 'width': 1000, 'rows': 4, 'price': 280000, 'type': 'торнадо'}
    ]

# Расчет теплопотерь
def calculate_heat_loss(params):
    total_loss = 0
    temp_diff = params['temp_inside'] - params['temp_outside']
    total_loss += params['wall_area'] * MATERIALS[params['wall_material']] / max(params['wall_thickness'],0.01) * temp_diff
    total_loss += params['window_area'] * U_VALUES[params['window_type']] * temp_diff
    total_loss += params['door_area'] * U_VALUES[params['door_type']] * temp_diff
    floor_type = 'пол_утепленный' if params['floor_insulated'] else 'пол_неутепленный'
    total_loss += params['floor_area'] * U_VALUES[floor_type] * temp_diff
    ceiling_type = 'потолок_утепленный' if params['ceiling_insulated'] else 'потолок_неутепленный'
    total_loss += params['ceiling_area'] * U_VALUES[ceiling_type] * temp_diff
    total_loss += params['room_volume'] * 0.3 * 1.2 * 1005 * temp_diff / 3600
    if params.get('has_radiators', False):
        total_loss += params.get('radiator_count',0) * U_VALUES['радиатор']
    return max(total_loss,0)

# Подбор теплообменников с каскадом
def select_heat_exchanger(required_power, room_volume, preferred_type="торнадо"):
    exchangers = load_heat_exchangers()
    suitable = []
    for unit in exchangers:
        if preferred_type != "все" and unit['type'] != preferred_type:
            continue
        count = max(1,int(np.ceil(required_power/unit['power'])))
        total_power = unit['power']*count
        air_exchange = (unit['air_flow']*count)/room_volume if room_volume>0 else 0
        power_reserve = round((total_power-required_power)/required_power*100,1)
        suitable.append({'model':unit['model'], 'unit_power':unit['power'], 'total_power':total_power,
                         'air_flow':unit['air_flow'], 'air_exchange':round(air_exchange,1), 'count':count,
                         'power_reserve':power_reserve, 'price':unit['price']*count, 'type':unit['type']})
    return sorted(suitable,key=lambda x:(-x['total_power'], x['price']))

# Визуализация
def create_visualization_fans_gradient(room_width, room_length, fans_info):
    fig, ax = plt.subplots(figsize=(10,6))
    ax.set_xlim(0, room_length)
    ax.set_ylim(0, room_width)
    ax.set_title('Эффективное расположение тепловентиляторов')
    ax.set_xlabel('Длина помещения (м)')
    ax.set_ylabel('Ширина помещения (м)')
    for idx, fan in enumerate(fans_info):
        x0 = (idx+1)*room_length/(len(fans_info)+1)
        y0 = 0.5
        ax.arrow(x0,y0,0,room_width*0.7,head_width=0.5,head_length=0.7,fc='orange',ec='orange')
        ax.text(x0,y0-0.5,f"{fan['model']} x{fan['count']}",ha='center')
        grad_radius = room_width*0.35
        resolution = 50
        for r in np.linspace(0,grad_radius,resolution):
            alpha = 0.15*(1-r/grad_radius)
            circle = plt.Circle((x0,y0+room_width*0.35),r,color='orange',alpha=alpha,fill=True)
            ax.add_patch(circle)
    plt.tight_layout()
    return fig

# Интерфейс
st.title('❄️ Калькулятор тепловентилятора Торнадо')
with st.sidebar:
    st.header('Параметры помещения')
    area = st.number_input('Площадь помещения (м²)',5.0,500.0,50.0,1.0)
    height = st.number_input('Высота помещения (м)',2.0,10.0,3.0,0.1)
    wall_material = st.selectbox('Материал стен',list(MATERIALS.keys()))
    wall_thickness = st.number_input('Толщина стен (м)',0.1,1.0,0.4,0.05)
    window_area = st.number_input('Площадь окон (м²)',0.0,50.0,5.0,0.5)
    window_type = st.selectbox('Тип окон',['окно_евро','окно_тройное','окно_двойное','окно_одинарное'])
    door_area = st.number_input('Площадь дверей (м²)',0.0,10.0,2.0,0.1)
    door_type = st.selectbox('Тип дверей',['дверь_утепленная','дверь_деревянная','дверь_металлическая'])
    floor_insulated = st.checkbox('Утепленный пол',True)
    ceiling_insulated = st.checkbox('Утепленный потолок',True)
    has_radiators = st.checkbox('Есть радиаторы',False)
    radiator_count = st.number_input('Количество секций радиаторов',1,50,5) if has_radiators else 0
    radiator_type = st.selectbox('Тип радиаторов',['Алюминиевые','Чугунные']) if has_radiators else None
    radiator_height = st.selectbox('Высота радиаторов (мм)',[350,500]) if has_radiators else None
    temp_inside = st.number_input('Температура внутри (°C)',15.0,30.0,20.0,0.5)
    temp_outside = st.number_input('Температура снаружи (°C)',-30.0,30.0,-5.0,0.5)

# Расчеты
room_volume = area*height
wall_area = area*height*2
floor_area = ceiling_area = area
params = {'room_volume':room_volume,'wall_area':wall_area,'floor_area':floor_area,'ceiling_area':ceiling_area,
          'wall_material':wall_material,'wall_thickness':wall_thickness,'window_area':window_area,'window_type':window_type,
          'door_area':door_area,'door_type':door_type,'floor_insulated':floor_insulated,'ceiling_insulated':ceiling_insulated,
          'has_radiators':has_radiators,'radiator_count':radiator_count,'temp_inside':temp_inside,'temp_outside':temp_outside}

heat_loss = calculate_heat_loss(params)
exchangers = select_heat_exchanger(heat_loss/1000, room_volume,'торнадо')

st.subheader('Результаты')
st.write(f'Теплопотери: {heat_loss/1000:.2f} кВт')

if exchangers:
    fans_info = [exchangers[0]]
    st.write('Рекомендуемые тепловентиляторы:')
    for fan in fans_info:
        st.write(f"{fan['model']} x{fan['count']}, общая мощность: {fan['total_power']:.2f} кВт, запас мощности: {fan['power_reserve']}%")
    fig = create_visualization_fans_gradient(height, area, fans_info)
    st.pyplot(fig)
else:
    st.warning('Нет подходящих моделей теплообменников')
"""

with open("app.py", "w", encoding="utf-8") as f:
    f.write(code)
print("Файл app.py создан!")
