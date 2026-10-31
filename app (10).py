# -*- coding: utf-8 -*-
"""
AI-ассистент для озеленения города — MVP (хакатон 28.10–01.11.2026)
Город по умолчанию: Павлодар.

Запуск:   pip install -r requirements.txt
          streamlit run app.py

Структура файла:
  1. Геометрия (UTM 43N / EPSG:32643 без внешних библиотек)
  2. Справочники: каталог растений, расценки, коэффициенты (с полем «источник»)
  3. Данные: синтетический демо-набор + загрузка OSM (Overpass) + импорт файлов
  4. Аналитика: подбор растений, смета, приживаемость (правила+ML), уход,
     эффект, проблемные зоны, приоритет, дорожная карта под бюджет
  5. Ассистент (разбор запросов + вызов функций модулей + поиск по каталогу)
  6. Отчёты: Excel, PDF, HTML
  7. Веб-интерфейс Streamlit (функция main)

Все расчёты — ОЦЕНОЧНЫЕ И РЕКОМЕНДАТЕЛЬНЫЕ. Окончательное решение принимает
специалист акимата. Синтетические данные помечены в интерфейсе.
"""
import io
import os
import re
import json
import math
import glob
import zipfile
import datetime
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data")
CENTER = (52.287, 76.967)                      # Павлодар (lat, lon)
BBOX = (52.240, 76.900, 52.335, 77.060)        # S, W, N, E
SYN = "🧪 синтетика"
DISCLAIMER = ("⚠️ Все расчёты (пригодность, приживаемость, стоимость, эффект) — оценочные и "
              "рекомендательные. Окончательное решение принимает специалист акимата.")

# =============================================================================
# 1. ГЕОМЕТРИЯ: UTM (формулы Крюгера, WGS84), площади, пересечения, расстояния
# =============================================================================
_A = 6378137.0
_F = 1 / 298.257223563
_K0 = 0.9996
_N = _F / (2 - _F)
_AA = _A / (1 + _N) * (1 + _N ** 2 / 4 + _N ** 4 / 64)
_AL = [_N / 2 - 2 * _N ** 2 / 3 + 5 * _N ** 3 / 16, 13 * _N ** 2 / 48 - 3 * _N ** 3 / 5, 61 * _N ** 3 / 240]
_BE = [_N / 2 - 2 * _N ** 2 / 3 + 37 * _N ** 3 / 96, _N ** 2 / 48 + _N ** 3 / 15, 17 * _N ** 3 / 480]
_DE = [2 * _N - 2 * _N ** 2 / 3 - 2 * _N ** 3, 7 * _N ** 2 / 3 - 8 * _N ** 3 / 5, 56 * _N ** 3 / 15]
UTM_ZONE = 43  # EPSG:32643


def to_utm(lon, lat, zone=UTM_ZONE):
    """WGS84 (EPSG:4326) -> UTM северного полушария (по умолчанию 43N, EPSG:32643)."""
    l0 = math.radians(-183 + 6 * zone)
    phi = math.radians(lat)
    lam = math.radians(lon) - l0
    c = 2 * math.sqrt(_N) / (1 + _N)
    t = math.sinh(math.atanh(math.sin(phi)) - c * math.atanh(c * math.sin(phi)))
    xi = math.atan2(t, math.cos(lam))
    eta = math.atanh(math.sin(lam) / math.sqrt(1 + t * t))
    e = 500000 + _K0 * _AA * (eta + sum(a * math.cos(2 * j * xi) * math.sinh(2 * j * eta) for j, a in enumerate(_AL, 1)))
    n = _K0 * _AA * (xi + sum(a * math.sin(2 * j * xi) * math.cosh(2 * j * eta) for j, a in enumerate(_AL, 1)))
    return e, n


def from_utm(e, n, zone=UTM_ZONE):
    """UTM северного полушария -> WGS84 (lon, lat)."""
    l0 = math.radians(-183 + 6 * zone)
    xi = n / (_K0 * _AA)
    eta = (e - 500000) / (_K0 * _AA)
    xi1 = xi - sum(b * math.sin(2 * j * xi) * math.cosh(2 * j * eta) for j, b in enumerate(_BE, 1))
    eta1 = eta - sum(b * math.cos(2 * j * xi) * math.sinh(2 * j * eta) for j, b in enumerate(_BE, 1))
    chi = math.asin(math.sin(xi1) / math.cosh(eta1))
    phi = chi + sum(d * math.sin(2 * j * chi) for j, d in enumerate(_DE, 1))
    lam = l0 + math.atan2(math.sinh(eta1), math.cos(xi1))
    return math.degrees(lam), math.degrees(phi)


def ring_xy(ring):
    return [to_utm(lo, la) for lo, la in ring]


def ring_area_xy(xy):
    s = 0.0
    for i in range(len(xy)):
        x1, y1 = xy[i]
        x2, y2 = xy[(i + 1) % len(xy)]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2


def ring_area_m2(ring):
    return ring_area_xy(ring_xy(ring))


def ring_centroid(ring):
    xy = ring_xy(ring)
    a = cx = cy = 0.0
    for i in range(len(xy)):
        x1, y1 = xy[i]
        x2, y2 = xy[(i + 1) % len(xy)]
        cr = x1 * y2 - x2 * y1
        a += cr
        cx += (x1 + x2) * cr
        cy += (y1 + y2) * cr
    if abs(a) < 1e-9:
        x, y = np.mean([p[0] for p in xy]), np.mean([p[1] for p in xy])
    else:
        x, y = cx / (3 * a), cy / (3 * a)
    return from_utm(x, y)  # (lon, lat)


def pip(x, y, poly):
    """Точка в многоугольнике (луч)."""
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-15) + xi:
            inside = not inside
        j = i
    return inside


def _orient(a, b, c):
    v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    return 0 if abs(v) < 1e-12 else (1 if v > 0 else -1)


def _onseg(a, b, c):
    return min(a[0], b[0]) - 1e-12 <= c[0] <= max(a[0], b[0]) + 1e-12 and min(a[1], b[1]) - 1e-12 <= c[1] <= max(a[1], b[1]) + 1e-12


def seg_intersect(a, b, c, d):
    o1, o2, o3, o4 = _orient(a, b, c), _orient(a, b, d), _orient(c, d, a), _orient(c, d, b)
    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and _onseg(a, b, c): return True
    if o2 == 0 and _onseg(a, b, d): return True
    if o3 == 0 and _onseg(c, d, a): return True
    if o4 == 0 and _onseg(c, d, b): return True
    return False


def pt_seg_dist(p, a, b):
    ax, ay = a; bx, by = b; px, py = p
    dx, dy = bx - ax, by - ay
    L = dx * dx + dy * dy
    t = 0 if L == 0 else max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / L))
    return math.hypot(px - ax - t * dx, py - ay - t * dy)


def seg_seg_dist(a, b, c, d):
    if seg_intersect(a, b, c, d):
        return 0.0
    return min(pt_seg_dist(a, c, d), pt_seg_dist(b, c, d), pt_seg_dist(c, a, b), pt_seg_dist(d, a, b))


def poly_line_dist(poly_xy, line_xy):
    if any(pip(x, y, poly_xy) for x, y in line_xy):
        return 0.0
    best = 1e18
    m = len(poly_xy)
    for i in range(m):
        a, b = poly_xy[i], poly_xy[(i + 1) % m]
        for k in range(len(line_xy) - 1):
            d = seg_seg_dist(a, b, line_xy[k], line_xy[k + 1])
            if d < best:
                best = d
                if best == 0:
                    return 0.0
    return best


def poly_point_dist(poly_xy, p):
    if pip(p[0], p[1], poly_xy):
        return 0.0
    m = len(poly_xy)
    return min(pt_seg_dist(p, poly_xy[i], poly_xy[(i + 1) % m]) for i in range(m))


def self_intersections(ring):
    """Возвращает число пар несмежных пересекающихся рёбер (самопересечения)."""
    xy = ring_xy(ring)
    n = len(xy)
    cnt = 0
    for i in range(n):
        a, b = xy[i], xy[(i + 1) % n]
        for j in range(i + 1, n):
            if j == i or (j + 1) % n == i or (i + 1) % n == j:
                continue
            if seg_intersect(a, b, xy[j], xy[(j + 1) % n]):
                cnt += 1
    return cnt


def bbox_of(xy):
    xs = [p[0] for p in xy]; ys = [p[1] for p in xy]
    return min(xs), min(ys), max(xs), max(ys)


def bbox_gap(b1, b2):
    dx = max(0, max(b1[0], b2[0]) - min(b1[2], b2[2]))
    dy = max(0, max(b1[1], b2[1]) - min(b1[3], b2[3]))
    return math.hypot(dx, dy)


def polys_overlap(r1, r2):
    a, b = ring_xy(r1), ring_xy(r2)
    if bbox_gap(bbox_of(a), bbox_of(b)) > 0:
        return False
    for i in range(len(a)):
        for j in range(len(b)):
            if seg_intersect(a[i], a[(i + 1) % len(a)], b[j], b[(j + 1) % len(b)]):
                return True
    return pip(a[0][0], a[0][1], b) or pip(b[0][0], b[0][1], a)


def line_len_m(line):
    xy = ring_xy(line)
    return sum(math.hypot(xy[i + 1][0] - xy[i][0], xy[i + 1][1] - xy[i][1]) for i in range(len(xy) - 1))


def sample_points_in(ring, n, seed=0):
    xy = ring_xy(ring)
    x0, y0, x1, y1 = bbox_of(xy)
    rng = np.random.default_rng(seed)
    pts = []
    tries = 0
    while len(pts) < n and tries < n * 40:
        tries += 1
        x, y = rng.uniform(x0, x1), rng.uniform(y0, y1)
        if pip(x, y, xy):
            pts.append(from_utm(x, y))
    return pts


def ring_to_geojson(ring):
    c = [[lo, la] for lo, la in ring]
    return {"type": "Polygon", "coordinates": [c + [c[0]]]}


def in_city(lon, lat, margin=0.05):
    S, W, N, E = BBOX
    return S - margin <= lat <= N + margin and W - margin <= lon <= E + margin


# =============================================================================
# 2. СПРАВОЧНИКИ (редактируются администратором, у каждой записи есть «источник»)
# =============================================================================
SRC_PLANT = "ориентировочно: обобщение открытых дендрологических справочников; требует проверки экспертом"
SRC_PRICE = "синтетика — заменить прайсом питомников / расценками акимата"

PLANT_COLS = ["name", "latin", "type", "usda", "drought", "salt_tol", "ph_min", "ph_max", "shade_tol",
              "pollution_tol", "wind_tol", "allergen", "toxic", "invasive", "height_m", "crown_m", "price",
              "base_surv", "co2_kg", "dust_kg", "mature_years"]
PLANT_RU = {"name": "Вид", "latin": "Латинское название", "type": "Ярус", "usda": "Зона зимостойкости USDA",
            "drought": "Засухоуст. (1–5)", "salt_tol": "Солеуст., дСм/м", "ph_min": "pH мин", "ph_max": "pH макс",
            "shade_tol": "Теневынос. (1–5)", "pollution_tol": "Газо-/пылеуст. (1–5)", "wind_tol": "Ветроуст. (1–5)",
            "allergen": "Сильный аллерген", "toxic": "Ядовит", "invasive": "Инвазивный", "height_m": "Высота, м",
            "crown_m": "Диаметр кроны, м", "price": "Цена, ₸ (шт или м²)", "base_surv": "Базовая приживаемость",
            "co2_kg": "CO₂, кг/год (взрослое; газон — на м²)", "dust_kg": "Пыль, кг/год", "mature_years": "Лет до зрелости",
            "source": "Источник"}

_PLANTS = [
    # name, latin, type, usda, drought, salt, phmin, phmax, shade, poll, wind, allergen, toxic, invasive, h, crown, price, base, co2, dust, mature
    ("Берёза повислая", "Betula pendula", "дерево", 2, 3, 3, 4.5, 7.5, 2, 3, 3, True, False, False, 20, 8, 18000, 0.85, 25, 0.25, 25),
    ("Тополь бальзамический (мужские экз.)", "Populus balsamifera", "дерево", 2, 2, 4, 5.5, 8.5, 1, 5, 3, False, False, False, 25, 10, 12000, 0.90, 40, 0.45, 15),
    ("Вяз мелколистный", "Ulmus pumila", "дерево", 3, 5, 8, 6.0, 8.8, 2, 5, 5, False, False, False, 15, 9, 14000, 0.90, 28, 0.35, 20),
    ("Клён ясенелистный", "Acer negundo", "дерево", 2, 4, 6, 6.0, 8.5, 3, 4, 3, True, False, True, 15, 8, 9000, 0.90, 22, 0.25, 15),
    ("Липа мелколистная", "Tilia cordata", "дерево", 3, 2, 3, 5.5, 7.8, 4, 3, 3, False, False, False, 20, 9, 25000, 0.78, 26, 0.40, 30),
    ("Ясень пенсильванский", "Fraxinus pennsylvanica", "дерево", 3, 4, 6, 6.0, 8.5, 2, 4, 4, False, False, False, 18, 9, 16000, 0.87, 27, 0.30, 25),
    ("Сосна обыкновенная", "Pinus sylvestris", "дерево", 2, 4, 2, 4.5, 7.0, 1, 2, 4, False, False, False, 25, 6, 30000, 0.75, 20, 0.50, 40),
    ("Ель колючая", "Picea pungens", "дерево", 3, 3, 3, 5.5, 7.5, 2, 4, 4, False, False, False, 20, 6, 35000, 0.78, 18, 0.55, 40),
    ("Лиственница сибирская", "Larix sibirica", "дерево", 1, 4, 3, 5.0, 8.0, 1, 3, 5, False, False, False, 25, 8, 28000, 0.80, 24, 0.35, 35),
    ("Рябина обыкновенная", "Sorbus aucuparia", "дерево", 2, 2, 2, 5.0, 7.5, 3, 3, 3, False, False, False, 8, 5, 12000, 0.80, 10, 0.15, 15),
    ("Черёмуха виргинская", "Padus virginiana", "дерево", 3, 3, 3, 5.5, 8.0, 3, 3, 3, False, True, False, 8, 5, 11000, 0.85, 9, 0.15, 12),
    ("Яблоня Недзвецкого", "Malus niedzwetzkyana", "дерево", 3, 3, 3, 6.0, 8.0, 2, 3, 3, False, False, False, 7, 5, 15000, 0.82, 8, 0.12, 15),
    ("Ива белая", "Salix alba", "дерево", 2, 1, 3, 5.5, 8.0, 2, 4, 3, True, False, False, 20, 10, 9000, 0.88, 30, 0.30, 15),
    ("Лох узколистный", "Elaeagnus angustifolia", "дерево", 3, 5, 10, 6.5, 9.0, 1, 5, 5, True, False, False, 7, 5, 7000, 0.88, 6, 0.12, 10),
    ("Карагана древовидная", "Caragana arborescens", "кустарник", 2, 5, 7, 6.0, 9.0, 2, 5, 5, False, False, False, 4, 3, 3500, 0.92, 3, 0.07, 6),
    ("Сирень обыкновенная", "Syringa vulgaris", "кустарник", 3, 4, 4, 6.0, 8.5, 2, 4, 4, False, False, False, 4, 3, 6000, 0.88, 3, 0.08, 8),
    ("Кизильник блестящий", "Cotoneaster lucidus", "кустарник", 3, 4, 4, 5.5, 8.5, 4, 5, 4, False, False, False, 2.5, 2, 2500, 0.90, 1.5, 0.06, 6),
    ("Спирея средняя", "Spiraea media", "кустарник", 2, 4, 4, 5.5, 8.5, 3, 4, 4, False, False, False, 2, 1.5, 3000, 0.90, 1.2, 0.04, 5),
    ("Шиповник морщинистый", "Rosa rugosa", "кустарник", 2, 4, 7, 5.5, 8.5, 2, 5, 5, False, False, False, 2, 2, 3500, 0.88, 1.5, 0.05, 5),
    ("Боярышник кроваво-красный", "Crataegus sanguinea", "кустарник", 2, 4, 4, 5.5, 8.5, 2, 4, 4, False, False, False, 5, 3, 5000, 0.87, 3, 0.08, 8),
    ("Облепиха крушиновидная", "Hippophae rhamnoides", "кустарник", 2, 5, 8, 6.0, 9.0, 1, 4, 5, False, False, False, 4, 3, 4500, 0.85, 2.5, 0.06, 6),
    ("Жимолость татарская", "Lonicera tatarica", "кустарник", 2, 4, 4, 6.0, 8.5, 3, 5, 4, False, True, False, 3, 3, 3000, 0.90, 2, 0.06, 6),
    ("Тамарикс многоветвистый", "Tamarix ramosissima", "кустарник", 4, 5, 12, 6.5, 9.0, 1, 4, 5, False, False, False, 4, 3, 6000, 0.80, 2, 0.05, 6),
    ("Газон засухоустойчивый (овсяница + житняк)", "Festuca / Agropyron", "газон", 2, 4, 5, 6.0, 8.8, 2, 4, 4, False, False, False, 0.1, 0, 350, 0.85, 0.15, 0.005, 1),
    ("Газон партерный (мятлик)", "Poa pratensis", "газон", 3, 2, 3, 6.0, 7.5, 2, 3, 3, False, False, False, 0.1, 0, 450, 0.80, 0.15, 0.004, 1),
]


def default_plants_df():
    df = pd.DataFrame(_PLANTS, columns=PLANT_COLS)
    df["source"] = SRC_PLANT + "; цена — " + SRC_PRICE
    return df


_RATES = [
    ("prep_soil_m2", "Подготовка территории и плодородный грунт (10 см)", 1200, "₸/м²"),
    ("fertilizer_m2", "Удобрения при посадке", 150, "₸/м²"),
    ("gypsum_m2", "Мелиорация засолённой почвы (гипсование/промывка)", 400, "₸/м²"),
    ("drainage_m2", "Дренаж (на 30% площади при переувлажнении)", 2500, "₸/м²"),
    ("plant_tree", "Посадка дерева (яма, замена грунта, опоры)", 9000, "₸/шт"),
    ("plant_shrub", "Посадка кустарника", 2500, "₸/шт"),
    ("lawn_work_m2", "Устройство газона (работы)", 600, "₸/м²"),
    ("delivery_pct", "Доставка посадочного материала", 8, "% от материала"),
    ("care_tree_year", "Уход за деревом (полив, обрезка, подкормка)", 15000, "₸/шт·год"),
    ("care_shrub_year", "Уход за кустарником", 4000, "₸/шт·год"),
    ("care_lawn_m2_year", "Уход за газоном", 500, "₸/м²·год"),
]


def default_rates_df():
    df = pd.DataFrame(_RATES, columns=["key", "name", "value", "unit"])
    df["source"] = SRC_PRICE
    return df


_COEFFS = [
    ("o2_per_co2", "Кислород на 1 т поглощённого CO₂", 0.727, "т/т", "стехиометрия фотосинтеза (32/44)"),
    ("cooling_per_pp", "Снижение t° воздуха на 1 п.п. покрытия кронами", 0.02, "°C", "оценка по обзорным исследованиям городского климата; разброс широк"),
    ("cooling_max", "Максимальное локальное снижение t°", 2.5, "°C", "ограничение модели (допущение)"),
    ("precip_mm", "Среднегодовые осадки", 300, "мм", "порядок величины для Павлодара; уточнить по РГП «Казгидромет»"),
    ("interception", "Доля осадков, задерживаемая кроной", 0.15, "доля", "оценка (допущение)"),
    ("lawn_co2_m2", "Поглощение CO₂ газоном", 0.15, "кг/м²·год", "оценка (допущение)"),
    ("growth_exp", "Показатель кривой роста кроны", 1.3, "—", "допущение модели"),
    ("annual_loss", "Ежегодный отпад после 3-го года", 0.02, "доля/год", "допущение"),
    ("max_share", "Лимит доли одного вида в ярусе", 0.35, "доля", "правило разнообразия (ориентир «10-20-30»)"),
    ("green_per_capita_norm", "Ориентир зелени на жителя в пределах участка", 6, "м²/чел", "ориентир; уточнить по СП РК"),
    ("bldg_min_dist_big_tree", "Мин. расстояние от крупного дерева до здания", 5, "м", "ориентир; уточнить по СП РК"),
]


def default_coeffs_df():
    return pd.DataFrame(_COEFFS, columns=["key", "name", "value", "unit", "source"])


# =============================================================================
# 3. ДАННЫЕ
# =============================================================================
SOIL_TYPES = {
    "Каштановая": dict(ph=(7.0, 8.0), sal=(0.5, 2.5), fert=(3, 4), moist=(2, 3), color="#c49a6c"),
    "Светло-каштановая": dict(ph=(7.3, 8.3), sal=(1.0, 3.5), fert=(2, 3), moist=(1, 3), color="#e0c39a"),
    "Солонец": dict(ph=(8.2, 9.3), sal=(4.0, 9.0), fert=(1, 2), moist=(1, 3), color="#d9d9d9"),
    "Аллювиальная луговая": dict(ph=(6.8, 7.8), sal=(0.5, 2.0), fert=(3, 5), moist=(3, 5), color="#8fbf7a"),
    "Техногенный грунт": dict(ph=(7.5, 8.8), sal=(1.5, 5.0), fert=(1, 2), moist=(1, 3), color="#9e8f8f"),
}
PURPOSES = ["сквер", "двор", "улица", "СЗЗ", "школа", "больница", "парк"]
CHILD = {"школа", "больница"}
# деревьев/га, кустарников/га, доля газона (на свободной площади)
DENSITY = {"сквер": (150, 500, 0.5), "двор": (100, 400, 0.4), "улица": (200, 600, 0.3), "СЗЗ": (400, 800, 0.2),
           "школа": (120, 400, 0.5), "больница": (150, 400, 0.5), "парк": (200, 500, 0.4)}
PURPOSE_COLOR = {"сквер": "#2e7d32", "двор": "#6a1b9a", "улица": "#455a64", "СЗЗ": "#8d6e63", "школа": "#1565c0",
                 "больница": "#c62828", "парк": "#1b5e20"}
ROAD_CLASSES = {  # класс OSM: (рус, вес загрязнения, ширина линии, цвет, порядок)
    "motorway": ("магистраль", 1.0, 7, "#e8743b", 9), "trunk": ("магистраль", 1.0, 7, "#e8743b", 9),
    "primary": ("главная улица", 0.9, 6, "#f3a64a", 8), "secondary": ("улица районного значения", 0.75, 5, "#f7d06b", 7),
    "tertiary": ("улица районного значения", 0.6, 4, "#fff2a8", 6), "unclassified": ("местный проезд", 0.35, 3, "#ffffff", 5),
    "residential": ("местный проезд", 0.3, 3, "#ffffff", 5), "living_street": ("жилая зона", 0.2, 2.5, "#ffffff", 4),
    "service": ("внутриквартальный проезд", 0.15, 2, "#f2f2f2", 3), "pedestrian": ("пешеходная", 0.05, 2, "#f6c6c6", 2),
    "footway": ("пешеходная дорожка", 0.02, 1.5, "#e88", 1),
}
MAJOR_ROADS = {"motorway", "trunk", "primary", "secondary", "tertiary"}
SOCIAL_RU = {"school": "школа", "kindergarten": "детсад", "hospital": "больница", "clinic": "поликлиника",
             "residential": "жилой комплекс / жилой массив"}


def district_of(lon, lat):
    if lat > 52.305 and lon > 77.0:
        return "Промышленный (СВ)"
    if lat > 52.302:
        return "Северный"
    if lat < 52.262:
        return "Южный"
    if lon > 77.0:
        return "Восточный"
    return "Центральный"


def _poly_around(cx, cy, R, n, rng):
    ang = np.sort(rng.uniform(0, 2 * math.pi, n))
    rad = R * rng.uniform(0.75, 1.15, n)
    return [from_utm(cx + r * math.cos(a), cy + r * math.sin(a)) for a, r in zip(ang, rad)]


def _street_strip(cx, cy, L, W, ang):
    dx, dy = math.cos(ang), math.sin(ang)
    nx, ny = -dy, dx
    pts = [(cx - dx * L / 2 + nx * W / 2, cy - dy * L / 2 + ny * W / 2), (cx + dx * L / 2 + nx * W / 2, cy + dy * L / 2 + ny * W / 2),
           (cx + dx * L / 2 - nx * W / 2, cy + dy * L / 2 - ny * W / 2), (cx - dx * L / 2 - nx * W / 2, cy - dy * L / 2 - ny * W / 2)]
    return [from_utm(x, y) for x, y in pts]


def generate_base_data(seed=42, n_zones=26):
    """Синтетический демо-набор (замените данными организатора). Всё помечено как синтетика."""
    rng = np.random.default_rng(seed)
    S, W, N, E = BBOX
    # --- слой почв: сетка ячеек
    soils = []
    nx_, ny_ = 6, 5
    lats = np.linspace(S, N, ny_ + 1)
    lons = np.linspace(W, E, nx_ + 1)
    types = list(SOIL_TYPES)
    for i in range(ny_):
        for j in range(nx_):
            t = rng.choice(types, p=[0.35, 0.25, 0.12, 0.13, 0.15])
            if lons[j + 1] <= 76.935:
                t = "Аллювиальная луговая"  # пойма Иртыша
            r = SOIL_TYPES[t]
            ring = [(lons[j], lats[i]), (lons[j + 1], lats[i]), (lons[j + 1], lats[i + 1]), (lons[j], lats[i + 1])]
            soils.append(dict(ring=ring, soil_type=str(t), ph=round(float(rng.uniform(*r["ph"])), 1),
                              salinity=round(float(rng.uniform(*r["sal"])), 1),
                              fertility=int(rng.integers(r["fert"][0], r["fert"][1] + 1)),
                              moisture=int(rng.integers(r["moist"][0], r["moist"][1] + 1)), src=SYN))
    # --- зоны
    pw = np.array([0.2, 0.25, 0.15, 0.1, 0.12, 0.06, 0.12])
    zones = []
    tries = 0
    while len(zones) < n_zones and tries < 2000:
        tries += 1
        lat = rng.uniform(52.250, 52.328)
        lon = rng.uniform(76.948, 77.045)
        purpose = str(rng.choice(PURPOSES, p=pw / pw.sum()))
        cx, cy = to_utm(lon, lat)
        if purpose == "улица":
            ring = _street_strip(cx, cy, rng.uniform(220, 420), rng.uniform(18, 30), rng.uniform(0, math.pi))
        else:
            R = {"парк": rng.uniform(110, 170), "СЗЗ": rng.uniform(80, 140)}.get(purpose, rng.uniform(40, 95))
            ring = _poly_around(cx, cy, R, int(rng.integers(6, 10)), rng)
        if any(polys_overlap(ring, z["ring"]) for z in zones) or self_intersections(ring):
            continue
        dist = district_of(lon, lat)
        industry = float(rng.uniform(0.55, 1.0)) if (dist.startswith("Пром") or purpose == "СЗЗ") else float(rng.uniform(0, 0.35))
        gs = float(rng.uniform(0.25, 0.5)) if purpose == "парк" else float(rng.uniform(0.02, 0.4))
        pop = float(rng.uniform(5, 40)) if (purpose == "СЗЗ" or dist.startswith("Пром")) else float(rng.uniform(40, 260))
        zid = f"Z-{len(zones) + 1:02d}"
        zones.append(dict(
            id=zid, name=f"{purpose.capitalize()} {zid}", ring=ring, purpose=purpose, district=dist,
            green_share=round(gs, 3), pop_density=round(pop, 1),
            lst=round(30 + 10 * (1 - min(gs, 0.5) / 0.5) + 3 * industry + float(rng.normal(0, 1)), 1),
            shade=int(rng.integers(1, 5)), wind=int(rng.integers(2, 6)), degradation=round(float(rng.uniform(0, 0.6)), 2),
            industry=round(industry, 2), bldg_dist=round(float(rng.uniform(3, 8) if purpose == "улица" else rng.uniform(2, 25)), 1),
            dist_road_syn=round(float(rng.uniform(3, 10) if purpose == "улица" else rng.uniform(10, 350)), 0),
            road_class_syn=str(rng.choice(["primary", "secondary", "tertiary", "residential"])),
            src=SYN))
    # --- существующие насаждения
    trees = [p[0] for p in _PLANTS if p[2] == "дерево"]
    greens = []
    for z in zones:
        cx, cy = to_utm(*ring_centroid(z["ring"]))
        for _ in range(int(rng.integers(4, 14))):
            x, y = cx + rng.uniform(-220, 220), cy + rng.uniform(-220, 220)
            lo, la = from_utm(x, y)
            cond = rng.choice(["хорошее", "удовлетворительное", "плохое"], p=[0.6 - z["degradation"] / 2, 0.4, z["degradation"] / 2])
            greens.append(dict(lon=lo, lat=la, species=str(rng.choice(trees)), type="дерево", condition=str(cond),
                               age=int(rng.integers(3, 60)), r=float(rng.uniform(3, 7)), src=SYN))
    # --- социальные объекты
    social = []
    for _ in range(45):
        t = str(rng.choice(list(SOCIAL_RU), p=[0.3, 0.25, 0.08, 0.12, 0.25]))
        social.append(dict(lon=float(rng.uniform(76.945, 77.05)), lat=float(rng.uniform(52.248, 52.33)), kind=t,
                           name=f"{SOCIAL_RU[t].capitalize()} (синт.)", src=SYN))
    # --- синтетические дороги (используются, только если OSM не загружен)
    roads = []
    k = 0
    for i, base_lat in enumerate(np.linspace(52.252, 52.325, 6)):
        cls = ["primary", "secondary", "tertiary"][i % 3]
        pts = [(lo, base_lat + 0.0025 * math.sin((lo - 76.94) * 60 + i)) for lo in np.linspace(76.94, 77.055, 40)]
        k += 1
        roads.append(dict(rid=f"R-{k}", name=f"Синт. улица {k}", highway=cls, line=pts, oneway="no", lanes="", bridge=False, tunnel=False, src=SYN))
    for i, base_lon in enumerate(np.linspace(76.955, 77.045, 6)):
        cls = ["secondary", "tertiary", "residential"][i % 3]
        pts = [(base_lon + 0.004 * math.sin((la - 52.24) * 50 + i), la) for la in np.linspace(52.245, 52.332, 40)]
        k += 1
        roads.append(dict(rid=f"R-{k}", name=f"Синт. проспект {k}", highway=cls, line=pts, oneway="no", lanes="", bridge=False, tunnel=False, src=SYN))
    return dict(zones=zones, soils=soils, greens=greens, social=social, roads=roads)


# ---------------- OSM / Overpass ----------------
OVERPASS = "https://overpass-api.de/api/interpreter"
OVERPASS_URLS = [OVERPASS, "https://overpass.kumi.systems/api/interpreter", "https://overpass.private.coffee/api/interpreter"]


def fetch_osm_roads(timeout=90):
    """Загружает дорожный граф Павлодара из OSM (без внутриквартальных проездов и тротуаров — для аналитики
    их не нужно) и кэширует в data/osm_roads.json. Пробует несколько серверов Overpass."""
    import requests
    S, W, N, E = BBOX
    q = (f'[out:json][timeout:{timeout}];way["highway"~"^(motorway|motorway_link|trunk|trunk_link|primary|primary_link|'
         f'secondary|secondary_link|tertiary|tertiary_link|unclassified|residential|living_street|pedestrian)$"]({S},{W},{N},{E});out geom tags;')
    last = None
    for url in OVERPASS_URLS:
        try:
            r = requests.post(url, data={"data": q}, timeout=(10, timeout), headers={"User-Agent": "greening-mvp/1.0"})
            r.raise_for_status()
            js = r.json()
            break
        except Exception as e:  # noqa
            last = e
    else:
        raise last
    roads = []
    for el in js.get("elements", []):
        g = el.get("geometry")
        if not g or len(g) < 2:
            continue
        t = el.get("tags", {})
        hw = str(t.get("highway", "")).replace("_link", "")
        roads.append(dict(rid=f"R-{el['id']}", name=t.get("name", "без названия"), highway=hw,
                          line=[(p["lon"], p["lat"]) for p in g], oneway=t.get("oneway", "no"), lanes=t.get("lanes", ""),
                          surface=t.get("surface", ""), maxspeed=t.get("maxspeed", ""),
                          bridge=t.get("bridge") == "yes", tunnel=t.get("tunnel") == "yes",
                          src="© OpenStreetMap contributors (ODbL)"))
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, "osm_roads.json"), "w", encoding="utf-8") as f:
        json.dump(roads, f, ensure_ascii=False)
    return roads


SOCIAL_QUERY = ('(nwr["amenity"~"^(school|kindergarten|hospital|clinic|doctors)$"]({bb});'
                'nwr["healthcare"~"^(hospital|clinic|centre)$"]({bb});'
                'nwr["landuse"="residential"]["name"]({bb});'
                'nwr["building"~"^(apartments|residential)$"]["name"~"ЖК|[Жж]илой комплекс|Residence|Резиденс"]({bb}););out center tags;')


def social_kind(t):
    """Тип социального объекта по тегам OSM."""
    am, hc = t.get("amenity", ""), t.get("healthcare", "")
    if am == "school":
        return "school"
    if am == "kindergarten":
        return "kindergarten"
    if am == "hospital" or hc == "hospital":
        return "hospital"
    if am in ("clinic", "doctors") or hc in ("clinic", "centre"):
        return "clinic"
    return "residential"


def osm_address(t):
    st_, hn = t.get("addr:street", ""), t.get("addr:housenumber", "")
    if st_ or hn:
        return f"{st_} {hn}".strip()
    return t.get("addr:full", "") or "адрес не указан в OSM"


def fetch_osm_social(timeout=40):
    """Школы, детсады, больницы/поликлиники и жилые комплексы Павлодара из OSM — с адресами.
    Кэшируется в data/osm_social_v2.json."""
    import requests
    S, W, N, E = BBOX
    q = f"[out:json][timeout:{timeout}];" + SOCIAL_QUERY.replace("{bb}", f"{S},{W},{N},{E}")
    last = None
    for url in OVERPASS_URLS:
        try:
            r = requests.post(url, data={"data": q}, timeout=(10, timeout), headers={"User-Agent": "greening-mvp/1.0"})
            r.raise_for_status()
            js = r.json()
            break
        except Exception as e:  # noqa
            last = e
    else:
        raise last
    res, seen = [], set()
    for el in js.get("elements", []):
        c = el.get("center") or {"lat": el.get("lat"), "lon": el.get("lon")}
        if c.get("lat") is None:
            continue
        t = el.get("tags", {})
        kind = social_kind(t)
        name = t.get("name") or SOCIAL_RU.get(kind, "")
        key = (kind, name, round(c["lat"], 4), round(c["lon"], 4))
        if key in seen:
            continue
        seen.add(key)
        res.append(dict(lon=c["lon"], lat=c["lat"], kind=kind, name=name, addr=osm_address(t),
                        osm=f"{el['type']}/{el['id']}", src="© OpenStreetMap contributors (ODbL)"))
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, "osm_social_v2.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False)
    return res


def load_cached(name):
    p = os.path.join(DATA_DIR, name)
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return None


def geocode(address):
    """Поиск адреса через Nominatim (OSM). Возвращает (lon, lat, подпись) или None."""
    import requests
    r = requests.get("https://nominatim.openstreetmap.org/search",
                     params={"q": f"{address}, Павлодар, Казахстан", "format": "json", "limit": 1},
                     headers={"User-Agent": "greening-mvp/1.0"}, timeout=15)
    r.raise_for_status()
    js = r.json()
    if not js:
        return None
    return float(js[0]["lon"]), float(js[0]["lat"]), js[0].get("display_name", address)


class RoadIndex:
    """Индекс дорог в метрах (UTM) с предфильтром по bbox."""

    def __init__(self, roads):
        self.roads = roads
        self.xy = [ring_xy(r["line"]) for r in roads]
        self.bb = [bbox_of(xy) for xy in self.xy]
        for r, xy in zip(roads, self.xy):
            r["len_m"] = round(sum(math.hypot(xy[i + 1][0] - xy[i][0], xy[i + 1][1] - xy[i][1]) for i in range(len(xy) - 1)), 0)

    def near(self, poly_xy, radius=400, classes=None):
        pb = bbox_of(poly_xy)
        out = []
        for r, xy, bb in zip(self.roads, self.xy, self.bb):
            if classes and r["highway"] not in classes:
                continue
            if bbox_gap(pb, bb) > radius:
                continue
            d = poly_line_dist(poly_xy, xy)
            if d <= radius:
                out.append((d, r))
        out.sort(key=lambda t: t[0])
        return out


# ---------------- Импорт файлов и проверка качества ----------------
def _reproject_ring(ring, crs_hint):
    """Приводит кольцо к EPSG:4326. Поддерживается WGS84 и UTM северного полушария (EPSG:326xx)."""
    issues = []
    xs = [p[0] for p in ring]
    if max(abs(x) for x in xs) <= 180:
        if all(in_city(lo, la, 0.5) for lo, la in ring):
            return ring, issues
        # возможно, перепутаны широта/долгота
        if all(in_city(la, lo, 0.5) for lo, la in ring):
            issues.append("Перепутаны долгота и широта — исправлено автоматически")
            return [(la, lo) for lo, la in ring], issues
        issues.append("Координаты вне города (проверьте систему координат)")
        return ring, issues
    zone = UTM_ZONE
    m = re.search(r"326(\d\d)", str(crs_hint or ""))
    if m:
        zone = int(m.group(1))
    else:
        issues.append("Координаты в метрах без указания СК — принята UTM 43N (EPSG:32643)")
    out = [from_utm(x, y, zone) for x, y in ring]
    issues.append(f"Перепроецировано из EPSG:326{zone:02d} в EPSG:4326")
    return out, issues


def _rings_from_geom(g):
    t = g.get("type")
    c = g.get("coordinates")
    if t == "Polygon":
        return [[tuple(p[:2]) for p in c[0]]]
    if t == "MultiPolygon":
        return [[tuple(p[:2]) for p in poly[0]] for poly in c]
    if t == "Point":
        return [("POINT", tuple(c[:2]))]
    return []


def _circle_ring(lon, lat, r=30, n=12):
    cx, cy = to_utm(lon, lat)
    return [from_utm(cx + r * math.cos(2 * math.pi * i / n), cy + r * math.sin(2 * math.pi * i / n)) for i in range(n)]


def parse_upload(name, data):
    """Возвращает (объекты [{'ring','props'}], список проблем)."""
    issues = []
    feats = []
    low = name.lower()
    try:
        if low.endswith((".geojson", ".json")):
            js = json.loads(data.decode("utf-8"))
            crs = json.dumps(js.get("crs", ""))
            fl = js.get("features", [js] if js.get("type") == "Feature" else [])
            for k, f in enumerate(fl):
                g = f.get("geometry")
                if not g:
                    issues.append(f"Объект {k + 1}: пустая геометрия — пропущен")
                    continue
                for ring in _rings_from_geom(g):
                    feats.append(dict(raw=ring, props=f.get("properties") or {}, crs=crs))
        elif low.endswith(".kml"):
            root = ET.fromstring(data)
            for pm in root.iter():
                if not pm.tag.endswith("Placemark"):
                    continue
                nm = next((e.text for e in pm.iter() if e.tag.endswith("name")), "")
                for el in pm.iter():
                    if el.tag.endswith("coordinates") and el.text:
                        pts = [tuple(map(float, s.split(",")[:2])) for s in el.text.split() if "," in s]
                        if len(pts) >= 3:
                            feats.append(dict(raw=pts, props={"name": nm}, crs="EPSG:4326"))
                        elif len(pts) == 1:
                            feats.append(dict(raw=("POINT", pts[0]), props={"name": nm}, crs="EPSG:4326"))
        elif low.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(data))
            cols = {c.lower().strip(): c for c in df.columns}
            la = next((cols[c] for c in ["lat", "latitude", "широта", "y"] if c in cols), None)
            lo = next((cols[c] for c in ["lon", "lng", "longitude", "долгота", "x"] if c in cols), None)
            if la is None or lo is None:
                issues.append("CSV: не найдены столбцы координат (lat/lon или широта/долгота)")
            else:
                miss = df[[la, lo]].isna().any(axis=1)
                if miss.any():
                    issues.append(f"CSV: пропуски координат в {int(miss.sum())} строках — строки пропущены")
                for _, row in df[~miss].iterrows():
                    feats.append(dict(raw=("POINT", (float(row[lo]), float(row[la]))), props=row.drop([la, lo]).to_dict(), crs=""))
        elif low.endswith(".zip"):
            try:
                import shapefile  # pyshp
            except ImportError:
                return [], ["Для Shapefile установите пакет pyshp (pip install pyshp)"]
            z = zipfile.ZipFile(io.BytesIO(data))
            names = z.namelist()
            base = next((n[:-4] for n in names if n.lower().endswith(".shp")), None)
            if not base:
                return [], ["В архиве нет .shp"]
            prj = next((z.read(n).decode("utf-8", "ignore") for n in names if n.lower().endswith(".prj")), "")
            crs = ""
            if prj:
                m = re.search(r"UTM[_ ]zone[_ ](\d+)N", prj, re.I)
                crs = f"EPSG:326{int(m.group(1)):02d}" if m else "EPSG:4326"
            else:
                issues.append("Shapefile без .prj — система координат не указана")
            rd = shapefile.Reader(shp=io.BytesIO(z.read(base + ".shp")), dbf=io.BytesIO(z.read(base + ".dbf")),
                                  shx=io.BytesIO(z.read(base + ".shx")) if base + ".shx" in names else None)
            flds = [f[0] for f in rd.fields[1:]]
            for sr in rd.shapeRecords():
                props = dict(zip(flds, sr.record))
                pts = sr.shape.points
                parts = list(sr.shape.parts) + [len(pts)]
                if sr.shape.shapeType in (1, 11, 21):
                    feats.append(dict(raw=("POINT", tuple(pts[0])), props=props, crs=crs))
                else:
                    feats.append(dict(raw=[tuple(p) for p in pts[parts[0]:parts[1]]], props=props, crs=crs))
        else:
            return [], ["Неподдерживаемый формат. Допустимо: GeoJSON, KML, CSV, Shapefile (zip)"]
    except Exception as e:  # noqa
        return [], [f"Ошибка чтения файла: {e}"]

    out = []
    for k, f in enumerate(feats):
        raw = f["raw"]
        tag = f"Объект {k + 1}"
        if isinstance(raw, tuple) and raw and raw[0] == "POINT":
            (x, y) = raw[1]
            ring0, iss = _reproject_ring([(x, y)], f["crs"])
            lo, la = ring0[0]
            ring = _circle_ring(lo, la)
            issues += [f"{tag}: {i}" for i in iss]
            issues.append(f"{tag}: точка — построен участок радиусом 30 м")
        else:
            ring = list(raw)
            if len(ring) > 1 and ring[0] == ring[-1]:
                ring = ring[:-1]
            if len(ring) < 3:
                issues.append(f"{tag}: меньше 3 вершин — пропущен")
                continue
            ring, iss = _reproject_ring(ring, f["crs"])
            issues += [f"{tag}: {i}" for i in iss]
            si = self_intersections(ring)
            if si:
                issues.append(f"{tag}: самопересечение ({si} пар рёбер) — используйте с осторожностью, площадь может быть неверной")
        miss = [c for c in ["purpose", "назначение"] if c in f["props"] and (f["props"][c] in (None, "") or pd.isna(f["props"][c]))]
        if not any(c in f["props"] for c in ["purpose", "назначение"]) or miss:
            issues.append(f"{tag}: нет атрибута «назначение» — будет «сквер» (по умолчанию)")
        out.append(dict(ring=ring, props=f["props"]))
    for i in range(len(out)):
        for j in range(i + 1, len(out)):
            if polys_overlap(out[i]["ring"], out[j]["ring"]):
                issues.append(f"Объекты {i + 1} и {j + 1} пересекаются")
    return out, issues


# =============================================================================
# 4. АНАЛИТИКА
# =============================================================================
def logit(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def sigm(x):
    return 1 / (1 + math.exp(-x))


# ---- 4.1 Подбор растений: жёсткие ограничения + ранжирование 0–100 ----
def suitability(p, z, coeffs):
    """Возвращает (оценка 0–100 или None, список причин / причина исключения)."""
    if p["usda"] > 4:
        return None, f"не выдерживает зиму (зона USDA {p['usda']} при климате ≈3)"
    if z["salinity"] > p["salt_tol"] + 3:
        return None, f"засоление {z['salinity']:.1f} дСм/м > допустимого {p['salt_tol']}"
    if z["ph"] < p["ph_min"] - 1 or z["ph"] > p["ph_max"] + 1:
        return None, f"pH {z['ph']:.1f} вне диапазона {p['ph_min']}–{p['ph_max']}"
    if z["purpose"] in CHILD and (p["allergen"] or p["toxic"]):
        return None, "сильный аллерген/ядовит — недопустим для детских и больничных территорий"
    if z["purpose"] == "двор" and p["toxic"]:
        return None, "ядовитые плоды/листья — нежелателен во дворе (детские площадки)"
    if p["type"] != "газон" and p["height_m"] >= 15 and z["bldg_dist"] < coeffs["bldg_min_dist_big_tree"]:
        return None, f"крупное дерево, до зданий {z['bldg_dist']:.0f} м (< {coeffs['bldg_min_dist_big_tree']:.0f} м)"
    if z["road_exposure"] > 0.6 and p["pollution_tol"] <= 2:
        return None, "не переносит загрязнение и реагенты у дороги"
    s = 100.0
    why = []
    if p["usda"] == 4:
        s -= 25; why.append("− зимостойкость на пределе (нужна защита)")
    else:
        why.append("+ зимостойкость достаточна")
    margin = p["salt_tol"] - z["salinity"]
    if margin < 0:
        s -= min(40, -margin * 10); why.append(f"− засоление выше нормы вида на {-margin:.1f}")
    elif margin > 2:
        why.append("+ солеустойчив для этой почвы")
    phd = max(0, p["ph_min"] - z["ph"], z["ph"] - p["ph_max"])
    if phd > 0:
        s -= phd * 15; why.append(f"− pH вне оптимума на {phd:.1f}")
    md = max(0, (6 - p["drought"]) - z["moisture"])
    if md > 0:
        s -= md * 8; why.append(f"− не хватает влаги (дефицит {md} балла, нужен полив)")
    elif p["drought"] >= 4:
        why.append("+ засухоустойчив")
    lm = max(0, z["shade"] - p["shade_tol"])
    if lm > 0:
        s -= lm * 7; why.append("− светолюбив, участок затенён")
    pg = z["road_exposure"] * max(0, 4 - p["pollution_tol"])
    if pg > 0.05:
        s -= pg * 10; why.append("− чувствителен к загрязнению у дороги")
    elif z["road_exposure"] > 0.3 and p["pollution_tol"] >= 4:
        why.append("+ газо-/пылеустойчив (у дороги)")
    wg = max(0, z["wind"] - p["wind_tol"])
    if wg > 0:
        s -= wg * 5; why.append("− ветровая нагрузка выше устойчивости")
    if z["purpose"] in ("СЗЗ", "улица") and p["pollution_tol"] >= 4:
        s += 5; why.append("+ подходит для защитной/уличной посадки")
    if p["invasive"]:
        s -= 15; why.append("− инвазивный вид (ограничить)")
    if p["allergen"] and z["purpose"] in ("двор", "сквер", "парк"):
        s -= 8; why.append("− аллерген (пыльца)")
    return max(0.0, min(100.0, s)), why


# ---- 4.2 Приживаемость: экспертные правила + ML ----
FEATS = ["salt_excess", "ph_dev", "moist_def", "light_mis", "road_load", "wind_gap", "winter_gap", "watering", "age",
         "fertility", "base_logit"]
FEAT_RU = {"salt_excess": "Засоление почвы", "ph_dev": "Кислотность (pH)", "moist_def": "Дефицит влаги",
           "light_mis": "Недостаток света", "road_load": "Загрязнение/реагенты от дорог", "wind_gap": "Ветер",
           "winter_gap": "Зимостойкость", "watering": "Режим полива", "age": "Возраст саженца",
           "fertility": "Плодородие почвы"}
WATER_RU = {0: "минимальный", 1: "стандартный", 2: "интенсивный"}


def plant_zone_features(p, z, watering, age):
    return dict(salt_excess=max(0.0, z["salinity"] - p["salt_tol"]),
                ph_dev=max(0.0, p["ph_min"] - z["ph"], z["ph"] - p["ph_max"]),
                moist_def=max(0, (6 - p["drought"]) - z["moisture"]),
                light_mis=max(0, z["shade"] - p["shade_tol"]),
                road_load=z["road_exposure"] * max(0, 5 - p["pollution_tol"]) / 4,
                wind_gap=max(0, z["wind"] - p["wind_tol"]),
                winter_gap=max(0, p["usda"] - 3), watering=watering, age=age, fertility=z["fertility"],
                base_logit=logit(p["base_surv"]))


def rule_logit(f):
    return (f["base_logit"] - 0.45 * f["salt_excess"] - 0.9 * f["ph_dev"] - 0.45 * f["moist_def"] * (1 - 0.3 * f["watering"])
            - 0.2 * f["light_mis"] - 1.0 * f["road_load"] - 0.15 * f["wind_gap"] - 1.0 * f["winter_gap"]
            + 0.3 * (f["watering"] - 1) - 0.15 * abs(f["age"] - 3) + 0.12 * (f["fertility"] - 3))


IDEAL = dict(salt_excess=0, ph_dev=0, moist_def=0, light_mis=0, road_load=0, wind_gap=0, winter_gap=0, watering=1, age=3, fertility=3)

_MODEL = {}


def train_survival_model(seed=7, n=8000):
    """ML-часть: градиентный бустинг на СИНТЕТИЧЕСКОЙ выборке, сгенерированной из экспертных правил
    + нелинейные взаимодействия + шум. Это демонстрирует конвейер; для реальной оценки нужны данные
    инвентаризаций приживаемости акимата (заменяются в одном месте — этой функции)."""
    if "m" in _MODEL:
        return _MODEL["m"]
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(dict(
        salt_excess=rng.exponential(1.5, n) * (rng.random(n) < 0.45),
        ph_dev=rng.exponential(0.5, n) * (rng.random(n) < 0.35),
        moist_def=rng.integers(0, 4, n), light_mis=rng.integers(0, 4, n),
        road_load=rng.random(n) * (rng.random(n) < 0.55), wind_gap=rng.integers(0, 4, n),
        winter_gap=(rng.random(n) < 0.12).astype(int), watering=rng.integers(0, 3, n), age=rng.integers(2, 6, n),
        fertility=rng.integers(1, 6, n), base_logit=[logit(v) for v in rng.uniform(0.72, 0.93, n)]))
    rl = np.array([rule_logit(r) for r in X.to_dict("records")])
    true = rl - 0.25 * X.salt_excess * X.moist_def + 0.4 * X.watering * X.road_load - 0.3 * (X.age == 2) * X.moist_def
    # шум умеренный: метки синтетические, AUC отражает воспроизводимость закономерностей генератора
    true = true + rng.normal(0, 0.35, n)
    y = (rng.random(n) < 1 / (1 + np.exp(-true))).astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X[FEATS], y, test_size=0.25, random_state=seed, stratify=y)
    m = GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=seed).fit(Xtr, ytr)
    auc_ml = roc_auc_score(yte, m.predict_proba(Xte)[:, 1])
    auc_rule = roc_auc_score(yte, [rule_logit(r) for r in Xte.to_dict("records")])
    _MODEL["m"] = dict(model=m, auc=round(float(auc_ml), 3), auc_rule=round(float(auc_rule), 3), n_train=len(Xtr),
                       n_test=len(Xte), pos_rate=round(float(y.mean()), 3))
    return _MODEL["m"]


def predict_survival(p, z, watering, age, care_years):
    M = train_survival_model()
    f = plant_zone_features(p, z, watering, age)
    pr = sigm(rule_logit(f))
    pm = float(M["model"].predict_proba(pd.DataFrame([f])[FEATS])[0, 1])
    p1 = 0.5 * pr + 0.5 * pm
    band = abs(pr - pm) / 2 + 0.04
    r = min(0.99, max(0.85, sigm(rule_logit(f) + 1.6)))
    p2 = p1 * r * (0.97 if care_years < 2 else 1.0)
    p3 = p2 * r * (0.97 if care_years < 3 else 1.0)
    contrib = {}
    for k, ideal in IDEAL.items():
        g = dict(f); g[k] = ideal
        contrib[k] = round((pr - sigm(rule_logit(g))) * 100, 1)  # п.п., отрицательное = снижает
    return dict(p1=p1, p2=p2, p3=p3, lo=max(0, p1 - band), hi=min(1, p1 + band), p_rule=pr, p_ml=pm, contrib=contrib,
                conf="высокая" if band < 0.07 else ("средняя" if band < 0.12 else "низкая"))


# ---- 4.3 План посадки, смета, эффект, уход ----
VARIANTS = {
    "Эконом": dict(dens=0.6, n_tree=3, n_shrub=2, care_years=1, watering=0, age=2, sort="cheap"),
    "Оптимальный": dict(dens=1.0, n_tree=4, n_shrub=3, care_years=2, watering=1, age=3, sort="score"),
    "Максимальный эффект": dict(dens=1.3, n_tree=5, n_shrub=4, care_years=3, watering=2, age=3, sort="eco"),
}
WATER_K = {0: 0.75, 1: 1.0, 2: 1.35}


def cap_shares(w, cap):
    w = np.array(w, float)
    w = w / w.sum()
    n = len(w)
    if n * cap < 1 - 1e-9:
        return np.ones(n) / n
    fixed = np.zeros(n, bool)
    for _ in range(n):
        over = (w > cap + 1e-12) & ~fixed
        if not over.any():
            break
        fixed |= over
        w[fixed] = cap
        rest = 1 - w[fixed].sum()
        free = ~fixed
        if free.any():
            w[free] = w[free] / w[free].sum() * rest
    return w


def distribute(total, shares):
    raw = np.array(shares) * total
    q = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - q))[: int(total - q.sum())]:
        q[i] += 1
    return q


def build_plan(z, variant, C, ov=None, care_years=None, watering=None):
    ov = ov or {}
    V = dict(VARIANTS[variant])
    if care_years:
        V["care_years"] = int(care_years)
    if watering is not None:
        V["watering"] = int(watering)
    co = C["coeffs"]
    excluded = set(ov.get("excluded", []))
    fixed = list(ov.get("fixed", []))
    for a, b in (ov.get("replace") or {}).items():
        if a == b:
            continue
        excluded.add(a)
        if b and b not in fixed:
            fixed.append(b)
    evals, rejected, warnings = [], [], []
    for p in C["plants"]:
        if p["name"] in excluded:
            continue
        sc, info = suitability(p, z, co)
        if sc is None:
            rejected.append({"Вид": p["name"], "Ярус": p["type"], "Причина исключения": info})
            if p["name"] in fixed:
                warnings.append(f"«{p['name']}» зафиксирован пользователем, но нарушает ограничение: {info}")
                evals.append((p, 0.0, ["⚠ зафиксирован вопреки ограничению: " + info]))
            continue
        evals.append((p, sc, info))
    plantable = z["area_m2"] * max(0.0, 1 - z["green_share"])
    tph, sph, lawn_share = DENSITY.get(z["purpose"], DENSITY["сквер"])
    lines = []
    for tier, per_ha, nkey in [("дерево", tph, "n_tree"), ("кустарник", sph, "n_shrub"), ("газон", None, None)]:
        cands = [e for e in evals if e[0]["type"] == tier]
        if not cands:
            warnings.append(f"Нет подходящих видов яруса «{tier}» для условий зоны")
            continue
        maxp = max(e[0]["price"] for e in cands) or 1
        maxc = max(e[0]["co2_kg"] for e in cands) or 1
        if V["sort"] == "cheap":
            key = lambda e: 0.6 * e[1] + 40 * (1 - e[0]["price"] / maxp)
        elif V["sort"] == "eco":
            key = lambda e: 0.6 * e[1] + 40 * e[0]["co2_kg"] / maxc
        else:
            key = lambda e: e[1]
        chosen = [e for e in cands if e[0]["name"] in fixed]
        rest = sorted([e for e in cands if e[0]["name"] not in fixed and e[1] >= 40], key=key, reverse=True)
        need = 1 if tier == "газон" else V[nkey]
        for e in rest:
            if len(chosen) >= need:
                break
            chosen.append(e)
        if not chosen:
            warnings.append(f"Ни один вид яруса «{tier}» не набрал пригодность ≥ 40")
            continue
        if tier == "газон":
            e = chosen[0]
            m2 = round(plantable * lawn_share)
            if m2 > 0:
                lines.append(dict(name=e[0]["name"], type=tier, qty=m2, unit="м²", score=round(e[1]), why=e[2], plant=e[0]))
            continue
        total = int(round(per_ha * plantable / 1e4 * V["dens"]))
        if plantable > 0:
            total = max(total, 1)
        if len(chosen) * co["max_share"] < 1 - 1e-9 and total > 3:
            warnings.append(f"Ярус «{tier}»: {len(chosen)} вид(а) — лимит доли одного вида {co['max_share']:.0%} не соблюдён")
        shares = cap_shares([max(e[1], 1) for e in chosen], co["max_share"])
        for e, q in zip(chosen, distribute(total, shares)):
            if q > 0:
                lines.append(dict(name=e[0]["name"], type=tier, qty=int(q), unit="шт", score=round(e[1]), why=e[2], plant=e[0]))
    for l in lines:
        l["surv"] = predict_survival(l["plant"], z, V["watering"], V["age"], V["care_years"])
    plan = dict(zone_id=z["id"], variant=variant, V=V, lines=lines, rejected=rejected, warnings=warnings,
                plantable_m2=plantable, lawn_share=lawn_share)
    plan["estimate"] = estimate(plan, z, C)
    plan["eco"] = eco_effect(plan, z, C)
    woody = [l for l in lines if l["type"] != "газон"]
    q = sum(l["qty"] for l in woody)
    plan["avg_surv"] = {k: (sum(l["surv"][k] * l["qty"] for l in woody) / q if q else None) for k in ("p1", "p2", "p3")}
    return plan


def estimate(plan, z, C):
    R = C["rates"]
    V = plan["V"]
    L = plan["lines"]
    trees = sum(l["qty"] for l in L if l["type"] == "дерево")
    shrubs = sum(l["qty"] for l in L if l["type"] == "кустарник")
    lawn = sum(l["qty"] for l in L if l["type"] == "газон")
    area = plan["plantable_m2"]
    mat = sum(l["qty"] * l["plant"]["price"] for l in L)
    items = []
    items.append(("Посадочный материал", mat, "Σ (кол-во × цена вида)"))
    items.append(("Подготовка территории и грунт", area * R["prep_soil_m2"], f"{area:,.0f} м² × {R['prep_soil_m2']:,.0f} ₸"))
    items.append(("Удобрения", area * R["fertilizer_m2"], f"{area:,.0f} м² × {R['fertilizer_m2']:,.0f} ₸"))
    if z["salinity"] > 4:
        items.append(("Мелиорация засолённой почвы", area * R["gypsum_m2"], f"засоление {z['salinity']} > 4: {area:,.0f} м² × {R['gypsum_m2']:,.0f} ₸"))
    if z["moisture"] >= 4:
        items.append(("Дренаж", area * 0.3 * R["drainage_m2"], f"переувлажнение: 30% × {area:,.0f} м² × {R['drainage_m2']:,.0f} ₸"))
    items.append(("Посадка", trees * R["plant_tree"] + shrubs * R["plant_shrub"] + lawn * R["lawn_work_m2"],
                  f"{trees}×{R['plant_tree']:,.0f} + {shrubs}×{R['plant_shrub']:,.0f} + {lawn:,.0f} м²×{R['lawn_work_m2']:,.0f}"))
    items.append(("Доставка", mat * R["delivery_pct"] / 100, f"{R['delivery_pct']:.0f}% от материала"))
    wk = WATER_K[V["watering"]]
    care_y = (trees * R["care_tree_year"] + shrubs * R["care_shrub_year"] + lawn * R["care_lawn_m2_year"]) * wk
    items.append((f"Полив и уход, {V['care_years']} г.", care_y * V["care_years"],
                  f"({trees}×{R['care_tree_year']:,.0f} + {shrubs}×{R['care_shrub_year']:,.0f} + {lawn:,.0f}×{R['care_lawn_m2_year']:,.0f}) × k полива {wk} × {V['care_years']} г."))
    res = 0.0
    for l in L:
        p_h = l["surv"]["p" + str(min(3, V["care_years"]))]
        pr = R["plant_tree"] if l["type"] == "дерево" else (R["plant_shrub"] if l["type"] == "кустарник" else R["lawn_work_m2"])
        k = 0.5 if l["type"] == "газон" else 1.0
        res += l["qty"] * (1 - p_h) * (l["plant"]["price"] + pr) * k
    items.append(("Резерв на замену неприжившихся", res, "Σ кол-во × (1 − приживаемость на конец ухода) × (цена + посадка)"))
    total = sum(v for _, v, _ in items)
    df = pd.DataFrame([dict(Статья=n, **{"Сумма, ₸": round(v)}, Доля=(v / total if total else 0), Формула=f) for n, v, f in items])
    return dict(table=df, total=total, care_per_year=care_y)


def eco_effect(plan, z, C, years=(5, 10, 20)):
    co = C["coeffs"]
    area = z["area_m2"]
    pop = z["pop_density"] * z["area_ha"]
    green0 = z["green_share"] * area

    def at(Y):
        co2 = dust = crown = 0.0
        lawn = 0.0
        for l in plan["lines"]:
            p = l["plant"]
            if l["type"] == "газон":
                alive = l["qty"] * l["surv"]["p3"]
                lawn += alive
                co2 += alive * co["lawn_co2_m2"]
                continue
            alive = l["qty"] * l["surv"]["p3"] * (1 - co["annual_loss"]) ** max(0, Y - 3)
            f = min(1.0, (Y + plan["V"]["age"]) / p["mature_years"]) ** co["growth_exp"]
            co2 += alive * p["co2_kg"] * f
            dust += alive * p["dust_kg"] * f
            crown += alive * math.pi * (p["crown_m"] / 2) ** 2 * f
        pp = min(100.0, crown / area * 100) if area else 0
        new_green = min(area - green0, crown + lawn) if area > green0 else 0
        return dict(co2_t=co2 / 1000, o2_t=co2 / 1000 * co["o2_per_co2"], dust_kg=dust,
                    cooling_c=min(co["cooling_max"], co["cooling_per_pp"] * pp), rain_m3=crown * co["precip_mm"] / 1000 * co["interception"],
                    green_add_m2=new_green, green_share_after=min(1.0, (green0 + new_green) / area) if area else 0,
                    gpc_after=(green0 + new_green) / pop if pop > 0 else None)

    res = {Y: at(Y) for Y in years}
    cum = sum(at(Y)["co2_t"] for Y in range(1, 21))
    total = plan["estimate"]["total"] if "estimate" in plan else 0
    return dict(by_year=res, co2_cum20_t=cum, cost_per_t=(total / cum if cum > 0 else None),
                before=dict(green_share=z["green_share"], gpc=(green0 / pop if pop > 0 else None), lst=z["lst"]))


MONTHS = ["Янв", "Фев", "Мар", "Апр", "Май", "Июн", "Июл", "Авг", "Сен", "Окт", "Ноя", "Дек"]
WATER_BASE = {4: 1, 5: 3, 6: 4, 7: 5, 8: 4, 9: 2, 10: 1}  # поливов в месяц, молодые деревья, резко-континентальный климат


def care_calendar(plan, z, max_visits=6):
    """Календарь ухода на 1-й год. Учитывает тип растений, засухоустойчивость, проблемы зоны и лимит выездов."""
    L = plan["lines"]
    woody = [l for l in L if l["type"] != "газон"]
    has_conif = any(l["plant"]["name"].startswith(("Сосна", "Ель", "Лиственница")) for l in L)
    dr = np.mean([l["plant"]["drought"] for l in woody]) if woody else 3
    k = (0.7 if dr >= 4 else 1.0) * (0.6 if z["moisture"] >= 4 else 1.0) * (1.2 if z["lst"] > 37 else 1.0) * WATER_K[plan["V"]["watering"]]
    rows, warns = [], []
    for m in range(1, 13):
        ops = []
        freq = round(WATER_BASE.get(m, 0) * k)
        if freq > max_visits:
            warns.append(f"{MONTHS[m - 1]}: нужно {freq} поливов, бригада — {max_visits} выездов; рекомендуются мульча и капельный полив")
            freq = max_visits
        if freq:
            ops.append(("Полив", f"{freq} раз/мес", "деревья 50–60 л, кустарники 20 л, газон 10 л/м² за полив"))
        if m == 3:
            ops.append(("Снятие зимней защиты, санитарная обрезка", "1 раз", "до сокодвижения"))
        if m == 4:
            ops.append(("Формирующая обрезка, подкормка азотная", "1 раз", "при плодородии ≤3 — обязательно" if z["fertility"] <= 3 else "по необходимости"))
            if z["salinity"] > 4:
                ops.append(("Промывной полив (засоление)", "1 раз", f"засоление {z['salinity']} дСм/м"))
        if m == 5:
            ops.append(("Мульчирование приствольных кругов", "1 раз", "слой 5–8 см, снижает испарение"))
            ops.append(("Осмотр и профилактика вредителей (тля, моль)", "1 раз", "биопрепараты"))
        if m in (6, 7):
            ops.append(("Контроль вредителей и болезней", "1 раз", "при обнаружении — обработка"))
        if m == 6:
            ops.append(("Комплексная подкормка", "1 раз", "NPK"))
        if m == 8:
            ops.append(("Фосфорно-калийная подкормка", "1 раз", "подготовка к зиме, без азота"))
        if m == 9 and z["salinity"] > 4:
            ops.append(("Промывной полив (засоление)", "1 раз", ""))
        if m == 10:
            ops.append(("Влагозарядковый полив, побелка стволов", "1 раз", "до заморозков"))
        if m == 11:
            if has_conif:
                ops.append(("Притенение хвойных от весенних ожогов", "1 раз", "укрывной материал"))
            if z["road_exposure"] > 0.4:
                ops.append(("Защита от противогололёдных реагентов", "1 раз", "ограждения, отвод снега с реагентами"))
            ops.append(("Утепление приствольных кругов", "1 раз", "молодые посадки"))
        if m in (12, 1, 2):
            ops.append(("Снегозадержание, осмотр после ветров", "1–2 раза", "сломанные ветви — удалить"))
        for op, fr, note in ops:
            rows.append(dict(Месяц=MONTHS[m - 1], Операция=op, Периодичность=fr, Примечание=note))
    return pd.DataFrame(rows), warns


# ---- 4.4 Проблемные зоны ----
SEV_ORDER = {"высокая": 3, "средняя": 2, "низкая": 1}


def detect_problems(z, coeffs):
    P = []
    gs = z["green_share"]
    if gs < 0.10:
        P.append(("Дефицит озеленения", "высокая", f"доля зелени {gs:.0%} (ориентир ≥ 25%)", "первоочередная посадка, вариант «Оптимальный» или «Максимальный»"))
    elif gs < 0.20:
        P.append(("Дефицит озеленения", "средняя", f"доля зелени {gs:.0%}", "дополнить посадки"))
    pop = z["pop_density"] * z["area_ha"]
    if pop > 50:
        gpc = gs * z["area_m2"] / pop
        if gpc < coeffs["green_per_capita_norm"]:
            P.append(("Дефицит зелени на жителя", "средняя" if gpc > coeffs["green_per_capita_norm"] / 2 else "высокая",
                      f"{gpc:.1f} м²/чел (ориентир {coeffs['green_per_capita_norm']:.0f}); население по участку — {SYN}", "увеличить площадь озеленения"))
    if z["degradation"] > 0.35:
        P.append(("Деградация насаждений", "высокая", f"{z['degradation']:.0%} насаждений в плохом состоянии", "санитарная рубка и замена, восстановительный уход"))
    elif z["degradation"] > 0.2:
        P.append(("Деградация насаждений", "средняя", f"{z['degradation']:.0%} в плохом состоянии", "обследование, уход"))
    if z["salinity"] > 6 or z["ph"] > 8.7:
        P.append(("Неблагоприятные почвы", "высокая", f"{z['soil_type']}, засоление {z['salinity']} дСм/м, pH {z['ph']}", "мелиорация (гипс, промывка), солеустойчивые виды"))
    elif z["salinity"] > 4 or z["ph"] > 8.4 or z["fertility"] <= 1:
        P.append(("Неблагоприятные почвы", "средняя", f"{z['soil_type']}, засоление {z['salinity']}, pH {z['ph']}, плодородие {z['fertility']}/5", "внесение грунта и удобрений"))
    if z["lst"] >= 38:
        P.append(("Остров тепла", "высокая", f"температура поверхности {z['lst']} °C ({SYN})", "деревья с широкой кроной, сомкнутость крон"))
    elif z["lst"] >= 35.5:
        P.append(("Остров тепла", "средняя", f"температура поверхности {z['lst']} °C ({SYN})", "увеличить затенение"))
    if z["purpose"] in ("школа", "больница", "двор") and z["dist_road"] < 60 and gs < 0.15:
        P.append(("Нет защитной полосы у дороги", "высокая", f"{z['purpose']}, до дороги {z['dist_road']:.0f} м ({z['road_class_ru']})", "двухрядная полоса из газоустойчивых деревьев + живая изгородь"))
    if z["industry"] > 0.6 and gs < 0.2:
        P.append(("Промнагрузка без СЗЗ", "средняя", f"индекс промнагрузки {z['industry']:.2f} ({SYN})", "санитарно-защитные посадки"))
    return [dict(zone=z["id"], Проблема=a, Острота=b, Причина=c, Действие=d) for a, b, c, d in P]


# ---- 4.5 Приоритет ----
FACTORS = [("deficit", "Дефицит зелени", 25), ("population", "Плотность населения и застройки", 20),
           ("social", "Социальные объекты рядом", 15), ("heat_air", "Тепловая и воздушная нагрузка", 20),
           ("soil", "Проблемы почвы", 5), ("benefit", "Выгода / затраты", 15)]


def priority_table(zones, opt_plans, weights):
    rows = []
    pops = [z["pop_density"] for z in zones] or [1]
    socs = [z["social_n"] for z in zones] or [1]
    bens = []
    for z in zones:
        p = opt_plans[z["id"]]
        cost = p["estimate"]["total"]
        bens.append(p["eco"]["co2_cum20_t"] / cost * 1e6 if cost > 0 else 0)
    mb = max(bens) or 1
    for z, b in zip(zones, bens):
        f = dict(deficit=min(1, max(0, 1 - z["green_share"] / 0.4)), population=z["pop_density"] / (max(pops) or 1),
                 social=z["social_n"] / (max(socs) or 1),
                 heat_air=0.5 * min(1, max(0, (z["lst"] - 30) / 12)) + 0.5 * max(z["road_exposure"], z["industry"]),
                 soil=(min(1, z["salinity"] / 8) + min(1, abs(z["ph"] - 7) / 2) + (5 - z["fertility"]) / 4) / 3,
                 benefit=b / mb)
        sw = sum(weights.values()) or 1
        contrib = {k: weights[k] * f[k] / sw * 100 for k in f}
        rows.append(dict(id=z["id"], name=z["name"], district=z["district"], purpose=z["purpose"],
                         priority=round(sum(contrib.values()), 1), **{f"c_{k}": round(v, 1) for k, v in contrib.items()},
                         **{f"f_{k}": round(v, 2) for k, v in f.items()}))
    df = pd.DataFrame(rows).sort_values("priority", ascending=False).reset_index(drop=True)
    df.insert(0, "Ранг", range(1, len(df) + 1))
    return df


def budget_roadmap(zones, all_plans, prio_df, budget):
    """Жадный выбор (зона, вариант) по эффекту на тенге: эффект = приоритет/100 × CO₂ за 20 лет."""
    pr = dict(zip(prio_df.id, prio_df.priority))
    cands = []
    for z in zones:
        for v, p in all_plans[z["id"]].items():
            cost = p["estimate"]["total"]
            eff = pr[z["id"]] / 100 * p["eco"]["co2_cum20_t"]
            if cost > 0:
                cands.append((eff / cost, z["id"], v, cost, eff, p))
    cands.sort(key=lambda t: -t[0])
    chosen, used, left = [], set(), budget
    for r, zid, v, cost, eff, p in cands:
        if zid in used or cost > left:
            continue
        chosen.append(dict(Этап=len(chosen) + 1, Зона=zid, Вариант=v, **{"Стоимость, ₸": round(cost)},
                           **{"CO₂ за 20 лет, т": round(p["eco"]["co2_cum20_t"], 1)}, Приоритет=pr[zid],
                           **{"Деревьев": sum(l["qty"] for l in p["lines"] if l["type"] == "дерево")}))
        used.add(zid)
        left -= cost
    return pd.DataFrame(chosen), budget - left


# ---- 4.6 «Мир»: данные + кэш расчётов ----
class World:
    def __init__(self, base, plants_df, rates_df, coeffs_df, roads=None, social=None, extra_zones=None):
        self.soils = base["soils"]
        self.greens = base["greens"]
        self.roads_real = roads is not None
        self.roads = roads if roads is not None else base["roads"]
        # Синтетические соцобъекты не используются: только реальные (OSM). Нет данных — так и пишем.
        self.social_real = social is not None
        self.social = social if social is not None else []
        self.rindex = RoadIndex(self.roads)
        self.soc_xy = [to_utm(s["lon"], s["lat"]) for s in self.social]
        self.plants_df = plants_df
        self.C = dict(plants=plants_df.to_dict("records"), rates=dict(zip(rates_df.key, rates_df.value.astype(float))),
                      coeffs=dict(zip(coeffs_df.key, coeffs_df.value.astype(float))))
        self.zones = []
        for z in list(base["zones"]) + list(extra_zones or []):
            self.zones.append(self.enrich(dict(z)))
        self._plans = {}

    def soil_at(self, lon, lat):
        for s in self.soils:
            if pip(lon, lat, s["ring"]):
                return s
        return min(self.soils, key=lambda s: math.hypot(np.mean([p[0] for p in s["ring"]]) - lon, np.mean([p[1] for p in s["ring"]]) - lat))

    def enrich(self, z):
        xy = ring_xy(z["ring"])
        z["area_m2"] = ring_area_xy(xy)
        z["area_ha"] = z["area_m2"] / 1e4
        lo, la = ring_centroid(z["ring"])
        z["lon"], z["lat"] = lo, la
        if not z.get("soil_from_file"):
            s = self.soil_at(lo, la)
            for k in ("soil_type", "ph", "salinity", "fertility", "moisture"):
                z[k] = s[k]
        near = self.rindex.near(xy, radius=400)
        if near:
            d, r = near[0]
            z["dist_road"], z["road_class"], z["road_name"] = d, r["highway"], r["name"]
            exp = max(max(0, 1 - dd / 150) * ROAD_CLASSES.get(rr["highway"], ("", 0.3))[1] for dd, rr in near[:15])
            z["near_roads"] = [(round(dd), rr["rid"]) for dd, rr in near[:10]]
        else:
            z["dist_road"], z["road_class"], z["road_name"] = 400.0, "residential", "нет дорог в радиусе 400 м"
            exp = 0.0
            z["near_roads"] = []
        z["road_exposure"] = round(exp, 2)
        z["road_class_ru"] = ROAD_CLASSES.get(z["road_class"], (z["road_class"],))[0]
        soc = [s for s, p in zip(self.social, self.soc_xy) if poly_point_dist(xy, p) <= 500]
        z["social_n"] = len(soc)
        z["social_list"] = (", ".join(sorted({SOCIAL_RU.get(s["kind"], s["kind"]) for s in soc})) or "нет") if self.social_real \
            else "нет данных (OSM ещё не загружен)"
        z["social_names"] = "; ".join(f"{s['name']} ({s.get('addr', '')})" for s in soc[:6])
        z["problems"] = detect_problems(z, self.C["coeffs"])
        return z

    def zone(self, zid):
        return next((z for z in self.zones if z["id"] == zid), None)

    def plan(self, zid, variant="Оптимальный", ov=None, care_years=None, watering=None):
        key = (zid, variant, json.dumps(ov or {}, sort_keys=True, ensure_ascii=False), care_years, watering)
        if key not in self._plans:
            self._plans[key] = build_plan(self.zone(zid), variant, self.C, ov, care_years, watering)
        return self._plans[key]

    def all_plans(self):
        return {z["id"]: {v: self.plan(z["id"], v) for v in VARIANTS} for z in self.zones}

    def opt_plans(self):
        return {z["id"]: self.plan(z["id"], "Оптимальный") for z in self.zones}

    def priorities(self, weights=None):
        weights = weights or {k: w for k, _, w in FACTORS}
        return priority_table(self.zones, self.opt_plans(), weights)

    def problems_df(self):
        rows = []
        for z in self.zones:
            for p in z["problems"]:
                rows.append(dict(Зона=z["id"], Район=z["district"], Назначение=z["purpose"], **{k: v for k, v in p.items() if k != "zone"}))
        df = pd.DataFrame(rows)
        if len(df):
            df["_s"] = df["Острота"].map(SEV_ORDER)
            df = df.sort_values(["_s", "Зона"], ascending=[False, True]).drop(columns="_s").reset_index(drop=True)
        return df

    def new_zone_from_ring(self, ring, props=None, src="пользователь"):
        props = props or {}
        n = len(self.zones) + 1
        zid = f"Z-{n:02d}"
        while self.zone(zid):
            n += 1
            zid = f"Z-{n:02d}"
        lo, la = ring_centroid(ring)
        purpose = str(props.get("purpose") or props.get("назначение") or "сквер")
        if purpose not in PURPOSES:
            purpose = "сквер"
        z = dict(id=zid, name=str(props.get("name") or f"{purpose.capitalize()} {zid}"), ring=ring, purpose=purpose,
                 district=district_of(lo, la), green_share=float(props.get("green_share", 0.1)),
                 pop_density=float(props.get("pop_density", 80)), lst=float(props.get("lst", 34)), shade=2, wind=3,
                 degradation=float(props.get("degradation", 0.1)), industry=0.2, bldg_dist=float(props.get("bldg_dist", 10)),
                 src=f"{src}; атрибуты по умолчанию — {SYN}")
        return z


# =============================================================================
# 5. АССИСТЕНТ: разбор запроса -> вызов функций модулей -> ответ с цифрами
# =============================================================================
def _norm(s):
    return s.lower().replace("ё", "е")


def find_species(text, plants):
    t = _norm(text)
    hits = []
    for p in plants:
        w = _norm(p["name"].split()[0])
        stem = w[:2] if len(w) <= 3 else w[:max(3, len(w) - 2)]  # «липа» -> «лип», «клён» -> «кле», «ива» -> «ив»
        mm = re.search(r"(?<![а-яa-z])" + re.escape(stem) + r"[а-я]*", t)
        if mm:
            hits.append((mm.start(), p))
    hits.sort(key=lambda h: h[0])
    seen, out = set(), []
    for i, p in hits:
        if p["name"] not in seen:
            out.append(p); seen.add(p["name"])
    return out


def fmt_tg(x):
    return f"{x:,.0f} ₸".replace(",", " ")


def parse_money(t):
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*(млрд|млн|миллион\w*|тыс\w*|т\.)?", t)
    if not m:
        return None
    v = float(m.group(1).replace(",", "."))
    u = m.group(2) or ""
    if u.startswith("млрд"):
        v *= 1e9
    elif u.startswith(("млн", "миллион")):
        v *= 1e6
    elif u.startswith(("тыс", "т.")):
        v *= 1e3
    return v


def assistant_answer(q, W, selected=None, weights=None):
    """Возвращает (markdown-ответ, [id зон для ссылок без повторов], название вызванного инструмента)."""
    a, zl, tool = _assistant_answer(q, W, selected, weights)
    return a, list(dict.fromkeys(zl)), tool


def _assistant_answer(q, W, selected=None, weights=None):
    t = _norm(q)
    m = re.search(r"\b(?:z-?|зон[аеуы]?\s*№?\s*)(\d{1,3})\b", t)
    zid = f"Z-{int(m.group(1)):02d}" if m else selected
    if m and not W.zone(zid):
        return f"Зона {zid} не найдена. Доступны: {', '.join(z['id'] for z in W.zones[:10])}…", [], "—"
    z = W.zone(zid) if zid else None

    if re.search(r"\d", t) and any(w in t for w in ["млн", "млрд", "тыс", "₸", "тенге", "бюджет", "на сумму"]) and \
            not any(w in t for w in ["прижив"]):
        b = parse_money(t)
        if not b:
            return "Не удалось распознать сумму. Пример: «Что можно озеленить на 5 млн?»", [], "budget_roadmap"
        pr = W.priorities(weights)
        df, spent = budget_roadmap(W.zones, W.all_plans(), pr, b)
        if df.empty:
            cheapest = min(p["estimate"]["total"] for zp in W.all_plans().values() for p in zp.values())
            return (f"На {fmt_tg(b)} не помещается ни один вариант. Минимальный проект стоит {fmt_tg(cheapest)}."), [], "budget_roadmap"
        lines = [f"**Дорожная карта на {fmt_tg(b)}** (максимум эффекта = приоритет × CO₂ за 20 лет на тенге):"]
        for r in df.itertuples():
            lines.append(f"- Этап {r.Этап}: **{r.Зона}** — «{r.Вариант}», {fmt_tg(r._4)}, деревьев {r.Деревьев}, CO₂ за 20 лет ≈ {r._5} т, приоритет {r.Приоритет}")
        lines.append(f"\nИспользовано {fmt_tg(spent)} из {fmt_tg(b)}, CO₂ за 20 лет ≈ {df["CO₂ за 20 лет, т"].sum():.0f} т. _Оценка._")
        return "\n".join(lines), list(df["Зона"]), "budget_roadmap"

    if "почему" in t and find_species(t, W.C["plants"]):
        if not z:
            return "Выберите зону, чтобы объяснить выбор видов.", [], "suitability"
        p = find_species(t, W.C["plants"])[0]
        sc, info = suitability(p, z, W.C["coeffs"])
        V = VARIANTS["Оптимальный"]
        sv = predict_survival(p, z, V["watering"], V["age"], V["care_years"])
        inplan = [l for l in W.plan(z["id"])["lines"] if l["name"] == p["name"]]
        txt = [f"**{p['name']} в зоне {z['id']}** ({z['purpose']}, {z['soil_type']}, засоление {z['salinity']} дСм/м, pH {z['ph']}, "
               f"нагрузка от дорог {z['road_exposure']:.2f}):"]
        if sc is None:
            txt.append(f"- ❌ Исключён жёстким ограничением: {info}.")
        else:
            txt.append(f"- Пригодность {sc:.0f}/100" + (" — вид уже в плане." if inplan else " — в план не вошёл: у других видов яруса оценка выше."))
            minus = [w for w in info if w.startswith("−")]
            if minus:
                txt.append("- Минусы: " + "; ".join(w[2:] for w in minus))
        if p["allergen"]:
            txt.append("- Сильный аллерген (пыльца/пух) — ограничен у школ, больниц и во дворах.")
        if p["invasive"]:
            txt.append("- Инвазивный вид — ограничивать в новых посадках.")
        txt.append(f"- Прогноз приживаемости через 3 года: {sv['p3']:.0%}. _Оценка._")
        return "\n".join(txt), [z["id"]], "suitability + predict_survival"

    if "аудит" in t or "анализ зон" in t:
        if not z:
            return "Выберите зону для экспресс-аудита.", [], "zone_audit"
        p = W.plan(z["id"])
        a = p["avg_surv"]
        txt = [f"**Экспресс-аудит зоны {z['id']} «{z['name']}»:**",
               f"- Площадь {fmt_tg(z['area_m2'])[:-2]} м², под посадку {fmt_tg(p['plantable_m2'])[:-2]} м², доля зелени {z['green_share']:.0%}.",
               f"- Почва: {z['soil_type']}, pH {z['ph']}, засоление {z['salinity']} дСм/м, плодородие {z['fertility']}/5.",
               f"- Ближайшая дорога: {z['road_name']} ({z['road_class_ru']}, {z['dist_road']:.0f} м).",
               f"- Приживаемость (1/2/3 г.): {a['p1']:.0%} / {a['p2']:.0%} / {a['p3']:.0%}." if a["p1"] is not None else "- В плане нет деревьев и кустарников.",
               f"- Смета «Оптимальный»: {fmt_tg(p['estimate']['total'])}; CO₂ за 20 лет ≈ {p['eco']['co2_cum20_t']:.1f} т."]
        if z["problems"]:
            txt.append("- Проблемы: " + "; ".join(f"{q['Проблема']} ({q['Острота']})" for q in z["problems"]))
        txt.append("_Оценка; решение принимает специалист._")
        return "\n".join(txt), [z["id"]], "zone_audit"

    if "замен" in t:
        sp = find_species(t, W.C["plants"])
        if len(sp) < 2:
            return "Укажите два вида, например: «Какая будет приживаемость, если заменить клён на липу?»", [], "predict_survival"
        if not z:
            return "Сначала выберите зону на карте или укажите её номер (например, «в зоне 5»).", [], "predict_survival"
        a, b = sp[0], sp[1]
        V = VARIANTS["Оптимальный"]
        sa = predict_survival(a, z, V["watering"], V["age"], V["care_years"])
        sb = predict_survival(b, z, V["watering"], V["age"], V["care_years"])
        sca, wa = suitability(a, z, W.C["coeffs"])
        scb, wb = suitability(b, z, W.C["coeffs"])
        txt = [f"**Зона {z['id']}** ({z['purpose']}, {z['soil_type']}, засоление {z['salinity']}, pH {z['ph']}):",
               f"- {a['name']}: приживаемость 1/2/3 года — {sa['p1']:.0%} / {sa['p2']:.0%} / {sa['p3']:.0%}; пригодность {('исключён: ' + wa) if sca is None else f'{sca:.0f}/100'}",
               f"- {b['name']}: {sb['p1']:.0%} / {sb['p2']:.0%} / {sb['p3']:.0%}; пригодность {('исключён: ' + wb) if scb is None else f'{scb:.0f}/100'}",
               f"\nЗамена изменит приживаемость через 3 года на **{(sb['p3'] - sa['p3']) * 100:+.1f} п.п.**"]
        worst = sorted(sb["contrib"].items(), key=lambda kv: kv[1])[:2]
        txt.append("Главные факторы риска для нового вида: " + "; ".join(f"{FEAT_RU[k]} {v:+.1f} п.п." for k, v in worst if v < 0) if any(v < 0 for _, v in worst) else "Существенных факторов риска для нового вида нет.")
        txt.append("_Оценка; применить замену можно во вкладке «Растения»._")
        return "\n".join(txt), [z["id"]], "predict_survival"

    if any(w in t for w in ["первую очередь", "приоритет", "требуют озел", "в первую", "очеред"]):
        pr = W.priorities(weights)
        dists = sorted({zz["district"] for zz in W.zones})
        hit = [d for d in dists if _norm(d)[:5] in t]
        if hit:
            pr = pr[pr.district.isin(hit)]
        if pr.empty:
            return "В указанном районе зон нет.", [], "priority"
        top = pr.head(5)
        lines = [f"**Первоочередные участки{(' — ' + ', '.join(hit)) if hit else ' по городу'}:**"]
        for r in top.itertuples():
            zz = W.zone(r.id)
            main = max([k for k, _, _ in FACTORS], key=lambda k: getattr(r, f"c_{k}"))
            lines.append(f"- **{r.id}** ({r.purpose}, {r.district}) — приоритет **{r.priority}**; главный фактор: {dict((k, n) for k, n, _ in FACTORS)[main]}; проблем: {len(zz['problems'])}")
        if not hit:
            lines.append(f"\nРайоны: {', '.join(dists)}. Уточните район, например «в Северном районе».")
        return "\n".join(lines), list(top.id), "priority"

    if "проблем" in t:
        df = W.problems_df()
        if df.empty:
            return "Проблемных зон не выявлено.", [], "problems"
        hi = df[df["Острота"] == "высокая"]
        cnt = df["Проблема"].value_counts()
        lines = [f"Выявлено проблем: **{len(df)}** (высокой остроты — {len(hi)}).", "По типам: " + "; ".join(f"{k} — {v}" for k, v in cnt.items())]
        for r in hi.head(5).itertuples():
            lines.append(f"- **{r.Зона}**: {r.Проблема} — {r.Причина}")
        return "\n".join(lines), list(hi.head(5)["Зона"]), "problems"

    if any(w in t for w in ["подбер", "растени", "смет", "стоим", "посчита", "посад"]):
        if not z:
            return "Выберите зону на карте или укажите номер: «Подбери растения для зоны 3».", [], "build_plan"
        ym = re.search(r"на\s*(\d)\s*(?:год|лет)", t)
        cy = max(1, min(3, int(ym.group(1)))) if ym else None
        p = W.plan(z["id"], "Оптимальный", care_years=cy)
        lines = [f"**{z['id']} — {z['purpose']}**, {z['area_m2']:,.0f} м², свободно для посадки {p['plantable_m2']:,.0f} м².".replace(",", " "),
                 "Подобранные растения:"]
        for l in p["lines"]:
            lines.append(f"- {l['name']} — {l['qty']} {l['unit']}, пригодность {l['score']}/100, приживаемость 3 г. {l['surv']['p3']:.0%}")
        lines.append(f"\n**Смета (уход {p['V']['care_years']} г.): {fmt_tg(p['estimate']['total'])}**")
        for r in p["estimate"]["table"].itertuples():
            lines.append(f"  - {r.Статья}: {fmt_tg(r._2)}")
        for w in p["warnings"]:
            lines.append(f"⚠ {w}")
        lines.append("_Оценка; итог утверждает специалист._")
        return "\n".join(lines), [z["id"]], "build_plan + estimate"

    if "прижив" in t:
        if not z:
            return "Выберите зону, чтобы оценить приживаемость.", [], "predict_survival"
        p = W.plan(z["id"])
        a = p["avg_surv"]
        if a["p1"] is None:
            return "В плане зоны нет деревьев и кустарников — оценивать нечего.", [z["id"]], "predict_survival"
        worst = min(p["lines"], key=lambda l: l["surv"]["p3"])
        return (f"Средняя приживаемость в {z['id']}: {a['p1']:.0%} / {a['p2']:.0%} / {a['p3']:.0%} (1/2/3 года). "
                f"Наиболее рискованный вид — {worst['name']} ({worst['surv']['p3']:.0%} к 3-му году). _Оценка._"), [z["id"]], "predict_survival"

    if any(w in t for w in ["эффект", "co2", "со2", "кислород", "углекисл"]):
        if not z:
            return "Выберите зону, чтобы оценить экологический эффект.", [], "eco_effect"
        e = W.plan(z["id"])["eco"]
        y = e["by_year"]
        return (f"Эффект в {z['id']} (вариант «Оптимальный», оценка): CO₂ — {y[5]['co2_t']:.1f} / {y[10]['co2_t']:.1f} / {y[20]['co2_t']:.1f} т/год "
                f"через 5/10/20 лет; снижение t° через 20 лет ≈ {y[20]['cooling_c']:.1f} °C; задержание осадков ≈ {y[20]['rain_m3']:.0f} м³/год. "
                f"Стоимость ≈ {fmt_tg(e['cost_per_t'] or 0)} за 1 т CO₂ (за 20 лет)."), [z["id"]], "eco_effect"

    if any(w in t for w in ["уход", "полив", "календар"]):
        if not z:
            return "Выберите зону для плана ухода.", [], "care_calendar"
        df, warns = care_calendar(W.plan(z["id"]), z)
        summer = df[(df["Операция"] == "Полив") & df["Месяц"].isin(["Июн", "Июл", "Авг"])]
        return (f"План ухода для {z['id']}: {len(df)} операций в год. Полив летом: " + ", ".join(f"{r.Месяц} — {r.Периодичность}" for r in summer.itertuples())
                + ("\n⚠ " + "\n⚠ ".join(warns) if warns else "") + "\nПолный календарь — вкладка «Приживаемость и уход»."), [z["id"]], "care_calendar"

    sp = find_species(t, W.C["plants"])
    if sp:
        p = sp[0]
        return (f"**{p['name']}** (_{p['latin']}_), {p['type']}: зона зимостойкости USDA {p['usda']}, засухоустойчивость {p['drought']}/5, "
                f"солеустойчивость до {p['salt_tol']} дСм/м, pH {p['ph_min']}–{p['ph_max']}, высота {p['height_m']} м, крона {p['crown_m']} м, "
                f"цена {fmt_tg(p['price'])}, CO₂ ≈ {p['co2_kg']} кг/год. Аллерген: {'да' if p['allergen'] else 'нет'}, ядовит: {'да' if p['toxic'] else 'нет'}. "
                f"Источник: {p.get('source', '')}"), [], "catalog_search"

    return ("Я помогаю с вопросами по зонам озеленения. Примеры:\n- «Какие участки в Северном районе требуют озеленения в первую очередь?»\n"
            "- «Подбери растения для сквера в зоне 3 и посчитай стоимость на 3 года»\n- «Какая будет приживаемость, если заменить клён на липу?»\n"
            "- «Что можно озеленить на 5 млн при максимальном эффекте?»\n- «Покажи проблемные зоны», «Какой эффект по CO₂ в зоне 2?»\n"
            "Если данных для ответа нет, я так и скажу."), [], "help"


# =============================================================================
# 6. ОТЧЁТЫ
# =============================================================================
def plan_tables(plan):
    sel = pd.DataFrame([{"Вид": l["name"], "Ярус": l["type"], "Кол-во": l["qty"], "Ед.": l["unit"], "Пригодность": l["score"],
                         "Цена, ₸": l["plant"]["price"], "Обоснование": "; ".join(l["why"])} for l in plan["lines"]])
    surv = pd.DataFrame([{"Вид": l["name"], "1 год": round(l["surv"]["p1"], 3), "2 год": round(l["surv"]["p2"], 3),
                          "3 год": round(l["surv"]["p3"], 3), "Диапазон (1 г.)": f"{l['surv']['lo']:.0%}–{l['surv']['hi']:.0%}",
                          "Уверенность": l["surv"]["conf"],
                          "Главные риски": "; ".join(f"{FEAT_RU[k]} {v:+.1f} п.п." for k, v in sorted(l["surv"]["contrib"].items(), key=lambda kv: kv[1])[:3] if v < -0.5) or "—"}
                         for l in plan["lines"]])
    e = plan["eco"]["by_year"]
    eco = pd.DataFrame([{"Горизонт": f"{Y} лет", "CO₂, т/год": round(v["co2_t"], 2), "O₂, т/год": round(v["o2_t"], 2),
                         "Пыль, кг/год": round(v["dust_kg"], 1), "Снижение t°, °C": round(v["cooling_c"], 2),
                         "Осадки, м³/год": round(v["rain_m3"]), "Прирост зелени, м²": round(v["green_add_m2"]),
                         "Доля зелени после": f"{v['green_share_after']:.0%}"} for Y, v in e.items()])
    return sel, surv, eco


def assumptions_df(W):
    M = train_survival_model()
    rows = [("Данные", "Зоны, почвы, насаждения, LST, население, соцобъекты — синтетика (до получения данных организатора); дороги и соцобъекты — OSM при загрузке"),
            ("Приживаемость", f"Гибрид: 50% экспертные правила (логит) + 50% GradientBoosting. Обучение на синтетике ({M['n_train']} строк), ROC-AUC на тесте {M['auc']} (правила: {M['auc_rule']}). Метрика показывает воспроизведение закономерностей генератора, а не реальную точность — требуется калибровка на инвентаризациях акимата."),
            ("Подбор", "Жёсткие ограничения (зима, засоление, pH, безопасность для детей, расстояние до зданий, загрязнение) + штрафы, пригодность 0–100"),
            ("Эффект", "Коэффициенты на вид для взрослого дерева × кривая роста × выживаемость; температура — от покрытия кронами"),
            ("Смета", "Расценки из справочника (синтетика) — заменить на сметные нормы РК / прайс")]
    rows += [(f"Коэффициент: {r.name}", f"{r.value} {r.unit} — источник: {r.source}") for r in pd.DataFrame(_COEFFS, columns=["key", "name", "value", "unit", "source"]).itertuples()]
    return pd.DataFrame(rows, columns=["Пункт", "Описание"])


def make_excel(W, zid, plan, prio_df, care_df):
    bio = io.BytesIO()
    sel, surv, eco = plan_tables(plan)
    with pd.ExcelWriter(bio, engine="openpyxl") as xw:
        pd.DataFrame({"": [DISCLAIMER, f"Зона {zid}, вариант «{plan['variant']}»", f"Сформировано {datetime.datetime.now():%d.%m.%Y %H:%M}"]}).to_excel(xw, sheet_name="Титул", index=False)
        sel.to_excel(xw, sheet_name="Подбор", index=False)
        plan["estimate"]["table"].to_excel(xw, sheet_name="Смета", index=False)
        surv.to_excel(xw, sheet_name="Приживаемость", index=False)
        care_df.to_excel(xw, sheet_name="Уход", index=False)
        eco.to_excel(xw, sheet_name="Эффект", index=False)
        W.problems_df().to_excel(xw, sheet_name="Проблемные зоны", index=False)
        prio_df.to_excel(xw, sheet_name="Приоритеты", index=False)
        assumptions_df(W).to_excel(xw, sheet_name="Допущения", index=False)
    return bio.getvalue()


def _find_font():
    cands = []
    try:
        import matplotlib
        cands.append(os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf"))
    except Exception:
        pass
    cands += ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf",
              r"C:\Windows\Fonts\arial.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial Unicode.ttf"]
    cands += glob.glob("/usr/share/fonts/**/DejaVuSans.ttf", recursive=True)
    return next((c for c in cands if os.path.exists(c)), None)


def _pdf_safe(x):
    """Убирает эмодзи (нет в шрифте) и экранирует спецсимволы для reportlab."""
    t = str(x).replace("🧪 ", "").replace("⚠️ ", "").replace("⚠ ", "")
    t = "".join(ch for ch in t if ord(ch) < 0x2600 or ch in "₸")
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def make_pdf(W, zid, plan, prio_df, care_df):
    """PDF-отчёт (нужны reportlab и шрифт с кириллицей). Возвращает bytes или None."""
    try:
        from reportlab.lib.pagesizes import A4, landscape
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
    except ImportError:
        return None
    font = _find_font()
    if not font:
        return None
    pdfmetrics.registerFont(TTFont("Cyr", font))
    ss = getSampleStyleSheet()
    H = ParagraphStyle("h", parent=ss["Heading2"], fontName="Cyr")
    Pn = ParagraphStyle("p", parent=ss["Normal"], fontName="Cyr", fontSize=8.5, leading=11)
    Cs = ParagraphStyle("c", parent=Pn, fontSize=7, leading=8.5)

    def tbl(df, widths=None):
        data = [[Paragraph(_pdf_safe(c), Cs) for c in df.columns]] + [[Paragraph(_pdf_safe(v), Cs) for v in row] for row in df.astype(str).values.tolist()]
        t = Table(data, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.3, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8f5e9")),
                               ("VALIGN", (0, 0), (-1, -1), "TOP")]))
        return t

    z = W.zone(zid)
    sel, surv, eco = plan_tables(plan)
    bio = io.BytesIO()
    doc = SimpleDocTemplate(bio, pagesize=landscape(A4), leftMargin=24, rightMargin=24, topMargin=24, bottomMargin=24,
                            title="Отчёт по озеленению")
    el = [Paragraph(_pdf_safe(f"Отчёт по озеленению: зона {zid} ({z['purpose']}, {z['district']})"), H),
          Paragraph(_pdf_safe(DISCLAIMER), Pn), Paragraph(_pdf_safe(f"Сформировано: {datetime.datetime.now():%d.%m.%Y %H:%M}. Вариант: «{plan['variant']}». "
                                               f"Площадь {round(z['area_m2']):,} м²".replace(",", " ") + f"; почва: {z['soil_type']}, засоление {z['salinity']} дСм/м, pH {z['ph']}; "
                                               f"доля зелени {z['green_share']:.0%}; ближайшая дорога: {z['road_name']} ({z['road_class_ru']}, {z['dist_road']:.0f} м). "
                                               f"Источник атрибутов: {z.get('src', '')}"), Pn), Spacer(1, 6),
          Paragraph("1. Подбор растений", H), tbl(sel.drop(columns=["Обоснование"]).assign(Обоснование=sel["Обоснование"].str.slice(0, 160))),
          Paragraph("2. Смета", H), tbl(plan["estimate"]["table"].assign(Доля=lambda d: (d["Доля"] * 100).round(1).astype(str) + "%")),
          Paragraph(f"Итого: {fmt_tg(plan['estimate']['total'])}", Pn),
          Paragraph("3. Прогноз приживаемости", H), tbl(surv),
          Paragraph("4. Экологический эффект (оценка)", H), tbl(eco),
          Paragraph(f"CO₂ за 20 лет ≈ {plan['eco']['co2_cum20_t']:.1f} т; стоимость ≈ {fmt_tg(plan['eco']['cost_per_t'] or 0)} за 1 т CO₂.", Pn),
          PageBreak(), Paragraph("5. Календарь ухода (1-й год)", H), tbl(care_df),
          PageBreak(), Paragraph("6. Проблемные зоны города", H), tbl(W.problems_df().head(40)),
          Paragraph("7. Приоритеты (топ-15)", H),
          tbl(prio_df.head(15)[["Ранг", "id", "district", "purpose", "priority"] + [f"c_{k}" for k, _, _ in FACTORS]]
              .rename(columns={"id": "Зона", "district": "Район", "purpose": "Назначение", "priority": "Индекс",
                               **{f"c_{k}": n for k, n, _ in FACTORS}})),
          Paragraph("8. Допущения и источники", H), tbl(assumptions_df(W))]
    doc.build(el)
    return bio.getvalue()


def make_html(W, zid, plan, prio_df, care_df):
    sel, surv, eco = plan_tables(plan)
    css = "body{font-family:Arial,sans-serif;margin:24px}table{border-collapse:collapse;font-size:12px;margin:8px 0}td,th{border:1px solid #ccc;padding:3px 6px}th{background:#e8f5e9}"
    parts = [f"<html><head><meta charset='utf-8'><title>Отчёт {zid}</title><style>{css}</style></head><body>",
             f"<h1>Отчёт по озеленению: зона {zid}</h1><p><b>{DISCLAIMER}</b></p>"]
    for h, df in [("Подбор растений", sel), ("Смета", plan["estimate"]["table"]), ("Приживаемость", surv), ("Эффект", eco),
                  ("Уход", care_df), ("Проблемные зоны", W.problems_df()), ("Приоритеты", prio_df), ("Допущения", assumptions_df(W))]:
        parts.append(f"<h2>{h}</h2>" + df.to_html(index=False))
    parts.append("</body></html>")
    return "".join(parts).encode("utf-8")


def make_passport_html(W, zid, plan):
    """«Паспорт объекта озеленения» — печатная форма (браузер → Печать → PDF)."""
    z = W.zone(zid)
    sel, surv, eco = plan_tables(plan)
    e10 = plan["eco"]["by_year"][10]
    css = ("body{font-family:Arial,sans-serif;margin:32px;color:#0f172a}h1{font-size:18px;border-bottom:2px solid #059669;padding-bottom:6px}"
           "h3{font-size:14px;margin-top:20px}table{border-collapse:collapse;width:100%;font-size:12px}td,th{border:1px solid #cbd5e1;padding:4px 6px;text-align:left}"
           "th{background:#ecfdf5}.note{font-size:11px;color:#64748b;margin-top:18px}")
    soil = pd.DataFrame([("Адрес / район", f"{z['name']}, {z['district']} район"), ("Назначение", z["purpose"]),
                         ("Площадь участка", f"{round(z['area_m2']):,} м²".replace(",", " ")),
                         ("Площадь под посадку", f"{round(plan['plantable_m2']):,} м²".replace(",", " ")),
                         ("Доля зелени сейчас", f"{z['green_share']:.0%}"),
                         ("Почва", f"{z['soil_type']}, pH {z['ph']}, засоление {z['salinity']} дСм/м, плодородие {z['fertility']}/5, влажность {z['moisture']}/5"),
                         ("Ближайшая дорога", f"{z['road_name']} ({z['road_class_ru']}), {z['dist_road']:.0f} м"),
                         ("Соцобъекты в 500 м (OSM)", f"{z['social_n']} ({z['social_list']})" + (f": {z.get('social_names')}" if z.get("social_names") else "")),
                         ("Источник данных", z.get("src", ""))], columns=["Параметр", "Значение"])
    summ = pd.DataFrame([("Вариант", plan["variant"]), ("Итого по смете", fmt_tg(plan["estimate"]["total"])),
                         ("Приживаемость 1/2/3 г.", " / ".join(f"{plan['avg_surv'][k]:.0%}" for k in ("p1", "p2", "p3")) if plan["avg_surv"]["p1"] is not None else "—"),
                         ("CO₂ через 10 лет", f"{e10['co2_t']:.2f} т/год"), ("Снижение t° через 10 лет", f"{e10['cooling_c']:.2f} °C"),
                         ("Стоимость 1 т CO₂ (20 лет)", fmt_tg(plan["eco"]["cost_per_t"] or 0))], columns=["Показатель", "Значение"])
    h = [f"<html><head><meta charset='utf-8'><title>Паспорт объекта озеленения {zid}</title><style>{css}</style></head><body>",
         f"<h1>ПАСПОРТ ОБЪЕКТА ОЗЕЛЕНЕНИЯ: {z['name'].upper()} ({zid})</h1>",
         f"<p>Отдел благоустройства и озеленения, г. Павлодар · сформировано {datetime.datetime.now():%d.%m.%Y %H:%M}</p>",
         "<h3>1. Физико-географические и почвенные параметры</h3>", soil.to_html(index=False),
         "<h3>2. Спецификация посадочного материала</h3>", sel.drop(columns=["Обоснование"]).to_html(index=False),
         "<h3>3. Смета</h3>", plan["estimate"]["table"].drop(columns=["Доля"]).to_html(index=False),
         "<h3>4. Сводка: приживаемость и экологический эффект (оценка)</h3>", summ.to_html(index=False),
         "<h3>5. Прогноз приживаемости по видам</h3>", surv.to_html(index=False),
         f"<p class='note'>{DISCLAIMER} Расценки и коэффициенты — из справочников системы, источники указаны там. "
         f"Данные с пометкой «синтетика» — демонстрационные.</p></body></html>"]
    return "".join(h).encode("utf-8")


# Слой дорог для карты: реальная геометрия OSM грузится В БРАУЗЕРЕ (Overpass API) и кэшируется в localStorage.
# Подложка Voyager без подписей + слой подписей поверх дорог (текст идёт по изгибу улиц).
ROADS_JS = r"""
(function(){
  var map = %MAP%;
  var BBOX = [52.215, 76.880, 52.360, 77.110];
  var URLS = ['https://overpass-api.de/api/interpreter','https://overpass.kumi.systems/api/interpreter','https://overpass.private.coffee/api/interpreter'];
  var CACHE = 'pvl_osm_roads_v2';
  var CL = {
    motorway:{ru:'Магистраль скоростного движения',w:7,c:'#e8590c',z:9,mz:0}, trunk:{ru:'Магистраль общегородского значения',w:7,c:'#e8590c',z:9,mz:0},
    primary:{ru:'Магистраль общегородского значения',w:6,c:'#f59f00',z:8,mz:0}, secondary:{ru:'Улица районного значения',w:5,c:'#fcc419',z:7,mz:0},
    tertiary:{ru:'Улица районного значения',w:4.5,c:'#ffe066',z:6,mz:12}, unclassified:{ru:'Местный проезд',w:3,c:'#ffffff',z:5,mz:14},
    residential:{ru:'Жилая улица',w:3,c:'#ffffff',z:5,mz:14}, living_street:{ru:'Жилая зона',w:2.5,c:'#f8f9fa',z:4,mz:15},
    pedestrian:{ru:'Пешеходная улица',w:2.5,c:'#fcc2d7',z:3,mz:15}, service:{ru:'Внутриквартальный проезд',w:1.8,c:'#f1f3f5',z:2,mz:16},
    footway:{ru:'Пешеходная дорожка',w:1.2,c:'#e64980',z:1,mz:16}
  };
  var MAIN = 'motorway|motorway_link|trunk|trunk_link|primary|primary_link|secondary|secondary_link|tertiary|tertiary_link|unclassified|residential|living_street|pedestrian';
  var MINOR = 'service|footway';
  var ROADS = [], ids = {}, cells = {};
  map.createPane('gzRoads'); map.getPane('gzRoads').style.zIndex = 350;
  map.createPane('gzLabels'); map.getPane('gzLabels').style.zIndex = 450; map.getPane('gzLabels').style.pointerEvents = 'none';
  if (%LABELS%) L.tileLayer('https://{s}.basemaps.cartocdn.com/rastertiles/voyager_only_labels/{z}/{x}/{y}{r}.png', {pane:'gzLabels', maxZoom:19, subdomains:'abcd'}).addTo(map);
  var R = L.canvas({pane:'gzRoads', padding:0.5, tolerance:6});
  var group = L.layerGroup().addTo(map);
  var st = L.control({position:'bottomright'});
  var stEl; st.onAdd = function(){ stEl = L.DomUtil.create('div'); stEl.style.cssText='background:rgba(15,23,42,.88);color:#fff;font:11px Inter,Arial,sans-serif;padding:4px 9px;border-radius:999px'; return stEl; }; st.addTo(map);
  function status(t, ok){ if (stEl) stEl.innerHTML = '<span style="color:'+(ok===false?'#fca5a5':ok?'#86efac':'#fde68a')+'">●</span> '+t; }
  function cls(hw){ hw = (hw||'').replace('_link',''); return CL[hw] ? hw : null; }
  function q(query){
    var i = 0;
    function next(){
      if (i >= URLS.length) return Promise.reject('all failed');
      var u = URLS[i++];
      return fetch(u, {method:'POST', body:'data='+encodeURIComponent(query), headers:{'Content-Type':'application/x-www-form-urlencoded'}})
        .then(function(r){ if(!r.ok) throw r.status; return r.json(); }).catch(function(){ return next(); });
    }
    return next();
  }
  function add(els){
    var n = 0;
    (els||[]).forEach(function(el){
      if (el.type!=='way' || !el.geometry || el.geometry.length<2 || ids[el.id]) return;
      var t = el.tags||{}, c = cls(t.highway); if (!c) return;
      ids[el.id] = 1; n++;
      ROADS.push({id:el.id, hw:t.highway, c:c, name:t.name||'', lanes:t.lanes||'', oneway:t.oneway==='yes',
        bridge:t.bridge==='yes'||t.bridge==='viaduct', tunnel:t.tunnel==='yes', maxspeed:t.maxspeed||'',
        pts:el.geometry.map(function(p){return [Math.round(p.lat*1e6)/1e6, Math.round(p.lon*1e6)/1e6];})});
    });
    return n;
  }
  function len(p){ var s=0; for (var i=1;i<p.length;i++) s += map.distance(p[i-1], p[i]); return s; }
  function info(r, ll){
    var s = CL[r.c], L0 = len(r.pts), major = ['motorway','trunk','primary','secondary','tertiary'].indexOf(r.c) >= 0;
    L.popup().setLatLng(ll).setContent('<div style="font:12px/1.45 Inter,Arial,sans-serif"><b>'+(r.name||'Улица без названия')+'</b><br>Класс: '+s.ru+
      ' <span style="color:#868e96">(highway='+r.hw+')</span><br>Длина сегмента: '+Math.round(L0).toLocaleString('ru-RU')+' м'+
      (r.lanes?'<br>Полос: '+r.lanes:'')+(r.oneway?'<br>Одностороннее движение':'')+(r.bridge?'<br>Мост / путепровод':'')+
      (r.maxspeed?'<br>Ограничение скорости: '+r.maxspeed:'')+
      (major?'<hr style="margin:4px 0">Защитная полоса вдоль сегмента (оценка): ~'+(Math.floor(L0/8)*2)+' деревьев, ~'+(Math.floor(L0/0.7)*2).toLocaleString('ru-RU')+' кустарников':'')+
      '<div style="color:#868e96;font-size:10px;margin-top:3px">© OpenStreetMap contributors</div></div>').openOn(map);
  }
  function draw(){
    group.clearLayers();
    var zm = map.getZoom(), k = Math.max(0.45, Math.min(2.6, Math.pow(2, (zm-15)*0.55)));
    var vis = ROADS.filter(function(r){ return zm >= CL[r.c].mz; });
    [-1,0,1].forEach(function(lv){
      var g = vis.filter(function(r){ return (r.bridge?1:(r.tunnel?-1:0)) === lv; }).sort(function(a,b){ return CL[a.c].z - CL[b.c].z; });
      g.forEach(function(r){ var s = CL[r.c]; if (s.w >= 2.5) L.polyline(r.pts, {renderer:R, color: lv===1?'#343a40':'#868e96', weight: s.w*k+(lv===1?3:2),
        opacity: lv===-1?0.3:0.85, interactive:false, lineCap:'round', lineJoin:'round'}).addTo(group); });
      g.forEach(function(r){ var s = CL[r.c];
        var pl = L.polyline(r.pts, {renderer:R, color:s.c, weight:s.w*k, opacity: lv===-1?0.45:1,
          dashArray: lv===-1?'6 6':(r.c==='footway'?'3 4':null), lineCap:'round', lineJoin:'round'}).addTo(group);
        pl.bindTooltip((r.name||'без названия')+' · '+s.ru, {sticky:true, className:'gz-road-label'});
        pl.on('click', function(e){ info(r, e.latlng); });
      });
    });
  }
  function minor(){
    if (map.getZoom() < 16 || !ROADS.length) return;
    var b = map.getBounds(), step = 0.01, todo = [];
    for (var la = Math.floor(b.getSouth()/step)*step; la < b.getNorth(); la += step)
      for (var lo = Math.floor(b.getWest()/step)*step; lo < b.getEast(); lo += step) {
        var key = la.toFixed(2)+','+lo.toFixed(2); if (!cells[key]) { cells[key] = 1; todo.push([la, lo]); } }
    if (!todo.length) return;
    var query = '[out:json][timeout:60];(' + todo.map(function(c){ return 'way["highway"~"^('+MINOR+')$"]('+c[0].toFixed(4)+','+c[1].toFixed(4)+','+(c[0]+step).toFixed(4)+','+(c[1]+step).toFixed(4)+');'; }).join('') + ');out tags geom;';
    q(query).then(function(js){ if (add(js.elements)) draw(); }).catch(function(){ todo.forEach(function(c){ delete cells[c[0].toFixed(2)+','+c[1].toFixed(2)]; }); });
  }
  function ready(src){ status('Дороги OSM: '+ROADS.length+' сегментов'+(src?' ('+src+')':''), true); draw(); minor(); }
  map.on('zoomend', draw); map.on('moveend', minor);
  var cached = null;
  try { cached = JSON.parse(localStorage.getItem(CACHE)||'null'); } catch(e) {}
  if (cached && Date.now() - cached.t < 7*24*3600*1000 && cached.r && cached.r.length) { cached.r.forEach(function(r){ ids[r.id]=1; ROADS.push(r); }); ready('кэш'); return; }
  status('Загрузка улиц из OpenStreetMap…');
  q('[out:json][timeout:90];way["highway"~"^('+MAIN+')$"]('+BBOX.join(',')+');out tags geom;').then(function(js){
    add(js.elements);
    try { localStorage.setItem(CACHE, JSON.stringify({t:Date.now(), r:ROADS})); } catch(e) {}
    ready('');
  }).catch(function(){ status('Не удалось загрузить дороги OSM — улицы видны на подложке', false); });
})();
"""


# Слой соцобъектов в браузере (если сервер ещё не получил их из OSM): школы, детсады, больницы, поликлиники, ЖК.
SOCIAL_JS = r"""
(function(){
  var map = %MAP%;
  var BB = '52.215,76.880,52.360,77.110';
  var URLS = ['https://overpass-api.de/api/interpreter','https://overpass.kumi.systems/api/interpreter','https://overpass.private.coffee/api/interpreter'];
  var CACHE = 'pvl_osm_social_v2';
  var Q = '[out:json][timeout:60];' + %QUERY%.split('{bb}').join(BB);
  var ICO = {school:'🎓', kindergarten:'🧸', hospital:'🏥', clinic:'⚕️', residential:'🏢'};
  var RU = {school:'школа', kindergarten:'детсад', hospital:'больница', clinic:'поликлиника', residential:'жилой комплекс / жилой массив'};
  var g = L.layerGroup().addTo(map);
  function kind(t){ var a=t.amenity||'', h=t.healthcare||'';
    if (a==='school') return 'school'; if (a==='kindergarten') return 'kindergarten';
    if (a==='hospital'||h==='hospital') return 'hospital'; if (a==='clinic'||a==='doctors'||h==='clinic'||h==='centre') return 'clinic';
    return 'residential'; }
  function addr(t){ var s=((t['addr:street']||'')+' '+(t['addr:housenumber']||'')).trim(); return s || t['addr:full'] || 'адрес не указан в OSM'; }
  function esc(x){ return String(x).replace(/[&<>"]/g, function(c){ return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]; }); }
  function draw(items){
    g.clearLayers();
    items.forEach(function(p){
      if (p.k !== 'residential') L.circle([p.lat, p.lon], {radius:150, color:'#3b82f6', weight:1, dashArray:'4, 4', fillColor:'#60a5fa', fillOpacity:0.10, interactive:false}).addTo(g);
      L.marker([p.lat, p.lon], {icon: L.divIcon({className:'', html:'<div class="gz-poi">'+ICO[p.k]+'</div>', iconSize:[24,24], iconAnchor:[12,12]})})
        .bindTooltip('<b>'+esc(p.n)+'</b><br>'+RU[p.k]+' · '+esc(p.a)+(p.k!=='residential'?'<br><span style="color:#dc2626;font-size:10px">Буфер 150 м: без аллергенов и ядовитых видов</span>':'')+'<br><span style="color:#94a3b8;font-size:10px">© OpenStreetMap</span>')
        .addTo(g);
    });
  }
  function q(){ var i=0; function next(){ if (i>=URLS.length) return Promise.reject('fail'); var u=URLS[i++];
    return fetch(u,{method:'POST', body:'data='+encodeURIComponent(Q), headers:{'Content-Type':'application/x-www-form-urlencoded'}})
      .then(function(r){ if(!r.ok) throw r.status; return r.json(); }).catch(function(){ return next(); }); } return next(); }
  var c = null; try { c = JSON.parse(localStorage.getItem(CACHE)||'null'); } catch(e) {}
  if (c && Date.now()-c.t < 7*24*3600*1000 && c.p && c.p.length) { draw(c.p); return; }
  q().then(function(js){
    var seen = {}, items = [];
    (js.elements||[]).forEach(function(el){
      var ct = el.center || {lat:el.lat, lon:el.lon}; if (ct.lat == null) return;
      var t = el.tags||{}, k = kind(t), n = t.name || RU[k], key = k+n+ct.lat.toFixed(4)+ct.lon.toFixed(4);
      if (seen[key]) return; seen[key] = 1;
      items.push({lat:ct.lat, lon:ct.lon, k:k, n:n, a:addr(t)});
    });
    try { localStorage.setItem(CACHE, JSON.stringify({t:Date.now(), p:items})); } catch(e) {}
    draw(items);
  }).catch(function(){});
})();
"""


# =============================================================================
# 7. ВЕБ-ИНТЕРФЕЙС (Streamlit)
# =============================================================================
def main():
    import streamlit as st
    import folium
    from folium.plugins import Draw
    from branca.element import MacroElement, Element
    from jinja2 import Template
    try:
        from streamlit_folium import st_folium
    except ImportError:
        st_folium = None
    import streamlit.components.v1 as components

    st.set_page_config(page_title="AI-озеленение Павлодара", page_icon="🌳", layout="wide", initial_sidebar_state="collapsed")
    ss = st.session_state

    # --- совместимость версий Streamlit ---
    try:
        _ver = tuple(int(x) for x in re.findall(r"\d+", st.__version__)[:2])
    except Exception:
        _ver = (1, 40)
    WIDE = {"width": "stretch"} if _ver >= (1, 50) else {"use_container_width": True}

    def chart(data, kind="bar", grouped=False, height=280):
        """Графики через Vega-Lite без altair (altair 5 не импортируется на Python 3.14)."""
        df = data.to_frame() if isinstance(data, pd.Series) else data.copy()
        df = df.reset_index()
        x = df.columns[0]
        if kind == "bar":
            df[x] = df[x].astype(str)
        long = df.melt(id_vars=x, var_name="Серия", value_name="Значение")
        multi = long["Серия"].nunique() > 1
        enc = {"x": {"field": x, "type": "nominal" if kind == "bar" else "quantitative", "sort": None, "title": None},
               "y": {"field": "Значение", "type": "quantitative", "title": None,
                     **({"stack": None} if (grouped or (kind != "bar" and multi)) else {})},
               "tooltip": [{"field": x}, {"field": "Серия"}, {"field": "Значение", "format": ",.2f"}]}
        if multi:
            enc["color"] = {"field": "Серия", "type": "nominal", "title": None, "legend": {"orient": "bottom"},
                            "scale": {"range": ["#059669", "#0d5c9c", "#f59e0b", "#e11d48", "#7c3aed", "#0891b2"]}}
            if grouped:
                enc["xOffset"] = {"field": "Серия"}
        mark = {"type": kind, "tooltip": True, **({"point": True} if kind == "line" else {}),
                **({"color": "#059669"} if not multi else {}), **({"cornerRadiusEnd": 3} if kind == "bar" else {})}
        try:
            st.vega_lite_chart(long, {"mark": mark, "encoding": enc, "height": height}, **WIDE)
        except Exception:
            st.dataframe(data, **WIDE)

    @st.cache_resource(show_spinner="Обучение модели приживаемости…")
    def _cached_model():
        return train_survival_model()

    _MODEL["m"] = _cached_model()

    @st.cache_resource(show_spinner=False)
    def _roads_loader():
        """Дороги OSM для расчётов грузятся в ФОНОВОМ потоке, чтобы страница открывалась сразу.
        Общий объект для всех пользователей; пока граф не загружен, расчёты помечены как синтетика."""
        holder = {"roads": load_cached("osm_roads.json"), "status": "ok", "err": "",
                  "social": load_cached("osm_social_v2.json"), "social_status": "ok"}

        def _run():
            if not holder["roads"]:
                holder["roads"], holder["status"] = None, "loading"
                try:
                    holder["roads"] = fetch_osm_roads(timeout=40)
                    holder["status"] = "ok"
                except Exception as e:  # noqa
                    holder["status"], holder["err"] = "error", str(e)[:200]
            if not holder["social"]:
                holder["social"], holder["social_status"] = None, "loading"
                try:
                    holder["social"] = fetch_osm_social(timeout=40)
                    holder["social_status"] = "ok"
                except Exception:  # noqa
                    holder["social_status"] = "error"

        if not holder["roads"] or not holder["social"]:
            if not holder["roads"]:
                holder["status"] = "loading"
            if not holder["social"]:
                holder["social_status"] = "loading"
            import threading
            threading.Thread(target=_run, daemon=True).start()
        return holder

    # ---------- стиль (по мотивам HTML-макета) ----------
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea, select { font-family: 'Inter', sans-serif; }
header[data-testid="stHeader"] { background: transparent; height: 2.2rem; }
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; max-width: 100%; }
.gz-head { background:#0f172a; color:#fff; border-radius:12px; padding:12px 16px; display:flex; align-items:center; justify-content:space-between; gap:12px; flex-wrap:wrap; }
.gz-logo { width:38px; height:38px; border-radius:10px; background:linear-gradient(45deg,#059669,#34d399); display:flex; align-items:center; justify-content:center; font-size:20px; }
.gz-title { font-weight:700; font-size:16px; }
.gz-badge { font:500 10px 'JetBrains Mono',monospace; background:rgba(16,185,129,.18); color:#6ee7b7; border:1px solid rgba(16,185,129,.35); padding:1px 7px; border-radius:5px; margin-left:8px; }
.gz-sub { font-size:11px; color:#94a3b8; margin-top:2px; }
.gz-sub b { color:#34d399; font-weight:500; }
.gz-right { text-align:right; font-size:11px; color:#94a3b8; }
.gz-right .v { font:700 13px 'JetBrains Mono',monospace; color:#fbbf24; }
.gz-role { display:inline-block; margin-left:10px; background:#1e293b; border:1px solid #334155; color:#cbd5e1; padding:3px 9px; border-radius:999px; font-size:11px; }
div.st-key-page { margin-top: 8px; }
div.st-key-page div[role="radiogroup"] { background:#1e293b; padding:4px; border-radius:10px; gap:2px; flex-wrap:wrap; }
div.st-key-page div[role="radiogroup"] label { background:transparent; padding:5px 11px; border-radius:7px; margin:0 !important; cursor:pointer; }
div.st-key-page div[role="radiogroup"] label p { color:#94a3b8 !important; font-size:12.5px; font-weight:500; }
div.st-key-page div[role="radiogroup"] label:hover p { color:#fff !important; }
div.st-key-page div[role="radiogroup"] label:has(input:checked) { background:#334155; box-shadow:0 1px 2px rgba(0,0,0,.3); }
div.st-key-page div[role="radiogroup"] label:has(input:checked) p { color:#fff !important; }
div.st-key-page div[role="radiogroup"] label > div:first-child { display:none; }
.gz-card { background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:9px 11px; height:100%; }
.gz-card .l { font-size:10.5px; color:#94a3b8; }
.gz-card .v { font:700 15px 'JetBrains Mono',monospace; color:#1e293b; }
.gz-card .s { font-size:10.5px; color:#64748b; }
.gz-grid { display:grid; grid-template-columns:1fr 1fr; gap:8px; margin:8px 0; }
.gz-panel-h { font-size:11px; font-weight:700; letter-spacing:.06em; text-transform:uppercase; color:#047857; }
.gz-panel-t { font-size:17px; font-weight:700; color:#1e293b; margin:2px 0 4px; }
.gz-sev { display:inline-block; padding:2px 8px; border-radius:6px; font-size:11px; font-weight:700; }
.gz-sev.h { background:#fee2e2; color:#b91c1c; border:1px solid #fecaca; } .gz-sev.m { background:#fef3c7; color:#b45309; border:1px solid #fde68a; } .gz-sev.l { background:#d1fae5; color:#047857; border:1px solid #a7f3d0; }
.gz-ai { background:rgba(236,253,245,.7); border:1px solid #a7f3d0; border-radius:12px; padding:11px 13px; margin:8px 0; font-size:12px; color:#475569; }
.gz-ai .h { display:flex; justify-content:space-between; font-weight:700; color:#064e3b; font-size:12.5px; margin-bottom:5px; }
.gz-ai .tag { background:rgba(167,243,208,.7); color:#065f46; font-size:10px; padding:1px 6px; border-radius:5px; }
.gz-ai .row { display:flex; justify-content:space-between; font-size:11.5px; padding-top:3px; }
.gz-ai .row b { font-family:'JetBrains Mono',monospace; color:#065f46; }
.gz-box { background:#fff; border:1px solid #e2e8f0; border-radius:12px; padding:11px 13px; margin:8px 0; box-shadow:0 1px 2px rgba(0,0,0,.04); }
.gz-box .h { display:flex; justify-content:space-between; font-weight:700; color:#1e293b; font-size:12.5px; }
.gz-box .h span { font:500 10px 'JetBrains Mono',monospace; color:#94a3b8; }
.gz-surv { display:grid; grid-template-columns:repeat(3,1fr); gap:8px; text-align:center; margin:8px 0; }
.gz-surv div { background:#f8fafc; border:1px solid #f1f5f9; border-radius:8px; padding:6px; }
.gz-surv .y { font-size:10px; color:#64748b; } .gz-surv .p { font:800 17px 'JetBrains Mono',monospace; color:#059669; } .gz-surv .n { font-size:9px; color:#94a3b8; }
.gz-risk { display:flex; justify-content:space-between; font-size:11.5px; color:#475569; padding:1px 0; }
.gz-risk b { font-family:'JetBrains Mono',monospace; font-weight:600; }
.gz-dark { background:#0f172a; color:#cbd5e1; border-radius:12px; padding:12px 14px; margin:8px 0; font-size:11.5px; }
.gz-dark .t { display:flex; justify-content:space-between; color:#e2e8f0; font-size:12.5px; margin-bottom:6px; }
.gz-dark .t b { font-family:'JetBrains Mono',monospace; color:#fbbf24; }
.gz-dark .r { display:flex; justify-content:space-between; padding:2px 0; } .gz-dark .r span:first-child { color:#94a3b8; } .gz-dark .r span:last-child { font-family:'JetBrains Mono',monospace; }
.gz-foot { font-size:10.5px; color:#94a3b8; font-style:italic; text-align:center; }
.gz-pill { background:rgba(15,23,42,.92); color:#fff; font-size:11.5px; border-radius:999px; padding:6px 14px; display:inline-flex; gap:8px; align-items:center; margin-top:6px; }
.gz-pill .dot { width:8px; height:8px; border-radius:50%; background:#34d399; display:inline-block; }
.gz-pill b { color:#fcd34d; } .gz-pill .m { color:#94a3b8; }
.gz-kpi { background:#fff; border:1px solid #e2e8f0; border-radius:12px; padding:13px 15px; height:100%; }
.gz-kpi .l { font-size:10.5px; text-transform:uppercase; letter-spacing:.05em; color:#64748b; font-weight:700; }
.gz-kpi .v { font:800 22px 'JetBrains Mono',monospace; color:#0f172a; margin:3px 0; }
.gz-kpi .s { font-size:11px; color:#059669; }
.gz-scn { border:1px solid #e2e8f0; border-radius:12px; padding:12px 14px; background:#fff; height:100%; font-size:12px; color:#475569; }
.gz-scn.best { border:2px solid #10b981; background:#f0fdf4; }
.gz-scn .n { font-weight:700; color:#0f172a; font-size:13px; } .gz-scn .c { font:800 18px 'JetBrains Mono',monospace; color:#0f172a; margin:4px 0; }
.gz-scn .tag { background:#10b981; color:#fff; font-size:10px; padding:1px 7px; border-radius:5px; font-weight:700; }
.gz-opt { background:#0f172a; color:#e2e8f0; border-radius:12px; padding:14px 16px; font-size:12px; }
.gz-opt .h { font-weight:700; font-size:13px; color:#f1f5f9; margin-bottom:8px; }
.gz-opt .r { display:flex; justify-content:space-between; padding:3px 0; border-bottom:1px solid #1e293b; } .gz-opt .r b { font-family:'JetBrains Mono',monospace; }
.gz-opt .g { color:#34d399; margin-top:8px; }
.gz-layers { font-size:12px; font-weight:700; color:#1e293b; margin-bottom:-6px; }
</style>""", unsafe_allow_html=True)

    # ---------- состояние ----------
    # Каждый ключ инициализируется отдельно: после обновления кода на Streamlit Cloud старая сессия
    # сохраняется, и новых полей в ней может не быть.
    defaults = dict(base=generate_base_data, plants_df=default_plants_df, rates_df=default_rates_df, coeffs_df=default_coeffs_df,
                    extra_zones=list, data_ver=lambda: 0, world_ver=lambda: -1, selected=lambda: "Z-01", ov=dict, history=list,
                    decisions=list, weights=lambda: {k: w for k, _, w in FACTORS}, last_tip=lambda: None, processed_drawings=set,
                    osm_roads=lambda: None, osm_social=lambda: None, found_point=lambda: None,
                    budget=lambda: 45_000_000, zone_ai=lambda: None, app_ver=lambda: 0)
    for k_, f_ in defaults.items():
        if k_ not in ss:
            ss[k_] = f_()
    if ss.get("page") not in PAGES:
        ss.page = PAGES[0]
    roads_holder = _roads_loader()
    if ss.osm_roads is None and roads_holder["roads"]:
        ss.osm_roads = roads_holder["roads"]
        ss.data_ver += 1
    if (ss.osm_social is None or (ss.osm_social and "addr" not in ss.osm_social[0])) and roads_holder["social"]:
        ss.osm_social = roads_holder["social"]
        ss.data_ver += 1
    if ss.app_ver != 3:  # соцобъекты стали реальными — пересобрать расчёты
        ss.app_ver = 3
        ss.world_ver = -1

    def bump():
        ss.data_ver += 1

    if ss.world_ver != ss.data_ver:
        with st.spinner("Пересчёт аналитики…"):
            ss.world = World(ss.base, ss.plants_df, ss.rates_df, ss.coeffs_df, ss.osm_roads, ss.osm_social, ss.extra_zones)
        ss.world_ver = ss.data_ver
    W = ss.world
    C = W.C
    if not W.zone(ss.selected):
        ss.selected = W.zones[0]["id"]

    def goto_zone(zid):
        ss.selected = zid
        ss.page = PAGES[0]

    def ask(q):
        ans, zl, tool = assistant_answer(q, W, ss.selected, ss.weights)
        ss.history.append(dict(t=datetime.datetime.now().strftime("%H:%M:%S"), q=q, a=ans, zones=zl, tool=tool))
        return ans

    def ov_of(zid):
        return ss.ov.setdefault(zid, {"fixed": [], "excluded": [], "replace": {}})

    # ---------- боковая панель: роль + чат ----------
    with st.sidebar:
        st.markdown("### 🤖 AI-консультант\nGIS-ассистент отдела благоустройства")
        role = st.selectbox("Роль (демо)", ["Специалист", "Администратор", "Просмотр"], key="role",
                            help="В MVP роль выбирается вручную. Аутентификация и HTTPS — на уровне развёртывания.")
        qc = st.columns(3)
        for c_, (lab, q_) in zip(qc, [("🚨 Что в первую очередь?", "Какие участки требуют озеленения в первую очередь?"),
                                      ("💰 План на 35 млн ₸", "Что можно озеленить на 35 млн при максимальном эффекте?"),
                                      ("🌳 Почему не тополь?", "Почему не тополь для этой зоны?")]):
            if c_.button(lab, key="q_" + lab, **WIDE):
                ask(q_)
        with st.form("side_chat", clear_on_submit=True):
            q = st.text_input("Вопрос", placeholder="Спросите об участке, растениях, бюджете…", label_visibility="collapsed")
            if st.form_submit_button("Спросить", **WIDE) and q:
                ask(q)
        for i, h in list(enumerate(ss.history))[-3:][::-1]:
            st.markdown(f"**Вы:** {h['q']}")
            st.markdown(h["a"])
            if h["zones"]:
                cols = st.columns(min(4, len(h["zones"])))
                for j, zid in enumerate(h["zones"][:8]):
                    cols[j % len(cols)].button(f"📍{zid}", key=f"sb_{i}_{zid}", on_click=goto_zone, args=(zid,))
            st.divider()
        _rs = {"ok": "OSM", "loading": "OSM загружается в фоне — пока " + SYN, "error": "Overpass недоступен с сервера — " + SYN}[roads_holder["status"]]
        st.caption(f"Дороги на карте: OSM (загружаются в браузере). Аналитика по дорогам: {'OSM' if W.roads_real else _rs}. "
                   f"Школы, детсады, больницы, ЖК: OSM ({'загружены' if W.social_real else 'на карте — из браузера; для расчётов ещё загружаются'}). Зоны/почвы/насаждения: {SYN}.")

    # ---------- шапка ----------
    n_prob = int((W.problems_df()["Острота"] == "высокая").sum()) if len(W.problems_df()) else 0
    st.markdown(f"""
<div class="gz-head">
  <div style="display:flex;align-items:center;gap:12px">
    <div class="gz-logo">🌳</div>
    <div><div class="gz-title">AI-Озеленение Павлодара<span class="gz-badge">MVP 2026.11</span></div>
    <div class="gz-sub">Акимат г. Павлодар • Отдел благоустройства и озеленения • <b>СК: EPSG:32643 (UTM 43N)</b></div></div>
  </div>
  <div class="gz-right">Лимит бюджета благоустройства:<br><span class="v">{fmt_tg(ss.budget)}</span>
    <span class="gz-role">👤 {role}</span> <span class="gz-role" style="color:#fca5a5">⚠ {n_prob} критич. проблем</span></div>
</div>""", unsafe_allow_html=True)
    st.radio("Раздел", PAGES, key="page", horizontal=True, label_visibility="collapsed")
    st.caption(DISCLAIMER)
    page = ss.page

    # ---------- карта ----------
    class OsmRoads(MacroElement):
        def __init__(self, labels=True):
            super().__init__()
            self._name = "OsmRoads"
            self.labels = labels
            self._template = Template("{% macro script(this, kwargs) %}" + ROADS_JS.replace("%MAP%", "{{ this._parent.get_name() }}")
                                      .replace("%LABELS%", "true" if labels else "false") + "{% endmacro %}")

    class OsmSocial(MacroElement):
        def __init__(self):
            super().__init__()
            self._name = "OsmSocial"
            self._template = Template("{% macro script(this, kwargs) %}" + SOCIAL_JS.replace("%MAP%", "{{ this._parent.get_name() }}")
                                      .replace("%QUERY%", json.dumps(SOCIAL_QUERY)) + "{% endmacro %}")

    BASEMAPS = {
        "Voyager (OSM) + подписи поверх дорог": ("https://{s}.basemaps.cartocdn.com/rastertiles/voyager_nolabels/{z}/{x}/{y}{r}.png", "&copy; OpenStreetMap contributors &copy; CARTO", True),
        "OpenStreetMap (стандарт)": ("OpenStreetMap", None, False),
        "Светлая (Carto Positron)": ("https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png", "&copy; OpenStreetMap contributors &copy; CARTO", True),
        "2ГИС (растровые тайлы)": ("https://tile2.maps.2gis.com/tiles?x={x}&y={y}&z={z}&v=1", "© 2ГИС (проверьте условия использования)", False),
    }

    def build_map(layers, basemap, center=None, zoom=13, plan=None):
        url, attr, labels = BASEMAPS.get(basemap, list(BASEMAPS.values())[0])
        if attr:
            m = folium.Map(location=center or CENTER, zoom_start=zoom, tiles=url, attr=attr, max_zoom=19)
        else:
            m = folium.Map(location=center or CENTER, zoom_start=zoom, tiles=url, max_zoom=19)
        m.get_root().header.add_child(Element("""<style>
.gz-road-label{background:rgba(15,23,42,.8);border:none;color:#fff;font:600 10px Inter,Arial,sans-serif;padding:1px 5px;border-radius:3px;box-shadow:0 1px 3px rgba(0,0,0,.3)}
.gz-road-label:before{display:none}
.gz-pulse{width:14px;height:14px;border-radius:50%;background:#dc2626;border:2px solid #fff;animation:gzp 1.8s cubic-bezier(.215,.61,.355,1) infinite}
@keyframes gzp{0%{transform:scale(.85);box-shadow:0 0 0 0 rgba(239,68,68,.7)}70%{transform:scale(1.1);box-shadow:0 0 0 12px rgba(239,68,68,0)}100%{transform:scale(.85);box-shadow:0 0 0 0 rgba(239,68,68,0)}}
.gz-poi{width:24px;height:24px;border-radius:50%;background:#fff;border:1px solid #cbd5e1;box-shadow:0 1px 3px rgba(0,0,0,.25);display:flex;align-items:center;justify-content:center;font-size:13px}
</style>"""))
        pr = W.priorities(ss.weights)
        prd = dict(zip(pr.id, pr.priority))
        if "Почвы" in layers:
            g = folium.FeatureGroup(name="Почвы")
            for s in W.soils:
                folium.GeoJson(ring_to_geojson(s["ring"]), style_function=lambda f, c=SOIL_TYPES[s["soil_type"]]["color"]: dict(fillColor=c, color=c, weight=0.5, fillOpacity=0.25),
                               tooltip=f"Почва: {s['soil_type']}, pH {s['ph']}, засоление {s['salinity']} дСм/м, плодородие {s['fertility']}/5, влажность {s['moisture']}/5 ({s['src']})").add_to(g)
            g.add_to(m)
        if "Существующая зелень" in layers:
            g = folium.FeatureGroup(name="Существующая зелень")
            cc = {"хорошее": "#2e7d32", "удовлетворительное": "#f9a825", "плохое": "#c62828"}
            for o in W.greens:
                folium.Circle([o["lat"], o["lon"]], radius=o["r"] * 3, color=cc[o["condition"]], fill=True, fill_opacity=0.6, weight=1,
                              tooltip=f"{o['species']}, {o['condition']}, {o['age']} лет ({o['src']})").add_to(g)
            g.add_to(m)
        zg = folium.FeatureGroup(name="Зоны озеленения")
        for z in W.zones:
            sev = max([SEV_ORDER[p["Острота"]] for p in z["problems"]] or [0])
            if "Приоритеты" in layers:
                v = prd.get(z["id"], 0)
                col = "#b91c1c" if v >= 60 else ("#ea580c" if v >= 45 else ("#f59e0b" if v >= 30 else "#10b981"))
            else:
                col = {3: "#ef4444", 2: "#f59e0b"}.get(sev, "#10b981")
            sel = z["id"] == ss.selected
            folium.GeoJson(ring_to_geojson(z["ring"]),
                           style_function=lambda f, c=col, s=sel: dict(fillColor=c, color=c, weight=3.5 if s else 2, dashArray=None if s else "3, 5",
                                                                       fillOpacity=0.45 if s else 0.25),
                           tooltip=f"{z['id']} · {z['name']} · приоритет {prd.get(z['id'], 0):.0f} · зелень {z['green_share']:.0%}").add_to(zg)
            if "Проблемные зоны" in layers and sev == 3:
                folium.Marker([z["lat"], z["lon"]], icon=folium.DivIcon(html='<div class="gz-pulse"></div>', icon_size=(14, 14), icon_anchor=(7, 7)),
                              tooltip=f"{z['id']} — критично: " + "; ".join(p["Проблема"] for p in z["problems"] if p["Острота"] == "высокая")).add_to(zg)
            elif "Проблемные зоны" in layers and sev == 2:
                folium.CircleMarker([z["lat"], z["lon"]], radius=5, color="#fff", weight=2, fill=True, fill_color="#f59e0b", fill_opacity=1,
                                    tooltip=f"{z['id']} — " + "; ".join(p["Проблема"] for p in z["problems"])).add_to(zg)
        zg.add_to(m)
        if "Школы, детсады, больницы, ЖК" in layers:
            if W.social_real:
                g = folium.FeatureGroup(name="Школы, детсады, больницы, ЖК")
                ico = {"school": "🎓", "kindergarten": "🧸", "hospital": "🏥", "clinic": "⚕️", "residential": "🏢"}
                for s_ in W.social:
                    if s_["kind"] != "residential":
                        folium.Circle([s_["lat"], s_["lon"]], radius=150, color="#3b82f6", weight=1, dash_array="4, 4", fill=True,
                                      fill_color="#60a5fa", fill_opacity=0.10, interactive=False).add_to(g)
                    folium.Marker([s_["lat"], s_["lon"]], icon=folium.DivIcon(html=f'<div class="gz-poi">{ico.get(s_["kind"], "📍")}</div>', icon_size=(24, 24), icon_anchor=(12, 12)),
                                  tooltip=f"{s_['name']} — {SOCIAL_RU.get(s_['kind'], s_['kind'])}, {s_.get('addr', '')}"
                                          + (" · буфер 150 м: без аллергенов и ядовитых видов" if s_["kind"] != "residential" else "") + " · © OpenStreetMap").add_to(g)
                g.add_to(m)
            else:
                m.add_child(OsmSocial())
        if "Предлагаемые посадки" in layers and plan:
            g = folium.FeatureGroup(name="Предлагаемые посадки")
            z = W.zone(plan["zone_id"])
            woody = [l for l in plan["lines"] if l["type"] != "газон"]
            n = sum(l["qty"] for l in woody)
            k = min(1.0, 250 / n) if n else 0
            pts = sample_points_in(z["ring"], int(sum(round(l["qty"] * k) for l in woody)), seed=sum(map(ord, z["id"])))
            i = 0
            for l in woody:
                for _ in range(int(round(l["qty"] * k))):
                    if i >= len(pts):
                        break
                    lo, la = pts[i]; i += 1
                    folium.CircleMarker([la, lo], radius=4 if l["type"] == "дерево" else 2.5, color="#14532d" if l["type"] == "дерево" else "#65a30d",
                                        fill=True, fill_opacity=0.9, weight=1, tooltip=f"{l['name']} (план)").add_to(g)
            g.add_to(m)
        if ss.found_point:
            lo, la, lab = ss.found_point
            folium.Marker([la, lo], tooltip=f"Найдено: {lab}", icon=folium.Icon(color="blue", icon="search")).add_to(m)
        if "Граф дорог (OSM)" in layers:
            m.add_child(OsmRoads(labels=labels))
        elif labels:
            folium.TileLayer(url.replace("_nolabels", "_only_labels"), attr=attr, overlay=True, name="Подписи").add_to(m)
        Draw(export=False, draw_options=dict(polyline=False, circle=False, marker=False, circlemarker=False, rectangle=True, polygon=True),
             edit_options=dict(edit=False, remove=True)).add_to(m)
        return m

    def show_map(m, key, height=620):
        if st_folium is None:
            components.html(m.get_root().render(), height=height)
            st.warning("Пакет streamlit-folium не установлен: клик по карте и рисование недоступны.")
            return {}
        try:
            return st_folium(m, height=height, use_container_width=True, key=key,
                             returned_objects=["last_object_clicked_tooltip", "last_active_drawing"]) or {}
        except TypeError:
            return st_folium(m, height=height, width=None, key=key, returned_objects=["last_object_clicked_tooltip", "last_active_drawing"]) or {}

    def handle_map_output(out):
        tip = (out or {}).get("last_object_clicked_tooltip")
        if tip and tip != ss.last_tip:
            ss.last_tip = tip
            mz = re.search(r"Z-\d+", tip)
            if mz and W.zone(mz.group(0)) and mz.group(0) != ss.selected:
                ss.selected = mz.group(0)
                ss.zone_ai = None
                st.rerun()
        drw = (out or {}).get("last_active_drawing")
        if drw and (drw.get("geometry") or {}).get("type") == "Polygon":
            h = json.dumps(drw["geometry"], sort_keys=True)
            if h not in ss.processed_drawings:
                ring = [tuple(p[:2]) for p in drw["geometry"]["coordinates"][0]][:-1]
                a = ring_area_m2(ring)
                c1, c2, c3 = st.columns([3, 1.4, 1])
                c1.success(f"Нарисован полигон: {a:,.0f} м²".replace(",", " "))
                pur = c2.selectbox("Назначение", PURPOSES, key="draw_purpose")
                if c3.button("➕ Создать зону"):
                    si = self_intersections(ring)
                    z = W.new_zone_from_ring(ring, {"purpose": pur}, src="нарисована на карте")
                    ss.extra_zones.append(z)
                    ss.processed_drawings.add(h)
                    ss.selected = z["id"]
                    if si:
                        st.warning("Полигон самопересекается — площадь может быть неверной.")
                    bump()
                    st.rerun()

    def zone_inspector(z, plan, variant):
        """Паспорт выбранного участка — по образцу HTML-макета."""
        sev = max([SEV_ORDER[p["Острота"]] for p in z["problems"]] or [0])
        badge = {3: ('h', 'Критический (срочно)'), 2: ('m', 'Средний дефицит'), 1: ('l', 'Низкий'), 0: ('l', 'Без проблем')}[sev]
        deficit = z["green_share"] - 0.25
        pol = "высокое" if z["road_exposure"] > 0.6 else ("среднее" if z["road_exposure"] > 0.3 else "низкое")
        polc = "#dc2626" if pol == "высокое" else ("#d97706" if pol == "среднее" else "#059669")
        st.markdown(f"""
<div class="gz-panel-h">🎯 Паспорт выбранного участка</div>
<div style="display:flex;justify-content:space-between;align-items:flex-start;gap:8px"><div class="gz-panel-t">{z['id']} · {z['name']}</div>
<span class="gz-sev {badge[0]}">{badge[1]}</span></div>
<div class="gz-grid">
 <div class="gz-card"><div class="l">Площадь (EPSG:32643)</div><div class="v">{fmt_tg(z['area_m2'])[:-2]} м²</div><div class="s">Посадочная: <b>{fmt_tg(plan['plantable_m2'])[:-2]} м²</b></div></div>
 <div class="gz-card"><div class="l">Текущая зелень / дефицит</div><div class="v" style="color:{'#dc2626' if deficit < 0 else '#059669'}">{z['green_share']:.0%}
   <span style="font-size:11px;color:#94a3b8">({deficit * 100:+.0f} п.п. к ориентиру)</span></div><div class="s">Ориентир: 25% · {z['district']} район</div></div>
 <div class="gz-card"><div class="l">Почвенный профиль</div><div style="font-weight:700;font-size:12.5px;color:#78350f">{z['soil_type']}</div>
   <div class="s">pH: <b>{z['ph']}</b> ({'щелочн.' if z['ph'] > 7.5 else 'нейтр.'}) · соль: <b style="color:{'#dc2626' if z['salinity'] > 4 else '#334155'}">{z['salinity']} дСм/м</b></div></div>
 <div class="gz-card"><div class="l">Транспортная нагрузка</div><div style="font-weight:700;font-size:12.5px;color:#1e293b">{z['road_class_ru'].capitalize()}, {z['dist_road']:.0f} м</div>
   <div class="s" style="color:{polc};font-weight:600">{z['road_name'][:38]} · загрязнение: {pol}</div></div>
</div>""", unsafe_allow_html=True)
        woody = [l for l in plan["lines"] if l["type"] != "газон"]
        if plan["lines"]:
            avg_sc = round(sum(l["score"] * l["qty"] for l in woody) / max(1, sum(l["qty"] for l in woody))) if woody else plan["lines"][0]["score"]
            tiers = {"дерево": "1-й ярус", "кустарник": "2-й ярус", "газон": "травы"}
            top = sorted(woody, key=lambda l: -l["score"])[:2]
            why = "; ".join(w[2:] for w in (top[0]["why"] if top else []) if w.startswith("+"))[:180]
            rows = "".join(f'<div class="row"><span>{i + 1}. {l["name"]}</span><b>{l["qty"]:,} {l["unit"]} ({tiers[l["type"]]})</b></div>'.replace(",", " ")
                           for i, l in enumerate(plan["lines"][:6]))
            st.markdown(f"""<div class="gz-ai"><div class="h"><span>✨ AI-рекомендация по породам · «{variant}»</span><span class="tag">Пригодность: {avg_sc}/100</span></div>
Для почвы «{z['soil_type']}» (pH {z['ph']}, засоление {z['salinity']}) и назначения «{z['purpose']}» подобрана {len(set(l['type'] for l in plan['lines']))}-ярусная посадка.
{('Ведущий вид — ' + top[0]['name'] + (': ' + why if why else '') + '.') if top else ''}
<div style="margin-top:8px;padding-top:6px;border-top:1px solid #a7f3d0">{rows}</div></div>""", unsafe_allow_html=True)
        a = plan["avg_surv"]
        if a["p1"] is not None:
            agg = {}
            q = sum(l["qty"] for l in woody)
            for l in woody:
                for k_, v_ in l["surv"]["contrib"].items():
                    agg[k_] = agg.get(k_, 0) + v_ * l["qty"] / q
            risks = sorted([(k_, v_) for k_, v_ in agg.items() if v_ < -0.5], key=lambda t: t[1])[:3]
            M = train_survival_model()
            col_ = lambda p: "#059669" if p >= 0.75 else ("#d97706" if p >= 0.6 else "#dc2626")
            rk = "".join(f'<div class="gz-risk"><span>➖ {FEAT_RU[k_]}</span><b style="color:#dc2626">{v_:+.1f} п.п.</b></div>' for k_, v_ in risks) or \
                '<div class="gz-risk"><span>Существенных факторов риска нет</span><b></b></div>'
            zi = dict(z); zi.update(salinity=round(z["salinity"] * 0.6, 1), fertility=min(5, z["fertility"] + 1))
            V = plan["V"]
            imp = sum(predict_survival(l["plant"], zi, V["watering"], V["age"], V["care_years"])["p3"] * l["qty"] for l in woody) / q - a["p3"]
            if imp > 0.005:
                rk += f'<div class="gz-risk"><span>➕ Мелиорация почвы (гипс, грунт, удобрения)</span><b style="color:#059669">{imp * 100:+.1f} п.п.</b></div>'
            st.markdown(f"""<div class="gz-box"><div class="h">❤️ Прогноз приживаемости (правила + ML)<span>ROC-AUC {M['auc']} · синт.</span></div>
<div class="gz-surv"><div><div class="y">1-й год</div><div class="p" style="color:{col_(a['p1'])}">{a['p1']:.0%}</div><div class="n">Критический период</div></div>
<div><div class="y">2-й год</div><div class="p" style="color:{col_(a['p2'])}">{a['p2']:.0%}</div><div class="n">Укоренение</div></div>
<div><div class="y">3-й год</div><div class="p" style="color:{col_(a['p3'])}">{a['p3']:.0%}</div><div class="n">Стабильный фонд</div></div></div>{rk}</div>""",
                        unsafe_allow_html=True)
        est = plan["estimate"]["table"].set_index("Статья")["Сумма, ₸"]
        g1 = est.get("Посадочный материал", 0)
        g2 = sum(est.get(k_, 0) for k_ in ["Подготовка территории и грунт", "Удобрения", "Мелиорация засолённой почвы", "Дренаж"])
        g3 = sum(v_ for k_, v_ in est.items() if k_ in ("Посадка", "Доставка") or k_.startswith("Полив"))
        g4 = est.get("Резерв на замену неприжившихся", 0)
        st.markdown(f"""<div class="gz-dark"><div class="t"><span>Смета (сценарий «{variant}»):</span><b>{fmt_tg(plan['estimate']['total'])}</b></div>
<div class="r"><span>Посадочный материал:</span><span>{fmt_tg(g1)}</span></div>
<div class="r"><span>Подготовка грунта, мелиорация, дренаж:</span><span>{fmt_tg(g2)}</span></div>
<div class="r"><span>Посадка, доставка, полив и уход ({plan['V']['care_years']} г.):</span><span>{fmt_tg(g3)}</span></div>
<div class="r"><span>Резерв на замену неприжившихся:</span><span>{fmt_tg(g4)}</span></div></div>""", unsafe_allow_html=True)
        b1, b2, b3 = st.columns([1.3, 1, 1])
        if b1.button("✅ Утвердить в план", type="primary", disabled=role == "Просмотр", **WIDE):
            ss.decisions.append(dict(t=datetime.datetime.now().strftime("%d.%m %H:%M"), zone=z["id"], variant=variant,
                                     cost=round(plan["estimate"]["total"]), trees=sum(l["qty"] for l in plan["lines"] if l["type"] == "дерево"),
                                     surv3=round(plan["avg_surv"]["p3"] or 0, 3), co2_20=round(plan["eco"]["co2_cum20_t"], 1),
                                     ov=json.dumps(ov_of(z["id"]), ensure_ascii=False)))
            st.toast(f"Участок {z['id']} включён в предварительный план (см. «История»)") if hasattr(st, "toast") else st.success("Сохранено")
        if b2.button("💬 AI-анализ", **WIDE):
            ss.zone_ai = (z["id"], ask(f"Экспресс-аудит зоны {z['id']}"))
        b3.download_button("📄 Паспорт", make_passport_html(W, z["id"], plan), f"pasport_{z['id']}.html", mime="text/html", **WIDE)
        if ss.get("zone_ai") and ss.zone_ai[0] == z["id"]:
            st.info(ss.zone_ai[1])
        st.markdown('<div class="gz-foot">* Оценочный расчёт. Решение принимает специалист отдела благоустройства. Расценки и коэффициенты — «Справочники».</div>',
                    unsafe_allow_html=True)

    # ===================== СТРАНИЦЫ =====================
    if page == PAGES[0]:
        z = W.zone(ss.selected)
        st.markdown('<div class="gz-layers">🗂️ Слои Павлодара</div>', unsafe_allow_html=True)
        LAYERS = ["Граф дорог (OSM)", "Проблемные зоны", "Предлагаемые посадки", "Школы, детсады, больницы, ЖК", "Почвы", "Существующая зелень", "Приоритеты"]
        lc = st.columns(len(LAYERS))
        layers = [nm for c_, nm in zip(lc, LAYERS) if c_.checkbox(nm, value=nm in LAYERS[:4], key=f"lay_{nm}")]
        t2, t3, t4 = st.columns([1.6, 1.2, 2])
        basemap = t2.selectbox("Подложка", list(BASEMAPS) + (["2ГИС MapGL (ключ)"] if os.getenv("DGIS_API_KEY") else []), key="basemap")
        ids = [zz["id"] for zz in W.zones]
        ss.zone_pick = ss.selected
        t3.selectbox("Участок", ids, key="zone_pick", format_func=lambda i: f"{i} · {W.zone(i)['purpose']}",
                     on_change=lambda: ss.__setitem__("selected", ss.zone_pick))
        with t4.form("addr", clear_on_submit=False):
            a1, a2 = st.columns([3, 1])
            addr = a1.text_input("Поиск адреса", placeholder="Адрес в Павлодаре, напр. ул. Сатпаева 50")
            a2.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            if a2.form_submit_button("🔎 Найти") and addr:
                try:
                    res = geocode(addr)
                    if res:
                        lo, la, lab = res
                        ss.found_point = res
                        ins = [zz for zz in W.zones if pip(lo, la, zz["ring"])]
                        if ins:
                            ss.selected = ins[0]["id"]
                        else:
                            zz = W.new_zone_from_ring(_circle_ring(lo, la, 40), {"name": lab[:40]}, src=f"адрес «{addr}»")
                            ss.extra_zones.append(zz); ss.selected = zz["id"]; bump()
                        st.rerun()
                    else:
                        st.warning("Адрес не найден")
                except Exception as e:
                    st.error(f"Геокодер недоступен: {e}")
        z = W.zone(ss.selected)
        variant = ss.get("variant", "Оптимальный")
        _o = ov_of(z["id"])
        for _k, _wk in (("excluded", f"ex_{z['id']}"), ("fixed", f"fx_{z['id']}")):
            if _wk in ss:
                _o[_k] = list(ss[_wk])
        plan = W.plan(z["id"], variant, _o, ss.get("care_years"), None)

        cmap, cpanel = st.columns([1.55, 1])
        with cmap:
            if basemap == "2ГИС MapGL (ключ)":
                key = os.getenv("DGIS_API_KEY")
                data = [dict(id=zz["id"], coords=[[[a, b] for a, b in zz["ring"]] + [[zz["ring"][0][0], zz["ring"][0][1]]]],
                             col=PURPOSE_COLOR.get(zz["purpose"], "#2e7d32")) for zz in W.zones]
                components.html(f"""<div id="c" style="width:100%;height:620px"></div>
<script src="https://mapgl.2gis.com/api/js/v1"></script>
<script>
const map = new mapgl.Map('c', {{center:[{z['lon']},{z['lat']}], zoom:14, key:'{key}'}});
{json.dumps(data)}.forEach(z => new mapgl.Polygon(map, {{coordinates: z.coords, color: z.col + '66', strokeColor: z.col, strokeWidth: 2}}));
</script>""", height=630)
                st.caption("2ГИС MapGL: дороги и подписи — из 2ГИС. Выбор зоны и рисование — на подложке OSM.")
            else:
                m = build_map(layers, basemap, center=(z["lat"], z["lon"]), zoom=15, plan=plan)
                handle_map_output(show_map(m, key=f"map_{ss.selected}_{ss.data_ver}_{basemap[:4]}"))
            near_road = [zz for zz in W.zones if zz["dist_road"] < 60 and zz["green_share"] < 0.15]
            st.markdown(f"""<div class="gz-pill"><span class="dot"></span><span>Защитные полосы: <b>{len(near_road)} из {len(W.zones)}</b> участков у дорог (&lt;60 м) с зеленью &lt;15%</span>
<span class="m">|</span><span class="m">Дороги: © OpenStreetMap · аналитика: {'OSM' if W.roads_real else 'синтетика'} · участки: синтетика</span></div>""", unsafe_allow_html=True)
            st.caption("Клик по участку — выбор; по улице — название, класс, длина и расчёт защитной полосы. Инструменты слева на карте — нарисовать новый участок.")

        with cpanel:
            vcol, ycol = st.columns(2)
            vcol.selectbox("Сценарий", list(VARIANTS), index=list(VARIANTS).index(variant), key="variant")
            ycol.selectbox("Уход, лет", [None, 1, 2, 3], format_func=lambda v: "по сценарию" if v is None else str(v), key="care_years")
            for w in plan["warnings"]:
                st.warning(w)
            zone_inspector(z, plan, variant)

        st.markdown("#### Подробно по участку")
        tabs = st.tabs(["Обзор", "Растения", "Смета", "Приживаемость и уход", "Эффект", "Вдоль дороги", "Отчёты"])
        with tabs[0]:
            st.markdown(f"""
- **Назначение:** {z['purpose']} · **Район:** {z['district']} · **Источник данных:** {z.get('src', '')}
- **Почва:** {z['soil_type']}, pH {z['ph']}, засоление {z['salinity']} дСм/м, плодородие {z['fertility']}/5, влажность {z['moisture']}/5
- **Ближайшая дорога:** {z['road_name']} — {z['road_class_ru']}, {z['dist_road']:.0f} м; нагрузка от дорог {z['road_exposure']:.2f} ({'OSM' if W.roads_real else SYN})
- **Соцобъекты в 500 м (OSM):** {z['social_n']} ({z['social_list']}){(' — ' + z.get('social_names', '')) if z.get('social_names') else ''}
- **Микроклимат:** t° поверхности {z['lst']} °C, затенение {z['shade']}/5, ветер {z['wind']}/5 ({SYN})
- **До зданий:** {z['bldg_dist']} м · **Население:** {z['pop_density']} чел/га ({SYN})""")
            if z["problems"]:
                st.dataframe(pd.DataFrame(z["problems"]).drop(columns="zone"), hide_index=True, **WIDE)
        with tabs[1]:
            sel, surv_t, eco_t = plan_tables(plan)
            st.dataframe(sel, hide_index=True, **WIDE,
                         column_config={"Пригодность": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%d")})
            o = ov_of(z["id"])
            names = [p["name"] for p in C["plants"]]
            o["excluded"] = st.multiselect("Исключить виды", names, default=[x for x in o["excluded"] if x in names], key=f"ex_{z['id']}")
            o["fixed"] = st.multiselect("Зафиксировать виды (всегда в плане)", names, default=[x for x in o["fixed"] if x in names], key=f"fx_{z['id']}")
            r1, r2, r3 = st.columns([2, 2, 1])
            frm = r1.selectbox("Заменить", ["—"] + [l["name"] for l in plan["lines"]], key=f"rf_{z['id']}")
            to = r2.selectbox("на", ["—"] + names, key=f"rt_{z['id']}")
            if r3.button("↔", key=f"rb_{z['id']}") and frm != "—" and to != "—" and frm != to:
                o["replace"][frm] = to
                st.rerun()
            if o["replace"]:
                st.caption("Замены: " + "; ".join(f"{a} → {b}" for a, b in o["replace"].items()))
                if st.button("Сбросить замены"):
                    o["replace"] = {}
                    st.rerun()
            with st.expander(f"Исключено ограничениями: {len(plan['rejected'])}"):
                st.dataframe(pd.DataFrame(plan["rejected"]), hide_index=True, **WIDE)
        with tabs[2]:
            est = plan["estimate"]
            st.metric("Итого (оценка)", fmt_tg(est["total"]))
            st.dataframe(est["table"], hide_index=True, **WIDE,
                         column_config={"Доля": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
                                        "Сумма, ₸": st.column_config.NumberColumn(format="%d")})
            chart(est["table"].set_index("Статья")["Сумма, ₸"])
            st.caption("Расценки — «Справочники». Источник: " + SRC_PRICE)
        with tabs[3]:
            M = train_survival_model()
            st.dataframe(surv_t, hide_index=True, **WIDE,
                         column_config={k: st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f") for k in ["1 год", "2 год", "3 год"]})
            st.caption(f"Метод: 50% экспертные правила + 50% GradientBoosting (scikit-learn). ROC-AUC на тесте: {M['auc']} (правила {M['auc_rule']}) — "
                       f"на СИНТЕТИЧЕСКИХ данных; нужна калибровка на данных акимата или подтверждение экспертом.")
            if plan["lines"]:
                sp = st.selectbox("Факторы и меры для вида", [l["name"] for l in plan["lines"]], key=f"surv_sp_{z['id']}_{variant}_{len(ov_of(z['id'])['replace'])}")
                l = next((x for x in plan["lines"] if x["name"] == sp), plan["lines"][0])
                cdf = pd.DataFrame([{"Фактор": FEAT_RU[k], "Вклад, п.п.": v} for k, v in l["surv"]["contrib"].items() if abs(v) >= 0.1])
                if len(cdf):
                    chart(cdf.sort_values("Вклад, п.п.").set_index("Фактор"))
                V = plan["V"]
                base = l["surv"]["p3"]
                zi = dict(z); zi.update(salinity=round(z["salinity"] * 0.6, 1), fertility=min(5, z["fertility"] + 1), ph=round(z["ph"] - 0.5 if z["ph"] > 7.5 else z["ph"], 1))
                meas = [("Улучшение почвы (гипс/промывка, грунт, удобрения)", predict_survival(l["plant"], zi, V["watering"], V["age"], V["care_years"])["p3"]),
                        ("Интенсивный полив", predict_survival(l["plant"], z, 2, V["age"], V["care_years"])["p3"]),
                        ("Саженец 3–4 года + уход 3 года", predict_survival(l["plant"], z, V["watering"], 3, 3)["p3"])]
                alt = [(p, predict_survival(p, z, V["watering"], V["age"], V["care_years"])["p3"]) for p in C["plants"]
                       if p["type"] == l["type"] and p["name"] != l["name"] and suitability(p, z, C["coeffs"])[0] is not None]
                if alt:
                    bp, bv = max(alt, key=lambda t: t[1])
                    meas.append((f"Замена на «{bp['name']}»", bv))
                st.dataframe(pd.DataFrame([{"Мера": a, "Приживаемость 3 г.": f"{b:.0%}", "Изменение, п.п.": round((b - base) * 100, 1)} for a, b in meas]),
                             hide_index=True, **WIDE)
            st.markdown("**Календарь ухода (1-й год)**")
            mv = st.slider("Допустимо выездов бригады в месяц", 2, 12, 6, key="max_visits")
            care_df, warns = care_calendar(plan, z, mv)
            for w in warns:
                st.warning(w)
            st.caption(f"Бюджет ухода по смете: {fmt_tg(plan['estimate']['care_per_year'])} в год · режим полива: {WATER_RU[plan['V']['watering']]}")
            st.dataframe(care_df, hide_index=True, **WIDE, height=300)
        with tabs[4]:
            e = plan["eco"]
            cols = st.columns(3)
            for c_, Y in zip(cols, (5, 10, 20)):
                v = e["by_year"][Y]
                c_.metric(f"CO₂ через {Y} лет", f"{v['co2_t']:.1f} т/год")
                c_.caption(f"O₂ {v['o2_t']:.1f} т/год · пыль {v['dust_kg']:.0f} кг/год · t° −{v['cooling_c']:.2f} °C · осадки {v['rain_m3']:.0f} м³/год")
            st.dataframe(eco_t, hide_index=True, **WIDE)
            b = e["before"]; a20 = e["by_year"][20]
            st.markdown(f"**До / после (20 лет):** доля зелени {b['green_share']:.0%} → {a20['green_share_after']:.0%}; "
                        + (f"зелень на жителя {b['gpc']:.1f} → {a20['gpc_after']:.1f} м²/чел; " if b["gpc"] else "")
                        + f"t° поверхности {b['lst']} → ≈{b['lst'] - a20['cooling_c']:.1f} °C · стоимость {fmt_tg(e['cost_per_t'] or 0)} за 1 т CO₂ (оценка)")
        with tabs[5]:
            st.caption("Расчёт защитной полосы вдоль ближайших дорог (буфер, виды, стоимость) — для школ, больниц и жилья у магистралей.")
            if not z["near_roads"]:
                st.info("В радиусе 400 м дорог нет.")
            else:
                rr = [(d, next(r for r in W.roads if r["rid"] == rid)) for d, rid in z["near_roads"]]
                st.dataframe(pd.DataFrame([{"Дорога": r["rid"], "Название": r["name"], "Класс": ROAD_CLASSES.get(r["highway"], (r["highway"],))[0],
                                            "До зоны, м": d, "Длина, м": r.get("len_m")} for d, r in rr]), hide_index=True, **WIDE)
                d0, r0 = rr[0]
                L_ = min(r0.get("len_m", 0), 2 * math.sqrt(z["area_m2"]))
                good = sorted([p for p in C["plants"] if p["type"] != "газон" and p["pollution_tol"] >= 4 and suitability(p, z, C["coeffs"])[0] is not None],
                              key=lambda p: -suitability(p, z, C["coeffs"])[0])
                tr = [p for p in good if p["type"] == "дерево"][:2]
                sh = [p for p in good if p["type"] == "кустарник"][:1]
                n_tr = int(L_ / 6) * 2 if tr else 0
                n_sh = int(L_ / 0.7) if sh else 0
                cost = n_tr * (np.mean([p["price"] for p in tr]) if tr else 0) + n_sh * (sh[0]["price"] if sh else 0) + n_tr * C["rates"]["plant_tree"] + n_sh * C["rates"]["plant_shrub"]
                st.markdown(f"**Защитная полоса со стороны «{r0['name']}»** ({'OSM' if W.roads_real else SYN}; длина ≈ {L_:.0f} м, ширина 10–15 м): "
                            f"2 ряда деревьев через 6 м — **{n_tr} шт.** ({', '.join(p['name'] for p in tr) or 'нет подходящих'}), "
                            f"живая изгородь — **{n_sh} шт.** ({', '.join(p['name'] for p in sh) or 'нет подходящих'}). Стоимость посадки ≈ **{fmt_tg(cost)}** (оценка).")
        with tabs[6]:
            rkey = (z["id"], variant, json.dumps(ov_of(z["id"]), sort_keys=True, ensure_ascii=False), ss.get("care_years"), ss.data_ver)
            if st.button("📄 Сформировать отчёты по участку"):
                pr = W.priorities(ss.weights)
                care_df, _ = care_calendar(plan, z, ss.get("max_visits", 6))
                with st.spinner("Формирование…"):
                    ss.reports = dict(key=rkey, xlsx=make_excel(W, z["id"], plan, pr, care_df), pdf=make_pdf(W, z["id"], plan, pr, care_df),
                                      html=make_html(W, z["id"], plan, pr, care_df))
            rep_ = ss.get("reports")
            if rep_ and rep_["key"] == rkey:
                st.download_button("⬇️ Excel (подбор, смета, приживаемость, уход, эффект, проблемы, приоритеты)", rep_["xlsx"], f"otchet_{z['id']}.xlsx")
                if rep_["pdf"]:
                    st.download_button("⬇️ PDF-отчёт", rep_["pdf"], f"otchet_{z['id']}.pdf", mime="application/pdf")
                st.download_button("⬇️ HTML-отчёт", rep_["html"], f"otchet_{z['id']}.html", mime="text/html")
            else:
                st.caption("Нажмите кнопку, чтобы сформировать Excel, PDF и HTML для текущего сценария.")

    elif page == PAGES[1]:
        st.subheader("Реестр проблемных зон озеленения Павлодара")
        df = W.problems_df()
        if df.empty:
            st.success("Проблем не выявлено")
        else:
            c1, c2, c3 = st.columns(3)
            f1 = c1.multiselect("Тип", sorted(df["Проблема"].unique()))
            f2 = c2.multiselect("Острота", ["высокая", "средняя", "низкая"])
            f3 = c3.multiselect("Район", sorted(df["Район"].unique()))
            mask = pd.Series(True, index=df.index)
            if f1: mask &= df["Проблема"].isin(f1)
            if f2: mask &= df["Острота"].isin(f2)
            if f3: mask &= df["Район"].isin(f3)
            d = df[mask]
            a, b, c = st.columns(3)
            a.metric("Проблем", len(d)); b.metric("Высокой остроты", int((d["Острота"] == "высокая").sum())); c.metric("Участков", d["Зона"].nunique())
            st.dataframe(d, hide_index=True, **WIDE, height=400)
            chart(d["Проблема"].value_counts())
            cols = st.columns(8)
            for i, zid in enumerate(d["Зона"].unique()[:16]):
                cols[i % 8].button(f"📍{zid}", key=f"pz_{zid}", on_click=goto_zone, args=(zid,))
            m = build_map(["Проблемные зоны", "Граф дорог (OSM)"], list(BASEMAPS)[0], zoom=12)
            handle_map_output(show_map(m, key=f"pmap_{ss.data_ver}", height=480))

    elif page == PAGES[2]:
        st.subheader("Каталог районированных растений")
        df = ss.plants_df
        c1, c2, c3, c4 = st.columns(4)
        qs = c1.text_input("Поиск")
        tp = c2.multiselect("Ярус", sorted(df["type"].unique()))
        dr = c3.slider("Засухоустойчивость ≥", 1, 5, 1)
        sf = c4.checkbox("Безопасно для детей")
        d = df[df["name"].str.lower().str.contains(qs.lower(), regex=False) | df["latin"].str.lower().str.contains(qs.lower(), regex=False)] if qs else df
        if tp: d = d[d["type"].isin(tp)]
        d = d[d["drought"] >= dr]
        if sf: d = d[~d["allergen"] & ~d["toxic"]]
        ic = {"дерево": "🌳", "кустарник": "🌿", "газон": "🌱"}
        cols = st.columns(4)
        for i, p in enumerate(d.to_dict("records")):
            flags = ("⚠️ аллерген " if p["allergen"] else "") + ("☠️ ядовит " if p["toxic"] else "") + ("🚫 инвазивный" if p["invasive"] else "")
            cols[i % 4].markdown(f"""<div class="gz-card" style="margin-bottom:8px"><div style="font-weight:700;color:#0f172a;font-size:13px">{ic.get(p['type'], '')} {p['name']}</div>
<div class="s"><i>{p['latin']}</i> · {p['type']}</div>
<div class="s" style="margin-top:4px">Зимостойкость USDA {p['usda']} · засухоуст. {p['drought']}/5 · соль до {p['salt_tol']} дСм/м · pH {p['ph_min']}–{p['ph_max']}</div>
<div class="s">Газоустойч. {p['pollution_tol']}/5 · высота {p['height_m']} м · CO₂ {p['co2_kg']} кг/год</div>
<div style="display:flex;justify-content:space-between;margin-top:4px"><b style="font-family:'JetBrains Mono';color:#047857">{fmt_tg(p['price'])}</b><span class="s" style="color:#dc2626">{flags}</span></div></div>""",
                                 unsafe_allow_html=True)
        with st.expander("Таблица"):
            st.dataframe(d.rename(columns=PLANT_RU), hide_index=True, **WIDE, height=420)
        st.caption(f"Источник: {SRC_PLANT}. Цены — {SRC_PRICE}. Редактирование — «Справочники» (администратор).")

    elif page == PAGES[3]:
        st.subheader("Матрица приоритетов и бюджетная оптимизация")
        st.caption("Многокритериальный индекс 0–100. Веса нормируются к 100%; пересчёт мгновенный.")
        lcol, rcol = st.columns([2, 1])
        with lcol:
            hc1, hc2 = st.columns([4, 1])
            if hc2.button("Сброс по умолчанию"):
                for k, _, w in FACTORS:
                    ss[f"w_{k}"] = w
            cols = st.columns(3)
            for i, (k, n, w0) in enumerate(FACTORS):
                if f"w_{k}" not in ss:
                    ss[f"w_{k}"] = int(ss.weights[k])
                ss.weights[k] = cols[i % 3].slider(n, 0, 50, key=f"w_{k}")
            sw = sum(ss.weights.values()) or 1
            hc1.markdown("**Весовые коэффициенты (сумма = 100%):** " + " · ".join(f"{n.split()[0]} {ss.weights[k] / sw:.0%}" for k, n, _ in FACTORS))
            pr = W.priorities(ss.weights)
            show = pr[["Ранг", "id", "district", "purpose", "priority"] + [f"c_{k}" for k, _, _ in FACTORS]].rename(
                columns={"id": "Зона", "district": "Район", "purpose": "Назначение", "priority": "Приоритет", **{f"c_{k}": n for k, n, _ in FACTORS}})
            st.dataframe(show, hide_index=True, **WIDE, height=360,
                         column_config={"Приоритет": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f")})
            st.markdown("**Вклад факторов (топ-12)**")
            chart(show.head(12).set_index("Зона")[[n for _, n, _ in FACTORS]])
        with rcol:
            ss.budget = st.number_input("Выделенный бюджет на сезон, ₸", min_value=1_000_000, value=int(ss.budget), step=5_000_000)
            rd, spent = budget_roadmap(W.zones, W.all_plans(), pr, ss.budget)
            if rd.empty:
                st.warning("Бюджета не хватает ни на один проект.")
            else:
                pops = sum(W.zone(zid)["pop_density"] * W.zone(zid)["area_ha"] for zid in rd["Зона"])
                near_kids = sum(1 for zid in rd["Зона"] if re.search("школ|детсад|больниц", W.zone(zid)["social_list"]))
                st.markdown(f"""<div class="gz-opt"><div class="h">⚡ Оптимизатор бюджета · жадный алгоритм (эффект на ₸)</div>
<div class="r"><span>Выделенный бюджет:</span><b style="color:#fbbf24">{fmt_tg(ss.budget)}</b></div>
<div class="r"><span>Охват участков:</span><b>{len(rd)} из {len(W.zones)}</b></div>
<div class="r"><span>Сумма по сметам:</span><b>{fmt_tg(spent)}</b></div>
<div class="r"><span>Остаток резерва:</span><b>{fmt_tg(ss.budget - spent)}</b></div>
<div class="r"><span>CO₂ за 20 лет:</span><b>{rd['CO₂ за 20 лет, т'].sum():.0f} т</b></div>
<div class="g">✔ Затронет ≈ {pops:,.0f} жителей участков ({'синтетика' if True else ''})</div>
<div class="g">✔ {near_kids} участк. рядом со школами, детсадами или больницами</div></div>""".replace(",", " "), unsafe_allow_html=True)
                st.dataframe(rd, hide_index=True, **WIDE)
                cols = st.columns(4)
                for i, zid in enumerate(rd["Зона"][:12]):
                    cols[i % 4].button(f"📍{zid}", key=f"bz_{zid}", on_click=goto_zone, args=(zid,))
                lst = rd.merge(pd.DataFrame([{"Зона": z_["id"], "Наименование": z_["name"], "Район": z_["district"]} for z_ in W.zones]), on="Зона")
                doc = (f"<html><head><meta charset='utf-8'><title>Перечень объектов озеленения</title><style>body{{font-family:Arial;margin:32px}}"
                       f"table{{border-collapse:collapse;font-size:12px}}td,th{{border:1px solid #ccc;padding:4px 6px}}</style></head><body>"
                       f"<h2>Проект перечня объектов озеленения на сезон (черновик)</h2><p>Бюджет: {fmt_tg(ss.budget)}; по сметам: {fmt_tg(spent)}.</p>"
                       + lst.to_html(index=False) + f"<p style='font-size:11px;color:#666'>{DISCLAIMER}</p></body></html>")
                st.download_button("📄 Перечень объектов (проект)", doc.encode("utf-8"), "perechen_obektov.html", mime="text/html", **WIDE)
                bio = io.BytesIO()
                with pd.ExcelWriter(bio, engine="openpyxl") as xw:
                    show.to_excel(xw, sheet_name="Приоритеты", index=False); rd.to_excel(xw, sheet_name="Дорожная карта", index=False)
                    W.problems_df().to_excel(xw, sheet_name="Проблемные зоны", index=False); assumptions_df(W).to_excel(xw, sheet_name="Допущения", index=False)
                st.download_button("⬇️ Excel: приоритеты, план, проблемы", bio.getvalue(), "prioritety.xlsx", **WIDE)

    elif page == PAGES[4]:
        st.subheader("Долгосрочный экологический эффект и структура затрат")
        scope = st.radio("Охват", ["Выбранный участок", "Дорожная карта под бюджет", "Все участки города"], horizontal=True, key="eco_scope")
        if scope == "Выбранный участок":
            zids = [ss.selected]
        elif scope == "Дорожная карта под бюджет":
            zids = list(budget_roadmap(W.zones, W.all_plans(), W.priorities(ss.weights), ss.budget)[0].get("Зона", []))
        else:
            zids = [z_["id"] for z_ in W.zones]
        if not zids:
            st.warning("Нет участков в выбранном охвате.")
        else:
            st.caption(f"Участков: {len(zids)} · сценарий для показателей: «Оптимальный» · прогноз на 5, 10 и 20 лет с учётом роста кроны и выживаемости. Все значения — оценка.")
            plans = {v: [W.plan(zid, v) for zid in zids] for v in VARIANTS}
            opt = plans["Оптимальный"]
            zs = [W.zone(zid) for zid in zids]
            A = sum(z_["area_m2"] for z_ in zs)
            pop = sum(z_["pop_density"] * z_["area_ha"] for z_ in zs)
            g0 = sum(z_["green_share"] * z_["area_m2"] for z_ in zs)
            y10 = [p["eco"]["by_year"][10] for p in opt]
            y20 = [p["eco"]["by_year"][20] for p in opt]
            co2_10 = sum(v["co2_t"] for v in y10)
            cost = sum(p["estimate"]["total"] for p in opt)
            cum = sum(p["eco"]["co2_cum20_t"] for p in opt)
            cool = sum(v["cooling_c"] * z_["area_m2"] for v, z_ in zip(y20, zs)) / A
            dust = sum(v["dust_kg"] for v in y10) / 1000
            g1 = g0 + sum(v["green_add_m2"] for v in y20)
            k1, k2, k3, k4 = st.columns(4)
            for c_, l_, v_, s_ in [(k1, "Поглощение CO₂ (10 лет)", f"{co2_10:.1f} т/год", f"≈ {fmt_tg(cost / cum if cum else 0)} за тонну CO₂"),
                                   (k2, "Охлаждение микроклимата", f"−{cool:.2f} °C", "через 20 лет, средневзвешенно"),
                                   (k3, "Удержание пыли (10 лет)", f"{dust:.2f} т/год", "защитный экран от дорог"),
                                   (k4, "Обеспеченность зеленью", f"{g1 / pop:.1f} м²/чел" if pop else "—",
                                    f"рост с {g0 / pop:.1f} м²/чел ({(g1 / g0 - 1) * 100:+.0f}%)" if pop and g0 else "")]:
                c_.markdown(f'<div class="gz-kpi"><div class="l">{l_}</div><div class="v">{v_}</div><div class="s">{s_}</div></div>', unsafe_allow_html=True)
            l_, r_ = st.columns([1.3, 1])
            with l_:
                st.markdown("**Динамика поглощения CO₂ и выработки O₂, т/год**")
                yrs = list(range(1, 21))
                co2s = [sum(eco_effect(p, W.zone(p["zone_id"]), C, (Y,))["by_year"][Y]["co2_t"] for p in opt) for Y in yrs]
                chart(pd.DataFrame({"CO₂": co2s, "O₂": [c * C["coeffs"]["o2_per_co2"] for c in co2s]}, index=pd.Index(yrs, name="Год")), "line")
                st.markdown("**Структура затрат («Оптимальный»)**")
                tot = pd.concat([p["estimate"]["table"].set_index("Статья")["Сумма, ₸"] for p in opt], axis=1).sum(axis=1)
                chart(tot)
            with r_:
                st.markdown("**Сравнение сценариев**")
                rows = []
                for v, ps in plans.items():
                    cst = sum(p["estimate"]["total"] for p in ps)
                    co = sum(p["eco"]["co2_cum20_t"] for p in ps)
                    q_ = sum(sum(l["qty"] for l in p["lines"] if l["type"] != "газон") for p in ps)
                    s3 = sum((p["avg_surv"]["p3"] or 0) * sum(l["qty"] for l in p["lines"] if l["type"] != "газон") for p in ps) / q_ if q_ else 0
                    rows.append((v, cst, co, s3))
                best = max(rows, key=lambda r: r[2] / r[1] if r[1] else 0)[0]
                DESC = {v: f"плотность посадки {VARIANTS[v]['dens']:.0%} от нормы, саженцы {VARIANTS[v]['age']} лет, полив «{WATER_RU[VARIANTS[v]['watering']]}», уход {VARIANTS[v]['care_years']} г."
                        for v in VARIANTS}
                for v, cst, co, s3 in rows:
                    risk = "высокий" if s3 < 0.65 else ("средний" if s3 < 0.8 else "низкий")
                    st.markdown(f"""<div class="gz-scn {'best' if v == best else ''}" style="margin-bottom:8px"><div style="display:flex;justify-content:space-between">
<span class="n">Сценарий «{v}»</span>{'<span class="tag">Лучший эффект на ₸</span>' if v == best else ''}</div>
<div class="c">{fmt_tg(cst)}</div><div>{DESC[v]}</div>
<div style="margin-top:5px">Приживаемость (3 года): <b>{s3:.0%}</b> · риск отпада: <b>{risk}</b><br>CO₂ за 20 лет: <b>{co:.0f} т</b> · {fmt_tg(cst / co if co else 0)} за т</div></div>""",
                                unsafe_allow_html=True)

    elif page == PAGES[5]:
        st.subheader("Состояние города")
        pr = W.priorities(ss.weights)
        pdf_ = W.problems_df()
        opt = W.opt_plans()
        a, b, c, d, e = st.columns(5)
        a.metric("Участков", len(W.zones))
        b.metric("Площадь участков", f"{sum(z['area_ha'] for z in W.zones):.1f} га")
        c.metric("Средняя доля зелени", f"{np.mean([z['green_share'] for z in W.zones]):.0%}")
        d.metric("Проблем высокой остроты", int((pdf_["Острота"] == "высокая").sum()) if len(pdf_) else 0)
        e.metric("«Оптимальный» по городу", f"{sum(p['estimate']['total'] for p in opt.values()) / 1e6:.0f} млн ₸")
        l, r = st.columns(2)
        with l:
            st.markdown("**Приоритет по участкам**")
            chart(pr.set_index("id")["priority"])
            st.markdown("**Проблемы по типам**")
            if len(pdf_): chart(pdf_["Проблема"].value_counts())
        with r:
            st.markdown("**Эффект по годам при реализации «Оптимальный» во всех участках (оценка)**")
            yrs = list(range(1, 21))
            chart(pd.DataFrame({"CO₂, т/год": [sum(eco_effect(p, W.zone(zid), C, (Y,))["by_year"][Y]["co2_t"] for zid, p in opt.items()) for Y in yrs]},
                               index=pd.Index(yrs, name="Год")), "area")
            st.markdown("**Доля зелени по районам: до / после (20 лет)**")
            rows = []
            for dist in sorted({z["district"] for z in W.zones}):
                zs = [z for z in W.zones if z["district"] == dist]
                A = sum(z["area_m2"] for z in zs)
                rows.append(dict(Район=dist, До=sum(z["green_share"] * z["area_m2"] for z in zs) / A,
                                 После=sum(opt[z["id"]]["eco"]["by_year"][20]["green_share_after"] * z["area_m2"] for z in zs) / A))
            chart(pd.DataFrame(rows).set_index("Район"), grouped=True)
        m = build_map(["Приоритеты", "Проблемные зоны", "Граф дорог (OSM)"], list(BASEMAPS)[0], zoom=12)
        handle_map_output(show_map(m, key=f"dmap_{ss.data_ver}", height=460))

    elif page == PAGES[6]:
        st.subheader("AI-консультант · GIS-ассистент")
        st.caption(f"Ассистент вызывает функции модулей (приоритет, подбор, смета, приживаемость, эффект, бюджет) и ищет по каталогу. "
                   f"Текущий участок: **{ss.selected}**. Если данных нет — ассистент сообщает об этом.")
        ex = ["Какие участки в Северном районе требуют озеленения в первую очередь?", "Подбери растения для сквера на этой зоне и посчитай стоимость на 3 года",
              "Какая будет приживаемость, если заменить клён на липу?", "Что можно озеленить на 5 млн при максимальном эффекте?", "Почему не тополь для этой зоны?"]
        cols = st.columns(len(ex))
        pending = None
        for c_, e_ in zip(cols, ex):
            if c_.button(e_, key="ex_" + e_[:12], **WIDE):
                pending = e_
        for i, h in enumerate(ss.history):
            with st.chat_message("user"):
                st.write(h["q"])
            with st.chat_message("assistant"):
                st.markdown(h["a"])
                st.caption(f"инструмент: {h['tool']}")
                if h["zones"]:
                    cc = st.columns(min(8, len(h["zones"])))
                    for j, zid in enumerate(h["zones"][:8]):
                        cc[j].button(f"📍 {zid}", key=f"ch_{i}_{zid}", on_click=goto_zone, args=(zid,))
        q = st.chat_input("Спросите об озеленении…") or pending
        if q:
            ask(q)
            st.rerun()

    elif page == PAGES[7]:
        st.subheader("Загрузка данных")
        st.markdown("**1. Дороги и соцобъекты из OpenStreetMap (для аналитики)**")
        st.caption("На карте улицы OSM загружаются автоматически в браузере. Для расчётов (расстояние до дороги, класс, загрязнение, защитные полосы) "
                   "сервер загружает граф через Overpass API при старте и кэширует его в data/osm_roads.json. Данные © OpenStreetMap contributors (ODbL).")
        c1, c2 = st.columns(2)
        if c1.button("🔄 Обновить дороги OSM", disabled=role == "Просмотр"):
            try:
                with st.spinner("Загрузка из Overpass…"):
                    ss.osm_roads = fetch_osm_roads()
                st.success(f"Загружено сегментов: {len(ss.osm_roads)}"); bump(); st.rerun()
            except Exception as e:
                st.error(f"Overpass недоступен с сервера: {e}. Аналитика использует синтетические расстояния (помечено).")
        if c2.button("🔄 Обновить школы, детсады, больницы, ЖК (OSM)", disabled=role == "Просмотр"):
            try:
                with st.spinner("Загрузка…"):
                    ss.osm_social = fetch_osm_social()
                st.success(f"Объектов: {len(ss.osm_social)}"); bump(); st.rerun()
            except Exception as e:
                st.error(f"Overpass недоступен: {e}")
        st.write(f"Сейчас для аналитики: дорог {len(W.roads)} ({'OSM' if W.roads_real else SYN}), соцобъектов {len(W.social)} ({'OSM' if W.social_real else SYN}).")
        st.markdown("**2. Участки из файла**")
        st.caption("GeoJSON, KML, CSV (lat/lon), Shapefile (zip). Метрические координаты приводятся к EPSG:4326 (по умолчанию UTM 43N, EPSG:32643).")
        f = st.file_uploader("Файл", type=["geojson", "json", "kml", "csv", "zip"], disabled=role == "Просмотр")
        if f:
            objs, issues = parse_upload(f.name, f.getvalue())
            for o in objs:
                for z in W.zones:
                    if polys_overlap(o["ring"], z["ring"]):
                        issues.append(f"«{o['props'].get('name', '?')}» пересекается с существующим участком {z['id']}")
                        break
            st.write(f"Объектов: **{len(objs)}**")
            if issues:
                st.warning("Проблемы качества данных:")
                st.dataframe(pd.DataFrame({"Проблема": issues}), hide_index=True, **WIDE)
            else:
                st.success("Проблем не обнаружено")
            if objs and st.button(f"➕ Добавить {len(objs)} участков"):
                for o in objs:
                    ss.extra_zones.append(W.new_zone_from_ring(o["ring"], o["props"], src=f"файл {f.name}"))
                bump(); st.rerun()
        st.download_button("Пример GeoJSON", json.dumps({"type": "FeatureCollection", "features": [
            {"type": "Feature", "properties": {"name": "Тестовый сквер", "purpose": "сквер"},
             "geometry": ring_to_geojson(_circle_ring(76.975, 52.29, 50))}]}, ensure_ascii=False), "primer.geojson")

    elif page == PAGES[8]:
        st.subheader("Справочники")
        can = role == "Администратор"
        if not can:
            st.info("Редактирование доступно роли «Администратор».")
        for title, key in [("Расценки (₸)", "rates_df"), ("Коэффициенты", "coeffs_df"), ("Каталог растений", "plants_df")]:
            st.markdown(f"**{title}**")
            ed = st.data_editor(ss[key], disabled=not can, **WIDE, key=f"ed_{key}", num_rows="dynamic" if key == "plants_df" else "fixed")
            if can and st.button(f"Сохранить: {title}", key=f"save_{key}"):
                ss[key] = ed.reset_index(drop=True)
                bump(); st.success("Сохранено, расчёты обновлены"); st.rerun()
        st.markdown("**Допущения и метод**")
        st.dataframe(assumptions_df(W), hide_index=True, **WIDE)

    elif page == PAGES[9]:
        st.subheader("История запросов и решений")
        st.markdown("**Утверждённые в план участки**")
        if ss.decisions:
            st.dataframe(pd.DataFrame(ss.decisions), hide_index=True, **WIDE)
        else:
            st.caption("Пока нет решений — нажмите «✅ Утвердить в план» в паспорте участка.")
        st.markdown("**Запросы к ассистенту**")
        if ss.history:
            st.dataframe(pd.DataFrame([{"Время": h["t"], "Вопрос": h["q"], "Инструмент": h["tool"], "Зоны": ", ".join(h["zones"])} for h in ss.history]),
                         hide_index=True, **WIDE)
            st.download_button("⬇️ Экспорт истории (JSON)", json.dumps(dict(history=ss.history, decisions=ss.decisions), ensure_ascii=False, indent=1), "history.json")
        else:
            st.caption("Пока нет запросов.")

    elif page == PAGES[10]:
        st.subheader("Инструкция (1 страница)")
        st.markdown(f"""
1. **ГИС-карта.** Выберите участок кликом, в списке, поиском адреса (🔎) или нарисуйте полигон инструментом слева на карте и нажмите «Создать зону».
   Улицы — реальная геометрия OpenStreetMap (загружается в браузере); клик по улице показывает класс, длину и расчёт защитной полосы.
2. **Слои** (дороги, проблемы, посадки, школы, детсады, больницы и ЖК из OSM с адресами и буфером 150 м, почвы, зелень, приоритеты) включаются галочками над картой.
3. **Паспорт участка** справа: параметры, AI-рекомендация, приживаемость, смета, кнопки «Утвердить в план», «AI-анализ», «Паспорт».
   Ниже — подробные вкладки: растения (исключить/зафиксировать/заменить), смета, приживаемость и уход, эффект, защитная полоса, отчёты.
4. **Проблемные зоны** — реестр с фильтрами. **Матрица приоритетов** — веса и оптимизатор бюджета. **Эффект и смета** — CO₂/O₂ и сценарии.
5. **AI-консультант** — в боковой панели (кнопка «>» слева вверху) и на отдельной вкладке; кнопки 📍 открывают участок.
6. **Данные** — обновление OSM и загрузка участков из файлов с проверкой качества. **Справочники** — расценки, коэффициенты, каталог (администратор).

{DISCLAIMER} Данные с пометкой «{SYN}» — демонстрационные; замените их данными организатора.""")


PAGES = ["🗺️ ГИС-карта", "⚠️ Проблемные зоны", "🌿 Каталог растений", "🎚️ Матрица приоритетов", "📈 Эффект и смета",
         "📊 Дашборд", "💬 AI-консультант", "📥 Данные", "📚 Справочники", "🕘 История", "❓ Инструкция"]


if __name__ == "__main__":
    main()
