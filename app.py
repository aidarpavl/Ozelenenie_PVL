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
             "residential": "жилой комплекс"}


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


def fetch_osm_roads(timeout=120):
    """Загружает дорожный граф Павлодара из OSM и кэширует в data/osm_roads.json."""
    import requests
    S, W, N, E = BBOX
    q = (f'[out:json][timeout:{timeout}];way["highway"~"^(motorway|trunk|primary|secondary|tertiary|unclassified|'
         f'residential|living_street|service|pedestrian|footway)$"]({S},{W},{N},{E});out geom tags;')
    r = requests.post(OVERPASS, data={"data": q}, timeout=timeout + 30, headers={"User-Agent": "greening-mvp/1.0"})
    r.raise_for_status()
    roads = []
    for el in r.json().get("elements", []):
        g = el.get("geometry")
        if not g or len(g) < 2:
            continue
        t = el.get("tags", {})
        roads.append(dict(rid=f"R-{el['id']}", name=t.get("name", "без названия"), highway=t.get("highway"),
                          line=[(p["lon"], p["lat"]) for p in g], oneway=t.get("oneway", "no"), lanes=t.get("lanes", ""),
                          surface=t.get("surface", ""), maxspeed=t.get("maxspeed", ""),
                          bridge=t.get("bridge") == "yes", tunnel=t.get("tunnel") == "yes",
                          src="© OpenStreetMap contributors (ODbL)"))
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, "osm_roads.json"), "w", encoding="utf-8") as f:
        json.dump(roads, f, ensure_ascii=False)
    return roads


def fetch_osm_social(timeout=90):
    import requests
    S, W, N, E = BBOX
    q = (f'[out:json][timeout:{timeout}];nwr["amenity"~"^(school|kindergarten|hospital|clinic)$"]({S},{W},{N},{E});'
         f'out center tags;')
    r = requests.post(OVERPASS, data={"data": q}, timeout=timeout + 30, headers={"User-Agent": "greening-mvp/1.0"})
    r.raise_for_status()
    res = []
    for el in r.json().get("elements", []):
        c = el.get("center") or {"lat": el.get("lat"), "lon": el.get("lon")}
        if c.get("lat") is None:
            continue
        t = el.get("tags", {})
        res.append(dict(lon=c["lon"], lat=c["lat"], kind=t.get("amenity"), name=t.get("name", SOCIAL_RU.get(t.get("amenity"), "")),
                        src="© OpenStreetMap contributors (ODbL)"))
    with open(os.path.join(DATA_DIR, "osm_social.json"), "w", encoding="utf-8") as f:
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
        self.social_real = social is not None
        self.social = social if social is not None else base["social"]
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
        z["social_list"] = ", ".join(sorted({SOCIAL_RU.get(s["kind"], s["kind"]) for s in soc})) or "нет"
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


# =============================================================================
# 7. ВЕБ-ИНТЕРФЕЙС (Streamlit)
# =============================================================================
def main():
    import streamlit as st
    import folium
    from folium.plugins import Draw
    try:
        from streamlit_folium import st_folium
    except ImportError:
        st_folium = None
    import streamlit.components.v1 as components

    st.set_page_config(page_title="AI-озеленение города", page_icon="🌳", layout="wide")
    ss = st.session_state

    @st.cache_resource(show_spinner="Обучение модели приживаемости…")
    def _cached_model():
        return train_survival_model()

    _MODEL["m"] = _cached_model()  # скрипт перезапускается при каждом действии — модель берём из кэша

    # ---------- состояние ----------
    if "base" not in ss:
        ss.base = generate_base_data()
        ss.plants_df = default_plants_df()
        ss.rates_df = default_rates_df()
        ss.coeffs_df = default_coeffs_df()
        ss.extra_zones = []
        ss.data_ver = 0
        ss.world_ver = -1
        ss.selected = "Z-01"
        ss.ov = {}
        ss.history = []
        ss.decisions = []
        ss.weights = {k: w for k, _, w in FACTORS}
        ss.last_tip = None
        ss.sel_road = None
        ss.processed_drawings = set()
        ss.osm_roads = load_cached("osm_roads.json")
        ss.osm_social = load_cached("osm_social.json")
        ss.page = "🗺️ Карта и зона"
        ss.found_point = None

    def bump():
        ss.data_ver += 1

    if ss.world_ver != ss.data_ver:
        with st.spinner("Пересчёт аналитики…"):
            ss.world = World(ss.base, ss.plants_df, ss.rates_df, ss.coeffs_df, ss.osm_roads, ss.osm_social, ss.extra_zones)
        ss.world_ver = ss.data_ver
    W = ss.world
    if not W.zone(ss.selected):
        ss.selected = W.zones[0]["id"]

    def goto_zone(zid):
        ss.selected = zid
        ss.page = "🗺️ Карта и зона"

    # ---------- боковая панель ----------
    PAGES = ["🗺️ Карта и зона", "⚠️ Проблемные зоны", "📊 Приоритеты и бюджет", "🌿 Каталог растений", "📈 Дашборд",
             "💬 Ассистент", "📥 Загрузка данных", "📚 Справочники", "🕘 История", "❓ Инструкция"]
    with st.sidebar:
        st.markdown("### 🌳 AI-ассистент озеленения\n**Павлодар** · MVP")
        st.radio("Раздел", PAGES, key="page", label_visibility="collapsed")
        role = st.selectbox("Роль (демо)", ["Специалист", "Администратор", "Просмотр"], key="role",
                            help="В MVP роль выбирается вручную. Аутентификация и HTTPS — на уровне развёртывания (reverse proxy).")
        st.divider()
        st.caption("Быстрый вопрос ассистенту")
        with st.form("side_chat", clear_on_submit=True):
            q = st.text_input("Вопрос", placeholder="Что озеленить на 5 млн?", label_visibility="collapsed")
            if st.form_submit_button("Спросить") and q:
                ans, zl, tool = assistant_answer(q, W, ss.selected, ss.weights)
                ss.history.append(dict(t=datetime.datetime.now().strftime("%H:%M:%S"), q=q, a=ans, zones=zl, tool=tool))
        if ss.history:
            h = ss.history[-1]
            st.markdown(f"**Вы:** {h['q']}")
            st.markdown(h["a"])
            cols = st.columns(min(4, max(1, len(h["zones"]))))
            for i, zid in enumerate(h["zones"][:8]):
                cols[i % len(cols)].button(f"📍{zid}", key=f"sb_{len(ss.history)}_{zid}", on_click=goto_zone, args=(zid,))
        st.divider()
        roads_src = "OSM (реальная геометрия)" if W.roads_real else f"{SYN} — загрузите OSM"
        st.caption(f"Дороги: {roads_src}\n\nСоцобъекты: {'OSM' if W.social_real else SYN}\n\nЗоны/почвы/насаждения: {SYN}")

    st.info(DISCLAIMER, icon="ℹ️")
    page = ss.page
    C = W.C

    def ov_of(zid):
        return ss.ov.setdefault(zid, {"fixed": [], "excluded": [], "replace": {}})

    # ---------- карта ----------
    def build_map(layers, basemap, center=None, zoom=13, plan=None):
        if basemap.startswith("2ГИС"):
            m = folium.Map(location=center or CENTER, zoom_start=zoom, tiles="https://tile2.maps.2gis.com/tiles?x={x}&y={y}&z={z}&v=1",
                           attr="© 2ГИС (растровые тайлы; проверьте условия использования)", max_zoom=18)
        elif basemap.startswith("Светлая"):
            m = folium.Map(location=center or CENTER, zoom_start=zoom, tiles="CartoDB positron")
        else:
            m = folium.Map(location=center or CENTER, zoom_start=zoom, tiles="OpenStreetMap")
        pr = W.priorities(ss.weights)
        prd = dict(zip(pr.id, pr.priority))
        if "Почвы" in layers:
            g = folium.FeatureGroup(name="Почвы")
            for s in W.soils:
                folium.GeoJson(ring_to_geojson(s["ring"]), style_function=lambda f, c=SOIL_TYPES[s["soil_type"]]["color"]: dict(fillColor=c, color=c, weight=0.5, fillOpacity=0.28),
                               tooltip=f"Почва: {s['soil_type']}, pH {s['ph']}, засоление {s['salinity']} дСм/м, плодородие {s['fertility']}/5, влажность {s['moisture']}/5 ({s['src']})").add_to(g)
            g.add_to(m)
        if "Существующая зелень" in layers:
            g = folium.FeatureGroup(name="Существующая зелень")
            cc = {"хорошее": "#2e7d32", "удовлетворительное": "#f9a825", "плохое": "#c62828"}
            for o in W.greens:
                folium.Circle([o["lat"], o["lon"]], radius=o["r"] * 3, color=cc[o["condition"]], fill=True, fill_opacity=0.6, weight=1,
                              tooltip=f"{o['species']}, {o['condition']}, {o['age']} лет ({o['src']})").add_to(g)
            g.add_to(m)
        if "Дороги" in layers:
            g = folium.FeatureGroup(name="Дороги (вектор, аналитика)")
            sel_xy = ring_xy(W.zone(ss.selected)["ring"])
            feats = []
            for r, xy, bb in zip(W.roads, W.rindex.xy, W.rindex.bb):
                major = r["highway"] in MAJOR_ROADS
                if not major and bbox_gap(bbox_of(sel_xy), bb) > 900:
                    continue  # мелкие дороги — только у выбранной зоны (быстрая загрузка)
                cr = ROAD_CLASSES.get(r["highway"], ("прочая", 0.2, 1.5, "#ddd", 0))
                order = cr[4] + (20 if r.get("bridge") else 0) - (20 if r.get("tunnel") else 0)
                feats.append(dict(type="Feature", geometry=dict(type="LineString", coordinates=[[a, b] for a, b in r["line"]]),
                                  properties=dict(rid=r["rid"], name=r["name"], cls=r["highway"], cls_ru=cr[0], len_m=int(r.get("len_m", 0)),
                                                  oneway="да" if r.get("oneway") == "yes" else "нет", w=cr[2], col=cr[3], order=order,
                                                  tunnel=bool(r.get("tunnel")))))
            feats.sort(key=lambda f: f["properties"]["order"])  # тоннели снизу, мосты сверху
            fc = dict(type="FeatureCollection", features=feats)
            folium.GeoJson(fc, style_function=lambda f: dict(color="#6b6b6b", weight=f["properties"]["w"] + 2, opacity=0.35 if f["properties"]["tunnel"] else 0.9)).add_to(g)
            folium.GeoJson(fc, style_function=lambda f: dict(color=f["properties"]["col"], weight=f["properties"]["w"],
                                                             opacity=0.4 if f["properties"]["tunnel"] else 1, dashArray="4 4" if f["properties"]["tunnel"] else None),
                           highlight_function=lambda f: dict(color="#0066ff", weight=f["properties"]["w"] + 2),
                           tooltip=folium.GeoJsonTooltip(fields=["rid", "name", "cls_ru", "len_m", "oneway"],
                                                         aliases=["Дорога", "Название", "Класс", "Длина, м", "Одностор."])).add_to(g)
            g.add_to(m)
        zg = folium.FeatureGroup(name="Зоны озеленения")
        for z in W.zones:
            if "Приоритеты" in layers:
                v = prd.get(z["id"], 0)
                col = "#b71c1c" if v >= 60 else ("#ef6c00" if v >= 45 else ("#fbc02d" if v >= 30 else "#2e7d32"))
            else:
                col = PURPOSE_COLOR.get(z["purpose"], "#2e7d32")
            sel = z["id"] == ss.selected
            folium.GeoJson(ring_to_geojson(z["ring"]),
                           style_function=lambda f, c=col, s=sel: dict(fillColor=c, color="#00e5ff" if s else c, weight=5 if s else 2, fillOpacity=0.45),
                           tooltip=f"{z['id']} · {z['purpose']} · {z['district']} · приоритет {prd.get(z['id'], 0)}").add_to(zg)
        zg.add_to(m)
        if "Проблемные зоны" in layers:
            g = folium.FeatureGroup(name="Проблемные зоны")
            for z in W.zones:
                if not z["problems"]:
                    continue
                sev = max(SEV_ORDER[p["Острота"]] for p in z["problems"])
                col = {3: "red", 2: "orange", 1: "beige"}[sev]
                folium.Marker([z["lat"], z["lon"]], icon=folium.Icon(color=col, icon="exclamation-sign"),
                              tooltip=f"{z['id']} — проблем: {len(z['problems'])}: " + "; ".join(p["Проблема"] for p in z["problems"])).add_to(g)
            g.add_to(m)
        if "Социальные объекты" in layers:
            g = folium.FeatureGroup(name="Социальные объекты")
            for s in W.social:
                folium.CircleMarker([s["lat"], s["lon"]], radius=4, color="#1565c0", fill=True,
                                    tooltip=f"{SOCIAL_RU.get(s['kind'], s['kind'])}: {s['name']} ({s['src']})").add_to(g)
            g.add_to(m)
        if "Предлагаемые посадки" in layers and plan:
            g = folium.FeatureGroup(name="Предлагаемые посадки")
            z = W.zone(plan["zone_id"])
            woody = [l for l in plan["lines"] if l["type"] != "газон"]
            n = sum(l["qty"] for l in woody)
            k = min(1.0, 250 / n) if n else 0
            pts = sample_points_in(z["ring"], int(sum(round(l["qty"] * k) for l in woody)), seed=hash(z["id"]) % 1000)
            i = 0
            for l in woody:
                for _ in range(int(round(l["qty"] * k))):
                    if i >= len(pts):
                        break
                    lo, la = pts[i]; i += 1
                    folium.CircleMarker([la, lo], radius=4 if l["type"] == "дерево" else 2.5, color="#1b5e20" if l["type"] == "дерево" else "#7cb342",
                                        fill=True, fill_opacity=0.9, weight=1, tooltip=f"{l['name']} (план)").add_to(g)
            g.add_to(m)
        if ss.found_point:
            lo, la, lab = ss.found_point
            folium.Marker([la, lo], tooltip=f"Найдено: {lab}", icon=folium.Icon(color="blue", icon="search")).add_to(m)
        Draw(export=False, draw_options=dict(polyline=False, circle=False, marker=False, circlemarker=False, rectangle=True, polygon=True),
             edit_options=dict(edit=False, remove=True)).add_to(m)
        folium.LayerControl(collapsed=True).add_to(m)
        return m

    def show_map(m, key, height=600):
        if st_folium is None:
            components.html(m.get_root().render(), height=height)
            st.warning("Пакет streamlit-folium не установлен: клик по карте и рисование недоступны (pip install streamlit-folium).")
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
            mr = re.search(r"R-\d+", tip)
            if mz and W.zone(mz.group(0)) and mz.group(0) != ss.selected:
                ss.selected = mz.group(0)
                st.rerun()
            elif mr:
                ss.sel_road = mr.group(0)
        drw = (out or {}).get("last_active_drawing")
        if drw and drw.get("geometry", {}).get("type") == "Polygon":
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

    # ===================== СТРАНИЦЫ =====================
    if page == "🗺️ Карта и зона":
        z = W.zone(ss.selected)
        t1, t2, t3, t4 = st.columns([2.2, 1.6, 1.6, 2])
        layers = t1.multiselect("Слои", ["Почвы", "Существующая зелень", "Проблемные зоны", "Приоритеты", "Предлагаемые посадки", "Дороги", "Социальные объекты"],
                                default=["Проблемные зоны", "Предлагаемые посадки", "Дороги"], key="layers")
        basemap = t2.selectbox("Подложка", ["OpenStreetMap", "Светлая (Carto)", "2ГИС (растровые тайлы)"] + (["2ГИС MapGL (ключ)"] if os.getenv("DGIS_API_KEY") else []), key="basemap")
        ids = [zz["id"] for zz in W.zones]
        ss.zone_pick = ss.selected
        t3.selectbox("Зона", ids, key="zone_pick", on_change=lambda: ss.__setitem__("selected", ss.zone_pick))
        with t4.form("addr", clear_on_submit=False):
            a1, a2 = st.columns([3, 1])
            addr = a1.text_input("Поиск адреса", placeholder="ул. Толстого 99", label_visibility="collapsed")
            if a2.form_submit_button("🔎") and addr:
                try:
                    res = geocode(addr)
                    if res:
                        lo, la, lab = res
                        ss.found_point = res
                        ins = [zz for zz in W.zones if pip(lo, la, zz["ring"])]
                        if ins:
                            ss.selected = ins[0]["id"]
                            st.success(f"Адрес внутри зоны {ins[0]['id']}")
                        else:
                            zz = W.new_zone_from_ring(_circle_ring(lo, la, 40), {"name": lab[:40]}, src=f"адрес «{addr}»")
                            ss.extra_zones.append(zz); ss.selected = zz["id"]; bump()
                            st.success(f"Создана зона {zz['id']} (круг 40 м) вокруг адреса")
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
        plan = W.plan(z["id"], variant, ov_of(z["id"]), ss.get("care_years"), None)

        cmap, cpanel = st.columns([1.35, 1])
        with cmap:
            if basemap == "2ГИС MapGL (ключ)":
                key = os.getenv("DGIS_API_KEY")
                data = [dict(id=zz["id"], coords=[[[a, b] for a, b in zz["ring"]] + [[zz["ring"][0][0], zz["ring"][0][1]]]],
                             col=PURPOSE_COLOR.get(zz["purpose"], "#2e7d32")) for zz in W.zones]
                html = f"""<div id="c" style="width:100%;height:600px"></div>
<script src="https://mapgl.2gis.com/api/js/v1"></script>
<script>
const map = new mapgl.Map('c', {{center:[{z['lon']},{z['lat']}], zoom:14, key:'{key}'}});
const Z = {json.dumps(data)};
Z.forEach(z => new mapgl.Polygon(map, {{coordinates: z.coords, color: z.col + '66', strokeColor: z.col, strokeWidth: 2}}));
</script>"""
                components.html(html, height=610)
                st.caption("2ГИС MapGL: просмотр. Выбор зоны и рисование — на подложке OSM.")
            else:
                m = build_map(layers, basemap, center=(z["lat"], z["lon"]), zoom=15, plan=plan)
                out = show_map(m, key=f"map_{ss.selected}_{ss.data_ver}")
                handle_map_output(out)
            st.caption("Клик по зоне — выбор; инструменты слева — нарисовать новый полигон; слои — в правом верхнем углу карты. "
                       + ("Дороги: OSM, реальная геометрия." if W.roads_real else "⚠ Векторные дороги — синтетика; реальные дороги видны на подложке. Загрузите OSM в «Загрузка данных»."))
            if ss.sel_road:
                r = next((rr for rr in W.roads if rr["rid"] == ss.sel_road), None)
                if r:
                    n_tr = int(r.get("len_m", 0) * 2 / 8)
                    cost = n_tr * (16000 + C["rates"]["plant_tree"]) + r.get("len_m", 0) * 2 * (3500 + C["rates"]["plant_shrub"])
                    st.info(f"🛣️ **{r['name']}** ({ROAD_CLASSES.get(r['highway'], (r['highway'],))[0]}), длина {r.get('len_m', 0):,.0f} м. "
                            f"Озеленение вдоль дороги (2 стороны, деревья через 8 м + живая изгородь): ≈ {n_tr} деревьев, ≈ {fmt_tg(cost)} (оценка).".replace(",", " "))

        with cpanel:
            st.subheader(f"{z['id']} · {z['name']}")
            st.caption(f"Источник данных: {z.get('src', '')}")
            vcol, ycol = st.columns(2)
            vcol.selectbox("Вариант", list(VARIANTS), index=list(VARIANTS).index(variant), key="variant")
            ycol.selectbox("Уход, лет", [None, 1, 2, 3], format_func=lambda v: "по варианту" if v is None else str(v), key="care_years")
            for w in plan["warnings"]:
                st.warning(w)
            tabs = st.tabs(["Обзор", "Растения", "Смета", "Приживаемость и уход", "Эффект", "Вдоль дороги", "Отчёт"])
            with tabs[0]:
                a, b, c = st.columns(3)
                a.metric("Площадь", f"{z['area_m2']:,.0f} м²".replace(",", " "))
                b.metric("Доля зелени", f"{z['green_share']:.0%}")
                pr = W.priorities(ss.weights)
                c.metric("Приоритет", f"{pr.set_index('id').loc[z['id'], 'priority']:.0f}/100")
                st.markdown(f"""
- **Назначение:** {z['purpose']} · **Район:** {z['district']}
- **Почва:** {z['soil_type']}, pH {z['ph']}, засоление {z['salinity']} дСм/м, плодородие {z['fertility']}/5, влажность {z['moisture']}/5
- **Ближайшая дорога:** {z['road_name']} — {z['road_class_ru']}, {z['dist_road']:.0f} м; нагрузка от дорог {z['road_exposure']:.2f}
- **Соцобъекты в 500 м:** {z['social_n']} ({z['social_list']})
- **Микроклимат:** t° поверхности {z['lst']} °C, затенение {z['shade']}/5, ветер {z['wind']}/5 ({SYN})
- **До зданий:** {z['bldg_dist']} м · **Население:** {z['pop_density']} чел/га ({SYN})""")
                if z["problems"]:
                    st.markdown("**Проблемы:**")
                    st.dataframe(pd.DataFrame(z["problems"]).drop(columns="zone"), hide_index=True, use_container_width=True)
                if role != "Просмотр" and st.button("💾 Сохранить решение по зоне"):
                    ss.decisions.append(dict(t=datetime.datetime.now().strftime("%d.%m %H:%M"), zone=z["id"], variant=variant,
                                             cost=round(plan["estimate"]["total"]), trees=sum(l["qty"] for l in plan["lines"] if l["type"] == "дерево"),
                                             surv3=round(plan["avg_surv"]["p3"] or 0, 3), co2_20=round(plan["eco"]["co2_cum20_t"], 1),
                                             ov=json.dumps(ov_of(z["id"]), ensure_ascii=False)))
                    st.success("Решение сохранено в «Историю»")
            with tabs[1]:
                sel, surv_t, eco_t = plan_tables(plan)
                st.dataframe(sel, hide_index=True, use_container_width=True,
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
                    st.dataframe(pd.DataFrame(plan["rejected"]), hide_index=True, use_container_width=True)
                st.markdown("**Сравнение вариантов**")
                rows = []
                for v in VARIANTS:
                    p = W.plan(z["id"], v, ov_of(z["id"]))
                    rows.append({"Вариант": v, "Стоимость, ₸": round(p["estimate"]["total"]),
                                 "Деревьев": sum(l["qty"] for l in p["lines"] if l["type"] == "дерево"),
                                 "Кустарников": sum(l["qty"] for l in p["lines"] if l["type"] == "кустарник"),
                                 "Приживаемость 3 г.": f"{(p['avg_surv']['p3'] or 0):.0%}", "CO₂ за 20 лет, т": round(p["eco"]["co2_cum20_t"], 1),
                                 "₸ за т CO₂": round(p["eco"]["cost_per_t"] or 0)})
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            with tabs[2]:
                est = plan["estimate"]
                st.metric("Итого (оценка)", fmt_tg(est["total"]))
                st.dataframe(est["table"], hide_index=True, use_container_width=True,
                             column_config={"Доля": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
                                            "Сумма, ₸": st.column_config.NumberColumn(format="%d")})
                st.bar_chart(est["table"].set_index("Статья")["Сумма, ₸"])
                st.caption("Расценки — «Справочники». Источник: " + SRC_PRICE)
            with tabs[3]:
                M = train_survival_model()
                st.dataframe(surv_t, hide_index=True, use_container_width=True,
                             column_config={k: st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f") for k in ["1 год", "2 год", "3 год"]})
                st.caption(f"Метод: 50% экспертные правила + 50% GradientBoosting (scikit-learn). ROC-AUC на тесте: {M['auc']} (правила {M['auc_rule']}) — "
                           f"на СИНТЕТИЧЕСКИХ данных; нужна калибровка на данных акимата или подтверждение экспертом.")
                if plan["lines"]:
                    sp = st.selectbox("Факторы и меры для вида", [l["name"] for l in plan["lines"]], key=f"surv_sp_{z['id']}_{variant}_{len(ss.ov.get(z['id'], {}).get('replace', {}))}")
                    l = next((x for x in plan["lines"] if x["name"] == sp), plan["lines"][0])
                    cdf = pd.DataFrame([{"Фактор": FEAT_RU[k], "Вклад, п.п.": v} for k, v in l["surv"]["contrib"].items() if abs(v) >= 0.1]).sort_values("Вклад, п.п.")
                    if len(cdf):
                        st.bar_chart(cdf.set_index("Фактор"))
                    st.caption(f"Вклад правил относительно идеальных условий. Диапазон 1-го года: {l['surv']['lo']:.0%}–{l['surv']['hi']:.0%}, уверенность — {l['surv']['conf']}.")
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
                                 hide_index=True, use_container_width=True)
                st.markdown("**Календарь ухода (1-й год)**")
                mv = st.slider("Допустимо выездов бригады в месяц", 2, 12, 6, key="max_visits")
                care_df, warns = care_calendar(plan, z, mv)
                for w in warns:
                    st.warning(w)
                st.caption(f"Бюджет ухода по смете: {fmt_tg(plan['estimate']['care_per_year'])} в год · режим полива: {WATER_RU[plan['V']['watering']]}")
                st.dataframe(care_df, hide_index=True, use_container_width=True, height=300)
            with tabs[4]:
                e = plan["eco"]
                st.caption("ОЦЕНКА. Коэффициенты и допущения — «Справочники».")
                cols = st.columns(3)
                for c_, Y in zip(cols, (5, 10, 20)):
                    v = e["by_year"][Y]
                    c_.metric(f"CO₂ через {Y} лет", f"{v['co2_t']:.1f} т/год")
                    c_.caption(f"O₂ {v['o2_t']:.1f} т/год · пыль {v['dust_kg']:.0f} кг/год · t° −{v['cooling_c']:.2f} °C · осадки {v['rain_m3']:.0f} м³/год")
                st.dataframe(eco_t, hide_index=True, use_container_width=True)
                b = e["before"]
                a20 = e["by_year"][20]
                st.markdown(f"**До / после (20 лет):** доля зелени {b['green_share']:.0%} → {a20['green_share_after']:.0%}; "
                            + (f"зелень на жителя {b['gpc']:.1f} → {a20['gpc_after']:.1f} м²/чел; " if b["gpc"] else "")
                            + f"t° поверхности {b['lst']} → ≈{b['lst'] - a20['cooling_c']:.1f} °C")
                st.metric("Стоимость единицы эффекта", f"{fmt_tg(e['cost_per_t'] or 0)} за 1 т CO₂", help=f"Смета / CO₂ за 20 лет ({e['co2_cum20_t']:.1f} т)")
                st.line_chart(pd.DataFrame({"CO₂, т/год": [eco_effect(plan, z, C, (Y,))["by_year"][Y]["co2_t"] for Y in range(1, 21)]}, index=range(1, 21)))
            with tabs[5]:
                st.caption("Собственная идея: расчёт защитной полосы вдоль ближайших дорог (буфер, виды, стоимость) — для школ, больниц и жилья у магистралей.")
                if not z["near_roads"]:
                    st.info("В радиусе 400 м дорог нет.")
                else:
                    rr = [(d, next(r for r in W.roads if r["rid"] == rid)) for d, rid in z["near_roads"]]
                    st.dataframe(pd.DataFrame([{"Дорога": r["rid"], "Название": r["name"], "Класс": ROAD_CLASSES.get(r["highway"], (r["highway"],))[0],
                                                "До зоны, м": d, "Длина, м": r.get("len_m")} for d, r in rr]), hide_index=True, use_container_width=True)
                    d0, r0 = rr[0]
                    L = min(r0.get("len_m", 0), 2 * math.sqrt(z["area_m2"]))
                    good = sorted([p for p in C["plants"] if p["type"] != "газон" and p["pollution_tol"] >= 4 and suitability(p, z, C["coeffs"])[0] is not None],
                                  key=lambda p: -suitability(p, z, C["coeffs"])[0])
                    tr = [p for p in good if p["type"] == "дерево"][:2]
                    sh = [p for p in good if p["type"] == "кустарник"][:1]
                    n_tr = int(L / 6) * 2 if tr else 0
                    n_sh = int(L / 0.7) if sh else 0
                    cost = n_tr * (np.mean([p["price"] for p in tr]) if tr else 0) + n_sh * (sh[0]["price"] if sh else 0) + n_tr * C["rates"]["plant_tree"] + n_sh * C["rates"]["plant_shrub"]
                    st.markdown(f"**Защитная полоса со стороны «{r0['name']}»** (длина участка ≈ {L:.0f} м, ширина 10–15 м): "
                                f"2 ряда деревьев через 6 м — **{n_tr} шт.** ({', '.join(p['name'] for p in tr) or 'нет подходящих'}), "
                                f"живая изгородь — **{n_sh} шт.** ({', '.join(p['name'] for p in sh) or 'нет подходящих'}). "
                                f"Стоимость посадки ≈ **{fmt_tg(cost)}** (оценка, без ухода).")
            with tabs[6]:
                rkey = (z["id"], variant, json.dumps(ov_of(z["id"]), sort_keys=True, ensure_ascii=False), ss.get("care_years"), ss.data_ver)
                if st.button("📄 Сформировать отчёты по зоне"):
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
                    else:
                        st.caption("PDF недоступен (нужны reportlab и шрифт с кириллицей). Используйте HTML → печать в PDF.")
                    st.download_button("⬇️ HTML-отчёт", rep_["html"], f"otchet_{z['id']}.html", mime="text/html")
                else:
                    st.caption("Нажмите кнопку, чтобы сформировать Excel, PDF и HTML для текущего варианта.")

    elif page == "⚠️ Проблемные зоны":
        st.header("Проблемные зоны")
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
            a.metric("Проблем", len(d)); b.metric("Высокой остроты", int((d["Острота"] == "высокая").sum())); c.metric("Зон", d["Зона"].nunique())
            st.dataframe(d, hide_index=True, use_container_width=True, height=420)
            st.bar_chart(d["Проблема"].value_counts())
            cols = st.columns(8)
            for i, zid in enumerate(d["Зона"].unique()[:16]):
                cols[i % 8].button(f"📍{zid}", key=f"pz_{zid}", on_click=goto_zone, args=(zid,))
            m = build_map(["Проблемные зоны", "Приоритеты"], "Светлая (Carto)", zoom=12)
            handle_map_output(show_map(m, key=f"pmap_{ss.data_ver}", height=480))

    elif page == "📊 Приоритеты и бюджет":
        st.header("Приоритет зон (0–100)")
        st.caption("Веса настраиваются; пересчёт мгновенный. Вклад факторов показан для каждой зоны.")
        cols = st.columns(len(FACTORS))
        for c_, (k, n, _) in zip(cols, FACTORS):
            ss.weights[k] = c_.slider(n, 0, 50, int(ss.weights[k]), key=f"w_{k}")
        pr = W.priorities(ss.weights)
        show = pr[["Ранг", "id", "district", "purpose", "priority"] + [f"c_{k}" for k, _, _ in FACTORS]].rename(
            columns={"id": "Зона", "district": "Район", "purpose": "Назначение", "priority": "Приоритет", **{f"c_{k}": n for k, n, _ in FACTORS}})
        st.dataframe(show, hide_index=True, use_container_width=True, height=380,
                     column_config={"Приоритет": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f")})
        st.markdown("**Вклад факторов (топ-12)**")
        st.bar_chart(show.head(12).set_index("Зона")[[n for _, n, _ in FACTORS]])
        st.divider()
        st.subheader("Дорожная карта под бюджет («лучшее на сумму N»)")
        b = st.number_input("Бюджет, ₸", min_value=1_000_000, value=50_000_000, step=5_000_000)
        rd, spent = budget_roadmap(W.zones, W.all_plans(), pr, b)
        if rd.empty:
            st.warning("Бюджета не хватает ни на один проект.")
        else:
            a, b2, c = st.columns(3)
            a.metric("Зон в плане", len(rd)); b2.metric("Использовано", fmt_tg(spent)); c.metric("CO₂ за 20 лет", f"{rd["CO₂ за 20 лет, т"].sum():.0f} т")
            st.dataframe(rd, hide_index=True, use_container_width=True)
            cols = st.columns(8)
            for i, zid in enumerate(rd["Зона"][:16]):
                cols[i % 8].button(f"📍{zid}", key=f"bz_{zid}", on_click=goto_zone, args=(zid,))
            bio = io.BytesIO()
            with pd.ExcelWriter(bio, engine="openpyxl") as xw:
                show.to_excel(xw, sheet_name="Приоритеты", index=False); rd.to_excel(xw, sheet_name="Дорожная карта", index=False)
                W.problems_df().to_excel(xw, sheet_name="Проблемные зоны", index=False); assumptions_df(W).to_excel(xw, sheet_name="Допущения", index=False)
            st.download_button("⬇️ Excel: приоритеты, дорожная карта, проблемные зоны", bio.getvalue(), "prioritety.xlsx")

    elif page == "🌿 Каталог растений":
        st.header("Каталог растений")
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
        st.dataframe(d.rename(columns=PLANT_RU), hide_index=True, use_container_width=True, height=520)
        st.caption(f"Источник: {SRC_PLANT}. Цены — {SRC_PRICE}. Редактирование — «Справочники» (администратор).")

    elif page == "📈 Дашборд":
        st.header("Состояние города")
        pr = W.priorities(ss.weights)
        pdf_ = W.problems_df()
        opt = W.opt_plans()
        a, b, c, d, e = st.columns(5)
        a.metric("Зон", len(W.zones))
        b.metric("Площадь зон", f"{sum(z['area_ha'] for z in W.zones):.1f} га")
        c.metric("Средняя доля зелени", f"{np.mean([z['green_share'] for z in W.zones]):.0%}")
        d.metric("Проблем высокой остроты", int((pdf_["Острота"] == "высокая").sum()) if len(pdf_) else 0)
        e.metric("Стоимость «Оптимальный» по городу", f"{sum(p['estimate']['total'] for p in opt.values()) / 1e6:.0f} млн ₸")
        l, r = st.columns(2)
        with l:
            st.markdown("**Приоритет по зонам**")
            st.bar_chart(pr.set_index("id")["priority"])
            st.markdown("**Проблемы по типам**")
            if len(pdf_): st.bar_chart(pdf_["Проблема"].value_counts())
        with r:
            st.markdown("**Эффект по годам, если реализовать «Оптимальный» во всех зонах (оценка)**")
            yrs = list(range(1, 21))
            st.area_chart(pd.DataFrame({"CO₂, т/год": [sum(eco_effect(p, W.zone(zid), C, (Y,))["by_year"][Y]["co2_t"] for zid, p in opt.items()) for Y in yrs]}, index=yrs))
            st.markdown("**Доля зелени по районам: до / после (20 лет)**")
            rows = []
            for dist in sorted({z["district"] for z in W.zones}):
                zs = [z for z in W.zones if z["district"] == dist]
                A = sum(z["area_m2"] for z in zs)
                rows.append(dict(Район=dist, До=sum(z["green_share"] * z["area_m2"] for z in zs) / A,
                                 После=sum(opt[z["id"]]["eco"]["by_year"][20]["green_share_after"] * z["area_m2"] for z in zs) / A))
            st.bar_chart(pd.DataFrame(rows).set_index("Район"))
        m = build_map(["Приоритеты", "Проблемные зоны"], "Светлая (Carto)", zoom=12)
        handle_map_output(show_map(m, key=f"dmap_{ss.data_ver}", height=460))

    elif page == "💬 Ассистент":
        st.header("Диалоговый ассистент")
        st.caption(f"Ассистент вызывает функции модулей (приоритет, подбор, смета, приживаемость, эффект, бюджет) и ищет по каталогу. "
                   f"Текущая зона: **{ss.selected}**. Ответы — оценки на основе данных системы; если данных нет — ассистент сообщает.")
        ex = ["Какие участки в Северном районе требуют озеленения в первую очередь?", "Подбери растения для сквера на этой зоне и посчитай стоимость на 3 года",
              "Какая будет приживаемость, если заменить клён на липу?", "Что можно озеленить на 5 млн при максимальном эффекте?"]
        cols = st.columns(4)
        pending = None
        for c_, e_ in zip(cols, ex):
            if c_.button(e_, key="ex_" + e_[:10], use_container_width=True):
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
            ans, zl, tool = assistant_answer(q, W, ss.selected, ss.weights)
            ss.history.append(dict(t=datetime.datetime.now().strftime("%H:%M:%S"), q=q, a=ans, zones=zl, tool=tool))
            st.rerun()

    elif page == "📥 Загрузка данных":
        st.header("Загрузка данных")
        st.subheader("1. Дороги и соцобъекты из OpenStreetMap")
        st.caption("Overpass API: highway=* (магистрали … пешеходные), атрибуты name, highway, lanes, surface, oneway, maxspeed, bridge, tunnel. "
                   "Кэшируется в data/osm_roads.json. Данные © OpenStreetMap contributors (ODbL).")
        c1, c2 = st.columns(2)
        if c1.button("⬇️ Загрузить дороги OSM", disabled=role == "Просмотр"):
            try:
                with st.spinner("Загрузка из Overpass (до 2 мин)…"):
                    ss.osm_roads = fetch_osm_roads()
                st.success(f"Загружено сегментов: {len(ss.osm_roads)}"); bump(); st.rerun()
            except Exception as e:
                st.error(f"Overpass недоступен: {e}. Остаются синтетические дороги (подложка карты всё равно показывает реальные).")
        if c2.button("⬇️ Загрузить школы/детсады/больницы OSM", disabled=role == "Просмотр"):
            try:
                with st.spinner("Загрузка…"):
                    ss.osm_social = fetch_osm_social()
                st.success(f"Объектов: {len(ss.osm_social)}"); bump(); st.rerun()
            except Exception as e:
                st.error(f"Overpass недоступен: {e}")
        st.write(f"Сейчас: дорог {len(W.roads)} ({'OSM' if W.roads_real else SYN}), соцобъектов {len(W.social)} ({'OSM' if W.social_real else SYN}).")
        st.subheader("2. Зоны из файла")
        st.caption("GeoJSON, KML, CSV (lat/lon), Shapefile (zip). Метрические координаты приводятся к EPSG:4326 (по умолчанию UTM 43N, EPSG:32643). "
                   "Атрибуты: purpose/назначение, name, green_share, pop_density.")
        f = st.file_uploader("Файл", type=["geojson", "json", "kml", "csv", "zip"], disabled=role == "Просмотр")
        if f:
            objs, issues = parse_upload(f.name, f.getvalue())
            for o in objs:
                for z in W.zones:
                    if polys_overlap(o["ring"], z["ring"]):
                        issues.append(f"«{o['props'].get('name', '?')}» пересекается с существующей зоной {z['id']}")
                        break
            st.write(f"Объектов: **{len(objs)}**")
            if issues:
                st.warning("Проблемы качества данных:")
                st.dataframe(pd.DataFrame({"Проблема": issues}), hide_index=True, use_container_width=True)
            else:
                st.success("Проблем не обнаружено")
            if objs and st.button(f"➕ Добавить {len(objs)} зон"):
                for o in objs:
                    z = W.new_zone_from_ring(o["ring"], o["props"], src=f"файл {f.name}")
                    ss.extra_zones.append(z)
                    W.zones.append(W.enrich(dict(z)))
                bump(); st.rerun()
        st.download_button("Пример GeoJSON", json.dumps({"type": "FeatureCollection", "features": [
            {"type": "Feature", "properties": {"name": "Тестовый сквер", "purpose": "сквер"},
             "geometry": ring_to_geojson(_circle_ring(76.975, 52.29, 50))}]}, ensure_ascii=False), "primer.geojson")

    elif page == "📚 Справочники":
        st.header("Справочники")
        can = role == "Администратор"
        if not can:
            st.info("Редактирование доступно роли «Администратор».")
        for title, key in [("Расценки (₸)", "rates_df"), ("Коэффициенты", "coeffs_df"), ("Каталог растений", "plants_df")]:
            st.subheader(title)
            ed = st.data_editor(ss[key], disabled=not can, use_container_width=True, key=f"ed_{key}", num_rows="dynamic" if key == "plants_df" else "fixed")
            if can and st.button(f"Сохранить: {title}", key=f"save_{key}"):
                ss[key] = ed.reset_index(drop=True)
                bump(); st.success("Сохранено, расчёты обновлены"); st.rerun()
        st.subheader("Допущения и метод")
        st.dataframe(assumptions_df(W), hide_index=True, use_container_width=True)

    elif page == "🕘 История":
        st.header("История запросов и решений (контекстная память сессии)")
        st.subheader("Решения")
        st.dataframe(pd.DataFrame(ss.decisions), hide_index=True, use_container_width=True) if ss.decisions else st.caption("Пока нет сохранённых решений.")
        st.subheader("Запросы к ассистенту")
        if ss.history:
            st.dataframe(pd.DataFrame([{"Время": h["t"], "Вопрос": h["q"], "Инструмент": h["tool"], "Зоны": ", ".join(h["zones"])} for h in ss.history]),
                         hide_index=True, use_container_width=True)
            st.download_button("⬇️ Экспорт истории (JSON)", json.dumps(dict(history=ss.history, decisions=ss.decisions), ensure_ascii=False, indent=1), "history.json")
        else:
            st.caption("Пока нет запросов.")

    elif page == "❓ Инструкция":
        st.header("Инструкция (1 страница)")
        st.markdown(f"""
1. **Выберите зону** на «Карте»: кликните по полигону, выберите в списке, найдите адрес (🔎) или нарисуйте полигон/прямоугольник инструментом слева на карте и нажмите «Создать зону».
2. **Слои** (почвы, зелень, проблемы, приоритеты, посадки, дороги, соцобъекты) включаются над картой; подложка — OSM/Carto/2ГИС.
3. В панели зоны выберите **вариант** (эконом / оптимальный / максимальный эффект) и срок ухода.
   - «Растения» — подбор с пригодностью 0–100 и обоснованием; можно исключить, зафиксировать или заменить вид — всё пересчитывается.
   - «Смета» — статьи, формулы и доли. «Приживаемость и уход» — прогноз на 1/2/3 года, факторы риска, меры, календарь.
   - «Эффект» — CO₂, O₂, пыль, t°, осадки на 5/10/20 лет, «до/после», ₸ за т CO₂. «Вдоль дороги» — защитная полоса.
   - «Отчёт» — Excel, PDF, HTML.
4. **Проблемные зоны** — таблица с фильтрами и картой. **Приоритеты** — ползунки весов и дорожная карта под бюджет.
5. **Ассистент** — вопросы на русском; кнопки 📍 открывают зону на карте.
6. **Загрузка данных** — реальные дороги и соцобъекты из OSM, зоны из файлов с проверкой качества.
7. **Справочники** — расценки, коэффициенты, каталог (роль «Администратор»).

{DISCLAIMER} Данные с пометкой «{SYN}» — демонстрационные; замените их данными организатора.""")


if __name__ == "__main__":
    main()
