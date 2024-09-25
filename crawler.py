import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import xml.etree.ElementTree as ET
from io import BytesIO
import openai
import json
import os
import re
from urllib.parse import urlparse, urljoin
import matplotlib.pyplot as plt
from pydantic import BaseModel, Field, ValidationError
import plotly.graph_objects as go



# Access API keys from st.secrets
OPENAI_API_KEY = st.secrets["openai_api_key"]
PAGESPEED_API_KEY = st.secrets["pagespeed_api_key"]

# Set OpenAI API key
openai.api_key = OPENAI_API_KEY


# URL gifa ładowania
LOADING_GIF_URL = "https://media.giphy.com/media/LML5ldpTKLPelFtBfY/giphy.gif"

class MetaTags(BaseModel):
    title: str = Field(..., max_length=60)
    description: str = Field(..., max_length=160)

def generate_meta_tags(content, context):
    try:
        # Tworzenie promptu
        prompt = f"Treść: {content}\n"
        if context:
            prompt += f"Kontekst: {context}\n"
        prompt += "\nWygeneruj meta title (maksymalnie 60 znaków) i meta description (maksymalnie 160 znaków) w formacie JSON. Użyj kluczy 'title' i 'description'."

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Jesteś ekspertem SEO. Twoim zadaniem jest generowanie zoptymalizowanych meta tagów na podstawie podanej treści i kontekstu.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )

        response_content = response.choices[0].message.content

        # Próba parsowania jako MetaTags
        try:
            meta_tags = MetaTags.model_validate_json(response_content)
            return meta_tags.title, meta_tags.description
        except ValidationError:
            # Jeśli nie udało się sparsować jako MetaTags, próbujemy elastycznego podejścia
            data = json.loads(response_content)

            title = data.get("title") or data.get("meta_title") or ""
            description = data.get("description") or data.get("meta_description") or ""

            # Upewniamy się, że długość jest odpowiednia
            title = title[:60]
            description = description[:160]

            return title, description

    except Exception as e:
        return (
            f"Błąd generowania meta tagów: {str(e)}",
            f"Błąd generowania meta tagów: {str(e)}",
        )



def fetch_url(url, elements, generate_new_meta, context, optimize_headings):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        result = {'URL': url}

        for element in elements:
            if element == 'H1':
                h1 = soup.find('h1')
                result['H1'] = h1.get_text(strip=True) if h1 else ''
            elif element == 'Wszystkie nagłówki':
                headers = []
                for header in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                    level = header.name.upper()
                    text = header.get_text(strip=True)
                    headers.append(f"{level}: {text}")
                result['Wszystkie nagłówki'] = '\n'.join(headers)
            elif element == 'Meta title':
                title = soup.find('title')
                result['Meta title'] = title.get_text(strip=True) if title else ''
            elif element == 'Meta description':
                meta_desc = soup.find('meta', attrs={'name': 'description'})
                result['Meta description'] = meta_desc['content'].strip() if meta_desc else ''
            elif element == 'Canonical':
                canonical = soup.find('link', attrs={'rel': 'canonical'})
                if canonical:
                    canonical_url = canonical['href']
                    if canonical_url == url:
                        result['Canonical'] = 'self reference'
                    else:
                        result['Canonical'] = 'other'
                else:
                    result['Canonical'] = 'brak'

        # Generowanie nowych meta tagów
        if generate_new_meta and OPENAI_API_KEY and ('Meta title' in elements or 'Meta description' in elements):
            content = soup.get_text()[:1000]  # Ograniczamy treść do pierwszych 1000 znaków
            new_meta_title, new_meta_description = generate_meta_tags(content, context)
            result['Nowy Meta title'] = new_meta_title
            result['Nowy Meta description'] = new_meta_description

        # Jeśli optymalizacja nagłówków jest włączona, zbieramy dane
        if optimize_headings:
            content = soup.get_text(separator='\n', strip=True)
            content = content[:5000]  # Ograniczamy do 5000 znaków

            existing_headings = []
            for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                level = tag.name.upper()
                text = tag.get_text(strip=True)
                existing_headings.append({
                    'level': level,
                    'text': text
                })

            result['content_for_optimization'] = content
            result['existing_headings'] = existing_headings

        return result
    except Exception as e:
        return {'URL': url, 'Error': str(e)}



    
def generate_optimized_headings(content, existing_headings):
    prompt = f"""Jesteś ekspertem SEO i copywriterem. Twoim zadaniem jest przeanalizowanie poniższej treści strony oraz istniejącej struktury nagłówków, a następnie zaproponowanie zoptymalizowanej struktury nagłówków (H1-H6), która poprawi SEO i rozszerzy pokrycie semantyczne tematu.

**UWAGA:** Zwróć **tylko** listę nagłówków w podanym formacie. **Nie dodawaj żadnych dodatkowych tekstów, wyjaśnień ani komentarzy** poza listą nagłówków.

Treść strony:
{content}

Istniejąca struktura nagłówków:
{json.dumps(existing_headings, ensure_ascii=False, indent=2)}

Na podstawie powyższej treści i istniejących nagłówków, zaproponuj zoptymalizowaną wersję tych nagłówków, uwzględniając najlepsze praktyki SEO. Nie dodawaj nowych nagłówków. **Zwróć tylko listę nagłówków w formacie:**

[
    {{"level": "H1", "text": "Twój tytuł H1"}},
    {{"level": "H2", "text": "Nagłówek H2"}},
    {{"level": "H3", "text": "Nagłówek H3"}},
    ...
]

**Nie dodawaj żadnych dodatkowych komentarzy ani tekstu.**

Twoja propozycja:
"""
    try:
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Jesteś ekspertem SEO i copywriterem specjalizującym się w optymalizacji struktury nagłówków."},
                {"role": "user", "content": prompt}
            ]
        )
        ai_response = response.choices[0].message.content.strip()
        
        # Próba parsowania odpowiedzi AI jako JSON
        try:
            optimized_headings_list = json.loads(ai_response)
            # Formatowanie nagłówków
            formatted_headings = ''
            for heading in optimized_headings_list:
                level = heading.get('level', '')
                text = heading.get('text', '')
                formatted_headings += f"{level}: {text}\n"
            return formatted_headings.strip()
        except json.JSONDecodeError:
            # Jeśli parsowanie się nie powiedzie, spróbuj wyciągnąć nagłówki za pomocą wyrażeń regularnych
            import re
            pattern = r'{"level":\s*"(?P<level>H[1-6])",\s*"text":\s*"(?P<text>.*?)"}'
            matches = re.findall(pattern, ai_response)
            if matches:
                formatted_headings = ''
                for level, text in matches:
                    formatted_headings += f"{level}: {text}\n"
                return formatted_headings.strip()
            else:
                return "Błąd: Nie udało się wyciągnąć nagłówków z odpowiedzi AI."
    except Exception as e:
        return f"Błąd podczas generowania zoptymalizowanej struktury nagłówków: {str(e)}"



def parse_sitemap(sitemap_url):
    try:
        response = requests.get(sitemap_url)
        response.raise_for_status()  # Sprawdź, czy nie było błędu HTTP
        content = response.content

        # Spróbuj sparsować jako XML
        try:
            root = ET.fromstring(content)
            namespaces = {'sm': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
            urls = [elem.text for elem in root.findall('.//sm:loc', namespaces)]
        except ET.ParseError:
            # Jeśli nie udało się sparsować jako XML, spróbuj jako zwykły tekst
            urls = [line.strip() for line in content.decode('utf-8').split('\n') if line.strip()]

        return urls
    except requests.RequestException as e:
        st.error(f"Błąd podczas pobierania sitemapy: {e}")
        return []

def extract_domain(url):
    parsed_uri = urlparse(url)
    domain = '{uri.netloc}'.format(uri=parsed_uri)
    return domain


def check_structured_data(url, page_type):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        scripts = soup.find_all('script', type='application/ld+json')
        
        if not scripts:
            errors = ['Brak danych strukturalnych na stronie.']
            recommendation = get_ai_recommendation(errors, url, page_type, schema_data=None)
            return {'Status': 'Brak danych strukturalnych', 'Rekomendacje': recommendation}
        
        schema_data = []
        errors = []
        
        for script in scripts:
            try:
                data = json.loads(script.string)
                schema_data.append(data)
            except json.JSONDecodeError as e:
                errors.append(f"Błąd parsowania JSON: {str(e)}")
        
        recommendation = get_ai_recommendation(errors, url, page_type, schema_data)
        
        return {'Status': 'Dane strukturalne znalezione', 'Rekomendacje': recommendation}
        
    except requests.RequestException as e:
        error = f'Błąd pobierania strony: {str(e)}'
        recommendation = get_ai_recommendation([error], url, page_type, schema_data=None)
        return {'Status': error, 'Rekomendacje': recommendation}
    except Exception as e:
        error = f'Nieoczekiwany błąd: {str(e)}'
        recommendation = get_ai_recommendation([error], url, page_type, schema_data=None)
        return {'Status': error, 'Rekomendacje': recommendation}
    
def get_ai_recommendation(errors, url, page_type, schema_data):
    domain = extract_domain(url)
    schema_data_str = json.dumps(schema_data, ensure_ascii=False, indent=2) if schema_data else "Brak danych"
    errors_str = '\n'.join(errors) if errors else "Brak błędów"

    prompt = f"""Jesteś specjalistą SEO z doświadczeniem w danych strukturalnych. Otrzymałeś dane schema ze strony {url}, która jest typu {page_type}. Oceń, czy obecne dane strukturalne są wystarczające i poprawne dla tego typu strony. Jeśli są błędy lub braki, zasugeruj, jakie dodatkowe dane strukturalne można dodać lub jak poprawić istniejące.

Dane schema z strony:
{schema_data_str}

Błędy podczas ekstrakcji danych:
{errors_str}

Twoja ocena i rekomendacje:"""
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Jesteś ekspertem SEO specjalizującym się w danych strukturalnych i optymalizacji stron internetowych."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Błąd podczas generowania rekomendacji: {str(e)}"

def find_menu_automatically(soup):
    # Próbujemy znaleźć elementy menu w typowych miejscach
    possible_menu_elements = [
        soup.find('nav'),
        soup.find(id=re.compile('menu', re.I)),
        soup.find(class_=re.compile('menu', re.I)),
        soup.find(id=re.compile('nav', re.I)),
        soup.find(class_=re.compile('nav', re.I))
    ]
    menu_element = next((elem for elem in possible_menu_elements if elem), None)
    return menu_element

def extract_menu(url, menu_selector=None):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # Szukamy głównego elementu menu
        menu_element = None
        if menu_selector:
            if menu_selector.startswith('#'):
                menu_element = soup.find(id=menu_selector[1:])
            elif menu_selector.startswith('.'):
                menu_element = soup.find(class_=menu_selector[1:])
            else:
                menu_element = soup.select_one(menu_selector)

        # Jeśli nie znaleziono menu za pomocą selektora, próbujemy znaleźć automatycznie
        if not menu_element:
            menu_element = find_menu_automatically(soup)

        if not menu_element:
            return "Nie znaleziono struktury menu. Spróbuj podać inny kod menu lub sprawdź strukturę strony."

        def parse_menu_item(item, parent=None):
            result = {}
            link = item.find('a')
            if link:
                result['text'] = link.get_text(strip=True)
                result['url'] = link.get('href')
                result['parent'] = parent

                submenu = item.find(['ul', 'ol'])
                if submenu:
                    result['children'] = []
                    for sub_item in submenu.find_all(['li', 'div'], recursive=False):
                        child = parse_menu_item(sub_item, result['text'])
                        if child:
                            result['children'].append(child)

            return result if result else None

        menu_structure = []
        for item in menu_element.find_all(['li', 'div'], recursive=False):
            parsed_item = parse_menu_item(item)
            if parsed_item:
                menu_structure.append(parsed_item)

        return menu_structure
    except Exception as e:
        return f"Wystąpił błąd podczas analizy menu: {str(e)}"

def extract_menu_from_code(menu_code):
    try:
        soup = BeautifulSoup(menu_code, 'html.parser')
        menu_element = soup

        def parse_menu_item(item, parent=None):
            result = {}
            link = item.find('a')
            if link:
                result['text'] = link.get_text(strip=True)
                result['url'] = link.get('href')
                result['parent'] = parent

                submenu = item.find(['ul', 'ol'])
                if submenu:
                    result['children'] = []
                    for sub_item in submenu.find_all(['li', 'div'], recursive=False):
                        child = parse_menu_item(sub_item, result['text'])
                        if child:
                            result['children'].append(child)

            return result if result else None

        menu_structure = []
        for item in menu_element.find_all(['li', 'div'], recursive=False):
            parsed_item = parse_menu_item(item)
            if parsed_item:
                menu_structure.append(parsed_item)

        return menu_structure
    except Exception as e:
        return f"Wystąpił błąd podczas analizy menu z kodu: {str(e)}"

def visualize_menu(menu):
    if isinstance(menu, str):
        # Zwracamy pustą strukturę w przypadku błędu
        return [menu], ['']

    nodes = []
    urls = []

    def process_item(item, level=0):
        prefix = "- " * level if level > 0 else ""
        nodes.append(f"{prefix}{item['text']}")
        urls.append(item.get('url', ''))

        if 'children' in item:
            for child in item['children']:
                process_item(child, level + 1)

    for item in menu:
        process_item(item)

    return nodes, urls

def analyze_menu_with_ai(menu_structure, html_content):
    prompt = f"""Jesteś ekspertem SEO specjalizującym się w optymalizacji menu na stronach internetowych. Twoim zadaniem jest przeanalizowanie dostarczonej struktury menu oraz kodu HTML i przygotowanie szczegółowych rekomendacji dotyczących optymalizacji menu pod kątem SEO, dostępności i użyteczności. W swojej analizie zwróć uwagę na następujące aspekty:

1. **Semantyka HTML**:
   - Czy menu wykorzystuje odpowiednie znaczniki HTML, takie jak `<nav>`, `<ul>`, `<li>`?
   - Czy struktura kodu jest zgodna z najlepszymi praktykami semantycznymi?

2. **Optymalizacja linków**:
   - Czy linki w menu mają przyjazne adresy URL?
   - Czy teksty kotwic (anchor text) są opisowe i zawierają istotne słowa kluczowe?

3. **Hierarchia i struktura menu**:
   - Czy menu jest logicznie zorganizowane z jasnym podziałem na kategorie i podkategorie?
   - Czy poziomy zagnieżdżenia są odpowiednie i nie powodują dezorientacji użytkownika?

4. **Dostępność (Accessibility)**:
   - Czy menu jest dostosowane do potrzeb osób z niepełnosprawnościami?
   - Czy zastosowano atrybuty ARIA oraz inne techniki zwiększające dostępność?

5. **Wydajność i optymalizacja**:
   - Czy menu nie jest zbyt rozbudowane, co może wpływać na czas ładowania strony?
   - Czy kod jest zoptymalizowany pod kątem szybkości renderowania?

6. **Responsywność**:
   - Czy menu jest przystosowane do urządzeń mobilnych i różnych rozdzielczości ekranu?
   - Czy nawigacja jest intuicyjna na ekranach dotykowych?

**Struktura menu:**
{menu_structure}

**Kod HTML menu:**
{html_content}

Przygotuj szczegółową ocenę aktualnego stanu menu oraz konkretne rekomendacje optymalizacyjne. Twoja odpowiedź powinna być:

- Napisana w sposób profesjonalny i rzeczowy.
- Zawierać konkretne przykłady i odniesienia do kodu, jeśli to możliwe.
- Skupiać się na praktycznych rozwiązaniach, które można wdrożyć.
- Unikać ogólników i marketingowego języka.

Rozpocznij swoją analizę poniżej."""

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Jesteś ekspertem SEO specjalizującym się w analizie i optymalizacji menu na stronach internetowych."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Wystąpił błąd podczas analizy AI: {str(e)}"

def visualize_menu(menu):
    if isinstance(menu, str):
        # Zwracamy pustą strukturę w przypadku błędu
        return [menu], ['']
    
    nodes = []
    urls = []
    
    def process_item(item, level=0):
        prefix = "- " * level if level > 0 else ""
        nodes.append(f"{prefix}{item['text']}")
        urls.append(item.get('url', ''))
        
        if 'children' in item:
            for child in item['children']:
                process_item(child, level + 1)
    
    for item in menu:
        process_item(item)
    
    return nodes, urls

def analyze_menu_with_ai(menu_structure, html_content):
    prompt = f"""Jesteś ekspertem SEO specjalizującym się w optymalizacji menu na stronach internetowych. Twoim zadaniem jest przeanalizowanie dostarczonej struktury menu oraz kodu HTML i przygotowanie szczegółowych rekomendacji dotyczących optymalizacji menu pod kątem SEO, dostępności i użyteczności. W swojej analizie zwróć uwagę na następujące aspekty:

1. **Semantyka HTML**:
   - Czy menu wykorzystuje odpowiednie znaczniki HTML, takie jak `<nav>`, `<ul>`, `<li>`?
   - Czy struktura kodu jest zgodna z najlepszymi praktykami semantycznymi?

2. **Optymalizacja linków**:
   - Czy linki w menu mają przyjazne adresy URL i czy mozna po nich domyślić się co jest zawartością strony?
   - Czy teksty kotwic (anchor text) są opisowe i zawierają istotne słowa kluczowe, czy mozna po nich domyślić się, co jest zawartością strony?

3. **Hierarchia i struktura menu**:
   - Czy menu jest logicznie zorganizowane z jasnym podziałem na kategorie i podkategorie?

4. **Dostępność (Accessibility)**:
   - Czy menu jest dostosowane do potrzeb osób z niepełnosprawnościami?
   - Czy zastosowano atrybuty ARIA oraz inne techniki zwiększające dostępność?

**Struktura menu:**
{menu_structure}

**Kod HTML menu:**
{html_content}

Przygotuj szczegółową ocenę aktualnego stanu menu oraz konkretne rekomendacje optymalizacyjne. Twoja odpowiedź powinna być:

- Napisana w sposób profesjonalny i rzeczowy.
- Zawierać konkretne przykłady i odniesienia do kodu, jeśli to możliwe.
- Skupiać się na praktycznych rozwiązaniach, które można wdrożyć.
- Unikać ogólników i marketingowego języka.

Rozpocznij swoją analizę poniżej."""
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Jesteś ekspertem SEO specjalizującym się w analizie i optymalizacji menu na stronach internetowych."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Wystąpił błąd podczas analizy AI: {str(e)}"
    
def extract_menu_advanced(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        all_menus = []
        content_start = soup.find(['h1', 'h2'])
        
        if content_start:
            search_area = []
            for element in content_start.previous_elements:
                if element.name == 'body':
                    break
                search_area.append(element)
            search_area.reverse()  # Odwracamy listę, aby zachować oryginalną kolejność
        else:
            search_area = soup.find_all()
        
        potential_menus = [elem for elem in search_area if elem.name in ['ul', 'ol', 'nav']]
        
        for menu in potential_menus:
            menu_items = []
            for item in menu.find_all('a', href=True):
                parent = item.find_parent('li')
                menu_items.append({
                    'text': item.get_text(strip=True),
                    'url': item['href'],
                    'parent': parent.get_text(strip=True) if parent else None
                })
            if menu_items:
                all_menus.append(menu_items)
        
        return all_menus
    except Exception as e:
        return f"Wystąpił błąd podczas analizy menu: {str(e)}"

def visualize_menu_advanced(menus):
    if isinstance(menus, str):  # Jeśli menus jest stringiem (komunikat o błędzie)
        return [menus], ['']
    
    all_nodes = []
    all_urls = []
    
    for i, menu in enumerate(menus, 1):
        nodes = [f"Menu {i}:"]
        urls = [""]
        for item in menu:
            prefix = "- " if item.get('parent') is None else "  - "
            nodes.append(f"{prefix}{item['text']}")
            urls.append(item['url'])
        all_nodes.extend(nodes)
        all_urls.extend(urls)
        all_nodes.append("")  # Pusta linia między menu
        all_urls.append("")
    
    return all_nodes, all_urls

def run_pagespeed_insights(url, strategy='mobile'):
    api_url = 'https://www.googleapis.com/pagespeedonline/v5/runPagespeed'
    params = {
        'url': url,
        'key': PAGESPEED_API_KEY,
        'strategy': strategy
    }
    try:
        response = requests.get(api_url, params=params)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.exceptions.HTTPError as http_err:
        st.error(f'Wystąpił błąd HTTP: {http_err}')
    except Exception as err:
        st.error(f'Wystąpił inny błąd: {err}')


def display_pagespeed_results(data):
    import plotly.graph_objects as go

    # Pobieranie wyniku Performance
    lighthouse_result = data.get('lighthouseResult', {})
    categories = lighthouse_result.get('categories', {})
    performance_score = categories.get('performance', {}).get('score', 0) * 100

    # Wyświetlanie wyniku Performance jako wykresu kołowego
    display_performance_gauge(performance_score)

    # Pobieranie metryk z audytów
    audits = lighthouse_result.get('audits', {})

    # Lista metryk do pobrania
    desired_metrics = {
        'first-contentful-paint': 'First Contentful Paint (FCP)',
        'largest-contentful-paint': 'Largest Contentful Paint (LCP)',
        'total-blocking-time': 'Total Blocking Time (TBT)',
        'cumulative-layout-shift': 'Cumulative Layout Shift (CLS)',
        'speed-index': 'Speed Index'
    }

    # Pobieranie i przechowywanie metryk
    metric_data = {}
    for audit_key, metric_name in desired_metrics.items():
        audit = audits.get(audit_key, {})
        display_value = audit.get('displayValue', '')
        score = audit.get('score', 0)
        numeric_value = audit.get('numericValue', 0)
        # Określenie kategorii na podstawie wyniku
        if score >= 0.9:
            category = 'Dobrze'
        elif score >= 0.5:
            category = 'Wymaga poprawy'
        else:
            category = 'Słabo'
        metric_data[metric_name] = {
            'display_value': display_value,
            'score': score,
            'numeric_value': numeric_value,
            'category': category
        }

    # Wyświetlanie metryk Core Web Vitals
    st.subheader('Core Web Vitals i metryki wydajności')

    num_metrics = len(metric_data)
    cols = st.columns(num_metrics)

    for idx, (metric_name, data) in enumerate(metric_data.items()):
        display_value = data['display_value']
        category = data['category']
        score = data['score']
        # Ustawienie strzałki i koloru na podstawie kategorii
        if category == 'Dobrze':
            delta_arrow = '↑'
            value_color = 'green'
        elif category == 'Wymaga poprawy':
            delta_arrow = '→'
            value_color = 'orange'
        else:
            delta_arrow = '↓'
            value_color = 'red'

        # Wyświetlanie metryki
        with cols[idx]:
            st.markdown(f"<h5 style='text-align: center;'>{metric_name}</h5>", unsafe_allow_html=True)
            st.markdown(f"<h3 style='color:{value_color}; text-align: center;'>{display_value}</h3>", unsafe_allow_html=True)
            st.markdown(f"<p style='text-align: center; color:{value_color};'>{delta_arrow} {category}</p>", unsafe_allow_html=True)

    # Zbieranie 'Opportunities' i 'Diagnostics' do analizy AI
    opportunity_audits = {k: v for k, v in audits.items() if v.get('details', {}).get('type') == 'opportunity'}
    opportunities = []
    for audit in opportunity_audits.values():
        title = audit.get('title')
        description = audit.get('description')
        display_value = audit.get('displayValue', '')
        opportunities.append(f"{title}: {display_value}\n{description}")

    diagnostics_audits = {k: v for k, v in audits.items() if v.get('scoreDisplayMode') == 'informative'}
    diagnostics = []
    for audit in diagnostics_audits.values():
        title = audit.get('title')
        description = audit.get('description')
        display_value = audit.get('displayValue', '')
        diagnostics.append(f"{title}: {display_value}\n{description}")

    # Przygotowanie danych CWV do analizy AI
    cwv_data_for_ai = {
        metric_name: {
            'value': data['display_value'],
            'category': data['category']
        } for metric_name, data in metric_data.items()
    }

    # Przekazanie danych do AI
    st.subheader("Rekomendacje AI dotyczące optymalizacji Core Web Vitals")
    ai_assessment = analyze_cwv_with_ai(cwv_data_for_ai, opportunities, diagnostics)
    st.info(ai_assessment)


def analyze_cwv_with_ai(cwv_data, opportunities, diagnostics):
    # Przygotowanie podsumowania metryk
    cwv_summary = "\n".join([f"{metric}: {data['value']} ({data['category']})" for metric, data in cwv_data.items()])

    # Przygotowanie sekcji 'Opportunities' i 'Diagnostics'
    opportunities_summary = "\n".join(opportunities) if opportunities else "Brak zaleceń w sekcji Opportunities."
    diagnostics_summary = "\n".join(diagnostics) if diagnostics else "Brak zaleceń w sekcji Diagnostics."

    # Przygotowanie promptu
    prompt = f"""Jesteś ekspertem w optymalizacji wydajności stron internetowych. Otrzymałeś następujące wyniki testu Pagespeed Insights dla strony:

**Core Web Vitals:**
{cwv_summary}

**Opportunities:**
{opportunities_summary}

**Diagnostics:**
{diagnostics_summary}

Na podstawie tych danych:

- Przeanalizuj obecną sytuację strony pod kątem wydajności.
- Wskaż, które obszary wymagają poprawy.
- Zaproponuj proste i zrozumiałe rekomendacje dotyczące optymalizacji strony, aby poprawić wyniki.

Twoja odpowiedź powinna być zwięzła i konkretna, napisana w języku polskim.
"""

    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Jesteś ekspertem w optymalizacji wydajności stron internetowych."},
                {"role": "user", "content": prompt}
            ]
        )
        ai_response = response.choices[0].message.content.strip()
        return ai_response
    except Exception as e:
        return f"Błąd podczas analizy AI: {str(e)}"



def display_performance_gauge(performance_score):
    import plotly.graph_objects as go

    # Określenie koloru na podstawie wyniku
    if performance_score < 50:
        color = "red"
    elif performance_score < 90:
        color = "orange"
    else:
        color = "green"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=performance_score,
        number={'suffix': "%"},
        title={'text': "Performance"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 50], 'color': 'red'},
                {'range': [50, 90], 'color': 'orange'},
                {'range': [90, 100], 'color': 'green'}
            ],
            'threshold': {
                'line': {'color': color, 'width': 4},
                'thickness': 0.75,
                'value': performance_score
            }
        }
    ))

    st.plotly_chart(fig, use_container_width=True)



def main():
    st.title('Rozszerzone narzędzie do audytu SEO')

    # Inicjalizacja zmiennych stanu sesji
    if 'stage' not in st.session_state:
        st.session_state.stage = 'input'
    if 'optimize_headings' not in st.session_state:
        st.session_state.optimize_headings = False
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = "Audyt SEO"
    if 'structured_data_results' not in st.session_state:
        st.session_state.structured_data_results = {}
    if 'results' not in st.session_state:
        st.session_state.results = []

    # Definiowanie zakładek
    tab1, tab2, tab3, tab4 = st.tabs(["Audyt SEO", "Dane strukturalne", "Tester menu", "Pagespeed Insights"])

    # Zakładka 1: Audyt SEO
    with tab1:
        if st.session_state.stage == 'input':
            # Umieszczamy elementy w kontenerze, aby móc je łatwo usunąć
            input_container = st.container()
            with input_container:
                input_type = st.radio("Wybierz typ wejścia:", ('Sitemap URL', 'Lista adresów URL'))

                urls = []
                if input_type == 'Sitemap URL':
                    sitemap_url = st.text_input('Wprowadź URL sitemapy:')
                    if sitemap_url:
                        urls = parse_sitemap(sitemap_url)
                        if not urls:
                            st.warning("Nie udało się pobrać adresów URL z podanej sitemapy. Upewnij się, że URL jest poprawny.")
                else:
                    urls_input = st.text_area('Wprowadź listę adresów URL (jeden na linię):')
                    urls = urls_input.split('\n')
                    urls = [url.strip() for url in urls if url.strip()]

                elements_to_fetch = st.multiselect(
                    'Wybierz elementy do pobrania:',
                    ['H1', 'Wszystkie nagłówki', 'Meta title', 'Meta description', 'Canonical'],
                    default=['H1', 'Wszystkie nagłówki', 'Meta title', 'Meta description', 'Canonical']
                )

                generate_new_meta = st.checkbox('Generuj nowe meta tagi za pomocą AI', value=False)
                optimize_headings = st.checkbox('Optymalizacja struktury nagłówków')

                context = ""
                if generate_new_meta:
                    context = st.text_area('Wprowadź kontekst dla generowania meta tagów:', 'np. strona produktowa dla butów męskich')

                if st.button('Rozpocznij audyt'):
                    if urls:
                        st.session_state.urls = urls
                        st.session_state.elements_to_fetch = elements_to_fetch
                        st.session_state.generate_new_meta = generate_new_meta
                        st.session_state.optimize_headings = optimize_headings
                        st.session_state.context = context
                        st.session_state.stage = 'crawling'
                        # Usuwamy kontener z elementami wejściowymi
                        input_container.empty()
                        st.rerun()  # Używamy st.rerun() zamiast st.experimental_rerun()
                    else:
                        st.warning('Proszę wprowadzić adresy URL do audytu.')

        elif st.session_state.stage == 'crawling':
            # Usuwamy elementy z poprzedniego etapu, jeśli istnieją
            st.empty()
            progress_placeholder = st.empty()
            status_placeholder = st.empty()
            gif_placeholder = st.empty()

            with gif_placeholder.container():
                st.image(LOADING_GIF_URL, width=200)

            results = []
            with ThreadPoolExecutor(max_workers=10) as executor:
                future_to_url = {
                    executor.submit(
                        fetch_url,
                        url,
                        st.session_state.elements_to_fetch,
                        st.session_state.generate_new_meta,
                        st.session_state.context,
                        st.session_state.optimize_headings  # Przekazujemy wartość optimize_headings
                    ): url for url in st.session_state.urls
                }
                for i, future in enumerate(as_completed(future_to_url)):
                    result = future.result()
                    results.append(result)
                    progress = (i + 1) / len(st.session_state.urls)
                    progress_placeholder.progress(progress)
                    status_placeholder.text(f'Przetworzono {i+1}/{len(st.session_state.urls)} URL-i')

            # Usuwanie placeholderów po zakończeniu
            progress_placeholder.empty()
            status_placeholder.empty()
            gif_placeholder.empty()

            # Przechowywanie wyników
            st.session_state.results = results

            # Jeśli optymalizacja nagłówków jest włączona
            if st.session_state.optimize_headings:
                optimization_progress_placeholder = st.empty()
                optimization_status_placeholder = st.empty()
                total_urls = len(results)
                optimization_progress_bar = optimization_progress_placeholder.progress(0)

                for i, result in enumerate(results):
                    if 'content_for_optimization' in result and 'existing_headings' in result:
                        with st.spinner(f'Optymalizacja nagłówków dla URL {result["URL"]}...'):
                            optimized_headings = generate_optimized_headings(
                                result['content_for_optimization'],
                                result['existing_headings']
                            )
                            result['Zoptymalizowana struktura nagłówków'] = optimized_headings
                            # Usuwamy niepotrzebne dane
                            del result['content_for_optimization']
                            del result['existing_headings']
                    else:
                        result['Zoptymalizowana struktura nagłówków'] = 'Brak danych do optymalizacji'

                    # Aktualizacja paska postępu
                    optimization_progress = (i + 1) / total_urls
                    optimization_progress_bar.progress(optimization_progress)
                    optimization_status_placeholder.text(f'Zoptymalizowano {i+1}/{total_urls} URL-i')

                # Usuwamy placeholdery
                optimization_progress_placeholder.empty()
                optimization_status_placeholder.empty()

            st.session_state.stage = 'results_ready'
            st.rerun()
            # Przejście do kolejnego etapu
            st.session_state.stage = 'results_ready'
            st.rerun()  # Używamy st.rerun() zamiast st.experimental_rerun()

        elif st.session_state.stage == 'results_ready':
            st.success("Audyt zakończony!")
            if st.button('Przejdź do wyników'):
                st.session_state.stage = 'show_results'
                st.rerun()  # Używamy st.rerun() zamiast st.experimental_rerun()

        elif st.session_state.stage == 'show_results':
            # Przygotowanie listy kolumn do wyświetlenia
            columns_to_display = ['URL'] + st.session_state.elements_to_fetch.copy()

            # Dodanie nowych meta tagów, jeśli były generowane
            if st.session_state.generate_new_meta:
                if 'Meta title' in st.session_state.elements_to_fetch:
                    columns_to_display.append('Nowy Meta title')
                if 'Meta description' in st.session_state.elements_to_fetch:
                    columns_to_display.append('Nowy Meta description')

            # Dodanie zoptymalizowanych nagłówków, jeśli były generowane
            if st.session_state.optimize_headings:
                columns_to_display.append('Zoptymalizowana struktura nagłówków')

            # Tworzenie DataFrame z wybranymi kolumnami
            df = pd.DataFrame([{k: v for k, v in result.items() if k in columns_to_display} for result in st.session_state.results])

            # Przygotowanie pliku Excel
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, sheet_name='SEO Audit', index=False)
                workbook = writer.book
                worksheet = writer.sheets['SEO Audit']

                # Formatowanie
                wrap_format = workbook.add_format({'text_wrap': True, 'valign': 'top'})
                for idx, col in enumerate(df.columns):
                    series = df[col]
                    max_len = max((
                        series.astype(str).map(len).max(),
                        len(str(series.name))
                    )) + 1
                    worksheet.set_column(idx, idx, max_len, wrap_format)

            excel_data = output.getvalue()

            st.download_button(
                label="Pobierz wyniki jako Excel",
                data=excel_data,
                file_name="seo_audit_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

            # Wyświetlanie interaktywnej tabeli z wynikami
            st.dataframe(df)




    # Zakładka 2: Dane strukturalne
    with tab2:
        st.header("Analiza danych strukturalnych")

        url = st.text_input("Wprowadź URL strony do analizy:", key='structured_data_url')
        page_type = st.radio("Wybierz typ strony:", ('Artykuł', 'Produkt', 'Strona usługowa'))

        if st.button("Sprawdź dane strukturalne"):
            if url:
                structured_data_result = check_structured_data(url, page_type)
                # Wyświetlanie wyników
                st.subheader("Wyniki analizy danych strukturalnych:")
                for key, value in structured_data_result.items():
                    if key == 'Status':
                        st.write(f"**Status:** {value}")
                    elif key == 'Rekomendacje':
                        st.write("**Rekomendacje AI:**")
                        st.info(value)
                    else:
                        st.write(f"- {key}: {value}")
            else:
                st.warning("Proszę wprowadzić URL do analizy.")

    # Zakładka 3: Tester menu
    with tab3:
        st.header("Tester menu")

        menu_url = st.text_input("Wprowadź URL strony do analizy menu:")
        menu_code = st.text_area("Wprowadź kod menu (opcjonalnie):")
        advanced_mode = st.checkbox("Tryb zaawansowany (analiza wszystkich list z linkami)")

        if st.button("Analizuj menu"):
            if menu_url:
                with st.spinner('Analizuję menu...'):
                    if menu_code.strip():
                        # Jeśli użytkownik podał kod menu, używamy go bezpośrednio
                        menu_html_str = menu_code
                        # Parsujemy menu z podanego kodu
                        menu_structure = extract_menu_from_code(menu_code)
                        if isinstance(menu_structure, str):  # Jeśli wystąpił błąd
                            st.error(menu_structure)
                            return

                        nodes, urls = visualize_menu(menu_structure)

                    else:
                        # Jeśli nie podano kodu menu, postępujemy jak wcześniej
                        if advanced_mode:
                            menu_structure = extract_menu_advanced(menu_url)
                            if isinstance(menu_structure, str):  # Jeśli wystąpił błąd
                                st.error(menu_structure)
                                return
                            nodes, urls = visualize_menu_advanced(menu_structure)
                            # Pobieramy kod HTML menu z pobranej strony
                            response = requests.get(menu_url)
                            soup = BeautifulSoup(response.content, 'html.parser')
                            menu_html = soup.find_all(['ul', 'ol', 'nav'])
                            menu_html_str = "\n".join(str(menu) for menu in menu_html)
                        else:
                            menu_structure = extract_menu(menu_url, menu_selector=None)
                            if isinstance(menu_structure, str):  # Jeśli wystąpił błąd
                                st.error(menu_structure)
                                return
                            nodes, urls = visualize_menu(menu_structure)
                            # Pobieramy kod HTML menu z pobranej strony
                            response = requests.get(menu_url)
                            soup = BeautifulSoup(response.content, 'html.parser')
                            menu_html = soup.find('nav') or soup.find('ul', class_='menu') or soup.find('ul', id='menu')
                            menu_html_str = str(menu_html) if menu_html else "Nie udało się wyodrębnić kodu HTML menu."

                    st.success("Analiza menu zakończona!")

                    # Tworzymy DataFrame z wynikami
                    df = pd.DataFrame({
                        "Struktura menu": nodes,
                        "URL": urls
                    })

                    # Wyświetlamy wyniki w Streamlit
                    st.dataframe(df)

                    # Przygotowujemy plik CSV do pobrania
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Pobierz wyniki jako CSV",
                        data=csv,
                        file_name="menu_structure.csv",
                        mime="text/csv",
                    )

                    # Analiza AI
                    with st.spinner('Przeprowadzam analizę AI...'):
                        # Przygotowujemy strukturę menu jako string
                        menu_structure_str = "\n".join(nodes)

                        ai_analysis = analyze_menu_with_ai(menu_structure_str, menu_html_str)
                        st.write("**Rekomendacje AI:**")
                        st.info(ai_analysis)
            else:
                st.warning("Proszę wprowadzić URL do analizy menu.")

    # Zakładka 4: Pagespeed Insights
    with tab4:
        st.header("Pagespeed Insights Test")

        psi_url = st.text_input("Wprowadź URL strony do analizy:", key='psi_url')
        strategy = st.radio("Wybierz strategię testowania:", ('mobile', 'desktop'))

        if st.button("Uruchom test Pagespeed Insights"):
            if psi_url:
                with st.spinner('Przeprowadzam test Pagespeed Insights...'):
                    data = run_pagespeed_insights(psi_url, strategy)
                    if data:
                        # Przetwarzanie i wyświetlanie wyników
                        display_pagespeed_results(data)
            else:
                st.warning("Proszę wprowadzić URL do analizy.")





if __name__ == '__main__':
    main()
