import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from urllib.parse import urljoin
import time
import re
import os

def get_pytorch_doc_urls():
    '''Crawl the PyTorch documentation site to find all relevant URLs; right now you have to manually add them 
    (since this is a short project) but should capture majority of what people would be using to learn PyTorch.

    This function returns a list containing all the scrapeable links FROM the index page
    '''

    base_url = "https://docs.pytorch.org/docs/stable/"
    navigation_pages = [
        "nn.html",  # Neural network modules
    ]
    
    doc_urls = []
    
    for nav_page in tqdm(navigation_pages, desc="Crawling nav pages"):
        url = urljoin(base_url, nav_page)
        print(f"\nCrawling: {url}")  # DEBUG
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            all_links = soup.find_all('a', href=True)

            for link in all_links:
                href = link['href']
                # Adjust this condition if needed
                if 'generated/' in href:
                    full_url = urljoin(base_url, href)
                    doc_urls.append(full_url)
            
            time.sleep(0.5)
            
        except requests.RequestException as e:
            print(f"Error crawling {url}: {e}")
            continue
    
    unique_urls = list(set(doc_urls))
    print(f"Found {len(unique_urls)} documentation pages")
    return unique_urls

# Skip the fancy AST parsing - just grab text
def scrape(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Try to find the main section for the API (e.g., torch-absolute)
    import re
    match = re.search(r'/([\w\.]+)\.htm', url)
    section_id = None
    if match:
        section_id = match.group(1).replace('.', '-')
    section = soup.find('div', id=section_id) if section_id else None
    if section:
        signature = section.find('dt')
        description = section.find('dd')
        parts = []
        if signature:
            parts.append(signature.get_text())
        if description:
            parts.append(description.get_text())
        return '\n\n'.join(parts)
    else:
        # Fallback: get the first function signature and description
        dl = soup.find('dl', class_='function')
        if dl:
            signature = dl.find('dt')
            description = dl.find('dd')
            parts = []
            if signature:
                parts.append(signature.get_text())
            if description:
                parts.append(description.get_text())
            return '\n\n'.join(parts)
        # Fallback: return all text
        return soup.get_text()
    
# NEW: Save to individual files
def scrape_and_save_pytorch_docs():
    # Create data directory
    os.makedirs("../data/raw", exist_ok=True)
    
    urls = get_pytorch_doc_urls()
    
    for url in tqdm(urls, desc="Scraping and saving docs"):
        try:
            content = scrape(url)
            cleaned_content = re.sub(r'\s+', ' ', content).strip()
            
            # Create filename from URL
            func_name = url.split('/')[-1].replace('.html', '')
            filename = f"../data/raw/{func_name}.txt"
            
            # Save to file
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"Source: {url}\n\n")
                f.write(cleaned_content)
            
            time.sleep(0.2)
            
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            continue

    print(f"Saved documentation to ../data/raw/ directory")

# Run the scraper once
scrape_and_save_pytorch_docs()