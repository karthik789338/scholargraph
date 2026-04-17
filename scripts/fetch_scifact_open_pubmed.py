from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List


EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def chunked(items: List[str], size: int) -> List[List[str]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def extract_text(elem) -> str:
    if elem is None:
        return ""
    return "".join(elem.itertext()).strip()


def parse_pubmed_xml(xml_text: str) -> List[Dict]:
    root = ET.fromstring(xml_text)
    records = []

    for article in root.findall(".//PubmedArticle"):
        medline = article.find("MedlineCitation")
        if medline is None:
            continue

        pmid = extract_text(medline.find("PMID"))
        article_node = medline.find("Article")
        if article_node is None:
            continue

        title = extract_text(article_node.find("ArticleTitle"))

        abstract_parts = []
        abstract = article_node.find("Abstract")
        if abstract is not None:
            for a in abstract.findall("AbstractText"):
                label = a.attrib.get("Label", "").strip()
                txt = extract_text(a)
                if txt:
                    if label:
                        abstract_parts.append(f"{label}: {txt}")
                    else:
                        abstract_parts.append(txt)

        journal = article_node.find("Journal")
        journal_title = ""
        year = None

        if journal is not None:
            journal_title = extract_text(journal.find("Title"))
            pub_date = journal.find(".//PubDate")
            if pub_date is not None:
                year_text = extract_text(pub_date.find("Year"))
                if year_text.isdigit():
                    year = int(year_text)

        records.append(
            {
                "pmid": pmid,
                "title": title,
                "abstract": " ".join(abstract_parts).strip(),
                "journal": journal_title,
                "year": year,
            }
        )

    return records


def main():
    base = Path("data/raw/scifact_open")
    needed_ids_path = base / "needed_doc_ids.json"
    out_path = base / "documents.jsonl"
    missing_path = base / "missing_doc_ids.json"

    if not needed_ids_path.exists():
        raise FileNotFoundError(f"Missing file: {needed_ids_path}")

    doc_ids = json.loads(needed_ids_path.read_text(encoding="utf-8"))
    doc_ids = [str(x) for x in doc_ids]

    fetched = {}
    missing = []

    batches = chunked(doc_ids, 100)

    with out_path.open("w", encoding="utf-8") as fout:
        for i, batch in enumerate(batches, start=1):
            params = {
                "db": "pubmed",
                "id": ",".join(batch),
                "retmode": "xml",
            }
            url = EFETCH_URL + "?" + urllib.parse.urlencode(params)

            print(f"Fetching batch {i}/{len(batches)} with {len(batch)} ids...")
            with urllib.request.urlopen(url) as resp:
                xml_text = resp.read().decode("utf-8", errors="replace")

            records = parse_pubmed_xml(xml_text)

            batch_found = set()
            for rec in records:
                pmid = rec["pmid"]
                if pmid:
                    batch_found.add(pmid)
                    fetched[pmid] = rec
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

            for doc_id in batch:
                if doc_id not in batch_found:
                    missing.append(doc_id)

            time.sleep(0.34)  # stay polite to NCBI

    with missing_path.open("w", encoding="utf-8") as f:
        json.dump(missing, f, indent=2)

    print("\nDone.")
    print(f"Fetched documents: {len(fetched)}")
    print(f"Missing documents: {len(missing)}")
    print(f"Wrote docs to: {out_path}")
    print(f"Wrote missing ids to: {missing_path}")


if __name__ == "__main__":
    main()