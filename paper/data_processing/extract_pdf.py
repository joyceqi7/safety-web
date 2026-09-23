"""
步骤1: 从PDF事故报告中提取文本
从107份真实事故报告中批量提取文本内容
"""

import os
import json
import fitz  # PyMuPDF
from tqdm import tqdm
from config import DATA_DIR, EXTRACTED_TEXT_DIR


def extract_text_from_pdf(pdf_path: str) -> str:
    """从单个PDF文件中提取文本"""
    doc = fitz.open(pdf_path)
    text_parts = []
    for page in doc:
        text = page.get_text()
        if text.strip():
            text_parts.append(text.strip())
    doc.close()
    return "\n\n".join(text_parts)


def extract_all_pdfs(data_dir: str = DATA_DIR,
                     output_dir: str = EXTRACTED_TEXT_DIR) -> list[dict]:
    """
    批量提取所有PDF文本

    Returns:
        list[dict]: [{"filename": ..., "text": ..., "char_count": ...}, ...]
    """
    pdf_files = sorted([
        f for f in os.listdir(data_dir) if f.lower().endswith(".pdf")
    ])
    print(f"找到 {len(pdf_files)} 份PDF文件")

    results = []
    for pdf_file in tqdm(pdf_files, desc="提取PDF文本"):
        pdf_path = os.path.join(data_dir, pdf_file)
        try:
            text = extract_text_from_pdf(pdf_path)
            record = {
                "filename": pdf_file,
                "text": text,
                "char_count": len(text),
                "pdf_path": pdf_path,
            }
            results.append(record)

            # 保存提取的文本
            txt_filename = os.path.splitext(pdf_file)[0] + ".txt"
            txt_path = os.path.join(output_dir, txt_filename)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)
        except Exception as e:
            print(f"  错误: 无法提取 {pdf_file}: {e}")
            results.append({
                "filename": pdf_file,
                "text": "",
                "char_count": 0,
                "pdf_path": pdf_path,
                "error": str(e),
            })

    # 保存元数据
    meta_path = os.path.join(output_dir, "_metadata.json")
    meta = [
        {"filename": r["filename"], "char_count": r["char_count"],
         "has_error": "error" in r}
        for r in results
    ]
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    successful = sum(1 for r in results if "error" not in r)
    print(f"成功提取 {successful}/{len(pdf_files)} 份PDF")
    return results


if __name__ == "__main__":
    extract_all_pdfs()
