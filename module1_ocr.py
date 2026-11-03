"""
Module 1: Enhanced Multimodal Document Ingestion
==================================================
Uses multiple OCR engines and advanced preprocessing for:
- Printed text (Tesseract)
- Handwritten text (EasyOCR fallback)
- Scanned documents with noise
- Direct PDF text extraction (for text-based PDFs)
"""

import pytesseract
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import pymupdf as fitz  # PyMuPDF for PDF support
import re
import warnings
import os

warnings.filterwarnings("ignore")

# Project-local Tesseract language data (eng, hin, tam, osd). The system install only ships English.
_LOCAL_TESSDATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tessdata")
if os.path.isdir(_LOCAL_TESSDATA) and os.path.exists(os.path.join(_LOCAL_TESSDATA, "eng.traineddata")):
    os.environ["TESSDATA_PREFIX"] = _LOCAL_TESSDATA

# UI language -> Tesseract language string
OCR_LANGUAGES = {"English": "eng", "Hindi": "eng+hin", "Tamil": "eng+tam"}

# Set Tesseract path for Windows
if os.name == 'nt':  # Windows
    pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'


class ImagePreprocessor:
    """
    Advanced image preprocessing for better OCR accuracy.
    Handles: rotation, noise, contrast, binarization, deskewing.
    """
    
    @staticmethod
    def enhance_for_ocr(image: Image.Image) -> Image.Image:
        """
        Comprehensive preprocessing pipeline for OCR.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.5)
        
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(1.2)
        
        image = image.filter(ImageFilter.SHARPEN)
        image = image.filter(ImageFilter.SHARPEN)
        
        image = image.convert("L")
        image = image.filter(ImageFilter.MedianFilter(size=3))
        
        threshold = 140
        image = image.point(lambda x: 0 if x < threshold else 255, '1')
        image = image.convert("RGB")
        
        return image
    
    @staticmethod
    def preprocess_for_handwriting(image: Image.Image) -> Image.Image:
        """
        Specialized preprocessing for handwritten/cursive text.
        Uses contrast + sharpen WITHOUT aggressive binarization.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Moderate contrast boost (not too aggressive)
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)
        
        # Sharpen to make cursive strokes clearer
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(2.0)
        
        # Convert to grayscale (keep as L, don't binarize — preserves cursive details)
        image = image.convert("L")
        
        # Convert back to RGB for Tesseract
        image = image.convert("RGB")
        
        return image
    
    @staticmethod
    def correct_orientation(image: Image.Image) -> Image.Image:
        """
        Attempt to correct image orientation using Tesseract's OSD.
        """
        if not os.path.exists(pytesseract.pytesseract.tesseract_cmd):
            return image
        
        try:
            import subprocess
            import tempfile
            
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                image.save(tmp.name)
                tmp_path = tmp.name
            
            try:
                result = subprocess.run(
                    [pytesseract.pytesseract.tesseract_cmd, tmp_path, '-', '--psm', '0'],
                    capture_output=True, text=True, timeout=10
                )
                for line in result.stdout.split('\n') + result.stderr.split('\n'):
                    if 'Rotate' in line:
                        angle = int(line.split()[-1]) if line.split()[-1].isdigit() else 0
                        if angle == 90:
                            image = image.rotate(-90, expand=True)
                        elif angle == 180:
                            image = image.rotate(-180, expand=True)
                        elif angle == 270:
                            image = image.rotate(-90, expand=True)
                        break
            finally:
                os.unlink(tmp_path)
                
        except Exception:
            pass
        
        return image


class MultiEngineOCR:
    """
    Multi-engine OCR system with automatic fallback.
    Tries Tesseract first, falls back to other engines if confidence is low.
    """
    
    def __init__(self):
        self.preprocessor = ImagePreprocessor()
        self._easyocr_reader = None
        self.easyocr_available = False
        self.tesseract_available = True
        
        try:
            pytesseract.get_tesseract_version()
        except Exception:
            self.tesseract_available = False
    
    def _get_easyocr(self):
        """Lazy-load EasyOCR only when needed."""
        if self._easyocr_reader is None:
            try:
                import easyocr
                self._easyocr_reader = easyocr.Reader(['en'], gpu=False)
                self.easyocr_available = True
            except ImportError:
                self.easyocr_available = False
        return self._easyocr_reader
    
    def ocr_with_tesseract(self, image: Image.Image, config: str = '', lang: str = 'eng') -> dict:
        """
        Perform OCR using Tesseract (a single run: text is rebuilt from the word-level data).
        """
        if not self.tesseract_available:
            return {"text": "", "confidence": 0, "engine": "tesseract_unavailable"}

        try:
            data = pytesseract.image_to_data(image, lang=lang, config=config, output_type=pytesseract.Output.DICT)
            confidences = [float(c) for c in data['conf'] if float(c) > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0

            lines, order = {}, []
            for i, word in enumerate(data['text']):
                if not str(word).strip():
                    continue
                key = (data['block_num'][i], data['par_num'][i], data['line_num'][i])
                if key not in lines:
                    lines[key] = []
                    order.append(key)
                lines[key].append(str(word))
            out, prev_par = [], None
            for key in order:
                if prev_par is not None and key[:2] != prev_par:
                    out.append("")  # blank line between paragraphs/blocks
                out.append(" ".join(lines[key]))
                prev_par = key[:2]
            return {
                "text": "\n".join(out).strip(),
                "confidence": avg_confidence / 100.0,
                "engine": "tesseract" if lang == "eng" else f"tesseract ({lang})"
            }
        except Exception:
            return {"text": "", "confidence": 0, "engine": "tesseract_error"}
    
    def ocr_with_easyocr(self, image: Image.Image) -> dict:
        """
        Perform OCR using EasyOCR (better for handwriting).
        """
        reader = self._get_easyocr()
        if not self.easyocr_available or reader is None:
            return {"text": "", "confidence": 0, "engine": "easyocr_unavailable"}
        
        try:
            import numpy as np
            img_array = np.array(image)
            results = reader.readtext(img_array)
            texts = [text for (bbox, text, confidence) in results]
            confidences = [confidence for (bbox, text, confidence) in results]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            return {"text": " ".join(texts), "confidence": avg_confidence, "engine": "easyocr"}
        except Exception:
            return {"text": "", "confidence": 0, "engine": "easyocr_error"}
    
    def extract_text(self, image: Image.Image, is_handwritten: bool = False, lang: str = 'eng') -> dict:
        """
        Multi-engine OCR with automatic fallback.
        """
        if image.mode not in ('RGB', 'L'):
            image = image.convert('RGB')
        
        # Large scans make every Tesseract pass slow; ~2600px on the long side is still ~200 DPI on A4.
        MAX_SIDE = 2600
        if max(image.size) > MAX_SIDE:
            scale = MAX_SIDE / max(image.size)
            image = image.resize((int(image.width * scale), int(image.height * scale)), Image.LANCZOS)

        results = []

        def run_jobs(jobs):
            """Run (image, config, confidence_multiplier) Tesseract jobs in parallel."""
            with ThreadPoolExecutor(max_workers=min(4, len(jobs))) as pool:
                outs = list(pool.map(lambda j: (self.ocr_with_tesseract(j[0], j[1], lang), j[2]), jobs))
            for r, mult in outs:
                if r['text'].strip():
                    r['confidence'] *= mult
                    results.append(r)

        if is_handwritten:
            # PSM 3 = fully automatic, PSM 4 = single column (good for prescription lists)
            hw = self.preprocessor.preprocess_for_handwriting(image)
            jobs = [(hw, '--psm 3', 1.0), (hw, '--psm 4', 1.0), (image, '--psm 3', 1.1)]
            w, h = image.size
            if w < 1500:  # only upscale small images
                jobs.append((image.resize((w * 2, h * 2), Image.LANCZOS), '--psm 3', 1.0))
            run_jobs(jobs)
        else:
            # Fast path: one clean pass. If it is confident, the extra preprocessing passes are skipped.
            first = self.ocr_with_tesseract(image, '--psm 6', lang)
            if first['text'].strip():
                results.append(first)
            if not (first['text'].strip() and first['confidence'] >= 0.85 and len(first['text'].strip()) > 20):
                run_jobs([
                    (image, '', 1.0),
                    (self._light_preprocess(image), '--psm 6', 1.0),
                    (self.preprocessor.enhance_for_ocr(image), '--psm 6', 1.0),
                ])

        # Try EasyOCR if Tesseract confidence is low
        best_tesseract_conf = max([r['confidence'] for r in results if r['text'].strip()] or [0])
        if self.easyocr_available and best_tesseract_conf < 0.6:
            easyocr_result = self.ocr_with_easyocr(image)
            if easyocr_result['text'].strip():
                results.append(easyocr_result)
        
        # Filter out empty results
        valid_results = [r for r in results if r['text'].strip()]
        
        if not valid_results:
            return {"text": "", "confidence": 0, "engine": "none"}
        
        # Pick best using medical-content-aware scoring
        best_result = max(valid_results, key=lambda x: self._score_medical_text(x))
        best_result['text'] = self.post_process_text(best_result['text'])
        
        return best_result
    
    def _score_medical_text(self, result: dict) -> float:
        """
        Score OCR result by confidence + prescription content.
        Heavily weights dosage/prescription keywords over header metadata.
        """
        text = result['text'].lower()
        confidence = result['confidence']
        
        # HIGH-VALUE: Prescription-specific keywords (weighted 3x)
        rx_keywords = [
            'tablet', 'tab.', 'cap.', 'inj.', 'injection',
            'mg', 'mcg', 'ml', 'units',
            'daily', 'twice daily', 'three times daily', 'times daily',
            'hours', 'orally', 'iv ',
        ]
        
        # LOW-VALUE: Generic medical/clinic keywords (weighted 1x)
        generic_keywords = [
            'patient', 'doctor', 'dr.', 'pharmacy', 'rx',
            'clinic', 'hospital', 'delhi', 'chennai',
            'address', 'phone', 'mobile', 'date', 'time',
            'street', 'road', 'post',
        ]
        
        # Count high-value (3x weight) and low-value (1x weight)
        rx_hits = sum(3 for kw in rx_keywords if kw in text)
        generic_hits = sum(1 for kw in generic_keywords if kw in text)
        total_score = rx_hits + generic_hits
        
        # Bonus for handwriting indicators (means it's a prescription)
        handwriting_bonus = 1.3 if ('claily' in text or 'fer ' in text or 'tab.' in text) else 1.0
        
        # Penalty for long text with few prescription terms
        text_words = len(text.split())
        length_penalty = 0.7 if text_words > 50 and rx_hits < 6 else 1.0
        
        return confidence * (1 + total_score * 0.1) * handwriting_bonus * length_penalty
    
    def _light_preprocess(self, image: Image.Image) -> Image.Image:
        """
        Light preprocessing — just grayscale + slight contrast.
        """
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.5)
        
        image = image.convert('L')
        image = image.convert('RGB')
        
        return image
    
    def post_process_text(self, text: str) -> str:
        """
        Post-process OCR text to fix common errors in medical/cursive text.
        """
        if not text:
            return text
        
        # Remove excessive whitespace but preserve line breaks
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Dose units: handwriting often OCRs 'mg' as 'mq' ("40 mq"), 'mcg' as 'mcq'
        text = re.sub(r'(?<=\d)(\s*)mq\b', r'\1mg', text)
        text = re.sub(r'(?<=\d)(\s*)mcq\b', r'\1mcg', text)
        text = re.sub(r'(?<=\d)(\s*)ma\b', r'\1mg', text)          # '75 ma' -> '75 mg'
        text = re.sub(r'\b([Vv]itamin)\s+0(\d)\b', r'\1 D\2', text)   # 'Vitamin 03' -> 'Vitamin D3'
        text = re.sub(r'\b[lI]0(\s?)ml\b|\bloml\b', r'10\1ml', text)  # 'loml' / 'l0 ml' -> '10ml'

        # Fix common cursive OCR errors (letter confusion patterns)
        cursive_fixes = [
            # "daily" confusion
            (r'\bclaily\b', 'daily'),
            (r'\bolaily\b', 'daily'),
            (r'\bclaiiy\b', 'daily'),
            
            # "for" confusion  
            (r'\bfer\b', 'for'),
            (r'\btor\b', 'for'),
            
            # "days" confusion
            (r'\bclays\b', 'days'),
            (r'\bclay\b', 'day'),
            (r'\bdaya\b', 'days'),
            
            # "tablet" confusion
            (r'\btab\b', 'tab'),
            (r'\bTab\b', 'Tab'),
            
            # "5" vs "S" confusion
            (r'\bSe clays\b', '5 days'),
            (r'\bSe days\b', '5 days'),
            
            # "oral" / "orally" confusion
            (r'\boraiiy\b', 'orally'),
            (r'\boraily\b', 'orally'),
            
            # "times" confusion
            (r'\btimes\b', 'times'),
            
            # "needed" confusion
            (r'\bneeded\b', 'needed'),
            
            # "patient" confusion
            (r'\bpatient\b', 'patient'),
            (r'\bPatient\b', 'Patient'),
            
            # "once" confusion
            (r'\boncc\b', 'once'),
            
            # "every" confusion
            (r'\bevery\b', 'every'),
            
            # "hours" confusion
            (r'\bhours\b', 'hours'),
            
            # "pain" confusion
            (r'\bpain\b', 'pain'),
        ]
        
        for pattern, replacement in cursive_fixes:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Fix dosage format
        text = re.sub(r'(\d+)\s*mg\b', r'\1 mg', text)
        text = re.sub(r'(\d+)\s*ml\b', r'\1 ml', text)
        text = re.sub(r'(\d+)\s*mcg\b', r'\1 mcg', text)
        
        # Fix numbered list items
        text = re.sub(r'^nae\s', '2. ', text, flags=re.MULTILINE)
        text = re.sub(r'^nac\s', '2. ', text, flags=re.MULTILINE)
        
        # Fix common drug name OCR errors
        drug_fixes = [
            (r'\bFlurbozine\b', 'Flurbozine'),
            (r'\bRabisure\b', 'Rabisure'),
            (r'\bDolo-650\b', 'Dolo-650'),
            (r'\bDolo\b', 'Dolo'),
        ]
        
        for pattern, replacement in drug_fixes:
            text = re.sub(pattern, replacement, text)
        
        return text.strip()


def extract_text_from_pdf_direct(pdf_path: str) -> dict:
    """
    Try to extract text directly from PDF using PyMuPDF.
    """
    try:
        pdf_document = fitz.open(pdf_path)
        num_pages = len(pdf_document)
        all_text = []
        total_chars = 0
        
        for page_num in range(num_pages):
            page = pdf_document.load_page(page_num)
            
            text = page.get_text("text")
            if text.strip():
                all_text.append(text.strip())
                total_chars += len(text.strip())
            
            blocks = page.get_text("blocks")
            if blocks:
                for block in blocks:
                    if block[4].strip():
                        if block[4].strip() not in text:
                            all_text.append(block[4].strip())
        
        pdf_document.close()
        
        combined_text = "\n".join(all_text)
        
        if total_chars > 100:
            confidence = min(0.95, 0.5 + (total_chars / 1000))
        elif total_chars > 0:
            confidence = 0.3
        else:
            confidence = 0
        
        return {
            "text": combined_text,
            "confidence": confidence,
            "pages": num_pages,
            "engine": "pymupdf_text",
            "method": "direct_text_extraction"
        }
        
    except Exception as e:
        return {
            "text": "",
            "confidence": 0,
            "pages": 0,
            "engine": "pymupdf_error",
            "method": "direct_text_extraction",
            "error": str(e)
        }


def preprocess_image(image_path: str) -> Image.Image:
    """
    Preprocess image for better OCR accuracy.
    """
    img = Image.open(image_path)
    preprocessor = ImagePreprocessor()
    return preprocessor.enhance_for_ocr(img)


def extract_text_from_image(image_path: str, is_handwritten: bool = False) -> dict:
    """
    Extract text from an image using multi-engine OCR.
    """
    image = Image.open(image_path)
    
    # Ensure RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')        # Skip orientation correction to avoid hangs — proceed directly to OCR
    
    ocr = MultiEngineOCR()
    result = ocr.extract_text(image, is_handwritten)
    
    return result


def extract_text_from_pil_image(image: Image.Image, is_handwritten: bool = False, lang: str = 'eng') -> dict:
    """
    Extract text directly from a PIL Image object (for Streamlit uploads).
    """
    if image.mode != "RGB":
        image = image.convert("RGB")        # Skip orientation correction to avoid hangs — proceed directly to OCR
    
    ocr = MultiEngineOCR()
    result = ocr.extract_text(image, is_handwritten, lang)
    
    return result


def pdf_to_images(pdf_path: str, dpi: int = 300) -> list:
    """
    Convert a PDF file to a list of PIL Images.
    """
    images = []
    
    pdf_document = fitz.open(pdf_path)
    
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    
    pdf_document.close()
    return images


def extract_text_from_pdf(pdf_path: str) -> dict:
    """
    Extract text from a PDF file.
    
    Strategy:
    1. Always try direct text extraction (for overlay/header text)
    2. Always run OCR on rendered images (for image-based prescription body)
    3. Merge both results — deduplicate overlapping text, combine unique parts
    """
    # Step 1: Direct text extraction
    direct_result = extract_text_from_pdf_direct(pdf_path)
    direct_text = direct_result.get('text', '').strip()
    
    # Step 2: OCR on rendered images
    images = pdf_to_images(pdf_path)
    ocr_texts = []
    total_confidence = 0
    
    for i, img in enumerate(images):
        result = extract_text_from_pil_image(img)
        ocr_texts.append(result['text'])
        total_confidence += result['confidence']
    
    ocr_text = '\n'.join(ocr_texts)
    avg_confidence = total_confidence / len(images) if images else 0
    
    # Step 3: Merge results
    if direct_text and ocr_text:
        direct_lines = [l.strip() for l in direct_text.split('\n') if l.strip()]
        ocr_lines = [l.strip() for l in ocr_text.split('\n') if l.strip()]
        
        extra_lines = []
        for ocr_line in ocr_lines:
            is_duplicate = False
            for d_line in direct_lines:
                if len(ocr_line) > 5 and len(d_line) > 5:
                    ocr_words = set(ocr_line.lower().split())
                    d_words = set(d_line.lower().split())
                    if ocr_words and d_words:
                        overlap = len(ocr_words & d_words) / min(len(ocr_words), len(d_words))
                        if overlap > 0.6:
                            is_duplicate = True
                            break
            if not is_duplicate:
                extra_lines.append(ocr_line)
        
        if extra_lines:
            merged = direct_text + '\n\n--- Additional Content (OCR) ---\n' + '\n'.join(extra_lines)
        else:
            merged = direct_text
        
        confidence = max(direct_result.get('confidence', 0), avg_confidence)
        
        return {
            'text': merged.strip(),
            'confidence': confidence,
            'pages': len(images),
            'engine': 'merged',
            'method': 'direct_text + ocr',
            'direct_chars': len(direct_text),
            'ocr_chars': len(ocr_text)
        }
    
    if direct_text:
        direct_result['text'] = direct_text
        direct_result['method'] = 'direct_text_extraction'
        return direct_result
    
    if ocr_text:
        return {
            'text': ocr_text.strip(),
            'confidence': avg_confidence,
            'pages': len(images),
            'engine': 'multi-engine',
            'method': 'ocr'
        }
    
    return {
        'text': '',
        'confidence': 0,
        'pages': 0,
        'engine': 'none',
        'method': 'none'
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        print(f"Processing: {file_path}")
        
        if file_path.lower().endswith(".pdf"):
            result = extract_text_from_pdf(file_path)
        else:
            result = extract_text_from_image(file_path)
        
        print(f"Engine: {result['engine']}")
        print(f"Method: {result.get('method', 'unknown')}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Extracted Text:\n{result['text']}")
    else:
        print("Usage: python module1_ocr.py <image_or_pdf_path>")
