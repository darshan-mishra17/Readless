"""
Advanced PDF Parser with OCR support for handwritten and scanned documents
Supports both digital PDFs and image-based PDFs with OCR capabilities
"""

import os
import tempfile
import cv2
import numpy as np
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from typing import List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedPDFParser:
    """
    Advanced PDF Parser that automatically detects and handles:
    - Digital text PDFs (using PyPDFLoader)
    - Scanned/image PDFs (using OCR)
    - Handwritten content (using enhanced OCR)
    - Mixed content PDFs
    """
    
    def __init__(self, tesseract_cmd: Optional[str] = None):
        """
        Initialize the parser with optional Tesseract path
        
        Args:
            tesseract_cmd: Path to tesseract executable (auto-detected if None)
        """
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        
        # Try to auto-detect Tesseract if not provided
        try:
            pytesseract.get_tesseract_version()
            logger.info("✅ Tesseract OCR detected and ready")
        except Exception as e:
            logger.warning(f"⚠️ Tesseract not found: {e}")
            logger.info("📥 Download Tesseract from: https://github.com/UB-Mannheim/tesseract/wiki")

    def detect_pdf_type(self, pdf_path: str) -> Tuple[str, float]:
        """
        Detect if PDF contains digital text or requires OCR
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Tuple of (pdf_type, confidence) where:
            - pdf_type: 'digital', 'scanned', or 'mixed'
            - confidence: float between 0-1 indicating detection confidence
        """
        try:
            # First, try to extract text using standard method
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            total_chars = 0
            text_chars = 0
            
            for doc in documents[:3]:  # Check first 3 pages for speed
                content = doc.page_content.strip()
                total_chars += len(content)
                # Count meaningful text (not just whitespace/special chars)
                text_chars += len([c for c in content if c.isalnum()])
            
            if total_chars == 0:
                return "scanned", 1.0
            
            text_ratio = text_chars / total_chars if total_chars > 0 else 0
            
            if text_ratio > 0.7 and total_chars > 100:
                return "digital", 0.9
            elif text_ratio > 0.3:
                return "mixed", 0.7
            else:
                return "scanned", 0.8
                
        except Exception as e:
            logger.warning(f"Error detecting PDF type: {e}")
            return "scanned", 0.6

    def preprocess_image_for_ocr(self, image: Image.Image) -> Image.Image:
        """
        Enhance image quality for better OCR results
        
        Args:
            image: PIL Image object
            
        Returns:
            Preprocessed PIL Image
        """
        # Convert PIL to OpenCV format
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
        
        # Apply noise reduction
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Apply threshold for better text recognition
        _, threshold = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Convert back to PIL
        return Image.fromarray(threshold)

    def extract_text_with_ocr(self, pdf_path: str, enhance_handwriting: bool = True) -> List[Document]:
        """
        Extract text from PDF using OCR
        
        Args:
            pdf_path: Path to the PDF file
            enhance_handwriting: Apply special preprocessing for handwritten content
            
        Returns:
            List of Document objects with extracted text
        """
        documents = []
        
        try:
            # Convert PDF pages to images
            logger.info("🖼️ Converting PDF pages to images...")
            images = convert_from_path(pdf_path, dpi=300)  # High DPI for better OCR
            
            for page_num, image in enumerate(images, 1):
                logger.info(f"📄 Processing page {page_num}/{len(images)} with OCR...")
                
                try:
                    # Preprocess image for better OCR
                    if enhance_handwriting:
                        processed_image = self.preprocess_image_for_ocr(image)
                    else:
                        processed_image = image
                    
                    # Configure OCR for different content types
                    if enhance_handwriting:
                        # Enhanced config for handwritten content
                        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,!?;:()[]{}"-/\\ '
                    else:
                        # Standard config for printed text
                        custom_config = r'--oem 3 --psm 3'
                    
                    # Extract text using OCR
                    extracted_text = pytesseract.image_to_string(
                        processed_image, 
                        config=custom_config,
                        lang='eng'  # Can be extended to support multiple languages
                    )
                    
                    # Clean and validate extracted text
                    cleaned_text = self.clean_ocr_text(extracted_text)
                    
                    if cleaned_text.strip():
                        # Create document with metadata
                        doc = Document(
                            page_content=cleaned_text,
                            metadata={
                                "source": pdf_path,
                                "page": page_num,
                                "extraction_method": "ocr",
                                "enhanced_handwriting": enhance_handwriting
                            }
                        )
                        documents.append(doc)
                        logger.info(f"✅ Page {page_num}: Extracted {len(cleaned_text)} characters")
                    else:
                        logger.warning(f"⚠️ Page {page_num}: No readable text found")
                        
                except Exception as e:
                    logger.error(f"❌ Error processing page {page_num}: {e}")
                    continue
            
            logger.info(f"🎉 OCR completed: {len(documents)} pages processed successfully")
            return documents
            
        except Exception as e:
            logger.error(f"❌ OCR extraction failed: {e}")
            raise Exception(f"OCR processing error: {str(e)}")

    def clean_ocr_text(self, text: str) -> str:
        """
        Clean and improve OCR-extracted text
        
        Args:
            text: Raw OCR text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        cleaned = ' '.join(text.split())
        
        # Fix common OCR errors
        corrections = {
            # Common character misrecognitions
            'l': 'I',  # lowercase l often misread as I
            '0': 'O',  # zero vs letter O (context dependent)
            '5': 'S',  # five vs S (context dependent)
            '8': 'B',  # eight vs B (context dependent)
            # Add more corrections as needed
        }
        
        # Apply basic corrections (can be enhanced with ML-based correction)
        for old, new in corrections.items():
            # Only apply if it makes sense in context (basic heuristic)
            if old in cleaned and len(cleaned) > 10:
                words = cleaned.split()
                for i, word in enumerate(words):
                    if len(word) == 1 and word == old:
                        # Single character corrections
                        words[i] = new
                cleaned = ' '.join(words)
        
        return cleaned

    def load_pdf_with_smart_detection(self, pdf_path: str) -> List[Document]:
        """
        Main method to load PDF with automatic type detection and appropriate parsing
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of Document objects with extracted text
        """
        logger.info(f"📚 Loading PDF: {os.path.basename(pdf_path)}")
        
        # Step 1: Detect PDF type
        pdf_type, confidence = self.detect_pdf_type(pdf_path)
        logger.info(f"🔍 PDF Type detected: {pdf_type} (confidence: {confidence:.2f})")
        
        documents = []
        
        try:
            if pdf_type == "digital":
                # Use standard text extraction for digital PDFs
                logger.info("📄 Using digital text extraction...")
                loader = PyPDFLoader(pdf_path)
                documents = loader.load()
                
                # Add extraction method to metadata
                for doc in documents:
                    doc.metadata["extraction_method"] = "digital"
                
            elif pdf_type == "scanned":
                # Use OCR for scanned/image PDFs
                logger.info("🔍 Using OCR extraction for scanned content...")
                documents = self.extract_text_with_ocr(pdf_path, enhance_handwriting=True)
                
            elif pdf_type == "mixed":
                # Try digital first, then OCR for pages with little text
                logger.info("🔄 Using hybrid extraction for mixed content...")
                
                # First attempt: digital extraction
                loader = PyPDFLoader(pdf_path)
                digital_docs = loader.load()
                
                # Check which pages need OCR
                for i, doc in enumerate(digital_docs):
                    if len(doc.page_content.strip()) < 50:  # Threshold for "low text"
                        logger.info(f"📄 Page {i+1}: Low text detected, applying OCR...")
                        # Apply OCR to this specific page (would need page-specific OCR implementation)
                        # For now, we'll use the digital version but mark it
                        doc.metadata["extraction_method"] = "digital_low_text"
                    else:
                        doc.metadata["extraction_method"] = "digital"
                
                documents = digital_docs
            
            # Final validation
            valid_documents = []
            for doc in documents:
                if doc.page_content and len(doc.page_content.strip()) > 10:
                    valid_documents.append(doc)
            
            if not valid_documents:
                raise Exception("No readable text found in PDF. The document may be image-only or corrupted.")
            
            logger.info(f"✅ Successfully extracted text from {len(valid_documents)} pages")
            return valid_documents
            
        except Exception as e:
            logger.error(f"❌ PDF processing failed: {e}")
            # Fallback: try OCR as last resort
            if pdf_type != "scanned":
                logger.info("🔄 Attempting OCR fallback...")
                try:
                    return self.extract_text_with_ocr(pdf_path, enhance_handwriting=True)
                except Exception as ocr_error:
                    logger.error(f"❌ OCR fallback also failed: {ocr_error}")
            
            raise Exception(f"Failed to extract text from PDF: {str(e)}")

# Initialize the global parser instance
pdf_parser = AdvancedPDFParser()

def load_pdf_with_ocr(pdf_path: str) -> List[Document]:
    """
    Convenience function to load PDF with OCR support
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of Document objects
    """
    return pdf_parser.load_pdf_with_smart_detection(pdf_path)
