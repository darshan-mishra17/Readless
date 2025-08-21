"""
Simplified OCR PDF Parser with fallback support
Handles both digital PDFs and basic OCR with minimal dependencies
"""

import os
import tempfile
from typing import List, Tuple, Optional
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimplifiedPDFParser:
    """
    Simplified PDF Parser with OCR support
    Prioritizes reliability over advanced OCR features
    """
    
    def __init__(self):
        """Initialize the parser with dependency checking"""
        self.ocr_available = self._check_ocr_dependencies()
        
    def _check_ocr_dependencies(self) -> bool:
        """Check if OCR dependencies are available"""
        try:
            import pytesseract
            from PIL import Image
            from pdf2image import convert_from_path
            
            # Try to auto-configure Tesseract path for Windows
            possible_paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                r"C:\Users\HP\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    logger.info(f"🔧 Configured Tesseract path: {path}")
                    break
            
            # Test basic pytesseract
            version = pytesseract.get_tesseract_version()
            logger.info(f"✅ OCR dependencies available - Tesseract v{version}")
            return True
        except Exception as e:
            logger.warning(f"⚠️ OCR not available: {e}")
            logger.info("📝 Falling back to digital PDF processing only")
            logger.info("💡 To enable OCR: Install Tesseract from https://github.com/UB-Mannheim/tesseract/wiki")
            return False

    def detect_pdf_type(self, pdf_path: str) -> Tuple[str, float]:
        """
        Simple PDF type detection based on text extraction
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Tuple of (pdf_type, confidence)
        """
        try:
            # Try standard text extraction
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            # Analyze text content
            total_chars = 0
            text_chars = 0
            
            for doc in documents[:3]:  # Check first 3 pages
                content = doc.page_content.strip()
                total_chars += len(content)
                text_chars += len([c for c in content if c.isalnum()])
            
            if total_chars == 0:
                return "scanned", 1.0
            
            text_ratio = text_chars / total_chars if total_chars > 0 else 0
            
            if text_ratio > 0.7 and total_chars > 100:
                return "digital", 0.9
            elif text_ratio > 0.2:
                return "mixed", 0.7
            else:
                return "scanned", 0.8
                
        except Exception as e:
            logger.warning(f"Error detecting PDF type: {e}")
            return "unknown", 0.5

    def extract_text_with_basic_ocr(self, pdf_path: str) -> List[Document]:
        """
        Basic OCR extraction with minimal preprocessing
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of Document objects
        """
        if not self.ocr_available:
            raise Exception("OCR dependencies not available. Install: pip install pytesseract pillow pdf2image")
        
        documents = []
        
        try:
            # Import OCR dependencies
            import pytesseract
            from PIL import Image
            from pdf2image import convert_from_path
            
            logger.info("🖼️ Converting PDF pages to images for OCR...")
            
            # Convert PDF to images with lower DPI for speed
            images = convert_from_path(pdf_path, dpi=200)
            
            for page_num, image in enumerate(images, 1):
                logger.info(f"📄 Processing page {page_num}/{len(images)} with OCR...")
                
                try:
                    # Basic OCR without complex preprocessing
                    extracted_text = pytesseract.image_to_string(
                        image,
                        config=r'--oem 3 --psm 3',  # Standard config
                        lang='eng'
                    )
                    
                    # Basic text cleaning
                    cleaned_text = ' '.join(extracted_text.split())
                    
                    if cleaned_text.strip() and len(cleaned_text.strip()) > 10:
                        doc = Document(
                            page_content=cleaned_text,
                            metadata={
                                "source": pdf_path,
                                "page": page_num,
                                "extraction_method": "basic_ocr"
                            }
                        )
                        documents.append(doc)
                        logger.info(f"✅ Page {page_num}: Extracted {len(cleaned_text)} characters")
                    else:
                        logger.warning(f"⚠️ Page {page_num}: No readable text found")
                        
                except Exception as e:
                    logger.error(f"❌ Error processing page {page_num}: {e}")
                    continue
            
            logger.info(f"🎉 OCR completed: {len(documents)} pages processed")
            return documents
            
        except Exception as e:
            logger.error(f"❌ OCR extraction failed: {e}")
            raise Exception(f"OCR processing error: {str(e)}")

    def load_pdf_with_smart_fallback(self, pdf_path: str) -> List[Document]:
        """
        Load PDF with smart fallback strategy
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of Document objects
        """
        logger.info(f"📚 Loading PDF: {os.path.basename(pdf_path)}")
        
        # Step 1: Try digital extraction first (fastest)
        try:
            logger.info("📄 Attempting digital text extraction...")
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            # Check if we got meaningful content
            total_content = sum(len(doc.page_content.strip()) for doc in documents)
            
            if total_content > 100:  # Threshold for "sufficient content"
                # Add metadata
                for doc in documents:
                    doc.metadata["extraction_method"] = "digital"
                
                logger.info(f"✅ Digital extraction successful: {len(documents)} pages, {total_content} characters")
                return documents
            else:
                logger.info("📊 Low text content detected, trying OCR...")
                
        except Exception as e:
            logger.warning(f"Digital extraction failed: {e}")
        
        # Step 2: Try OCR if digital extraction failed or gave poor results
        if self.ocr_available:
            try:
                logger.info("🔍 Attempting OCR extraction...")
                documents = self.extract_text_with_basic_ocr(pdf_path)
                
                if documents:
                    logger.info(f"✅ OCR extraction successful: {len(documents)} pages")
                    return documents
                    
            except Exception as e:
                logger.error(f"OCR extraction failed: {e}")
        
        # Step 3: Final fallback - return whatever we can get
        try:
            logger.info("🔄 Final fallback: returning minimal digital extraction...")
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            
            # Add metadata
            for doc in documents:
                doc.metadata["extraction_method"] = "digital_fallback"
            
            if documents:
                return documents
            
        except Exception as e:
            logger.error(f"Final fallback failed: {e}")
        
        # If everything fails
        raise Exception("Failed to extract any text from PDF. The document may be corrupted or contain only images.")

# Initialize the global parser
pdf_parser = SimplifiedPDFParser()

def load_pdf_with_ocr(pdf_path: str) -> List[Document]:
    """
    Convenience function to load PDF with OCR support and smart fallbacks
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of Document objects
    """
    return pdf_parser.load_pdf_with_smart_fallback(pdf_path)
