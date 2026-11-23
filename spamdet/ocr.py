"""Optional OCR: no heavyweight imports, uploads, caches, or temporary files."""
import io
import warnings


class OCRError(RuntimeError):
    pass


def extract_text(content, reader=None):
    if not isinstance(content, bytes) or not content:
        raise OCRError('Berkas gambar kosong atau tidak valid.')
    if len(content) > 10 * 1024 * 1024:
        raise OCRError('Gambar maksimal 10 MiB.')
    try:
        from PIL import Image
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(content)) as image:
                if image.width * image.height > 20_000_000:
                    raise OCRError('Gambar maksimal 20 juta piksel.')
                image.verify()
    except OCRError:
        raise
    except Exception:
        raise OCRError('Format gambar tidak valid atau dimensi terlalu besar.') from None
    try:
        import cv2
        import numpy as np
        import easyocr
    except ImportError:
        raise OCRError('OCR opsional belum tersedia. Instal requirements-ocr.txt; deteksi teks tetap dapat digunakan.') from None
    try:
        image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError()
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        if reader is None:
            reader = easyocr.Reader(['id', 'en'], gpu=False)
        return ' '.join(str(result[1]) for result in reader.readtext(gray))
    except Exception:
        raise OCRError('OCR gagal memproses gambar atau mengunduh model. Periksa dependensi/koneksi; deteksi teks tetap tersedia.') from None
