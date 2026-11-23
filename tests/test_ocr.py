import builtins
import io
import pytest
from PIL import Image
from spamdet.ocr import extract_text, OCRError


def test_size_before_import():
    with pytest.raises(OCRError, match='10 MiB'):
        extract_text(b'x' * (10*1024*1024+1))


def test_invalid_image():
    with pytest.raises(OCRError, match='gambar'):
        extract_text(b'not image')


def test_pixels_before_decode():
    buf = io.BytesIO()
    Image.new('1', (5000, 4001)).save(buf, format='PNG')
    with pytest.raises(OCRError, match='20 juta'):
        extract_text(buf.getvalue())


def test_memory_preprocessing_and_download_failure(monkeypatch):
    import sys
    import types
    import numpy as np
    calls = []
    cv = types.SimpleNamespace(IMREAD_COLOR=1, COLOR_BGR2GRAY=2, THRESH_BINARY=4, THRESH_OTSU=8)
    def decode(data, mode):
        assert isinstance(data, np.ndarray)
        calls.append('decode')
        return 'image'
    def gray(image, code):
        calls.append('gray')
        return 'gray'
    def blur(image, kernel, sigma):
        assert kernel == (3, 3) and sigma == 0
        calls.append('blur')
        return 'blurred'
    def threshold(image, lower, upper, flags):
        assert flags == 12
        calls.append('threshold')
        return 0, 'binary'
    cv.imdecode, cv.cvtColor, cv.GaussianBlur, cv.threshold = decode, gray, blur, threshold
    class Reader:
        def __init__(self, languages, gpu):
            assert languages == ['id','en'] and gpu is False
        def readtext(self, image):
            assert image == 'binary'
            return [(None, 'hello', .9), (None, 'world', .8)]
    monkeypatch.setitem(sys.modules, 'cv2', cv)
    monkeypatch.setitem(sys.modules, 'easyocr', types.SimpleNamespace(Reader=Reader))
    buf = io.BytesIO()
    Image.new('RGB', (10,10)).save(buf, format='PNG')
    assert extract_text(buf.getvalue()) == 'hello world'
    assert calls == ['decode', 'gray', 'blur', 'threshold']
    def unavailable(*args, **kwargs):
        raise RuntimeError('download failed: private details')
    monkeypatch.setitem(sys.modules, 'easyocr', types.SimpleNamespace(Reader=unavailable))
    with pytest.raises(OCRError, match='mengunduh model'):
        extract_text(buf.getvalue())


def test_missing_optional(monkeypatch):
    original = builtins.__import__
    def blocked(name, *args, **kwargs):
        if name in ('cv2', 'easyocr'):
            raise ImportError('optional unavailable')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', blocked)
    buf = io.BytesIO()
    Image.new('RGB', (10, 10)).save(buf, format='PNG')
    with pytest.raises(OCRError, match='requirements-ocr'):
        extract_text(buf.getvalue())
