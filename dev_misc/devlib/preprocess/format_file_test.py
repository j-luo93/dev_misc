from pathlib import Path
from unittest import TestCase

from .format_file import FormatFile, UnsupportedError

path = Path('.')


class TestFormatFile(TestCase):

    def test_basic(self):
        f = FormatFile(path, 'train', 'en', 'txt')
        self.assertEqual(str(f), 'train.en.txt')

        f = FormatFile(path, 'train', 'de', 'txt', pair='de-en')
        self.assertEqual(str(f), 'train.de-en.de.txt')

        f = FormatFile(path, 'train', 'de', 'txt', pair='de-en', ops=['tok', 'bpe'])
        self.assertEqual(str(f), 'train.de-en.de.tok.bpe.txt')

    def test_unsupported_error(self):
        with self.assertRaises(UnsupportedError):
            FormatFile(path, 'train', 'en', 'txtblah')
        with self.assertRaises(UnsupportedError):
            FormatFile(path, 'train', 'en', 'txt', ops=['tokblah'])

    def test_propagated_methods(self):
        f = FormatFile(path, 'train', 'en', 'txt', pair='de-en')
        f = f.remove_pair()
        self.assertEqual(str(f), 'train.en.txt')
