from unittest import TestCase
from .format_path import FormatPath


class TestFormatPath(TestCase):

    def test_basic(self):
        fp = FormatPath()
        fp.add(['folder'])
        fp.add(['subfolder'])
        fp.add(['file'])
        self.assertEqual(str(fp.format()), 'folder/subfolder/file')

    def test_format(self):
        fp = FormatPath()
        fp.add(['folder'])
        fp.add(['subfolder'])
        fp.add(['file{idx}'])
        self.assertEqual(str(fp.format(idx=1)), 'folder/subfolder/file1')

    def test_init_with_str(self):
        fp = FormatPath()
        fp.add('new folder')
        fp.add('subfolder')
        fp.add('file')
        self.assertEqual(str(fp.format()), 'new_folder/subfolder/file')
