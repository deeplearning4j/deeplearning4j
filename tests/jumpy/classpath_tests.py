import os, unittest
try:
    from unittest.mock import MagicMock
except ImportError:
    from mock import MagicMock
from jumpy import get_classpath

class TestClassPath(unittest.TestCase):

    def test_simple_classpath(self):
        os.listdir = MagicMock(return_value=['nd4j-buffer-0.8.1-20170627.172301-977.jar', 'leptonica-1.73-1.3-linux-x86_64.jar'])

        res = '/opt/skil/lib/nd4j-buffer-0.8.1-20170627.172301-977.jar:/opt/skil/lib/leptonica-1.73-1.3-linux-x86_64.jar'
        # this test won't work on Windows (though the function it's testing will)
        if (os.path.sep == '/'):
            # the directory argument is moot
            self.assertEqual(get_classpath('/opt/skil/lib/*'), res)

    def test_mix_jar_wildcard(self):
        os.listdir = MagicMock(return_value=['nd4j-buffer-0.8.1-20170627.172301-977.jar', 'leptonica-1.73-1.3-linux-x86_64.jar'])

        res = '/opt/skil/lib/nd4j-buffer-0.8.1-20170627.172301-977.jar:/opt/skil/lib/nd4j-buffer-0.8.1-20170627.172301-977.jar:/opt/skil/lib/leptonica-1.73-1.3-linux-x86_64.jar'
        # this test won't work on Windows (though the function it's testing will)
        if (os.path.sep == '/'):
            # the directory argument is moot
            self.assertEqual(get_classpath('/opt/skil/lib/nd4j-buffer-0.8.1-20170627.172301-977.jar:/opt/skil/lib/*'), res)
