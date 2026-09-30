"""The viewer must render actual ALE frames and pause without advancing the game."""
import io
from pathlib import Path
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from PIL import Image
from game_viewer import GameViewer


class GameViewerTest(unittest.TestCase):
    def test_controls_and_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            viewer = GameViewer(Path(directory))
            try:
                with self.assertRaises(ValueError):
                    viewer.start('../outside.zip')
                with self.assertRaises(ValueError):
                    viewer.set_speed(.75)
                viewer.set_speed(2)
                self.assertEqual(viewer.snapshot()['speed'], 2)
                viewer.set_speed(1)
                viewer.start('random', 123)
                deadline = time.monotonic() + 30
                while viewer.snapshot()['length'] < 4 and time.monotonic() < deadline:
                    if viewer.snapshot()['error']:
                        self.fail(viewer.snapshot()['error'])
                    time.sleep(.1)
                self.assertGreaterEqual(viewer.snapshot()['length'], 4)
                image = Image.open(io.BytesIO(viewer.image()))
                self.assertEqual(image.size, (160, 210))
                self.assertEqual(image.format, 'PNG')
                frame_id, jpeg = viewer.stream_frame(-1)
                self.assertGreater(frame_id, viewer.snapshot()['length'])
                self.assertEqual(Image.open(io.BytesIO(jpeg)).format, 'JPEG')
                viewer.pause(True)
                time.sleep(.15)  # let an in-flight inference finish
                steps = viewer.snapshot()['length']
                paused_frame = viewer.snapshot()['frame_id']
                time.sleep(.2)
                self.assertEqual(viewer.snapshot()['length'], steps)
                self.assertEqual(viewer.snapshot()['frame_id'], paused_frame)
                self.assertEqual(viewer.snapshot()['status'], 'paused')
                viewer.pause(False)
                time.sleep(.3)
                self.assertGreater(viewer.snapshot()['length'], steps)
            finally:
                viewer.stop()
            self.assertEqual(viewer.snapshot()['status'], 'stopped')


if __name__ == '__main__':
    unittest.main()
