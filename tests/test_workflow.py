import json
import tempfile
import unittest
from pathlib import Path
from PIL import Image
from train import inspect_dataset, build_command
from evaluate import load_prompts

class WorkflowTests(unittest.TestCase):
    def test_empty_dataset_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                inspect_dataset(folder)

    def test_dataset_manifest_records_image_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            Image.new('RGB', (32, 16), 'blue').save(Path(folder)/'image.png')
            manifest = inspect_dataset(folder)
            self.assertEqual(manifest[0]['width'], 32)
            self.assertEqual(len(manifest[0]['sha256']), 64)

    def test_duplicates_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            a = Path(folder)/'a.png'
            Image.new('RGB', (16, 16), 'blue').save(a)
            (Path(folder)/'b.png').write_bytes(a.read_bytes())
            with self.assertRaises(ValueError):
                inspect_dataset(folder)

    def test_prompt_is_one_argument_and_original_hyperparameters_are_retained(self):
        with tempfile.TemporaryDirectory() as folder:
            script = Path(folder)/'train_dreambooth.py'
            script.write_text('')
            prompt = 'a sofa; text is data'
            command = build_command(script, folder, folder, 'model', prompt)
            self.assertEqual(command[command.index('--instance_prompt')+1], prompt)
            self.assertEqual(command[command.index('--max_train_steps')+1], '800')
            self.assertEqual(command[command.index('--learning_rate')+1], '2e-6')

    def test_invalid_prompt_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'prompts.json'
            path.write_text(json.dumps(['valid', '']))
            with self.assertRaises(ValueError):
                load_prompts(path)
