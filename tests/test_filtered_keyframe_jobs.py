import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_filtered_keyframe_jobs import build_jobs, write_jobs
from run_s2v_multiview_hunyuan_distil import load_and_assign_jobs, artifact_stem
from s2v_object_only_runtime import bbox_crop


class FilteredJobsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.name = 'clip___cup'
        frames = {}
        for fid, box in [(3, [20, 10, 60, 30]), (8, [0, 0, 100, 50]),
                         (9, [10, 10, 40, 40])]:
            filename = f'frame_{fid:06d}.jpg'
            path = self.root / self.name / filename
            path.parent.mkdir(exist_ok=True)
            Image.new('RGB', (200, 100), 'white').save(path)
            frames[filename] = dict(width=200, height=100, detections=[
                dict(label='cup', bbox_2d=box, motion_blur=False),
                dict(label='other', bbox_2d=[0, 0, 2, 2])])
        self.bbox = {'results': {'clip': dict(keyframe_dir=self.name, frames=frames)}}
        self.filtered = {'dirs': [dict(keyframe_dir=self.name, n_kept=2, kept_frame_ids=[3, 8])]}
        self.bp = self.root / 'bbox.json'
        self.fp = self.root / 'filtered.json'

    def build(self):
        self.bp.write_text(json.dumps(self.bbox))
        self.fp.write_text(json.dumps(self.filtered))
        return build_jobs([self.fp], self.bp, self.root)

    def test_kept_only_primary_box_and_runner_contract(self):
        jobs = self.build()
        self.assertEqual([r['frame_index'] for r in jobs], [8, 3])
        self.assertEqual(jobs[1]['bbox_norm'], [.1, .1, .3, .3])
        self.assertEqual({r['object_name'] for r in jobs}, {'cup'})
        self.assertEqual([r['is_anchor'] for r in jobs], [True, False])
        self.assertEqual(len({(r['sample_id'], r['entity_id']) for r in jobs}), 1)
        path = self.root / 'jobs.jsonl'
        write_jobs(path, jobs)
        assigned, groups, missing = load_and_assign_jobs(path, 1, 0)
        self.assertEqual((len(assigned), groups, missing), (2, 1, 0))
        self.assertEqual(len({artifact_stem(r) for r in jobs}), 2)
        with Image.open(jobs[1]['frame_path']) as image:
            crop, box = bbox_crop(image, jobs[1]['bbox_norm'], 0)
            self.assertEqual(box, (20, 10, 60, 30))
            self.assertEqual(crop.size, (40, 20))

    def test_missing_image_does_not_replace_output(self):
        output = self.root / 'jobs.jsonl'
        output.write_text('previous output')
        (self.root / self.name / 'frame_000003.jpg').unlink()
        with self.assertRaises(FileNotFoundError):
            write_jobs(output, self.build())
        self.assertEqual(output.read_text(), 'previous output')

    def test_invalid_boxes_and_dimensions_fail(self):
        original = copy.deepcopy(self.bbox)
        for patch in [{'width': 100}, {'detections': []},
                      {'detections': [dict(label='cup', bbox_2d=[0, 0, float('nan'), 50])]},
                      {'detections': [dict(label='cup', bbox_2d=[201, 0, 230, 50])]}]:
            with self.subTest(patch=patch):
                self.bbox = copy.deepcopy(original)
                self.bbox['results']['clip']['frames']['frame_000003.jpg'].update(patch)
                with self.assertRaises(ValueError):
                    self.build()

    def test_clips_box_to_image(self):
        self.bbox['results']['clip']['frames']['frame_000003.jpg']['detections'][0]['bbox_2d'] = [-10, -5, 220, 110]
        row = next(r for r in self.build() if r['frame_index'] == 3)
        self.assertEqual(row['bbox_norm'], [0, 0, 1, 1])

    def test_duplicate_directory_or_frame_fails(self):
        self.filtered['dirs'].append(copy.deepcopy(self.filtered['dirs'][0]))
        with self.assertRaisesRegex(ValueError, 'Duplicate filtered'):
            self.build()
        self.filtered['dirs'].pop()
        self.filtered['dirs'][0]['kept_frame_ids'] = [3, 3]
        with self.assertRaisesRegex(ValueError, 'unique'):
            self.build()

    def test_shards_and_empty_directories(self):
        expected = self.build()
        empty = self.root / 'empty.json'
        empty.write_text(json.dumps({'dirs': [dict(keyframe_dir='empty', kept_frame_ids=[], n_kept=0)]}))
        self.assertEqual(build_jobs([empty, self.fp], self.bp, self.root), expected)


if __name__ == '__main__':
    unittest.main()
