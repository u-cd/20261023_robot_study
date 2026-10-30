import unittest
from vision.evaluate_detectors import operational_metrics


class DetectionMetricsTests(unittest.TestCase):
    def test_wrong_class_counts_as_false_positive_and_miss(self):
        result = operational_metrics({1: [(1, .9, [0, 0, 10, 10])]},
            {1: [(0, [0, 0, 10, 10])]}, {'confidence_threshold': .25, 'iou_threshold': .5})
        self.assertEqual(result['per_class']['target']['fp'], 1)
        self.assertEqual(result['per_class']['sponge']['fn'], 1)
        self.assertEqual(result['both_found_images'], 0)

    def test_duplicate_is_not_a_second_true_positive(self):
        result = operational_metrics({1: [(0, .9, [0, 0, 10, 10]), (0, .8, [0, 0, 10, 10]), (1, .1, [0, 0, 10, 10])]},
            {1: [(0, [0, 0, 10, 10])]}, {'confidence_threshold': .25, 'iou_threshold': .5})
        self.assertEqual(result['per_class']['sponge']['tp'], 1)
        self.assertEqual(result['per_class']['sponge']['fp'], 1)
        self.assertEqual(result['per_class']['target']['fp'], 0)
        self.assertEqual(result['both_found_images'], 1)
        self.assertEqual(result['both_found_without_extra_detections'], 0)


if __name__ == '__main__':
    unittest.main()
