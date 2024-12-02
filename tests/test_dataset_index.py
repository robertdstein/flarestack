import unittest
import logging

from flarestack.data.dataset_index import dataset_index

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestDatasetIndex(unittest.TestCase):
    def setUp(self) -> None:
        pass

    def test_index_content(self) -> None:
        n = len(dataset_index.get_dataset_list())
        logger.info(f"There are {n} datasets in the index.")
        self.assertGreater(n, 0)

    def test_index_consistency(self) -> None:
        for key in dataset_index.get_dataset_list():
            dataset = dataset_index.get_dataset(key)
            self.assertEqual(key, dataset.name)


if __name__ == "__main__":
    unittest.main()
