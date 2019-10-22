import unittest
from optimizer import GenerateNeurons

class TestNeuronGeneration(unittest.TestCase):

    def test_for_one_layer(self):
        self.assertEqual(
            GenerateNeurons(number_of_layers=1, min_neurons=1, max_neurons=3),
            [(1,), (2,), (3,)]
        )

    def test_for_two_layers(self):
        self.assertEqual(
            GenerateNeurons(number_of_layers=2, min_neurons=1, max_neurons=3),
            [(1,), (2,), (3,),
            (1, 1), (1, 2), (1, 3),
            (2, 1), (2, 2), (2, 3),
            (3, 1), (3, 2), (3, 3),]
        )


if __name__ == '__main__':
    unittest.main()
