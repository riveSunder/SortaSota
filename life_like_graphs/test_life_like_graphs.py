import unittest

import numpy as np

        
class TestGetCAMLP(unittest.TestCase):

    def setup(self):
        
        pass

    def test_life(self):

        b = [3]
        s = [2,3]

        life_mlp = get_ca_mlp(b, s)

        self.assertEqual(life_mlp(2. + 9), 1.0)
        self.assertEqual(life_mlp(3. + 9), 1.0)
        self.assertEqual(life_mlp(3.), 1.0)

        self.assertEqual(life_mlp(np.array([[2. + 9]])), 1.0)
        self.assertEqual(life_mlp(np.array([[3. + 9]])), 1.0)
        self.assertEqual(life_mlp(np.array([[3.]])), 1.0)

        self.assertNotEqual(life_mlp(4. + 9), 1.0)
        self.assertNotEqual(life_mlp(1. + 9), 1.0)
        self.assertNotEqual(life_mlp(2.), 1.0)
        self.assertNotEqual(life_mlp(4.), 1.0)

    def test_general(self):

        for run in range(10):
            # nb - number off birth rules to use
            nb = np.random.randint(9)
            ns = np.random.randint(9)
            survival = np.random.choice(np.arange(9), size=(ns,), replace=False)
            birth = np.random.choice(np.arange(9), size=(nb,), replace=False)

            ca_mlp = get_ca_mlp(birth, survival)

            for bb in np.arange(9):
                self.assertEqual(ca_mlp(bb), 1.0 * bb in birth) 
                self.assertEqual(ca_mlp(np.array([[bb]])), 1.0 * bb in birth) 

            for ss in np.arange(9):
                self.assertEqual(ca_mlp(ss + 9), 1.0 * ss in survival) 
                self.assertEqual(ca_mlp(np.array([[ss + 9]])), \
                        1.0 * ss in survival) 


if __name__ == "__main__":

    unittest.main(verbosity = 2)

