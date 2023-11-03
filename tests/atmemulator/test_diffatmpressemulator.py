import unittest
from diffatmpressemulator.diffatmpressemulator import DiffAtmPressEmulator
import numpy as np 



class DiffAtmPressEmulatorTestCase(unittest.TestCase):
    """A test case for the diffatmpressemularor package."""


    def test_transparency(self):
        e =  DiffAtmPressEmulator('LSST')
        array = e.GetAllTransparencies([300.,400.,600.,800.,1000.],1.,0.,0.)
        self.assertTrue(np.allclose(array,[0.41483073, 0.76797745, 0.95170851, 0.98476396, 0.9937721 ]),msg="ObsAtmo.GetAllTransparencies error")

        


if __name__ == "__main__":
    unittest.main()
