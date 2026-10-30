import unittest
import numpy as np
from control.arm_kinematics import forward,fit,inverse_preview


class ArmKinematicsTests(unittest.TestCase):
    def setUp(self):
        self.p=[-100,80,-np.pi/2,50,100,140,np.pi/4,-np.pi/2]
        self.signs=[-1,1]

    def test_forward_known_geometry(self):
        xyz=forward(self.p,[90,0,0],self.signs)[0]
        c=np.sqrt(.5)
        np.testing.assert_allclose(xyz,[-100+240*c,80,50-40*c],atol=1e-10)

    def model_for(self,q):
        xyz=forward(self.p,q,self.signs)[0]
        x,y=xyz[:2]
        return xyz,{'parameters':self.p,'signs':self.signs,
                    'reference_xy_mm':[[x-10,y-10],[x+10,y-10],[x+10,y+10],[x-10,y+10]],
                    'reference_commands_deg':[q,q,q,q]}

    def test_inverse_round_trip_and_no_motion(self):
        xyz,model=self.model_for([90,25,40])
        result=inverse_preview(model,xyz,allow_unvalidated_height=True)
        np.testing.assert_allclose(forward(self.p,result['command_degrees_float'],self.signs)[0],xyz,atol=.01)
        self.assertFalse(result['motion_executed'])
        self.assertFalse(result['height_generalization_verified'])

    def test_rejects_height_and_xy_extrapolation(self):
        xyz,model=self.model_for([90,25,40])
        with self.assertRaises(ValueError):inverse_preview(model,xyz)
        with self.assertRaises(ValueError):inverse_preview(model,[xyz[0]+100,xyz[1],0])

    def test_fits_synthetic_varied_xyz_data(self):
        rng=np.random.default_rng(2)
        q=rng.uniform([35,15,20],[100,40,70],size=(20,3))
        xyz=forward(self.p,q,self.signs)
        result=fit(q,xyz,self.signs,starts=1,initial=self.p)
        self.assertLess(result['max_error_mm'],.001)
        self.assertEqual(result['jacobian_rank'],8)


if __name__=='__main__':unittest.main()
