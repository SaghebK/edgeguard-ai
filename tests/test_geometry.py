import numpy as np
from edgeguard.detector import polygon_pixels, point_in_polygon

def test_polygon_conversion():
    p=polygon_pixels([[0,0],[1,0],[1,1]],100,200)
    assert p.tolist()==[[0,0],[100,0],[100,200]]

def test_point_inside_outside():
    p=np.array([[10,10],[90,10],[90,90],[10,90]],dtype=np.int32)
    assert point_in_polygon((50,50),p)
    assert not point_in_polygon((2,2),p)
