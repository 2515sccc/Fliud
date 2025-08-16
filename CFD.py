import numpy as np
from Mesh import renumber_mesh_nodes
import pylab as plt
import fipy as fp
import itasca as it
from itasca import ballarray as ba
from itasca import cfdarray as ca
from itasca.element import cfd
import math

def createmesh():
    mesh = fp.Gmsh3D("""
        la=0.0025;
        lc=0.0025;
        Point(1) = {0.001, 0.0125, 0.0    , lc};
        Point(2) = {0.001, 0.025,  0.0125 , lc};
        Point(3) = {0.001, 0.0125, 0.0125 , lc};
        Point(4) = {0.001, 0.0125, 0.025  , lc};
        Point(5) = {0.001, 0.0,    0.0125 , lc};

        Circle(1) = {1, 3, 2,la};
        Circle(2) = {2, 3, 4,la};
        Circle(3) = {4, 3, 5,la};
        Circle(4) = {5, 3, 1,la};

        Line Loop(1) = {1, 2, 3, 4};
        Plane Surface(1) = {1};

        Transfinite Surface{1};
        Recombine Surface{1};

        Extrude {0.055, 0.0, 0} {Surface{1};Layers{15};Recombine;}
        """)
    return mesh
class DarcyFlowSolution(object):
    def __init__(self,mesh):
        self.mesh = mesh
        self.pressure = fp.CellVariable(mesh=self.mesh,
                                        name='pressure', value=0.0)
        self.mobility = fp.CellVariable(mesh=self.mesh,
                                        name='mobility', value=0.0)
        self.pressure.equation = (fp.DiffusionTerm(coeff=self.mobility) == 0.0)
        self.mu = it.fish.get("FluidDyV")  # dynamic viscosity
        self.inlet_mask = None
        self.outlet_mask = None        # create the FiPy grid into the PFC CFD module
        
        if it.ball.count() == 0:
            self.grain_size = 5e-4
        else:
            self.grain_size = 2*ba.radius().mean()
            
        it.command("""
        configure cfd
        element cfd attribute density [FluidDens]
        element cfd attribute viscosity {}
        cfd porosity polyhedron
        cfd interval 20
        """.format(self.mu))
        
    def set_pressure(self, value, where):
        """Dirichlet boundary condition. value is a pressure in Pa and where
        is a mask on the element faces."""
        print "setting pressure to {} on {} faces".format(value, where.sum())
        self.pressure.constrain(value, where)

    def set_inflow_rate(self, flow_rate):
        """
        Set inflow rate in m^3/s.  Flow is in the positive y direction and is specfified
        on the mesh faces given by the inlet_mask.
        """
        assert self.inlet_mask.sum()
        assert self.outlet_mask.sum()
        print "setting inflow on %i faces" % (self.inlet_mask.sum())
        print "setting outflow on %i faces" % (self.outlet_mask.sum())

        self.flow_rate = flow_rate
        self.inlet_area = (self.mesh.scaledFaceAreas*self.inlet_mask).sum()
        self.outlet_area = (self.mesh.scaledFaceAreas*self.outlet_mask).sum()
        self.Uin = flow_rate/self.inlet_area
        inlet_mobility = (self.mobility.getFaceValue() * \
                              self.inlet_mask).sum()/(self.inlet_mask.sum()+0.0)
        self.pressure.faceGrad.constrain(
            ((self.Uin/inlet_mobility,),(self.Uin/inlet_mobility,),(self.Uin/inlet_mobility,),), self.inlet_mask)

    def solve(self):
        """Solve the pressure equation and find the velocities."""
        self.pressure.equation.solve(var=self.pressure)
        # once we have the solution we write the values into the CFD elements
        ca.set_pressure(self.pressure.value)
        ca.set_pressure_gradient(self.pressure.grad.value.T)
        self.construct_cell_centered_velocity()

    def read_porosity(self):
        """Read the porosity from the PFC cfd elements and calculate a
        permeability."""
        porosity_limit = 0.7
        B = 1.0/180.0
        phi = ca.porosity()
        phi[phi>porosity_limit] = porosity_limit
        K = B*phi**3*self.grain_size**2/(1-phi)**2
        self.mobility.setValue(K/self.mu)
        ca.set_extra(1,self.mobility.value.T)

    def test_inflow_outflow(self):
        """Test continuity."""
        a = self.mobility.getFaceValue()*np.array([np.dot(a,b) for a,b in
                                      zip(self.mesh._faceNormals.T,
                                          self.pressure.getFaceGrad().value.T)])
        self.inflow = (self.inlet_mask * a * self.mesh.scaledFaceAreas).sum()
        self.outflow = (self.outlet_mask * a * self.mesh.scaledFaceAreas).sum()
        print "Inflow: {} outflow: {} tolerance: {}".format(
            self.inflow,  self.outflow,  self.inflow +  self.outflow)
        assert abs(self.inflow +  self.outflow) < 1e-6

    def construct_cell_centered_velocity(self):
        """The FiPy solver finds the velocity (fluxes) on the element faces,
        to calculate a drag force PFC needs an estimate of the
        velocity at the element centroids. """

        assert not self.mesh.cellFaceIDs.mask
        efaces = self.mesh.cellFaceIDs.data.T
        fvel = -(self.mesh._faceNormals*\
                 self.mobility.faceValue.value*np.array([np.dot(a,b) \
                 for a,b in zip(self.mesh._faceNormals.T, \
                               self.pressure.faceGrad.value.T)])).T
        def max_mag(a,b):
            if abs(a) > abs(b): return a
            else: return b
        for i, element in enumerate(cfd.list()):
            xmax, ymax, zmax = fvel[efaces[i][0]][0], fvel[efaces[i][0]][1],\
                               fvel[efaces[i][0]][2]
            for face in efaces[i]:
                xv,yv,zv = fvel[face]
                xmax = max_mag(xv, xmax)
                ymax = max_mag(yv, ymax)
                zmax = max_mag(zv, zmax)
            element.set_vel((xmax, ymax, zmax))

if __name__ == '__main__':
    it.command("res model")
    mesh = createmesh()
    data = renumber_mesh_nodes(mesh, extrude_direction='x')
    ca.create_mesh(mesh.vertexCoords.T, data[:,(0,1,2,3,4,5,6,7)].astype(np.int64))
    solver = DarcyFlowSolution(mesh)
    fx,fy,fz = solver.mesh.getFaceCenters()
    
    solver.inlet_mask = fx == it.fish.get("InletMaskX")
    solver.outlet_mask = fx == it.fish.get("OutletMaskX")
    
    solver.set_inflow_rate(it.fish.get("FluidQ")/3600.0)
    solver.set_pressure(0.0, solver.outlet_mask)
    solver.read_porosity()
    solver.solve()
    solver.test_inflow_outflow()
    it.command("cfd update")

    flow_solve_interval = 100
    def update_flow(*args):
        if it.cycle() % flow_solve_interval == 0:
            solver.read_porosity()
            solver.solve()
            solver.test_inflow_outflow()

    it.set_callback("update_flow",1)

    it.command("""
    
    def Force
        if FluidDens < 1.0e2
            loop foreach local bp ball.cfd.list
                ball.cfd.force(bp) = ball.cfd.force(bp) * 100
            endloop
        endif
    end
    set fish callback CFD_AFTER_UPDATE @Force
    def load
        loop i(1,Pnum)
            command
                solve time [SimulationDt]
                save [i]
            endcommand
        endloop
    end
    @load
    """)
