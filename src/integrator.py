import torch
from functional import Functional

"""
    # Diffusion: 

          zeta    -D       d*
        u ---> U ---> Phi ---> du 
        |                       |
       (+=) <-------------------'

    Some uses may only want to get `u(n)` given `u(0)` but 
    we may also want to introduce a writer instance along
    the loop to track and monitor the diffusion flow.
"""

class Orbit (Record): 

    def __init__(self, u):
        pass


class Diffusion:

    def __init__(self, K, bethe=True): 
        self.name = "Bethe" if bethe else "GBP"
        self.K = K
        self.flux = - K.Deff[0] @ K.Zeta[0]
        if bethe :
            self.flux = K.Mu[1] @ self.flux
        self.transport = K.Delta[1] @ self.flux

    def integrate(self, u, dt=0.5, nit=10):
        """ u -> (q, Phi) """
        K = self.K
        zeta, Deff = K.Zeta[0], K.Deff[0]
        free = lambda t: - torch.logsumexp(-t, [*range(t.dim())])
        if self.name == "Bethe": 
            for i in range(nit): 
                u += dt * self.transport(u)
            U = zeta @ u
            U -= U.fmap(free)
            q = U.fmap(lambda Ua: torch.exp(-Ua))
            Phi = Deff @ U
            return (q, Phi)

        elif self.name == "GBP":
            delta = K.Delta[1]
            for i in range(nit):
                U = zeta @ u
                U -= U.fmap(free)
                Phi = Deff (U) 
                u -= dt * delta @ Phi
            q = U.fmap(lambda Ua: torch.exp(-Ua))
            return (q, Phi)
