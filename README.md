# Statistics and Topology 

```python
from system import System
from field import Field

K = System(('i:j', 'j:k'))

# Degree-0 field: tensors indexed by faces a
f = K.gaussian(0)
# Tensor {
#   (i:j) :->   [[* , *],
#                [* , *]]
#   
#   (j:k) :->   [[* , *],
#                [* , *]] 
#
#   (j) :->     [* , *]       
# }

# Degree-1 field: tensors indexed by pairs a > b
phi = K.gaussian(1)
# Tensor {
#   (i:j) . (j) :-> [* , *]
#   (j:k) . (j) :-> [* , *]
# }

# Codifferential delta : A[n] <- A[n + 1]
g = f + K.delta[1] @ phi
assert g.degree == 0

# Combinatorial operations zeta and mu
f1 = K.mu @ K.zeta @ f
assert (f1 - f).trim(1e-6) == 0
``` 

See [zeta.py](zeta.py) for examples. 
