import numpy as np
import gbpy.byxtal.integer_manipulations as iman

## Create vectors of arbitrary integers
n1 = 5
vec1 = np.random.randint(-10, 10, size=3)
gcd = iman.gcd_array(vec1)

## Create arrays of arbitrary dimensions
##     Populate with arbitrary integers
n1 = np.random.randint(2, 5, size=1)
n2 = np.random.randint(2, 5, size=1)

arr1 = np.random.randint(-100, 100, size=(n1[0], n2[0]))
print(arr1)

Agcd1 = iman.gcd_array(arr1, 'rows')
Agcd2 = iman.gcd_array(arr1, 'cols')

print(Agcd1)
print(Agcd2)
## order:'all'

## order:'rows'


## order:'columns'
