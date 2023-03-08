import numpy as np

##################################### UTILITY 3d Polygon fucntion

#determinant of matrix a
def det(a):
    return a[0][0]*a[1][1]*a[2][2] + a[0][1]*a[1][2]*a[2][0] + a[0][2]*a[1][0]*a[2][1] - a[0][2]*a[1][1]*a[2][0] - a[0][1]*a[1][0]*a[2][2] - a[0][0]*a[1][2]*a[2][1]

#unit normal vector of plane defined by points a, b, and c
def unit_normal(a, b, c):
    x = det([[1,a[1],a[2]],
             [1,b[1],b[2]],
             [1,c[1],c[2]]])
    y = det([[a[0],1,a[2]],
             [b[0],1,b[2]],
             [c[0],1,c[2]]])
    z = det([[a[0],a[1],1],
             [b[0],b[1],1],
             [c[0],c[1],1]])
    magnitude = (x**2 + y**2 + z**2)**.5
    if magnitude == 0:
        return (0,0,0)
    return (x/magnitude, y/magnitude, z/magnitude)

#dot product of vectors a and b
def dot(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

#cross product of vectors a and b
def cross(a, b):
    x = a[1] * b[2] - a[2] * b[1]
    y = a[2] * b[0] - a[0] * b[2]
    z = a[0] * b[1] - a[1] * b[0]
    return (x, y, z)

#area of polygon poly
def area(poly):
    if len(poly) < 3: # not a plane - no area
        return 0

    total = [0, 0, 0]
    for i in range(len(poly)):
        vi1 = poly[i]
        if i is len(poly)-1:
            vi2 = poly[0]
        else:
            vi2 = poly[i+1]
        prod = cross(vi1, vi2)
        total[0] += prod[0]
        total[1] += prod[1]
        total[2] += prod[2]
    result = dot(total, unit_normal(poly[0], poly[1], poly[2]))
    return abs(result/2)


#####################################


# from LELEC2885 - Image Proc. & Comp. Vision
def resize_and_fix_origin(kernel, size):
    pad0, pad1 = size[0]-kernel.shape[0], size[1]-kernel.shape[1]
    shift0, shift1 = (kernel.shape[0]-1)//2, (kernel.shape[1]-1)//2
    kernel = np.pad(kernel, ((0, pad0), (0, pad1)), mode='constant')
    kernel = np.roll(kernel, (-shift0, -shift1), axis=(0, 1))
    return kernel



def corr_kernel(m, n, sigma):
    kernel = np.zeros((2*n+1, 2*n+1))
    kernel[n-m:n+m+1, n-m:n+m+1] = get_gaussian_kernel(sigma, m*2+1, divX=3)
    seuil = 1e-10
    kernel0 = (kernel < seuil)
    kernel[kernel0] = -1 / np.sum(kernel0)
    return kernel

def get_gaussian_kernel(sigma, n, divX=1, divY=1):
    indices = np.linspace(-n/2, n/2, n)
    [X, Y] = np.meshgrid(indices, indices)
    X, Y = X/divX, Y/divY
    h = np.exp(-(X**2+Y**2)/(2.0*(sigma)**2))
    h /= h.sum()
    return h

# from LELEC2885 - Image Proc. & Comp. Vision
def fast_convolution(image, kernel):
    kernel_resized = resize_and_fix_origin(kernel, image.shape)
    image_fft = np.fft.fft2(image)
    kernel_fft = np.fft.fft2(kernel_resized)
    result = np.fft.ifft2(image_fft * kernel_fft)
    return np.real(result)

def triangle_kernel(kerlenx, kerleny):
    """Generate a 2D triangle kernel given the length

    Args:
        kerlen (int): length of the kernel

    Returns:
        np.array([kerlen]float): Kernel
    """
    r = np.arange(kerlenx)
    kernel1d = (kerlenx + 1 - np.abs(r - r[::-1])) / 2
    kernel2d = np.expand_dims(kernel1d, axis=0)
    kernel2d /= kernel2d.sum()
    return kernel2d
    