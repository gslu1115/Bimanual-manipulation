"""Scalar-first quaternion geometry for the fixed inverted-carton experiment."""
import math
import numpy as np


def multiply(a, b):
    w, x, y, z = a
    v, i, j, k = b
    return np.array([w*v-x*i-y*j-z*k, w*i+x*v+y*k-z*j,
                     w*j-x*k+y*v+z*i, w*k+x*j-y*i+z*v])


def inverse(q):
    return np.asarray(q)*np.array([1., -1., -1., -1.])/np.dot(q, q)


def turn_y(angle):
    return multiply([math.cos(angle/2), 0., math.sin(angle/2), 0.], [0., 1., 0., 0.])


def angle_between(a, b):
    dot = abs(float(np.dot(a, b)/np.linalg.norm(a)/np.linalg.norm(b)))
    return 2*math.acos(min(1., dot))


def slerp(a, b, t):
    a, b = np.asarray(a)/np.linalg.norm(a), np.asarray(b)/np.linalg.norm(b)
    dot = float(np.dot(a, b))
    if dot < 0: b, dot = -b, -dot
    if dot > .9995:
        q = a+t*(b-a)
        return q/np.linalg.norm(q)
    theta = math.acos(min(1., dot))
    return (math.sin((1-t)*theta)*a+math.sin(t*theta)*b)/math.sin(theta)
