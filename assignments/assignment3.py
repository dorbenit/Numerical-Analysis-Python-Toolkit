"""
In this assignment you should find the area enclosed between the two given functions.
The rightmost and the leftmost x values for the integration are the rightmost and 
the leftmost intersection points of the two functions. 

The functions for the numeric answers are specified in MOODLE. 


This assignment is more complicated than Assignment1 and Assignment2 because: 
    1. You should work with float32 precision only (in all calculations) and minimize the floating point errors. 
    2. You have the freedom to choose how to calculate the area between the two functions. 
    3. The functions may intersect multiple times. Here is an example: 
        https://www.wolframalpha.com/input/?i=area+between+the+curves+y%3D1-2x%5E2%2Bx%5E3+and+y%3Dx
    4. Some of the functions are hard to integrate accurately. 
       You should explain why in one of the theoretical questions in MOODLE. 

"""
from assignment2 import Assignment2
import numpy as np
import time
import random


class Assignment3:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before 
        solving the assignment for specific functions. 
        """

        self.assignment2 = Assignment2()

    def integrate(self, f: callable, a: float, b: float, n: int) -> np.float32:
        """
        Integrate the function f in the closed range [a,b] using at most n 
        points. Your main objective is minimizing the integration error. 
        Your secondary objective is minimizing the running time. The assignment
        will be tested on variety of different functions. 
        
        Integration error will be measured compared to the actual value of the 
        definite integral. 
        
        Note: It is forbidden to call f more than n times. 
        
        Parameters
        ----------
        f : callable. it is the given function
        a : float
            beginning of the integration range.
        b : float
            end of the integration range.
        n : int
            maximal number of points to use.

        Returns
        -------
        np.float32
            The definite integral of f between a and b
        """

        # replace this line with your solution
        if a == b:
            return 0

        if n == 1:
            return np.float32(f(a) * np.float32(b - a))

        if n <= 3:
            fa = np.float32(f(a))
            fb = np.float32(f(b))
            return np.float32(0.5 * (np.float32(b - a)) * (fa + fb))

        intervals_number = n - 1
        if intervals_number % 2 != 0:
            intervals_number -= 1

        h = (b - a) / intervals_number

        x_vals = np.linspace(a, b, intervals_number + 1)
        fX_vals = [f(x) for x in x_vals]

        F0 = fX_vals[0]
        F1 = 0
        F2 = fX_vals[-1]

        for i in range(1, intervals_number):
            if i % 2 == 0:
               F2 += fX_vals[i]
               F0 += fX_vals[i]
            else:
               F1 += fX_vals[i]

        integral = np.float32((h / 3.0) * (F0 +4*F1+F2))
        return integral





    def areabetween(self, f1: callable, f2: callable) -> np.float32:
        """
        Finds the area enclosed between two functions. This method finds 
        all intersection points between the two functions to work correctly. 
        
        Example: https://www.wolframalpha.com/input/?i=area+between+the+curves+y%3D1-2x%5E2%2Bx%5E3+and+y%3Dx

        Note, there is no such thing as negative area. 
        
        In order to find the enclosed area the given functions must intersect 
        in at least two points. If the functions do not intersect or intersect 
        in less than two points this function returns NaN.  
        This function may not work correctly if there is infinite number of 
        intersection points. 
        

        Parameters
        ----------
        f1,f2 : callable. These are the given functions

        Returns
        -------
        np.float32
            The area between function and the X axis

        """

        # replace this line with your solution
        intersections = sorted(self.assignment2.intersections(f1, f2, 1, 100))

        number_of_roots = len(intersections)
        if number_of_roots < 2:
            return np.float32(np.nan)

        total_area = np.float32(0.0)
        for i in range(number_of_roots - 1):
            a = intersections[i]
            b = intersections[i + 1]

            mid = (a + b) / 2
            upper_func = f1 if f1(mid) > f2(mid) else f2
            lower_func = f2 if f1(mid) > f2(mid) else f1

            def diff_function(x):
                return upper_func(x) - lower_func(x)

            area = self.integrate(diff_function, a, b, 1000)
            total_area += np.float32(area)

        return total_area


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


# class TestAssignment3(unittest.TestCase):
#
#     def test_integrate_float32(self):
#         ass3 = Assignment3()
#         f1 = np.poly1d([-1, 0, 1])
#         r = ass3.integrate(f1, -1, 1, 10)
#
#         self.assertEqual(r.dtype, np.float32)
#
#     def test_integrate_hard_case(self):
#         ass3 = Assignment3()
#         f1 = strong_oscilations()
#         r = ass3.integrate(f1, 0.09, 10, 20)
#         true_result = -7.78662 * 10 ** 33
#         self.assertGreaterEqual(0.001, abs((r - true_result) / true_result))

class TestAssignment3(unittest.TestCase):
    def setUp(self):
        self.ass3 = Assignment3()

    def test_polynomial_product(self):
        f1 = lambda x: (x - 2) * (x - 4)
        f2 = lambda x: 0
        expected_result = 1.33333
        result = self.ass3.areabetween(f1, f2)
        print(f"Test polynomial_product: Result = {result}")
        if np.isnan(result):
            print("No intersections in the range [1, 100].")
        self.assertAlmostEqual(result, expected_result, places=2)

    def test_polynomial_multiple_roots(self):
        f1 = lambda x: (x - 2) * (x - 4) * (x - 7) * (x - 3)
        f2 = lambda x: 0
        expected_result = 43.4
        result = self.ass3.areabetween(f1, f2)
        roots = sorted(self.ass3.assignment2.intersections(f1, f2, 1, 100))
        print(f"Roots found: {roots}")
        print(f"Test polynomial_multiple_roots: Result = {result}")
        self.assertAlmostEqual(result, expected_result, places=2)

if __name__ == "__main__":
    unittest.main()
