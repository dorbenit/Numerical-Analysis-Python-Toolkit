"""
In this assignment you should find the intersection points for two functions.
"""

import numpy as np
import time
import random
from collections.abc import Iterable


class Assignment2:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before
        solving the assignment for specific functions.
        """

        pass

    def intersections(self, f1: callable, f2: callable, a: float, b: float, maxerr=0.001) -> Iterable:
        """
        Find as many intersection points as you can. The assignment will be
        tested on functions that have at least two intersection points, one
        with a positive x and one with a negative x.

        This function may not work correctly if there is infinite number of
        intersection points.


        Parameters
        ----------
        f1 : callable
            the first given function
        f2 : callable
            the second given function
        a : float
            beginning of the interpolation range.
        b : float
            end of the interpolation range.
        maxerr : float
            An upper bound on the difference between the
            function values at the approximate intersection points.


        Returns
        -------
        X : iterable of approximate intersection Xs such that for each x in X:
            |f1(x)-f2(x)|<=maxerr.

        """

        def find_root(f, sub_a, sub_b, maxerr=0.001):
            """
            Find a root of the function f in the interval [sub_a, sub_b] using Brent's method.
            Assumes f(sub_a) and f(sub_b) have opposite signs (a bracketed root).

            Parameters
            ----------
             f : callable
             A one-dimensional function, f(x).
            sub_a : float
            Left endpoint of the interval (where f(sub_a)*f(sub_b) < 0).
            sub_b : float
             Right endpoint of the interval.
             maxerr : float, optional
            The required accuracy (default is 0.001). Stops when |f(x)| <= maxerr
            or the interval is sufficiently small.

            Returns
            -------
            float
            Approximation of the root where f(root) = 0 within the tolerance maxerr.

            Raises
            ------
            ValueError
            If f(sub_a)*f(sub_b) >= 0 (no root is bracketed) or if it fails to converge
            within the maximum number of iterations.
        """

            max_iter = 100  # Maximum number of iterations
            small_eps = 1e-14  # A small threshold to avoid division by zero

            a = sub_a
            b = sub_b

            fa = f(a)
            fb = f(b)

            # Ensure that the root is indeed bracketed
            if fa * fb > 0:
                raise ValueError("error: f(a)*f(b)>0")

            # Make (b, fb) the point with smaller absolute function value if needed
            if abs(fa) < abs(fb):
                a, b = b, a
                fa, fb = fb, fa

            # c is a copy of a
            c = a
            fc = fa

            # d,e track previous steps
            d = b - a
            e = d

            for iteration in range(max_iter):

                # If |fb| is already within maxerr, we're done
                if abs(fb) <= maxerr:
                    return b
                # If the interval is extremely small, stop
                if abs(b - a) < small_eps:
                    return b

                # Swap if needed so that (b, fb) is the point with the smallest function value
                if abs(fa) < abs(fb):
                    a, b = b, a
                    fa, fb = fb, fa

                # Midpoint for bisection
                m = 0.5 * (a - b)

                # Another check for extremely small midpoint
                if abs(m) < small_eps:
                    return b

                # Decide whether to do interpolation or fallback to bisection
                use_bisection = False

                # Attempt inverse quadratic interpolation if possible
                if abs(fc) > small_eps and fc != fb and fc != fa:
                    # Inverse Quadratic Interpolation
                    s_num1 = a * fb * fc / ((fa - fb) * (fa - fc))
                    s_num2 = b * fa * fc / ((fb - fa) * (fb - fc))
                    s_num3 = c * fa * fb / ((fc - fa) * (fc - fb))
                    s = s_num1 + s_num2 + s_num3
                else:
                    # Secant Method
                    if (fb - fa) != 0:
                        s = b - fb * (b - a) / (fb - fa)
                    else:
                        # Fallback: just use the midpoint if the secant slope is zero
                        s = 0.5 * (a + b)

                # fallback conditions
                cond1 = (not ((3 * a + b) / 4 < s < b or (b < s < (3 * a + b) / 4))
                         )
                cond2 = (abs(s - b) >= abs(b - c) / 2.0)
                cond3 = (abs(b - c) < small_eps)

                if cond1 or cond2 or cond3:
                    # Fallback to bisection
                    s = 0.5 * (a + b)
                    use_bisection = True

                fs = f(s)

                d, c = c, b
                fc = fb

                # If fa*fs < 0, root is in [a, s]; otherwise [s, b]
                if fa * fs < 0:
                    b = s
                    fb = fs
                else:
                    a = s
                    fa = fs
                # Ensure (b, fb) remains the best approximation
                if abs(fa) < abs(fb):
                    a, b = b, a
                    fa, fb = fb, fa

            # If we exit the loop, we've failed to converge
            raise ValueError(f"failed to converge after {max_iter} iterations.")

        def bisection_method(f, a, b, maxerr):
            """
            Simple implementation of the Bisection method.
            Finds a root of f(x) in the interval [a, b] with a maximum error maxerr.
            """
            small_eps = 1e-14  # A small threshold to avoid division by zero
            if abs(f(a)) <= maxerr:
                return a
            if abs(f(b)) <= maxerr:
                return b
            if f(a) * f(b) >= 0:
                return None  # No root in this interval
            while abs(b - a) > maxerr:
                mid = (a + b) / 2
                if f(mid) <= small_eps:  # root found
                    return mid
                elif f(a) * f(mid) < 0:
                    b = mid
                else:
                    a = mid

            return (a + b) / 2

        def f(x):
            return f1(x) - f2(x)

        def coarse_split(f, a, b, num_intervals=1000):
            """
            Splits the interval [a, b] into a specified number of sub-intervals
            and checks for sign changes using NumPy for efficiency.

            Parameters
            ----------
            f : callable
                The function f(x) to check.
            a : float
                Left endpoint of the interval.
            b : float
                Right endpoint of the interval.
            num_intervals : int, optional
                Number of intervals to use for coarse splitting (default is 1000).

            Returns
            -------
            list of tuples
                A list of (sub_a, sub_b) pairs where a sign change occurs, indicating a possible root.
            """
            if (b - a) > 2000:  # Adjust for large intervals
                num_intervals = 100

            # Create evenly spaced points using NumPy
            x_points = np.linspace(a, b, num_intervals + 1)
            f_values = np.array([f(x) for x in x_points])  # Evaluate f at all points

            # Check for sign changes using vectorized operations
            sign_changes = np.where(f_values[:-1] * f_values[1:] <= 0)[0]

            # Create intervals for each detected sign change
            intervals = [(x_points[i], x_points[i + 1]) for i in sign_changes]

            return intervals

        def adaptive_refinement(f, a, b, maxerr, threshold=0.1):
            """
            Performs an adaptive interval refinement approach within [a, b].
            This method adjusts step sizes dynamically based on changes in function values.

            Parameters
            ----------
            f : callable
                The function f(x) for which we seek roots.
            a : float
                Start of the interval.
            b : float
                End of the interval.
            maxerr : float
                The desired maximum error.
            threshold : float, optional
                Threshold for determining how to adapt the step size (default is 0.1).

            Returns
            -------
            list of tuples
                A list of sub-intervals [(x_start, x_end), ...] to explore further.
            """
            intervals = []
            step = (b - a) / 10  # Initial step size

            # Adjust step size for large ranges
            if (b - a) > 2000:
                threshold *= 2
                big_factor = 1.5
                small_factor = 1.5
            else:
                big_factor = 2
                small_factor = 2

            # Generate initial points with NumPy
            x_points = [a]  # Start with the initial point
            x = a

            while x < b:
                fx = f(x)
                fx_next = f(x + step) if x + step <= b else f(b)

                # Adjust step size based on function behavior
                if abs(fx - fx_next) > threshold:
                    step /= small_factor
                else:
                    step *= big_factor

                # Keep the step size within reasonable bounds
                step = max(step, maxerr / 10)
                step = min(step, (b - a) / 10)

                # Append the new point
                x = min(x + step, b)
                x_points.append(x)

            # Convert to NumPy array for optimization
            x_points = np.array(x_points)
            f_values = np.array([f(x) for x in x_points])

            # Iterate through points and create intervals
            for i in range(len(x_points) - 1):
                if abs(f_values[i] - f_values[i + 1]) > threshold:  # Significant change detected
                    intervals.append((x_points[i], x_points[i + 1]))
                elif f_values[i] * f_values[i + 1] <= 0:  # Sign change detected
                    intervals.append((x_points[i], x_points[i + 1]))

            return intervals

        def hybrid_intervals(f, a, b, maxerr, coarse_intervals=1000, threshold=0.1):
            """
            Combines coarse splitting and adaptive refinement to identify sub-intervals
            in [a, b] where roots are likely to exist.

            Parameters
            ----------
            f : callable
                The function to analyze, f(x).
            a : float
                The start of the interval.
            b : float
                The end of the interval.
            maxerr : float
                The desired maximum error for refinement.
            coarse_intervals : int, optional
                Number of intervals for the initial coarse split (default 1000).
            threshold : float, optional
                Threshold for adaptive refinement sensitivity (default 0.1).

            Returns
            -------
            list of tuples
                A list of sub-intervals [(start, end), ...] where roots are likely to exist.
            """
            # Adjust splitting and threshold parameters for large intervals
            if (b - a) > 2000:
                coarse_intervals = 100  # Reduce number of intervals for large ranges
                threshold *= (b - a) / 100  # Increase sensitivity threshold

            # Perform coarse splitting to find major sign changes
            coarse_intervals = coarse_split(f, a, b, num_intervals=coarse_intervals)
            refined_intervals = []

            # Refine each coarse interval adaptively
            for sub_a, sub_b in coarse_intervals:
                refined_intervals.extend(adaptive_refinement(f, sub_a, sub_b, maxerr, threshold=threshold))

            return refined_intervals

            # Identify intervals containing potential roots using the hybrid approach

        intervals = hybrid_intervals(f, a, b, maxerr=maxerr, coarse_intervals=1000, threshold=0.1)

        # Use a set to collect unique roots without duplication
        roots = set()

        # Iterate through each sub-interval to locate roots
        for sub_a, sub_b in intervals:
            fa = f(sub_a)  # Evaluate function at the left endpoint
            fb = f(sub_b)  # Evaluate function at the right endpoint

            # Check if there is a sign change in the interval
            if fa * fb < 0:
                try:
                    # Use find_root to find the root in the interval
                    root = find_root(f, sub_a, sub_b, maxerr)
                except ValueError:
                    # Skip the interval if root finding fails
                    continue
            # Check if either endpoint is already close enough to a root
            elif abs(fa) <= maxerr:
                root = sub_a
            elif abs(fb) <= maxerr:
                root = sub_b
            else:
                # Skip intervals where no valid root is detected
                continue

            # Validate and add the root if it is distinct and within the error tolerance
            if root is not None and abs(f(root)) <= maxerr:
                # Ensure the root is not too close to any existing root
                if all(abs(root - r) > maxerr for r in roots):
                    roots.add(root)

        # Return the set of unique roots found in the interval
        return roots


##########################################################################


import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment2(unittest.TestCase):

    def test_sqr(self):

        ass2 = Assignment2()

        f1 = np.poly1d([-1, 0, 1])
        f2 = np.poly1d([1, 0, -1])

        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))

    def test_poly(self):

        ass2 = Assignment2()

        f1, f2 = randomIntersectingPolynomials(10)

        X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)

        for x in X:
            self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))


# class TestAssignment2(unittest.TestCase):

# def test_polynomial_intersection(self):
#     """
#     Test intersections for two simple polynomials with known roots.
#     """
#     ass2 = Assignment2()
#     f1 = np.poly1d([-1, 0, 1])
#     f2 = np.poly1d([1, 0, -1])
#     X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)
#     print("test_polynomial_intersection: ", len(X))
#     for x in X:
#         self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))
#
# def test_random_polynomials(self):
#     """
#     Test intersections for two random polynomials of degree 10.
#     """
#     ass2 = Assignment2()
#     f1, f2 = randomIntersectingPolynomials(10)
#     X = ass2.intersections(f1, f2, -1, 1, maxerr=0.001)
#     print("test_random_polynomials: ", len(X))
#     for x in X:
#         self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))
#
# def test_sine_vs_zero(self):
#     """
#     Test intersections between sine function and zero line.
#     """
#     ass2 = Assignment2()
#     f1 = np.poly1d([0])
#     f2 = sinus()
#     X = ass2.intersections(f1, f2, -10, 10, maxerr=0.001)
#     print("test_sine_vs_zero: ", len(X))
#     for x in X:
#         self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))
#
# def test_high_degree_polynomial(self):
#     """
#     Test intersections for a high-degree polynomial and zero.
#     """
#     ass2 = Assignment2()
#     f1 = np.poly1d([5, -2, 1, 0, -7, 3, 0])
#     f2 = np.poly1d([0])
#     X = ass2.intersections(f1, f2, -10, 10, maxerr=0.001)
#     print("test_high_degree_polynomial: ", len(X))
#     for x in X:
#         self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))
#
# def test_strong_changes(self):
#     """
#     Test intersections for a function with strong changes near zero.
#     """
#     ass2 = Assignment2()
#     f1 = strong_oscilations()
#     f2 = np.poly1d([0])
#     X = ass2.intersections(f1, f2, -10, -0.07, maxerr=0.001)
#     print("test_strong_oscillations: ", len(X))
#     for x in X:
#         self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))
#
# def test_sine_power_3(self):
#     """
#     Test intersections for sin(x^3) and zero in a narrow range.
#     """
#     ass2 = Assignment2()
#     f1 = np.poly1d([0])
#
#     def f2(x):
#         return np.sin(x**3)
#
#     X = ass2.intersections(f1, f2, 8.4, 8.8, maxerr=0.001)
#     print("test_sine_power_3: ", len(X))
#     for x in X:
#         self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))
#
# def test_sine_vs_cosine(self):
#     """
#     Test intersections between sine and a modified cosine function.
#     """
#     ass2 = Assignment2()
#     f1 = np.poly1d([0])
#
#     def f2(x):
#         return 5 * np.sin(x * 6) - np.cos(x / 3) * x + x + 3.2
#
#     X = ass2.intersections(f1, f2, -20, 20, maxerr=0.001)
#     print("test_sine_vs_cosine: ", len(X))
#     for x in X:
#         self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))
#
# def test_complex_polynomial(self):
#     """
#     Test intersections for two complex polynomials.
#     """
#     ass2 = Assignment2()
#     f1 = np.poly1d([0.2, 8, 0, -7, 0, 0, 0, 0.5])
#     f2 = np.poly1d([0])
#     X = ass2.intersections(f1, f2, -10, 10, maxerr=0.001)
#     print("test_complex_polynomial: ", len(X))
#     for x in X:
#         self.assertGreaterEqual(0.001, abs(f1(x) - f2(x)))
if __name__ == "__main__":
    unittest.main()
