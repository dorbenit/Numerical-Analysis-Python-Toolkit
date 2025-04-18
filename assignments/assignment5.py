"""
In this assignment you should fit a model function of your choice to data 
that you sample from a contour of given shape. Then you should calculate
the area of that shape. 

The sampled data is very noisy so you should minimize the mean least squares 
between the model you fit and the data points you sample.  

During the testing of this assignment running time will be constrained. You
receive the maximal running time as an argument for the fitting method. You 
must make sure that the fitting function returns at most 5 seconds after the 
allowed running time elapses. If you know that your iterations may take more 
than 1-2 seconds break out of any optimization loops you have ahead of time.

Note: You are allowed to use any numeric optimization libraries and tools you want
for solving this assignment. 
Note: !!!Despite previous note, using reflection to check for the parameters 
of the sampled function is considered cheating!!! You are only allowed to 
get (x,y) points from the given shape by calling sample(). 
"""
from shapely.geometry import Polygon
import numpy as np
from scipy.interpolate import splev, splprep
from scipy.integrate import simpson
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
import time
import random
from functionUtils import AbstractShape


class MyShape(AbstractShape):
    # change this class with anything you need to implement the shape
    def __init__(self, tck):

        self.tck = tck  # (t, [cx, cy], k)

    def sample(self):

        t_rand = np.random.rand()
        x_val, y_val = splev(t_rand, self.tck)
        return (x_val, y_val)

    def contour(self, n: int):

        t_vals = np.linspace(0, 1, n, endpoint=False)
        x_vals, y_vals = splev(t_vals, self.tck)
        contour_points = np.column_stack((x_vals, y_vals)).astype(np.float32)
        return contour_points

    def area(self):
        """
        Calculate the area of the shape using the polygonal approximation.
        """
        def compute_area(pts):
            """Compute the area using the Shoelace formula."""
            pts = np.array(pts)
            x, y = pts[:, 0], pts[:, 1]
            x_next, y_next = np.roll(x, -1), np.roll(y, -1)
            return 0.5 * np.abs(np.dot(x, y_next) - np.dot(y, x_next))

        num_points = 10
        prev_area = 0

        while True:
            # Sample points
            points = self.contour(num_points)

            # Compute area
            current_area = compute_area(points)

            # Check error
            if abs(current_area - prev_area) <= 0.001:
                break

            prev_area = current_area
            num_points *= 2  # Double the number of points for accuracy

            # Safety: Stop if points exceed reasonable limit
            if num_points > 10000:
                break

        return np.float32(current_area)


class Assignment5:
    def __init__(self):
        """
        Here goes any one time calculation that need to be made before
        solving the assignment for specific functions.
        """

        pass

    def option1(self, contour: callable, maxerr=0.001) -> np.float32:

        num_points = 10
        prev_area = 0

        while True:
            # Sample points
            points = contour(num_points)
            polygon = Polygon(points)

            # Compute area
            current_area = polygon.area

            # Check error
            if abs(current_area - prev_area) <= maxerr:
                break

            prev_area = current_area
            num_points *= 2  # Double the number of points for accuracy

            # Safety: Stop if points exceed reasonable limit
            if num_points > 10000:
                break

        return np.float32(current_area)

    def option2(self, contour: callable, maxerr=0.001) -> np.float32:

        def compute_area(pts):
            """Compute the area using the Shoelace formula."""
            s = 0.0
            n = len(pts)
            for i in range(n):
                x0, y0 = pts[i]
                x1, y1 = pts[(i + 1) % n]
                s += x0 * y1 - y0 * x1
            return abs(s) * 0.5

        n_points = 1000  # Fixed number of sampled points
        points = contour(n_points)  # Get the sampled points
        return np.float32(compute_area(points))
    def option_simpson(self, contour: callable, maxerr=0.001) -> np.float32:
        """
        Compute the area of the shape with the given contour using Simpson's Rule.

        Parameters
        ----------
        contour : callable
            A function that returns a list of points (x, y) when called with a number of samples.
        maxerr : float, optional
            The target error of the area computation. The default is 0.001.

        Returns
        -------
        np.float32
            The computed area of the shape.
        """
        num_points = 11  # Start with an odd number of points for Simpson's rule
        prev_area = 0

        while True:
            # Sample points
            points = contour(num_points)

            # Extract x and y coordinates
            x = [p[0] for p in points]
            y = [p[1] for p in points]

            # Compute area using Simpson's rule
            current_area = simpson(y, x)

            # Check error
            if abs(current_area - prev_area) <= maxerr:
                break

            prev_area = current_area
            num_points = num_points * 2 - 1  # Ensure an odd number of points

            # Safety: Stop if points exceed reasonable limit
            if num_points > 10001:
                break

        return np.float32(abs(current_area))

    def option4(self, contour: callable, maxerr=0.001) -> np.float32:
        num_points = 10
        prev_area = 0
        factor = 1.5  # Start with a moderate growth factor

        while True:
            # Sample points
            points = contour(num_points)
            polygon = Polygon(points)

            # Compute area
            current_area = polygon.area

            # Check error
            if abs(current_area - prev_area) <= maxerr:
                break

            # Adjust the growth factor based on error
            error = abs(current_area - prev_area)
            growth = max(2, min(int(num_points * (error / maxerr)), 1000))  # Limit growth
            prev_area = current_area
            num_points += growth

            # Safety: Stop if points exceed reasonable limit
            if num_points > 10000:
                break

        return np.float32(current_area)
    def area(self, contour: callable, maxerr=0.001)->np.float32:
        """
        Compute the area of the shape with the given contour. 

        Parameters
        ----------
        contour : callable
            Same as AbstractShape.contour 
        maxerr : TYPE, optional
            The target error of the area computation. The default is 0.001.

        Returns
        -------
        The area of the shape.

        """
        return self.option2(contour, maxerr)


##########################################################################
    def _spline_fit(self, points, smooth_factor):
        if len(points) < 5:
            return None

        if len(points) > 3000:
            step = len(points) // 3000
            points = points[::step]

        x_arr = np.array([p[0] for p in points])
        y_arr = np.array([p[1] for p in points])
        cx, cy = x_arr.mean(), y_arr.mean()

        theta_arr = np.arctan2(y_arr - cy, x_arr - cx)
        order_idx = np.argsort(theta_arr)
        x_sorted = x_arr[order_idx]
        y_sorted = y_arr[order_idx]

        x_smooth = gaussian_filter1d(x_sorted, sigma=2)
        y_smooth = gaussian_filter1d(y_sorted, sigma=2)

        s_val = smooth_factor * len(points)
        try:
            tck, u = splprep([x_smooth, y_smooth], s=s_val, per=True, k=3)
            return MyShape(tck)
        except:
            return None

    def _spline_fit_with_time_limit(self, points, smooth_factor, maxtime):

        time_left = maxtime - time.time()
        if time_left < 2.0:
            return None
        return self._spline_fit(points, smooth_factor)

    def option_spline_fit(self, sample: callable, maxtime: float):

        start_time = time.time()
        end_time = start_time + maxtime
        collect_deadline = start_time + 0.8 * maxtime

        points = []
        max_points = 300000
        while time.time() < collect_deadline and len(points) < max_points:
            points.append(sample())

        if len(points) < 5:
            return MyShape(None)

        x_arr = np.array([p[0] for p in points])
        y_arr = np.array([p[1] for p in points])
        cx, cy = x_arr.mean(), y_arr.mean()
        r_arr = np.sqrt((x_arr - cx) ** 2 + (y_arr - cy) ** 2)
        r_mean = r_arr.mean()
        r_std = r_arr.std() if r_arr.size > 1 else 0
        cutoff = r_mean + 3 * r_std
        filtered = []
        for (xx, yy), rr in zip(points, r_arr):
            if rr < cutoff:
                filtered.append((xx, yy))
        if len(filtered) < 5:
            return MyShape(None)

        points = filtered

        shape_coarse = self._spline_fit_with_time_limit(points, smooth_factor=0.002, maxtime=end_time)
        if not shape_coarse or not shape_coarse.tck:
            return MyShape(None)
        best_shape = shape_coarse

        time_left = end_time - time.time()
        if time_left > 2.0:
            shape_fine = self._spline_fit_with_time_limit(points, smooth_factor=0.007, maxtime=end_time)
            if shape_fine and shape_fine.tck:
                best_shape = shape_fine

        time_left = end_time - time.time()
        if time_left > 2.0:
            shape_extra_fine = self._spline_fit_with_time_limit(points, smooth_factor=0.003, maxtime=end_time)
            if shape_extra_fine and shape_extra_fine.tck:
                best_shape = shape_extra_fine

        return best_shape

    def fit_shape(self, sample: callable, maxtime: float) -> AbstractShape:
        return self.option_spline_fit(sample, maxtime)




import unittest
from sampleFunctions import *
from tqdm import tqdm


class TestAssignment5(unittest.TestCase):

    def test_return(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=5)
        T = time.time() - T
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertLessEqual(T, 5)

    def test_delay(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)

        def sample():
            time.sleep(7)
            return circ()

        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=sample, maxtime=5)
        T = time.time() - T
        self.assertTrue(isinstance(shape, AbstractShape))
        self.assertGreaterEqual(T, 5)

    def test_circle_area(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLessEqual(T, 32)

    def test_bezier_fit(self):
        circ = noisy_circle(cx=1, cy=1, radius=1, noise=0.1)
        ass5 = Assignment5()
        T = time.time()
        shape = ass5.fit_shape(sample=circ, maxtime=30)
        T = time.time() - T
        a = shape.area()
        self.assertLess(abs(a - np.pi), 0.01)
        self.assertLessEqual(T, 32)


if __name__ == "__main__":
    unittest.main()
