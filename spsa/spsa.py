import matplotlib.pyplot as plt
import time
import numpy as np
from random import random
from random import sample
from copy import deepcopy
import pandas as pd

__version__ = "0.1.4"
print(f"Version: {__version__}")


class Parameters:
    n: int  # number of sensors
    N: list  # indexes of sensors
    s: dict  # sensors coordinates

    m: int  # nuber of targets
    M: list  # targets indexes
    r: dict  # targets coordinates

    beta_1: float
    beta_2: float
    alpha: float
    gamma: float
    weight: list

    moment_alpha: float
    track_coef: float


class Agent:
    targets: dict

class TargetInfo:
    theta_hat


def rho(point_1, point_2):
    """Calculate distance between point_1 and point_2"""
    diff = point_1 - point_2
    return sum(diff * diff)


def cond_num(matrix):
    eig = np.linalg.eig(matrix)[0]
    eig = sorted([abs(n) for n in eig if abs(n) > 0.00001])
    return eig[-1] / eig[0]

class Result:
    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)


class SPSA:
    colors = ["blue", "red", "green", "orange", "yellow", "purple", "black", "gray"]

    def __init__(self, **kwargs):
        self.__dict__.update(**kwargs)

        self.s_norms = {i: sum(val * val) for i, val in self.s.items()}
        self.Delta_abs_value = 1 / np.sqrt(self.d)
        self.beta = self.beta_1 + self.beta_2
        self.meas = {targ: {sens: rho(self.r.get(targ), self.s.get(sens)) for sens in self.N} for targ in self.M}

    def init_plot(self):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot()
        plt.ion()

        self.fig.show()
        self.fig.canvas.draw()

        # draw targets
        x = [i for i, j in self.target_path[1].values()]
        y = [j for i, j in self.target_path[1].values()]
        self.ax.plot(x, y, 'bx', markersize=8)

        x = [i for i, j in self.s.values()]
        y = [j for i, j in self.s.values()]
        self.ax.plot(x, y, 'gx', markersize=3)

    def init_random_vars(self, num_steps, friends_num, tracking):
        # set up random variables
        self.neibors = {k: self.get_random_neibors(self.weight, max=friends_num) for k in range(1, num_steps + 1)}
        self.random_coef = {
            k: {
                sensor: [random(), random()] for sensor in self.N
            } for k in range(1, num_steps + 1)
        }  # todo: for multiple targets

        self.target_path = {1: deepcopy(self.r)}

        if not tracking:
            return

        coef = self.track_coef
        for k in range(2, num_steps + 1):
            self.target_path[k] = {l:
                                       self.target_path[k - 1][l] + [2 * coef * random() - coef,
                                                                     2 * coef * random() - coef]
                                   for l in self.M
                                   }

    def run(self, method, num_steps=20, eps=0.01, friends_num=2, accelerate=False, tracking=False, check=False):
        self.method = method
        print(self.moment_alpha)

        if not hasattr(self, "neibors"):
            print("Init random variables")
            self.init_random_vars(num_steps, friends_num, tracking)

        self.init_plot()

        self.update_target_position(self.target_path[1], tracking)
        theta_hat = {
            target: {
                sensor: self.s.get(sensor) + np.array([np.sqrt(self.meas.get(target).get(sensor) / 2),
                                                       np.sqrt(self.meas.get(target).get(sensor) / 2)])
                for sensor in self.N
            } for target in self.M
        }
        err_history = {target: {} for target in self.M}
        errors = {target: {} for target in self.M}
        history_val = pd.DataFrame()

        # nesterov coef
        L = 2
        h = 0.08
        H = h - pow(h, 2) * L / 2
        gamma_nest_next = 0.1
        mu = 2
        eta = 0.95
        alpha_nest = 0.5
        alpha_x = 0.5

        for step in range(1, num_steps + 1):  # шаги
            theta_new = {}
            gamma_nest = gamma_nest_next
            gamma_nest_next = (1 - alpha_nest) * gamma_nest + alpha_nest * (mu - eta)
            # diff = np.sqrt(H * 2 * gamma_nest) - alpha_x
            diff = 0
            if diff < 0:
                diff = 0
                print("diff < 0")
            alpha_nest = alpha_x + diff

            for target in self.M:
                theta_new[target] = {}  # Estimator of target position
                err_history[target][step] = {}  # Error history for each step step for each target target

                for ind, sensor in enumerate(self.N):
                    if step == 0:
                        self.ax.plot(theta_hat[target][sensor][0], theta_hat[target][sensor][1], 'o', markersize=3,
                                     color=self.colors[ind])

                    spsa = self.spsa_step(sensor, step, target, theta_hat)

                    # nesterov acceleraion
                    # firstly go in the direction of previous theta_diff then find new one
                    # theta_diff_sum can be changed to sum so we will acc spsa step to
                    if step > 1 and accelerate:
                        prev_step_info = history_val.loc[
                            (history_val["target"] == target)
                            & (history_val["step"] == step - 1)
                            & (history_val["sensor"] == sensor)]

                        v_n = prev_step_info["v_n"].iloc[0]
                        x_n = 1 / (gamma_nest + alpha_nest * (mu - eta)) * (alpha_nest * gamma_nest * v_n
                                                                            + gamma_nest_next * theta_hat[target][
                                                                                sensor])
                        coef = h
                    else:
                        x_n = theta_hat[target][sensor]
                        v_n = 0
                        coef = self.gamma

                    theta_diff = self.local_vote_step(sensor, step, target, x_n, theta_hat)
                    nesterov_step = coef * theta_diff

                    theta_new[target][sensor] = x_n - (self.alpha * spsa + nesterov_step)

                    if step > 1 and accelerate:
                        v_n = 1 / gamma_nest * ((1 - alpha_nest) * gamma_nest * v_n + alpha_nest * (
                                    mu - eta) * x_n - alpha_nest * nesterov_step)

                    self.update_plot_step(ind, sensor, target, theta_hat, theta_new)

                    err_l_i = self.compute_error(theta_new[target][sensor], self.r[target])
                    history_val = history_val.append({
                        "sensor": sensor,
                        "target": target,
                        "step": step,
                        "neibors": self.neibors[step][sensor],
                        "old": theta_hat[target][sensor],
                        "spsa": self.alpha * spsa,
                        "theta_diff": theta_diff,
                        "theta_diff_sum": self.gamma * theta_diff,
                        "sum": self.alpha * spsa + self.gamma * theta_diff,
                        "new": theta_new[target][sensor],
                        "err": err_l_i,
                        "nesterov_step": nesterov_step,
                        "v_n": v_n,
                        "alpah_nest": alpha_nest,
                        "gamma_nest": gamma_nest
                    }, ignore_index=True)
                    err_history[target][step][sensor] = err_l_i

                self.fig.canvas.draw()
                print(
                    f"Error - {sum(err_history[target][step].values()) / self.n:.2f} on {step} step for {target} target")
                time.sleep(1)
                errors[target][step] = sum(err_history[target][step].values()) / self.n

            if errors[1][step] < eps or errors[1][step] > 1e+9:  # todo: for multiple targets
                break

            theta_hat = deepcopy(theta_new)

            if tracking:
                self.update_target_position(self.target_path[step], tracking)

        self.errors = errors
        self.err_history = err_history

        # Compute error for each target and sensor separately
        target_err = {
            target: {
                sensor: self.compute_error(theta_hat[target][sensor], self.r[target]) for sensor in self.N
            } for target in self.M
        }

        return Result(
            check=check,
            tracking=tracking,
            accelerate=accelerate,
            method=method,
            errors=errors,
            theta_hat=theta_hat,
            moment_alpha=self.moment_alpha,
            target_err=target_err,
            history=history_val,
            err_history=err_history)

    def update_plot_step(self, ind, sensor, target, theta_hat, theta_new):
        if sensor == 3:
            self.ax.plot([theta_hat[target][sensor][0], theta_new[target][sensor][0]],
                         [theta_hat[target][sensor][1], theta_new[target][sensor][1]],
                         markersize=2, color=self.colors[ind])
        self.ax.plot(theta_new[target][sensor][0], theta_new[target][sensor][1], 'o', markersize=3,
                     color=self.colors[ind])

    def local_vote_step(self, sensor, step, target, theta_sensor, theta_hat):
        neibors_i = self.neibors[step].get(sensor, [])
        b = self.weight[sensor - 1]

        theta_diff = sum([abs(b[j - 1]) * (theta_sensor - theta_hat[target][j]) for j in neibors_i])
        return theta_diff

    def spsa_step(self, sensor, step, target, theta_hat):
        coef1 = 1 if self.random_coef[step][sensor][0] < 0.5 else -1
        coef2 = 1 if self.random_coef[step][sensor][1] < 0.5 else -1

        delta = np.array([coef1 * self.Delta_abs_value, coef2 * self.Delta_abs_value])

        # spsa step
        x1 = theta_hat[target][sensor] + self.beta_1 * delta
        x2 = theta_hat[target][sensor] - self.beta_2 * delta

        y1 = self.f_l_i(target, sensor, x1, self.neibors[step])
        y2 = self.f_l_i(target, sensor, x2, self.neibors[step])

        spsa = (y1 - y2) / (2 * self.beta) * delta
        return spsa

    def f_l_i(self, l, i, r_hat_l, neibors):
        """ Calculate function f for target l and for sensor i
        :param l: target index
        :param i: i sensor index
        :param r_hat_l: x point at which calculate
        :return: matrixe D for i sensor and l target
        """
        C = self.C_i(i, neibors)
        D = self.D_l_i(l, i, neibors)

        try:
            C_i_inv = np.linalg.inv(C)
        except Exception:
            C_i_inv = np.linalg.pinv(C)

        diff = r_hat_l - np.matmul(C_i_inv, D)
        return sum(diff * diff)

    def C_i(self, i, neibors):
        """ Calculate matrix C for sensor i
        :param i: index of i sensor
        :return: matrixe C for i sensor
        """
        C_i = [self.s.get(j) - self.s.get(i) for j in neibors.get(i)]
        return 2 * np.array(C_i)

    def D_l_i(self, l, i, neibors):
        """ Calculate matrix D for target l and for sensor i
        :param l: target index
        :param i: i sensor index
        :return: matrixe D for i sensor and l target
        """
        Dli = [self.calc_D_l_i_j(self.meas.get(l), i, j) for j in neibors.get(i)]
        return Dli

    def calc_D_l_i_j(self, meas_l: dict, i, j):
        """Calculate value of D_l_i_j
        :param meas_l: distances between l target and each sensor
        :param i: index of 1st sensor
        :param j: index of 2nd sensor
        :return: D_l_i[j] for vector D_l_i
        """
        return self.rho_overline(meas_l.get(i), meas_l.get(j)) + self.s_norms.get(j) - self.s_norms.get(i)

    def rho_overline(self, meas_1: float, meas_2: float):
        """Calculate difference between meas_1 and meas_2"""
        return meas_1 - meas_2

    def gen_new_coordinates(self, coords: np.array, R: float = 1):
        """ Add shift to coordinats within specified radius
        """
        phi = 2 * np.pi * random()
        rad = R * random()

        shift = rad * np.array([np.sin(phi), np.cos(phi)])
        return coords + shift

    def update_target_position(self, new_positions, tracking):
        if tracking:
            self.ax.plot([self.r[1][0], new_positions[1][0]],
                         [self.r[1][1], new_positions[1][1]], 'bx', markersize=8)
        self.r = new_positions
        self.meas = {
            target:
                {
                    sensor: rho(self.r.get(target), self.s.get(sensor)) for sensor in self.N
                    # measurments  from target to sensor
                } for target in self.M
        }

    def compute_error(self, vector_1, vector_2):
        return pow(sum(vector_1 - vector_2), 2)

    def get_random_neibors(self, weight, max=2):
        neibors_mat = (weight != 0).astype(int)
        np.fill_diagonal(neibors_mat, 0)

        # choose num random neibors from all neibors
        neibors = {}
        for sensor in self.N:
            neib = [ind + 1 for ind, sens in enumerate(neibors_mat[sensor - 1]) if sens == 1]

            if len(neib) > max:
                neib = sample(neib, max)

            neibors[sensor] = neib

        return neibors

    #  Optimization functions
    def cheb_polyn_mat(self, n, x, c2):
        cheb = [0] * 3
        cheb[0] = np.identity(len(x))
        cheb[1] = c2 * x

        for i in range(2, self.n + 1):
            next_cheb = 2 * c2 * np.matmul(x, cheb[1]) - cheb[0]
            cheb[0] = cheb[1]
            cheb[1] = next_cheb
        return cheb[min([n, 1])]

    def cheb_acceleration(self, mat):
        eigens = sorted(np.linalg.eig(mat)[0])
        cond = abs(eigens[-1]) / abs(eigens[1])

        c2 = (cond + 1) / (cond - 1)
        c3 = 2 / (eigens[-1] + eigens[1])

        eye_mat = np.identity(len(mat))
        k = int(np.floor(np.sqrt(cond)))

        mat_k = self.cheb_polyn_mat(k, eye_mat - c3 * mat, c2)
        a_k = self.cheb_polyn_mat(k, np.array([1]), c2)
        cheb_pol = eye_mat - mat_k / a_k

        return cheb_pol


if __name__ == "__main__":
    par = Parameters()

    par.n = 5  # number of sensors
    par.N = {i for i in range(1, par.n + 1)}  # indexes of sensors
    par.s = {1: np.array([1, 2]),  # sensors coordinates
             2: np.array([3, 20]),
             3: np.array([10, 3]),
             4: np.array([20, 3]),
             5: np.array([3, 10])}

    par.m = 1  # nuber of targets
    par.M = {i for i in range(1, par.m + 1)}  # targets indexes
    par.r = {1: np.array([40, 10])}  # targets coordinates

    par.d = 2  # number of dimensions

    par.beta_1 = 0.5
    par.beta_2 = 0.5

    par.alpha = 1 / 4
    par.gamma = 1 / 4

    par.weight = np.array([[3., -1., -1., -1., 0.],
                           [-1., 3., 0., -1., -1.],
                           [-1., 0., 2., 0., -1.],
                           [-1., -1., 0., 3., -1.],
                           [0., -1., -1., -1., 3.]])

    par.nest_alpha = 1 / 4

    par.b = 1
    par.moment_alpha = 0.6


    def K(u):
        coef1 = 1 if u[0] > 0 else -1
        coef2 = 1 if u[1] > 0 else -1
        return np.array([1 / 4, 1 / 4])


    spsa = SPSA(**par.__dict__)
    print("\n\nMain method\n")
    spsa_res = spsa.run("main", accelerate=True, tracking=True)

    print(spsa_res)
