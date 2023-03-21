import time
from copy import deepcopy
from random import random
from random import sample

import matplotlib.pyplot as plt
import numpy as np
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
    id: int
    position: list
    targets_info: dict

    def __init__(self, sensor_id, position, targets):
        self.id = sensor_id
        self.position = position
        self.targets_info = {
            target_id: TargetInfo(target, self.position) for target_id, target in targets.items()
        }

    def get_target_info(self, target_id):
        return self.targets_info.get(target_id)

    def update_target(self, tracking):
        [tar.update(self.position, tracking) for tar in self.targets_info.values()]


class Target:
    id: int
    position: np.array

    def __init__(self, tar_id, position):
        self.id = tar_id
        self.position = position

    def update_position(self, new_positions: dict):
        self.position = new_positions.get(self.id)


class TargetInfo:
    target: Target
    meas: np.array
    theta_hat: np.array
    theta_new: np.array
    nest_mem: float = 0
    error: np.array

    def __init__(self, target, agent_pos):
        self.target = target
        self.meas = rho(self.target.position, agent_pos)
        self.theta_hat = agent_pos + np.array([np.sqrt(self.meas / 2), np.sqrt(self.meas / 2)])

    def update(self, agent_pos, tracking):
        self.theta_hat = self.theta_new  # todo: maybe remove and use just theta_hat or parallel approach
        if tracking:
            self.meas = rho(self.target.position, agent_pos)
        return 1


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
        self.neighbors = {k: self.get_random_neibors(self.weight, max=friends_num) for k in range(1, num_steps + 1)}
        self.random_coef = {
            k: {
                sensor: [random(), random()] for sensor in self.N
            } for k in range(1, num_steps + 1)
        }  # todo: for multiple targets

        self.target_path = {1: {target_id: deepcopy(self.r.get(target_id)) for target_id in self.M}}

        if not tracking:
            return

        coef = self.track_coef
        for k in range(2, num_steps + 1):
            self.target_path[k] = {
                l: self.target_path[k - 1][l] + [2 * coef * random() - coef, 2 * coef * random() - coef]
                for l in self.M
            }

    def run(self, method, num_steps=20, eps=0.01, friends_num=2, accelerate=False, tracking=False, check=False):
        if not hasattr(self, "neighbors"):
            print("Init random variables")
            self.init_random_vars(num_steps, friends_num, tracking)

        self.init_plot()
        history_val = pd.DataFrame()

        targets = {target_id: Target(target_id, self.target_path[1][target_id]) for target_id in self.M}
        agents = {sensor_id: Agent(sensor_id, self.s[sensor_id], targets) for sensor_id in self.N}
        self.update_target_position(self.target_path[1], targets)

        err_history = {sensor: {} for sensor in self.N}
        errors = {target: {} for target in self.M}

        # nesterov coef
        L = self.L
        h = self.h
        H = h - pow(h, 2) * L / 2
        gamma_nest_next = self.gamma_nest
        mu = self.mu
        eta = self.eta
        alpha_nest = self.alpha_nest
        alpha_x = 0.1

        for step in range(1, num_steps + 1):  # шаги
            gamma_nest = gamma_nest_next
            gamma_nest_next = (1 - alpha_nest) * gamma_nest + alpha_nest * (mu - eta)
            if accelerate and H - pow(alpha_nest, 2) / 2 * gamma_nest_next < 0:
                print(f"h = {h}")
                print(f"H = {H}")
                print(f"Check: {H - pow(alpha_nest, 2) / 2 * gamma_nest_next} > 0?")
                return None


            for agent in agents.values():
                err_history[agent.id][step] = {}  # Error history for each step step for each target target

                for target_info in agent.targets_info.values():

                    if step == 0:
                        self.ax.plot(target_info.theta_hat[0], target_info.theta_hat[1],
                                     'o', markersize=3, color=self.colors[agent.id])
                        print(target_info.theta_hat)

                    spsa = self.spsa_step(step, agent.id, target_info.target.id, agents)

                    # nesterov acceleraion
                    # firstly go in the direction of previous theta_diff then find new one
                    # theta_diff_sum can be changed to sum so we will acc spsa step to
                    if step > 1 and accelerate:
                        v_n = target_info.nest_mem
                        x_n = 1 / (gamma_nest + alpha_nest * (mu - eta)) * (alpha_nest * gamma_nest * v_n
                                                                            + gamma_nest_next * target_info.theta_hat)
                        coef = h
                    else:
                        v_n = 0
                        x_n = target_info.theta_hat
                        coef = h

                    theta_diff = self.local_vote_step(step, agent.id, target_info.target.id, x_n, agents)
                    nesterov_step = coef * theta_diff

                    target_info.theta_new = x_n - (self.alpha * spsa + nesterov_step)

                    if step > 1 and accelerate:
                        v_n = 1 / gamma_nest * ((1 - alpha_nest) * gamma_nest * v_n
                                                + alpha_nest * (mu - eta) * x_n - alpha_nest * nesterov_step)

                    self.update_plot_step(agent.id, target_info.theta_hat, target_info.theta_new)

                    err_l_i = self.compute_error(target_info.theta_new, target_info.target.position)

                    target_info.error = err_l_i
                    target_info.nest_mem = v_n

                    err_history[agent.id][step][target_info.target.id] = err_l_i

                    history_val = history_val.append({
                        "sensor": agent.id,
                        "target": target_info.target.id,
                        "step": step,
                        "neibors": self.neighbors[step][agent.id],
                        "old": target_info.theta_hat,
                        "spsa": self.alpha * spsa,
                        "theta_diff": theta_diff,
                        "theta_diff_sum": self.gamma * theta_diff,
                        "sum": self.alpha * spsa + self.gamma * theta_diff,
                        "new": target_info.theta_new,
                        "err": err_l_i,
                        "nesterov_step": nesterov_step,
                        "v_n": v_n,
                        "alpah_nest": alpha_nest,
                        "gamma_nest": gamma_nest,
                        "agent_pos": agent.position,
                        "tar_pos": target_info.target.position,
                        "tar_meas": target_info.meas
                    }, ignore_index=True)

                self.fig.canvas.draw()

            print(
                f"Error - {sum([pow(err_history[sensor][step][1], 2) for sensor in self.N]) / self.n:.2f} on {step} step for {1} target")
            # time.sleep(1)

            errors[1][step] = sum([pow(err_history[sensor][step][1], 2) for sensor in self.N]) / self.n

            if errors[1][step] < eps or errors[1][step] > 1e+9:  # todo: for multiple targets
                break

            if tracking:
                self.update_target_position(self.target_path[step], agents)

                for target_info in targets.values():
                    target_info.update_position(self.target_path[step])

            for agent in agents.values():
                agent.update_target(tracking)

        self.errors = errors
        self.err_history = err_history

        return Result(
            check=check,
            tracking=tracking,
            accelerate=accelerate,
            method=method,
            errors=errors,
            err_history=err_history,
            history=history_val,
            agents=agents)

    def update_plot_step(self, sensor, theta_hat, theta_new):
        if sensor == 3:
            self.ax.plot([theta_hat[0], theta_new[0]],
                         [theta_hat[1], theta_new[1]],
                         markersize=2, color=self.colors[sensor - 1])
        self.ax.plot(theta_new[0], theta_new[1], 'o', markersize=3,
                     color=self.colors[sensor - 1])

    def local_vote_step(self, step, agent_id, target_id, agent_theta, agents_info):
        # get estimation from neighbors
        neighbors = self.neighbors[step].get(agent_id, [])
        neighb_theta = {neighb_id: agents_info[neighb_id].targets_info[target_id].theta_hat for neighb_id in neighbors}

        b = self.weight[agent_id - 1]

        # step of local vote protocol
        theta_diff = sum([abs(b[neib_id - 1]) * (agent_theta - theta) for neib_id, theta in neighb_theta.items()])
        return theta_diff

    def spsa_step(self, step, agent_id, target_id, agents):
        coef1 = 1 if self.random_coef[step][agent_id][0] < 0.5 else -1
        coef2 = 1 if self.random_coef[step][agent_id][1] < 0.5 else -1

        delta = np.array([coef1 * self.Delta_abs_value, coef2 * self.Delta_abs_value])

        theta_hat = agents[agent_id].targets_info[target_id].theta_hat
        x1 = theta_hat + self.beta_1 * delta
        x2 = theta_hat - self.beta_2 * delta

        y1 = self.f_l_i(target_id, agent_id, x1, self.neighbors[step], agents)
        y2 = self.f_l_i(target_id, agent_id, x2, self.neighbors[step], agents)

        spsa = (y1 - y2) / (2 * self.beta) * delta
        return spsa

    def f_l_i(self, target_id, agent_id, r_hat_l, neibors, agents_info):
        """ Calculate function f for target l and for sensor i
        :param target_id: target index
        :param agent_id: sensor index
        :param r_hat_l: x point at which calculate
        :return: matrixe D for sensor agent_id and target target_id
        """
        C = self.C_i(agent_id, neibors)
        D = self.D_l_i(target_id, agent_id, neibors, agents_info)

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

    def D_l_i(self, target_id, agent_id, neibors, agents_info):
        """ Calculate matrix D for target taret_id and for sensor agent_id
        :param target_id: target index
        :param agent_id: sensor index
        :param agents: agents info
        :return: matrixe D for agent_id sensor and target_id target
        """
        Dli = [self.calc_D_l_i_j(target_id, agent_id, neib_id, agents_info) for neib_id in neibors.get(agent_id)]
        return Dli

    def calc_D_l_i_j(self, target_id, agent_id, neib_id, agents_info: dict):
        """Calculate value of D_l_i_j
        :param target_id: target id
        :param agent_id: index of 1st sensor
        :param neib_id: index of 2nd sensor
        :param agents_info: info about agents
        :return: D_l_i[j] for vector D_l_i
        """
        meas_agent = agents_info[agent_id].targets_info[target_id].meas
        meas_neib = agents_info[neib_id].targets_info[target_id].meas
        return self.rho_overline(meas_agent, meas_neib) + self.s_norms.get(neib_id) - self.s_norms.get(agent_id)

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

    def update_target_position(self, new_positions, targets):
        self.ax.plot([targets[1].position[0], new_positions[1][0]],
                     [targets[1].position[1], new_positions[1][1]], 'bx', markersize=8)

    def compute_error(self, vector_1, vector_2):
        return np.sqrt(sum(np.power(vector_1 - vector_2, 2)))

    def get_random_neibors(self, weight, max=2):
        neibors_mat = (weight != 0).astype(int)
        np.fill_diagonal(neibors_mat, 0)

        # choose num random neibors from all neibors
        neighbors = {}
        for sensor in self.N:
            neib = [ind + 1 for ind, sens in enumerate(neibors_mat[sensor - 1]) if sens == 1]

            if len(neib) > max:
                neib = sample(neib, max)

            neighbors[sensor] = neib

        return neighbors


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
