"""

Frenet optimal trajectory generator

author: Atsushi Sakai (@Atsushi_twi)

Ref:

- [Optimal Trajectory Generation for Dynamic Street Scenarios in a Frenet Frame](https://www.researchgate.net/profile/Moritz_Werling/publication/224156269_Optimal_Trajectory_Generation_for_Dynamic_Street_Scenarios_in_a_Frenet_Frame/links/54f749df0cf210398e9277af.pdf)

- [Optimal trajectory generation for dynamic street scenarios in a Frenet Frame](https://www.youtube.com/watch?v=Cj6tAQe7UCY)

"""

import numpy as np
import matplotlib.pyplot as plt
import copy
import math
import os
import sys
import h5py
#from Frenet import cubic_spline_planner

SIM_LOOP = 500

# Parameter
MAX_SPEED = 50.0 / 3.6  # maximum speed [m/s]
MAX_ACCEL = 2.0  # maximum acceleration [m/ss]
MAX_CURVATURE = 1.0  # maximum curvature [1/m]

# Original
MAX_ROAD_WIDTH = 8.75  # maximum road width [m] # 7.0
D_ROAD_W = 3.5  # road width sampling length [m]

# Modified
MAX_ROAD_WIDTH = 10.0  # maximum road width [m] # 7.0
D_ROAD_W = 1.0  # road width sampling length [m]

DT = 0.2  # time tick [s]
MAXT = 5.0  # max prediction time [m]
MINT = 4.0  # min prediction time [m]
TARGET_SPEED = 30.0 / 3.6  # target speed [m/s]
D_T_S = 5.0 / 3.6  # target speed sampling length [m/s]
N_S_SAMPLE = 1  # sampling number of target speed
ROBOT_RADIUS = 2.0  # robot radius [m]

# cost weights
KJ = 0.1
KT = 0.1
KD = 1.0
KLAT = 1.0
KLON = 1.0

KCE_LAT = 0.01
KCE_LON = 0.01
KCA_LAT = 0.1
KCA_LON = 0.1
KCJ_LAT = 0.1
KCJ_LON = 0.1
KUNV = 1.0

KCCA_LON = 0.01
KCCA_LAT = 0.01

show_animation = True

def check_dir(dir_list):
    for d in dir_list:
        if not os.path.isdir(d):
            print('Create directory :\n' + d)
            os.makedirs(d)


class quintic_polynomial:

    def __init__(self, xs, vxs, axs, xe, vxe, axe, T):

        # calc coefficient of quintic polynomial
        self.xs = xs
        self.vxs = vxs
        self.axs = axs
        self.xe = xe
        self.vxe = vxe
        self.axe = axe

        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[T**3, T**4, T**5],
                      [3 * T ** 2, 4 * T ** 3, 5 * T ** 4],
                      [6 * T, 12 * T ** 2, 20 * T ** 3]])
        b = np.array([xe - self.a0 - self.a1 * T - self.a2 * T**2,
                      vxe - self.a1 - 2 * self.a2 * T,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]
        self.a5 = x[2]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t**2 + \
            self.a3 * t**3 + self.a4 * t**4 + self.a5 * t**5

        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + \
            3 * self.a3 * t**2 + 4 * self.a4 * t**3 + 5 * self.a5 * t**4

        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t**2 + 20 * self.a5 * t**3

        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t + 60 * self.a5 * t**2

        return xt


class quartic_polynomial:

    def __init__(self, xs, vxs, axs, vxe, axe, T):

        # calc coefficient of quintic polynomial
        self.xs = xs
        self.vxs = vxs
        self.axs = axs
        self.vxe = vxe
        self.axe = axe

        self.a0 = xs
        self.a1 = vxs
        self.a2 = axs / 2.0

        A = np.array([[3 * T ** 2, 4 * T ** 3],
                      [6 * T, 12 * T ** 2]])
        b = np.array([vxe - self.a1 - 2 * self.a2 * T,
                      axe - 2 * self.a2])
        x = np.linalg.solve(A, b)

        self.a3 = x[0]
        self.a4 = x[1]

    def calc_point(self, t):
        xt = self.a0 + self.a1 * t + self.a2 * t**2 + \
            self.a3 * t**3 + self.a4 * t**4

        return xt

    def calc_first_derivative(self, t):
        xt = self.a1 + 2 * self.a2 * t + \
            3 * self.a3 * t**2 + 4 * self.a4 * t**3

        return xt

    def calc_second_derivative(self, t):
        xt = 2 * self.a2 + 6 * self.a3 * t + 12 * self.a4 * t**2

        return xt

    def calc_third_derivative(self, t):
        xt = 6 * self.a3 + 24 * self.a4 * t

        return xt


class Frenet_path:

    def __init__(self):
        self.t = []
        self.d = []
        self.d_d = []
        self.d_dd = []
        self.d_ddd = []
        self.s = []
        self.s_d = []
        self.s_dd = []
        self.s_ddd = []
        self.cd = 0.0
        self.cv = 0.0
        self.ce = 0.0
        self.ca = 0.0
        self.cj = 0.0
        self.cf = 0.0
        self.cca = 0.0

        self.x = []
        self.y = []
        self.yaw = []
        self.ds = []
        self.c = []


def calc_frenet_paths(c_speed, c_d, c_d_d, c_d_dd, s0):

    frenet_paths = []

    # Check lane
    # for di in np.arange(-MAX_ROAD_WIDTH, MAX_ROAD_WIDTH+0.1, D_ROAD_W):
    #     print("di = ", di)

    # generate path to each offset goal
    for di in np.arange(-MAX_ROAD_WIDTH, MAX_ROAD_WIDTH+0.1, D_ROAD_W):

        # Lateral motion planning
        for Ti in np.arange(MINT, MAXT, DT):
            fp = Frenet_path()

            lat_qp = quintic_polynomial(c_d, c_d_d, c_d_dd, di, 0.0, 0.0, Ti)

            fp.t = [t for t in np.arange(0.0, Ti, DT)]
            fp.d = [lat_qp.calc_point(t) for t in fp.t]
            fp.d_d = [lat_qp.calc_first_derivative(t) for t in fp.t]
            fp.d_dd = [lat_qp.calc_second_derivative(t) for t in fp.t]
            fp.d_ddd = [lat_qp.calc_third_derivative(t) for t in fp.t]

            # Loongitudinal motion planning (Velocity keeping)
            for tv in np.arange(TARGET_SPEED - D_T_S * N_S_SAMPLE, TARGET_SPEED + D_T_S * N_S_SAMPLE, D_T_S):
                tfp = copy.deepcopy(fp)
                lon_qp = quartic_polynomial(s0, c_speed, 0.0, tv, 0.0, Ti)

                tfp.s = [lon_qp.calc_point(t) for t in fp.t]
                tfp.s_d = [lon_qp.calc_first_derivative(t) for t in fp.t]
                tfp.s_dd = [lon_qp.calc_second_derivative(t) for t in fp.t]
                tfp.s_ddd = [lon_qp.calc_third_derivative(t) for t in fp.t]

                Jp = sum(np.power(tfp.d_ddd, 2))  # square of jerk
                Js = sum(np.power(tfp.s_ddd, 2))  # square of jerk

                # square of diff from target speed
                ds = (TARGET_SPEED - tfp.s_d[-1])**2

                # Original
                tfp.cd = KJ * Jp + KT * Ti + KD * tfp.d[-1]**2
                tfp.cv = KJ * Js + KT * Ti + KD * ds

                # From paper
                # Speed Cost Functional
                power_kecepatan = True
                if power_kecepatan == True:
                    # Use power
                    kecepatan_lat = sum(np.power(tfp.d_d, 2))
                    kecepatan_lon = sum(np.power(tfp.s_d, 2))
                else:
                    # w/o power
                    kecepatan_lat = sum(tfp.d_d)
                    kecepatan_lon = sum(tfp.s_d)

                tfp.ce = KCE_LAT *  kecepatan_lat + KCE_LON * kecepatan_lon

                # Acceleration Cost Functional
                power_akselerasi = True
                if power_akselerasi == True:
                    # Use power
                    akselerasi_lat = sum(np.power(tfp.d_dd, 2))
                    akselerasi_lon = sum(np.power(tfp.s_dd, 2))
                else:
                    # w/o power
                    akselerasi_lat = sum(tfp.d_dd)
                    akselerasi_lon = sum(tfp.s_dd)

                tfp.ca = KCA_LAT *  akselerasi_lat + KCA_LON * akselerasi_lon

                # Jerk Cost Functional
                power_jerk = True
                if power_jerk == True:
                    # Use power
                    jerk_lat = sum(np.power(tfp.d_ddd, 2))
                    jerk_lon = sum(np.power(tfp.s_ddd, 2))
                else:
                    # w/o power
                    jerk_lat = sum(tfp.d_ddd)
                    jerk_lon = sum(tfp.s_ddd)

                tfp.cj = KCJ_LAT *  jerk_lat + KCJ_LON * jerk_lon
                tfp.cj = 0

                # Total Cost Functional
                # Original added by ce, ca, cj
                tfp.cf = KLAT * tfp.cd + KLON * tfp.cv + KUNV * tfp.ce + KUNV * tfp.ca + KUNV * tfp.cj

                # Original total cost functional
                #tfp.cf = KLAT * tfp.cd + KLON * tfp.cv

                # print("jerk_lat = ", jerk_lat)
                # print("jerk_lon = ", jerk_lon)
                # print("tfp.ce = ", tfp.ce)
                # print("tfp.ca = ", tfp.ca)
                # print("tfp.cj = ", tfp.cj)
                # print("tfp.cf = ", tfp.cf)
                # print("")

                frenet_paths.append(tfp)

    return frenet_paths


def calc_global_paths(fplist, csp):

    for fp in fplist:

        # calc global positions
        for i in range(len(fp.s)):
            ix, iy = csp.calc_position(fp.s[i])
            if ix is None:
                break
            iyaw = csp.calc_yaw(fp.s[i])
            di = fp.d[i]
            fx = ix + di * math.cos(iyaw + math.pi / 2.0)
            fy = iy + di * math.sin(iyaw + math.pi / 2.0)
            fp.x.append(fx)
            fp.y.append(fy)

        # calc yaw and ds
        for i in range(len(fp.x) - 1):
            dx = fp.x[i + 1] - fp.x[i]
            dy = fp.y[i + 1] - fp.y[i]
            fp.yaw.append(math.atan2(dy, dx))
            fp.ds.append(math.sqrt(dx**2 + dy**2))

        fp.yaw.append(fp.yaw[-1])
        fp.ds.append(fp.ds[-1])

        # calc curvature
        for i in range(len(fp.yaw) - 1):
            fp.c.append((fp.yaw[i + 1] - fp.yaw[i]) / fp.ds[i])

    return fplist

def calc_addition_cost(fplist):
    for i,_ in enumerate(fplist):
        total_curv = sum(fplist[i].c, 2)

        # Read speed
        centripetal_lat = sum(np.power(fplist[i].d_d, 2) * total_curv)
        centripetal_lon = sum(np.power(fplist[i].s_d, 2) * total_curv)

        fplist[i].cca = centripetal_lat * KCCA_LAT + centripetal_lon * KCCA_LON

        fplist[i].cf = fplist[i].cf + KUNV * fplist[i].cca

    return fplist

def check_collision(fp, ob):

    for i in range(len(ob[:, 0])):
        d = [((ix - ob[i, 0])**2 + (iy - ob[i, 1])**2)
             for (ix, iy) in zip(fp.x, fp.y)]

        collision = any([di <= ROBOT_RADIUS**2 for di in d])

        if collision:
            return False

    return True


def check_paths(fplist, ob):

    okind = []
    for i, _ in enumerate(fplist):
        if any([v > MAX_SPEED for v in fplist[i].s_d]):  # Max speed check
            continue
        elif any([abs(a) > MAX_ACCEL for a in fplist[i].s_dd]):  # Max accel check
            continue
        elif any([abs(c) > MAX_CURVATURE for c in fplist[i].c]):  # Max curvature check
            continue
        elif not check_collision(fplist[i], ob):
            continue

        okind.append(i)

    return [fplist[i] for i in okind]


def frenet_optimal_planning(csp, s0, c_speed, c_d, c_d_d, c_d_dd, ob):

    fplist = calc_frenet_paths(c_speed, c_d, c_d_d, c_d_dd, s0)
    fplist = calc_global_paths(fplist, csp)
    fplist = calc_addition_cost(fplist)
    fplist = check_paths(fplist, ob)

    # find minimum cost path
    mincost = float("inf")
    bestpath = None
    #print("fplist = ", len(fplist))

    for fp in fplist:
        #print("fp.cf = ", fp.cf)
        if mincost >= fp.cf:
            mincost = fp.cf
            bestpath = fp

    return bestpath


def generate_target_course(x, y):
    csp = cubic_spline_planner.Spline2D(x, y)
    s = np.arange(0, csp.s[-1], 0.1)

    rx, ry, ryaw, rk = [], [], [], []
    for i_s in s:
        ix, iy = csp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(csp.calc_yaw(i_s))
        rk.append(csp.calc_curvature(i_s))

    return rx, ry, ryaw, rk, csp

def check_new_lane(pos_ekor):
    if pos_ekor <= -8.75:
        heading = -7
    elif -8.75 <= pos_ekor < -5.25:
        heading = -7
    elif -5.25 <= pos_ekor < -1.75:
        heading = -3.5
    elif -1.75 <= pos_ekor < 1.75:
        heading = 0
    elif 1.75 <= pos_ekor < 5.25:
        heading = 3.5
    elif 5.25 <= pos_ekor < 8.75:
        heading = 7
    elif pos_ekor >= 8.75:
        heading = 7
    return heading

def main_frenet(data_dir):
    print(__file__ + " start!!")


    free_area_dir = os.path.join(data_dir, 'Free_Area/')
    free = os.listdir(free_area_dir)
    free.sort()

    path_dir = data_dir+'Frenet_Path/'
    frenet_dir = data_dir+'Frenet_Path_Vis/'
    check_dir([path_dir, frenet_dir])

    # Use real-world coordinate information
    position_path = os.path.join(data_dir,"position_all.txt")

    with open(position_path) as position:
        position_info = position.readlines()

    # Save trajectory
    save = os.path.join(data_dir, 'pseudo.txt')
    position_pseudo = open(save, 'w+')

    # Obstacles
    Obs = []

    # For drawing
    gambar_ego_x = []
    gambar_ego_y = []

    gambar_pseu_x = []
    gambar_pseu_y = []

    #################################################################################################################
    # Mencari titik tujuan
    # Menggunakan data dari RRT STAR
    position_path2 = os.path.join(data_dir,"pseudo_trajectory.txt")

    with open(position_path2) as position2:
        position_info2 = position2.readlines()

    flag1 = 0
    for iwak in range(len(position_info2)-1, 0, -1):
        # Read the movement
        info2 = position_info2[iwak].strip().split()

        if len(info2) > 3:
            if flag1 == 0:
                object_info2 = info2[3:]
                object_id = object_info2[::3]
                object_x = object_info2[1::3]
                object_y = object_info2[2::3]
                flag1 = 1
            else:
                continue
        else:
            continue

    object_xx = float(object_x[0])
    object_yy = float(object_y[0])

    print("object_xx = ", object_xx)
    print("object_yy = ", object_yy)

    # Mencari titik mulai
    # Menggunakan data dari RRT STAR
    with h5py.File(data_dir+'pseudo_fix.h5','r') as ra:
        starting_frame = ra['starting_frame'].value
        fix_pos = ra['fix_pos'].value

    starting_frame = starting_frame.lstrip('0')

    position_path2 = os.path.join(data_dir,"pseudo_trajectory.txt")

    with open(position_path2) as position2:
        position_info2 = position2.readlines()

    flag1 = 0
    for iwak in range(int(starting_frame)):
        # Read the movement
        info2 = position_info2[iwak].strip().split()

        if len(info2) > 3:
            if flag1 == 0:
                object_info2 = info2[3:]
                object_id = object_info2[::3]
                object_x = object_info2[1::3]
                object_y = object_info2[2::3]
                flag1 = 1
            else:
                continue
        else:
            continue

    object_xxx = float(object_x[0])
    object_yyy = float(object_y[0])

    print("object_xxx = ", object_xxx)
    print("object_yyy = ", object_yyy)
    print("starting_frame = ", starting_frame)
    #################################################################################################################
    print("starting_frame 1 = ", starting_frame)
    print("fix_pos 1 = ", fix_pos)


    #################################################################################################################
    free_area_dir = free_area_dir = data_dir+'Free_Area/'

    with h5py.File(data_dir+'free_area.h5','r') as ra:
        start_frame = ra['start_frame'].value
        position = ra['position'].value
        list_save = ra['list_save'].value

    print("starting_frame 2 = ", start_frame)
    print("fix_pos 2 = ", position)

    # Read free area information
    with h5py.File(free_area_dir+start_frame+'.h5','r') as fa:
        frame_id = fa['frame_id'].value
        range_const = fa['range_const'].value
        pos_x = fa['pos_x'].value
        pos_y = fa['pos_y'].value

    selected_pos_x = pos_x[position]
    selected_pos_y = pos_y[position]

    print("selected_pos_x = ", selected_pos_x)
    print("selected_pos_y = ", selected_pos_y)

    pos_x = ((8.75 + 8.75) / (175 - 0))*(selected_pos_x - 175) + 8.75
    pos_y = ((60 - 0) / (600 - 0))*(selected_pos_y - 600) + 60

    print("pos_x = ", pos_x)
    print("pos_y = ", pos_y)

    starting_frame = start_frame.lstrip('0')

    #################################################################################################################
    # Starting point using free area
    wx = [pos_x[0], 0]
    wy = [pos_y[0], pos_y[0] + 60]

    # Starting point using RRT
    # wx = [object_xxx, 0]
    # wy = [object_yyy, object_yy + 60]

    # Experiment 1
    # wx = [3.5, 3.5]
    # wy = [object_yyy-20, object_yy + 45]

    # Experiment 2
    # wx = [object_xx, 3.5]
    # wy = [object_yyy-20, object_yy + 45]

    c_speed = 10 / 3.6
    c_d = 0.0  # current lateral position [m]
    c_d_d = 0.0  # current lateral speed [m/s]
    c_d_dd = 0.0  # current latral acceleration [m/s]
    s0 = 0.0  # current course position

    area = 30.0  # animation area length [m]

    for frame_ids in range(int(starting_frame), len(free)-2):
        num = str(frame_ids).zfill(4)
        print("frame = ", num)

        print("num = ", num)

        with h5py.File(free_area_dir+num+'.h5','r') as fa:
            frame_id = fa['frame_id'].value
            range_const = fa['range_const'].value
            pos_x = fa['pos_x'].value
            pos_y = fa['pos_y'].value
            check_pos = fa['check_pos'].value

        # Read the ego movement
        info = position_info[frame_ids].strip().split()

        nomor_frame = info[0]
        ego_x = float(info[1])
        ego_y = float(info[2])
        gambar_ego_x.append(ego_x)
        gambar_ego_y.append(ego_y)

        object_info = info[3:]

        # Isi dari tiap objek
        object_id = object_info[::3]
        object_x = object_info[1::3]
        object_y = object_info[2::3]

        # Save the ego trajectory
        tulis_1 = info[0] + " " + info[1] + " " +info[2] + " "
        position_pseudo.write(tulis_1)

        Obs = []

        for i in range(len(object_id)):
            object_position = [float(object_x[i]), float(object_y[i])]
            Obs.append(object_position)

        print(Obs)

        if len(Obs) == 0:
            object_position = [float(0.0), float(0.0)]
            Obs.append(object_position)

        ob = np.array(Obs)

        #print("wx = ", wx)
        #print("wy = ", wy)

        tx, ty, tyaw, tc, csp = generate_target_course(wx, wy)

        if frame_ids == 0:
            # initial state
            c_speed = 10 / 3.6  # current speed [m/s]

        c_d = 0.0  # current lateral position [m]
        c_d_d = 0.0  # current lateral speed [m/s]
        c_d_dd = 0.0  # current latral acceleration [m/s]
        s0 = 0.0  # current course position

        area = 30.0  # animation area length [m]


        for i in range(1):
            path = frenet_optimal_planning(csp, s0, c_speed, c_d, c_d_d, c_d_dd, ob)

            #print("x = ", path.x[1], "    y = ", path.y[1])

            if hasattr(path, 'x') == False:
                continue

            s0 = path.s[1]
            c_d = path.d[1]
            c_d_d = path.d_d[1]
            c_d_dd = path.d_dd[1]
            c_speed = path.s_d[1]

            if np.hypot(path.x[1] - tx[-1], path.y[1] - ty[-1]) <= 1.0:
                print("Goal")
                break

            if show_animation:  # pragma: no cover
                plt.cla()
                plt.plot(tx, ty)
                plt.plot(ob[:, 0], ob[:, 1], "xk")
                plt.plot(path.x[1:], path.y[1:], "-or")
                plt.plot(path.x[1], path.y[1], "vc")

                # Ego
                plt.plot(ego_x, ego_y, "vb")

                #print("merah = ", len(path.x[1:]))

                # Original X
                # plt.xlim(path.x[1] - area, path.x[1] + area)

                # Modified X
                plt.xlim(-8.76, 8.76)

                # Add ticks
                major_ticks = np.arange(-10.5, 10.5, 3.5)
                plt.xticks(major_ticks)

                # Draw lane
                major_ticks2 = np.arange(-8.75, 8.76, 3.5)
                for q in range(len(major_ticks)):
                    x1, x2 = [major_ticks2[q], major_ticks2[q]], [path.y[1] - area, path.y[1] + area]
                    plt.plot(x1, x2, 'k')

                plt.ylim(path.y[1] - area, path.y[1] + area)
                plt.title("v[km/h]:" + str(c_speed * 3.6)[0:4])
                plt.grid(False)
                plt.savefig(os.path.join(data_dir, frenet_dir+num+".png"))
                plt.pause(0.0001)

        if hasattr(path, 'x') == False:
            continue

        # Check heading toward the target point
        # if check_way > 3.5 --> harus segera pindah lane ke kiri
        # if check_way < -3.5 --> harus segera pindah lane ke kanan
        #print("path.x[1] = ", path.x[1])
        #print("wx[1] = ", wx[1])
        check_way = path.x[1] - wx[1]
        #print("check_way = ", check_way)

        if 3.5 <= check_way < 4.5:
            adder_way = 0.1
            cond_way = -1
            #print("agak kiri")
        elif 4.5 <= check_way < 5.5:
            adder_way = 0.2
            cond_way = -1
            #print("kiri")
        elif check_way >= 5.5:
            adder_way = 0.3
            cond_way = -1
            #print("kiri banget")
        elif -4.5 < check_way <= -3.5:
            adder_way = 0.1
            cond_way = 1
            #print("agak kanan")
        elif -5.5 < check_way <= -4.5:
            adder_way = 0.2
            cond_way = 1
            #print("kanan")
        elif check_way <= -5.5:
            adder_way = 0.3
            cond_way = 1
            #print("agak kanan")
        else:
            adder_way = 0
            cond_way = 0  
            #print("lurus")     

        check_change_lane = path.x[1] - path.x[-1]

        # Check the lane condition, if in fornt of there are car it needs to turn left or right
        # -1 --> turn left
        # 0 --> straight
        # 1 --> turn right
        if check_change_lane > 0:
            cond = -1
            adder_lane = 0.05
            #print("left")
        elif check_change_lane == 0:
            cond = 0
            #print("straight")
        else:
            cond = 1
            adder_lane = 0.05
            #print("right")

        # Modified
        if cond == -1 and cond_way == -1:               # kiri - kiri
            wx[0] = path.x[1] - adder_lane - adder_way
            wy[0] = path.y[1]
            #print("1")
        elif cond == 1 and cond_way == 1:               # kanan - kanan
            wx[0] = path.x[1] + adder_lane + adder_way
            wy[0] = path.y[1]
            #print("2")
        elif cond == -1 and cond_way == 1:              # lane --> kiri , way --> kanan
            wx[0] = path.x[1] - adder_lane  
            wy[0] = path.y[1]
            #print("3")
        elif cond == 1 and cond_way == -1:              # lane --> kanan , way --> kiri
            wx[0] = path.x[1] + adder_lane
            wy[0] = path.y[1]
            #print("4")
        elif cond == 0 and cond_way == 0:               # lurus - lurus
            wx[0] = path.x[1]
            wy[0] = path.y[1]
            #print("5")
        elif cond == 0 and cond_way == -1:              # lurus - kiri
            wx[0] = path.x[1] - adder_way
            wy[0] = path.y[1]
            #print("6")
        elif cond == 0 and cond_way == 1:               # lurus - kanan
            wx[0] = path.x[1] + adder_way
            wy[0] = path.y[1]
            #print("7")
        elif cond == -1 and cond_way == 0:              # kiri - lurus
            wx[0] = path.x[1] - adder_lane
            wy[0] = path.y[1]
            #print("8")
        elif cond == 1 and cond_way == 0:               # kanan - lurus
            wx[0] = path.x[1] + adder_lane
            wy[0] = path.y[1]
            #print("9")

        # Change waypoint
        # Biar ga usa ganti lane
        if frame_ids == 23:
            wx[1] = wx[0]

        if 0 < frame_ids < 24:
            c_speed = 10 / 3.6

        if frame_ids >= 24:
            heading = check_new_lane(path.x[-1])
            print("heading = ", heading)
            wx[1] = heading

            if abs(path.x[1] - path.x[-1]) > 3.5:
                c_speed = 6 / 3.6
            else:
                c_speed = 12 / 3.6

        
        gambar_pseu_x.append(wx[0])
        gambar_pseu_y.append(wy[0])

        tulis_2 = '1' + " " + str(wx[0]) + " " + str(wy[0]) + " "
        position_pseudo.write(tulis_2)

        print("")

        position_pseudo.write("\n")

    position_pseudo.close()

    plt.figure(figsize = (12.8,12.8))
    plt.title("Pseudo Car Trajectory Path", fontsize = 30)
    plt.xlabel("X(m)",fontsize = 20)
    plt.ylabel("Y(m)",fontsize = 20)
    plt.xlim(-100,100)
    plt.plot(gambar_ego_x,gambar_ego_y,label = "Ego-motion",color='green',linewidth = 3)
    plt.plot(gambar_pseu_x,gambar_pseu_y,label = "Pseudo-motion",color='red',linewidth = 3)
    plt.legend(loc = "lower left")
    plt.savefig(os.path.join(data_dir, "pseudo_trajectory.png"))
    

if __name__ == '__main__':
    import cubic_spline_planner
    data_dir = "/media/ferdyan/LocalDiskE/Hasil/dataset/New/X_ooc14/"
    main_frenet(data_dir)
