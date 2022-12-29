
import numpy as np
np.random.seed(20211121)
import ApproxPH
import matplotlib.pyplot as plt
from gudhi.wasserstein.barycenter import lagrangian_barycenter as bary

def compute_mean(original_set, nb_subs, nb_sub_points, max_edge_length, min_persistence, scenario):
    subs = ApproxPH.get_subsample(original_set, nb_sub_points, nb_subs)
    diags = []
    for points in subs:
        diag = ApproxPH.get_PD(points, max_edge_length=max_edge_length, min_persistence=min_persistence)
        diag[np.isinf(diag)] = max_edge_length
        diags.append(diag)

    if scenario == 'mpm':
        sub_pers = np.array([[0,0]])
        for diag in diags:
            sub_pers = np.append(sub_pers, diag, axis=0)
        unit_mass = 1/nb_subs
        mean_mesr, mean_mesr_vis = ApproxPH.diag_to_mesr(sub_pers, unit_mass)
        return mean_mesr, mean_mesr_vis

    if scenario == 'fm':
        wmean, log = bary(diags, init=0, verbose=True)
        return wmean

    if scenario == 'both':
        wmean, log = bary(diags, init=0, verbose=True)
        sub_pers = np.array([[0,0]])
        for diag in diags:
            sub_pers = np.append(sub_pers, diag, axis=0)
        unit_mass = 1/nb_subs
        mean_mesr, mean_mesr_vis = ApproxPH.diag_to_mesr(sub_pers, unit_mass)
        return mean_mesr, mean_mesr_vis, wmean
X_torus = ApproxPH.sample_torus(50000, 0.8, 0.3)
'''
X_torus = ApproxPH.sample_torus(50000, 0.8, 0.3)
np.save('outputs/true-torus-points.npy', X_torus)
diag_torus = ApproxPH.get_PD(X_torus, max_edge_length=0.9,min_persistence=0)
np.save('outputs/true-torus-diagram.npy', diag_torus)

ApproxPH.plot_diag(diag_torus)

nb_simulates = 3
for i in range(nb_simulates):
    mean_mesr, mean_mesr_vis = compute_mean(original_set = X_torus,
                                            nb_subs = 20*(i+2),
                                            nb_sub_points = 200*(i+2),
                                            max_edge_length = 0.9,
                                            min_persistence = 0.01,
                                            scenario = 'mpm'
                                           )
    np.save('outputs/mean_mesr_nb%d.npy' %(i), mean_mesr)
    print('mean persistence measure for %dth simulation' %(i))
mesr_list = []
for i in range(15):
    mesr = np.load('outputs/mean_mesr_nb%d.npy' %(i))
    mesr_list.append(mesr)

true_PD = np.load('outputs/true-torus-diagram.npy')

true_mesr, true_mesr_vis = ApproxPH.diag_to_mesr(true_PD, 1)

power_index = 3
grid = ApproxPH.mesh_gen()
Mp = ApproxPH.dist_mat(grid, power_index)
dist_list = []
point_list = []
for i in range(len(mesr_list)):
    distance = ApproxPH.wass_dist(mesr_list[i], true_mesr, Mp)
    point_list.append(200*(i+2))
    dist_list.append(distance.tolist())

'''
nb_points = 5000
true_set = ApproxPH.sample_annulus(nb_points, r1=0.2, r2=0.5)
true_PD = ApproxPH.get_PD(true_set, max_edge_length=0.4, min_persistence=0.01)
true_mesr, true_mesr_vis = ApproxPH.diag_to_mesr(true_PD, 1)
ApproxPH.plot_mesr(true_mesr_vis)

nb_subs = 20
unit_mass  = 1/nb_subs
nb_sub_points_list = [400]
power_index = 2
w_list = []
permesr_list = []

for nb_sub_points in nb_sub_points_list:
    print('number of points in each subset: %d' %(nb_sub_points))
    mean_mesr, mean_mesr_vis, wmean = compute_mean(original_set = true_set,
                                            nb_subs = nb_subs,
                                            nb_sub_points = nb_sub_points,
                                            max_edge_length = 0.4,
                                            min_persistence = 0.01,
                                            scenario = 'both'
                                           )
    wmean_mesr, wmean_mesr_vis = ApproxPH.diag_to_mesr(wmean, 1)
    print(len(wmean_mesr))
    ApproxPH.plot_mesr(wmean_mesr_vis)
    ApproxPH.plot_diag(wmean)
'''
fig = plt.figure(figsize=(8,8))
plt.plot(nb_sub_points_list, permesr_list, linestyle='-', color='blue',\
         linewidth=2, label='Mean Persistence Measure')
plt.scatter(nb_sub_points_list, permesr_list, s=70, color='red', marker='o')
plt.plot(nb_sub_points_list, w_list, linestyle='--', color='green',\
         linewidth=2, label='Frechet Mean')
plt.scatter(nb_sub_points_list, w_list, s=70, color='black', marker='P')
plt.xlabel('Number of Points')
plt.ylabel('2-Wasserstein distance')
plt.title('Comparison of Frechet mean\n and mean persistence measure')
plt.legend()
plt.show()
'''
