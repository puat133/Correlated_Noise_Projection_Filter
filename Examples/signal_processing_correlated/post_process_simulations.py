import argparse
import numpy as onp
import matplotlib.pyplot as plt
import seaborn as sns
import sympy as sp
from functools import partial
from utils.boolean_parser import add_boolean_argument
from utils.hdf_io import load_hdf_file
from utils.plotter import create_animation_from_density, plot_natural_statistics, plot_hellinger_distance


def process_hdf_file(file_name: str,
                     relative_path: str = './simulation_result',
                     nice_look: bool = False,
                     minimal_font_size: int = 10,
                     include_static: bool = False):
    hdf_dataset_names = ['tspan',
                         'simulation_id',
                         'moments_particle_filter',
                         'em_pf/_level',
                         'nt',
                         'empirical_den_pf_history',
                         'essl_history',
                         'rel_ent_histories',
                         'grid_limits',
                         'dt',
                         'X_integrated',
                         'projection_filter_spg_level'
                         ]
    box_ratio = 1.0
    linewidth = 0.5
    linestyles = ['-r', '-b', '-g', '-k']
    markers = ['1', '2', '3', '-4']
    markevery = 40
    markersize = 10
    num_stats = 14
    for i in range(num_stats):
        hdf_dataset_names.append('statistics_str/{}'.format(i))

    num_srules = 3

    for i in range(num_srules):
        hdf_dataset_names.append('srules/{}'.format(i))
        hdf_dataset_names.append('integrators/{}'.format(i))
        # hdf_dataset_names.append('bijection_types/{}'.format(i))
        hdf_dataset_names.append('moments_projection_filters/{}'.format(i))
        hdf_dataset_names.append('density_histories/{}'.format(i))
        hdf_dataset_names.append('bijected_points_histories/{}'.format(i))
        hdf_dataset_names.append('hell_dist_hists/{}'.format(i))

    var_ = load_hdf_file(file_name, relative_path, hdf_dataset_names, hdf_dataset_names)
    moments_particle_filter = var_['moments_particle_filter']
    rel_ent_histories = var_['rel_ent_histories']
    essl_history = var_['essl_history']
    tspan = var_['tspan']
    nt = var_['nt']
    grid_limits = var_['grid_limits']
    dt = var_['dt']
    simulation_id = var_['simulation_id'].decode("utf-8")
    slevel = var_['projection_filter_spg_level']
    animeframe = 200
    sampled_state = var_['X_integrated'][:, :2]

    # this is hardcoded
    @partial(onp.vectorize, signature='(2)->(m)')
    def extended_statistics(x_: onp.ndarray):
        return onp.array([x_[1],
                          x_[1] ** 2,
                          x_[1] ** 3,
                          x_[1] ** 4,
                          x_[0],
                          x_[0] * x_[1],
                          x_[0] * x_[1] ** 2,
                          x_[0] * x_[1] ** 3,
                          x_[0] ** 2,
                          x_[0] ** 2 * x_[1],
                          x_[0] ** 2 * x_[1] ** 2,
                          x_[0] ** 3,
                          x_[0] ** 3 * x_[1],
                          x_[0] ** 4

                          ])

    sampled_polynomial_values = extended_statistics(sampled_state)

    srules = []
    integrators = []
    moments_projection_filters = []
    density_histories = []
    bijected_points_histories = []
    hell_dist_hists = []
    for i in range(num_srules):
        if not var_['srules/{}'.format(i)]:
            break
        srules.append(var_['srules/{}'.format(i)].decode("utf-8"))
        integrators.append(var_['integrators/{}'.format(i)].decode("utf-8"))
        moments_projection_filters.append(var_['moments_projection_filters/{}'.format(i)])
        density_histories.append(var_['density_histories/{}'.format(i)])
        bijected_points_histories.append(var_['bijected_points_histories/{}'.format(i)])
        hell_dist_hists.append(var_['hell_dist_hists/{}'.format(i)])

    empirical_den_pf_history = var_['empirical_den_pf_history']
    statistics_str = []
    for i in range(num_stats):
        if var_['statistics_str/{}'.format(i)]:
            stat_str = var_['statistics_str/{}'.format(i)].decode("utf-8")
            statistics_str.append(stat_str)

    # rename spg integrator to gpq

    integrators = ["GPQ", "GHQ", "QMC"]

    x, dw, dv = sp.symbols(('x1:3', 'dw1:3', 'dv1:3'))
    print('Sparse-Grid-level = {}, GPQ nodes = {}, GHQ nodes = {}'.format(slevel,
                                                                          bijected_points_histories[0].shape[1]
                                                                          , bijected_points_histories[1].shape[1]))
    if nice_look:
        # set some matplotlib variables
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "sans-serif",
            "font.sans-serif": "Helvetica",
            "font.size": minimal_font_size,
            'text.latex.preamble': r'\usepackage{amsfonts}'
        })

    if essl_history is not None:
        print("Plotting ESSL of weights ...")
        plt.plot(tspan, essl_history, linewidth=0.5)
        plt.xlabel('t')
        plt.ylabel('ESSL')
        plt.tight_layout()
        plt.savefig('./simulation_result/{}/essl.pdf'.format(simulation_id))
        plt.close()

    print("Plotting empirical KL divergence ...")
    for i, integrator in enumerate(integrators):
        plt.plot(tspan, rel_ent_histories[i, :],
                 linestyles[i],
                 label=integrator,
                 linewidth=linewidth,
                 marker=markers[i],
                 markevery=markevery,
                 markersize=markersize
                 )
    plt.xlabel('$t$')
    plt.ylabel('empirical KL divergence')
    plt.grid(False)
    plt.legend()
    plt.gca().set_box_aspect(box_ratio)
    plt.tight_layout(pad=0.1)
    plt.savefig('./simulation_result/{}/rel_ent.pdf'.format(simulation_id))
    plt.close()

    print("Plotting Hellinger Distance ...")

    # if "SIR" not in simulation_id:
    #     plot_hellinger_distance(hell_dist_hists=hell_dist_hists,
    #                             tspan=tspan,
    #                             integrators=integrators,
    #                             simulation_id=simulation_id,
    #                             box_ratio=0.333
    #                             )
    # else:
    #     plot_hellinger_distance(hell_dist_hists=[hell_dist_hists[0], ],
    #                             tspan=tspan,
    #                             integrators=[integrators[0], ],
    #                             simulation_id=simulation_id,
    #                             box_ratio=0.333
    #                             )
    if not include_static:
        if "SIR" in simulation_id:
            hell_dist_hists = [hell_dist_hists[0], ]
    else:
        if "SIR" in simulation_id:
            integrators = ["GPQ", "Static"]
            hell_dist_hists = [hell_dist_hists[0], hell_dist_hists[2]]

    plot_hellinger_distance(hell_dist_hists=hell_dist_hists,
                            tspan=tspan,
                            integrators=integrators,
                            simulation_id=simulation_id,
                            box_ratio=box_ratio,
                            linestyles=linestyles,
                            title_prefix='vdp_1d',
                            markevery=markevery,
                            markersize=markersize,
                            make_it_squared=True
                            )

    row_col = (2, 2)

    print("Plotting Moments ...")
    plot_natural_statistics(moments_projection_filters=moments_projection_filters,
                            moments_particle_filter=moments_particle_filter,
                            statistics_str=statistics_str,
                            tspan=tspan,
                            simulation_id=simulation_id,
                            integrators=integrators,
                            linestyles=linestyles,
                            markevery=markevery,
                            markersize=markersize,
                            title_prefix='vdp_1d'
                            )

    print("Generating Density Animation ...")
    sns.set_context("paper")
    create_animation_from_density(bijected_points_histories=bijected_points_histories,
                                  integrators=integrators,
                                  density_histories=density_histories,
                                  empirical_den_pf_history=empirical_den_pf_history,
                                  grid_limits=grid_limits,
                                  skip_frame=nt // animeframe,
                                  dt=dt,
                                  show_nodes=True,
                                  simulation_id=simulation_id,
                                  row_col=row_col,
                                  figsize=(6, 6))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default='HeteroSe_VDP28_09_2023_12_07_48/variables.hdf', type=str, help='file_name')
    parser.add_argument('--path', default='./simulation_result', type=str, help='relative_path')
    parser.add_argument('--font', default=10, type=int, help='minimal_font_size')
    add_boolean_argument(parser, 'nicelook', default=True, messages="whether to set some fonts on matplotlib or not")
    add_boolean_argument(parser, 'static', default=False,
                         messages="whether to include the static bijection result or not")
    args = parser.parse_args()

    process_hdf_file(file_name=args.file,
                     relative_path=args.path,
                     minimal_font_size=args.font,
                     nice_look=args.nicelook,
                     include_static=args.static)
