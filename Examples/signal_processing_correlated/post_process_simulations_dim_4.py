import argparse

import matplotlib.pyplot as plt
import seaborn as sns
import sympy as sp

from utils.boolean_parser import add_boolean_argument
from utils.hdf_io import load_hdf_file
from utils.plotter import create_animation_from_density, plot_natural_statistics, plot_hellinger_distance


def process_hdf_file(file_name: str,
                     relative_path: str = './simulation_result',
                     nice_look: bool = False,
                     minimal_font_size: int = 10):
    hdf_dataset_names = ['tspan',
                         'simulation_id',
                         'moments_particle_filter',
                         'nt',
                         'empirical_den_pf_history',
                         'essl_history',
                         'rel_ent_histories',
                         'grid_limits',
                         'dt',
                         ]
    box_ratio = 1.0
    linewidth = 0.5
    linestyles = ['-.k', '-.r', '-.g', '-k']
    markers = ['1', '2', '3', '-4']
    markevery = 40
    markersize = 10
    num_stats = 69
    for i in range(num_stats):
        hdf_dataset_names.append('statistics_str/{}'.format(i))

    num_srules = 1
    for i in range(num_srules):
        hdf_dataset_names.append('integrators/{}'.format(i))
        hdf_dataset_names.append('moments_projection_filters/{}'.format(i))

    var_ = load_hdf_file(file_name, relative_path, hdf_dataset_names, hdf_dataset_names)
    moments_particle_filter = var_['moments_particle_filter']
    rel_ent_histories = var_['rel_ent_histories']
    essl_history = var_['essl_history']
    tspan = var_['tspan']
    simulation_id = var_['simulation_id'].decode("utf-8")



    integrators = []
    moments_projection_filters = []

    for i in range(num_srules):
        integrators.append(var_['integrators/{}'.format(i)].decode("utf-8"))
        moments_projection_filters.append(var_['moments_projection_filters/{}'.format(i)])


    statistics_str = []
    for i in range(num_stats):
        if var_['statistics_str/{}'.format(i)]:
            statistics_str.append(var_['statistics_str/{}'.format(i)].decode("utf-8"))

    # rename spg integrator to gpq
    integrators = ["GPQ"]

    x, dw, dv = sp.symbols(('x1:5', 'dw1:3', 'dv1:3'))

    if nice_look:
        # set some matplotlib variables
        # sns.set_context("talk")
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
        plt.savefig(relative_path + '/{}/essl.pdf'.format(simulation_id))
        plt.close()

    print("Plotting empirical KL divergencec ...")
    if len(integrators)>1:
        for i, integrator in enumerate(integrators):
            plt.plot(tspan, rel_ent_histories[i, :],
                     linestyles[i],
                     label=integrator,
                     linewidth=linewidth,
                     marker=markers[i],
                     markevery=markevery,
                     markersize=markersize
                     )
    else:
        plt.plot(tspan, rel_ent_histories[0, :],
                 '-k',
                 label=integrators[0],
                 linewidth=linewidth,
                 )
    plt.xlabel('$t$')
    plt.ylabel('empirical KL divergence')
    plt.legend()
    plt.gca().set_box_aspect(box_ratio)
    plt.tight_layout(pad=0.1)
    plt.savefig('./simulation_result/{}/rel_ent.pdf'.format(simulation_id))
    plt.close()

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
                            title_prefix='vdp_2d'
                            )



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default='Hetero_VDP_431_05_2023_16_04_05/variables.hdf', type=str, help='file_name')
    parser.add_argument('--path', default='./simulation_result', type=str, help='relative_path')
    parser.add_argument('--font', default=10, type=int, help='minimal_font_size')
    add_boolean_argument(parser, 'nicelook', default=True, messages="whether to set some fonts on matplotlib or not")
    args = parser.parse_args()

    process_hdf_file(file_name=args.file,
                     relative_path=args.path,
                     minimal_font_size=args.font,
                     nice_look=args.nicelook)
