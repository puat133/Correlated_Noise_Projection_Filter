from functools import partial
from typing import List

import jax.numpy as jnp
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as onp
from jax import jit
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from tqdm import tqdm
from matplotlib.ticker import FuncFormatter

TEXT_FONT_SIZE = 10


def format_scientific_to_latex(number):
    # Check if the number is in scientific notation
    if "e" in "{:.1e}".format(number):
        mantissa, exponent = "{:.1e}".format(number).split("e")
        mantissa = float(mantissa)
        exponent = int(exponent)
        if mantissa == int(mantissa):
            mantissa = int(mantissa)
        return f"${mantissa} \\times 10^{{{exponent}}}$"
    else:
        return f"${number}$"

def set_colorbar(ax, im, fg_color):
    cbaxes = inset_axes(ax, width="40%", height="5%", loc='upper left', borderpad=1.5)
    fig = ax.get_figure()
    cb = fig.colorbar(im, cax=cbaxes, orientation='horizontal')
    cb.outline.set_edgecolor(fg_color)
    cb.ax.xaxis.set_tick_params(color=fg_color)
    plt.setp(plt.getp(cb.ax.axes, 'xticklabels'), color=fg_color)
    cb.ax.axes.get_xticklabels()
    plt.setp(cb.ax.axes.get_xticklabels(), fontsize=TEXT_FONT_SIZE)


@jit
def extent_distance(extent: jnp.ndarray):
    return extent[0, 1] - extent[0, 0], extent[1, 1] - extent[1, 0]


@jit
def extent_middle(extent: jnp.ndarray):
    return extent.mean(axis=-1)


def get_ratio(extent: jnp.ndarray):
    x_dist, y_dist = extent_distance(extent)
    return x_dist / y_dist


@partial(jnp.vectorize, signature='(n,n)->(n,n)', excluded=(1,))
def equalize_extent(extent: jnp.ndarray, scale: float):
    x_dist, y_dist = extent_distance(extent)
    middle = extent_middle(extent)
    max_dist = jnp.maximum(x_dist, y_dist)
    new_extent = jnp.vstack((middle - scale * max_dist, middle + scale * max_dist)).T
    return new_extent


def text_position(extent: jnp.ndarray, fractions: jnp.ndarray):
    dists = extent_distance(extent)
    a_position = (extent[0, 0] + fractions[0] * dists[0], extent[1, 0] + fractions[1] * dists[1])
    return a_position


def create_animation_from_density(bijected_points_histories: List[jnp.ndarray],
                                  integrators: List[str],
                                  density_histories: List[jnp.ndarray],
                                  empirical_den_pf_history: jnp.ndarray,
                                  grid_limits: jnp.ndarray,
                                  skip_frame: int,
                                  dt: float,
                                  simulation_id: str,
                                  aspect: str = "equal",
                                  show_nodes: bool = False,
                                  row_col: tuple = None,
                                  figsize: tuple = (6, 6)
                                  ):
    t_index = 0
    fg_color = 'black'

    if not row_col:
        total_plots = 1 + len(integrators)
        sqrt_total_ceiled = int(onp.ceil(onp.sqrt(total_plots)))
        col = sqrt_total_ceiled
        row = sqrt_total_ceiled
    else:
        row, col = row_col
    fig, ax = plt.subplots(row, col,
                           dpi=350,
                           sharex='all',
                           sharey='all',
                           gridspec_kw=dict(wspace=0.15,
                                            hspace=0.15,
                                            ),
                           figsize=figsize
                           )
    ax = ax.flatten()
    for i, ax_ in enumerate(ax):
        ax_.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        ax_.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        if i % col == 0:
            ax_.set_ylabel(r'$x_2$')
        if i // col == row - 1:
            ax_.set_xlabel(r'$x_1$')

    density_par_f = empirical_den_pf_history[t_index]
    time_text_position_frac = jnp.array([0.1, 0.1])
    title_text_position_frac = jnp.array([0.7, 0.9])
    if grid_limits.ndim == 2:
        extent = grid_limits
    else:
        extent = grid_limits[0]

    im_par_f = ax[0].imshow(density_par_f, cmap=plt.cm.Blues, extent=extent.flatten(),
                            origin='lower', alpha=1)
    ax[0].set_xlim(extent[0])
    ax[0].set_ylim(extent[1])
    ax[0].set_box_aspect(1)

    time_text_par_f = ax[0].text(*text_position(extent,
                                                time_text_position_frac),
                                 r'$t={:.3f}$'.format(t_index),
                                 fontsize=TEXT_FONT_SIZE, color=fg_color)
    ax[0].text(*text_position(extent, title_text_position_frac), "Particle",
               fontsize=TEXT_FONT_SIZE, color=fg_color)
    ax[0].set_aspect(aspect)
    # ax[0].set_title('Particle Filter')
    vmax_par_f = jnp.max(density_par_f)
    vmin = 0.
    im_par_f.set_clim(vmin, vmax_par_f)
    set_colorbar(ax[0], im_par_f, fg_color)
    ims = []
    sc_nodes = []
    time_texts = []
    for i in range(len(integrators)):
        density = density_histories[i][t_index]
        bijected_points = bijected_points_histories[i][t_index]
        vmax = jnp.max(density)
        time_text = ax[i + 1].text(*text_position(extent,
                                                  time_text_position_frac), r'$t={:.3f}$'.format(t_index),
                                   fontsize=TEXT_FONT_SIZE, color=fg_color)
        time_texts.append(time_text)
        im = ax[i + 1].imshow(density, cmap=plt.cm.Blues, extent=extent.flatten(),
                              origin='lower', alpha=1)
        ims.append(im)
        # ax[i + 1].set_xlim(extent[0])
        # ax[i + 1].set_ylim(extent[1])
        ax[i + 1].set_aspect(aspect)
        ax[i + 1].set_box_aspect(1)
        # ax[i + 1].set_title('Projection \n {}'.format(srules[i]))
        if show_nodes:
            sc_node = ax[i + 1].scatter(bijected_points[:, 0], bijected_points[:, 1],
                                        s=1.0, marker='2', color='black',
                                        alpha=0.1 / onp.sqrt(bijected_points.shape[0] / 500))
            sc_nodes.append(sc_node)
        im.set_clim(vmin, vmax)
        set_colorbar(ax[i + 1], im, fg_color)
        ax[i + 1].text(*text_position(extent, title_text_position_frac), integrators[i],
                       fontsize=TEXT_FONT_SIZE, color=fg_color)

    def animate(t_):
        # the particle filter part

        den_par_f = empirical_den_pf_history[t_]
        im_par_f.set_data(den_par_f)
        vmax_par_f_ = jnp.max(den_par_f)
        im_par_f.set_clim(vmin, vmax_par_f_)

        if grid_limits.ndim == 2:
            extent_ = grid_limits
        else:
            extent_ = grid_limits[t_]

        time_text_par_f.set_text(r'$t={:.3f}$'.format(t_ * dt))
        time_text_par_f.set_position(text_position(extent_,
                                                   time_text_position_frac))
        im_par_f.set_extent(extent_.flatten())
        ax[0].set_xlim(extent_[0])
        ax[0].set_ylim(extent_[1])

        for i_ in range(len(integrators)):
            im_ = ims[i_]
            ax_ = ax[i_ + 1]
            time_text_ = time_texts[i_]
            sc_node_ = sc_nodes[i_]
            bijected_p = bijected_points_histories[i_][t_]
            den_ = density_histories[i_][t_]
            im_.set_data(den_)
            vmax_ = jnp.max(den_)
            im_.set_clim(vmin, vmax_)
            if show_nodes:
                sc_node_.set_offsets(bijected_p)
            time_text_.set_text(r'$t={:.3f}$'.format(t_ * dt))
            time_text_.set_position(text_position(extent_,
                                                  time_text_position_frac))
            im_.set_extent(extent_.flatten())

    anim = animation.FuncAnimation(fig, animate, frames=tqdm(range(0, empirical_den_pf_history.shape[0], skip_frame)))
    anim.save('./simulation_result/{}/comparison_particle_vs_projection.mp4'.format(simulation_id))


def plot_hellinger_distance(hell_dist_hists: List[jnp.ndarray],
                            tspan: jnp.ndarray,
                            integrators: List[str],
                            simulation_id: str,
                            pde_included: bool = False,
                            box_ratio: float = 0,
                            linewidth=0.5,
                            linestyles=['--k', '-.k', '-k'],
                            markers=['1', '2', '3'],
                            markevery=40,
                            markersize=5,
                            title_prefix='',
                            make_it_squared=False  # use this if the stored hellinger distance is actually the square of it
                            ):
    ground_truth = 'Particle'
    shift = 0

    if pde_included:
        ground_truth = 'Crank-Nicholson'
        shift = 1
        hell_dist_hist = hell_dist_hists[0]
        if make_it_squared:
            hell_dist_hist = jnp.sqrt(hell_dist_hists[0])
        plt.semilogy(tspan, hell_dist_hist, linestyles[-1],
                     label='{} vs Proj-{}'.format(ground_truth,
                                                  'Particle'),
                     linewidth=linewidth,
                     markers=markers[0],
                     markevery=markevery,
                     markersize=markersize)

    for i in range(shift, len(hell_dist_hists)):
        hell_dist_hist = hell_dist_hists[i]
        if make_it_squared:
            hell_dist_hist = jnp.sqrt(hell_dist_hists[i])
        plt.semilogy(tspan, hell_dist_hist,
                     linestyles[i],
                     label='{} vs Proj-{}'.format(ground_truth,
                                                  integrators[i - shift]),

                     linewidth=linewidth, marker=markers[i], markevery=markevery, markersize=markersize)

    ax = plt.gca()  # Get the current axis

    # # Define a formatting function
    # def minor_tick_formatter(x, pos):
    #     if x in [5e-4, 1e-3, 5e-3]:  # Specify the ticks you want to label
    #         return format_scientific_to_latex(x)
    #     else:
    #         return ""
    #
    # # Apply the formatter
    # ax.yaxis.set_minor_formatter(FuncFormatter(minor_tick_formatter))
    plt.legend()
    plt.xlabel(r'$t$')
    plt.ylabel('Hellinger distance')

    if box_ratio > 0:
        plt.gca().set_box_aspect(box_ratio)

    plt.tight_layout(pad=0.1)
    plt.grid(False)
    plt.savefig('./simulation_result/{}/{}_Hellinger-Distance.pdf'.format(simulation_id, title_prefix))
    plt.close()


def plot_natural_statistics(moments_projection_filters: List[jnp.ndarray],
                            moments_particle_filter: jnp.ndarray,
                            statistics_str,
                            tspan: jnp.ndarray,
                            simulation_id: str,
                            integrators: List[str],
                            relative_path: str = './simulation_result',
                            sampled_polynomial_values: jnp.ndarray = None,
                            linestyles=['--k', '-.k', '-k'],
                            linewidth=0.5,
                            markers=['1', '2', '3'],
                            markevery=40,
                            markersize=5,
                            title_prefix='',
                            ):
    for natural_statistics_index in tqdm(range(len(statistics_str))):
        plt.figure(dpi=150)
        natural_statistic_str = statistics_str[natural_statistics_index]
        for i in range(len(moments_projection_filters)):
            plt.plot(tspan, moments_projection_filters[i][:, natural_statistics_index], linestyles[i],
                     label='proj-{}'.format(integrators[i]), linewidth=linewidth, marker=markers[i],
                     markevery=markevery,
                     markersize=markersize)
        plt.plot(tspan, moments_particle_filter[:, natural_statistics_index], linestyles[-1], color='black',
                 label='particle', linewidth=linewidth,
                 markevery=markevery,
                 markersize=markersize)
        if sampled_polynomial_values is not None:
            plt.plot(tspan, sampled_polynomial_values[:, natural_statistics_index], linestyles[2], color='black',
                     linestyle='dashed',
                     label='realization', linewidth=linewidth, marker=markers[1],
                     markevery=markevery,
                     markersize=markersize)

        plt.legend()
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        plt.ylabel(r'$\mathbb E [{}]$'.format(natural_statistic_str), rotation=90)
        plt.xlabel(r'$t$')
        plt.tight_layout(pad=0.1)
        plt.savefig(
            relative_path + '/{}/{}_Moments_{}.pdf'.format(simulation_id,
                                                           title_prefix,
                                                           natural_statistic_str.replace('^', 'p')))
        plt.close()
        plt.figure(dpi=150)
        for i in range(len(moments_projection_filters)):
            plt.semilogy(tspan, jnp.abs(
                moments_particle_filter[:, natural_statistics_index] -
                moments_projection_filters[i][:, natural_statistics_index])
                         , linestyles[i],
                         label='proj-{}'.format(integrators[i]),
                         linewidth=linewidth,
                         marker=markers[i],
                         markevery=markevery,
                         markersize=markersize)
        plt.legend()
        plt.ylabel(r'$| \mathbb E_a[{}] - \mathbb E_b [{}]|$'.format(natural_statistic_str,
                                                                     natural_statistic_str), rotation=90)
        plt.xlabel('$t$')
        plt.tight_layout(pad=0.1)
        plt.savefig(relative_path + '/{}/{}_Moments_Error_{}.pdf'.format(simulation_id,
                                                                         title_prefix,
                                                                         natural_statistic_str.replace('^', 'p')))
        plt.close()
