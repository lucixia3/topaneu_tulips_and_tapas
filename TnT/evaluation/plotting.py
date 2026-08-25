import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


def heatmap(data, title, path):
    df = pd.DataFrame(data).T
    n_rows, n_cols = df.shape
    fig_w = max(6, n_cols * 1.1 + 2)
    fig_h = max(8, n_rows * 0.42 + 2)
    
    plt.figure(figsize=(fig_w, fig_h))
    ax = sns.heatmap(
        df,
        vmin=0, vmax=1,
        cmap='inferno',
        annot=True,
        fmt='.2f',              # round to 2 decimals
        annot_kws={'size': 8},
        linewidths=0.5,
        linecolor='white',
        cbar_kws={'label': 'Score'},
    )
    
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()

def spider(data, suptitle, path): 
    # Derive spoke labels from the first subplot's variable names. All
    # inner dicts are expected to share the same keys/order.
    subplot_titles = list(data.keys())
    spoke_labels = list(next(iter(data.values())).keys())
    N = len(spoke_labels)
 
    theta = radar_factory(N, frame='polygon')
 
    n_subplots = len(data)
    ncols = 7
    nrows = int(np.ceil(n_subplots / ncols))
    subplot_size = 3
 
    fig, axs = plt.subplots(figsize=(ncols * subplot_size, nrows * subplot_size), nrows=nrows, ncols=ncols,
                            subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)
 
    # Normalize axs to a flat array regardless of nrows/ncols
    axs_flat = np.atleast_1d(axs).flatten()
 
    color = 'b'
    for ax, title in zip(axs_flat, subplot_titles):
        values = list(data[title].values())
        if all(v == 0 for v in values):
            # Opaque, blanked-out panel with an N/A marker
            ax.set_facecolor('#d9d9d9')
            ax.patch.set_alpha(1.0)          # force fully opaque
            ax.plot(theta, [0] * N, color='none')  # "empty" placeholder line, nothing visible
            ax.set_rgrids([0.2, 0.4, 0.6, 0.8], labels=[])  # keep grid but hide numbers
            ax.set_varlabels([''] * N)       # blank spoke labels to reduce clutter
            ax.set_title(title, weight='bold', size='small', position=(0.5, 1.1),
                        horizontalalignment='center', verticalalignment='center')
            ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes,
                    ha='center', va='center', fontsize=14, weight='bold', color='dimgray')
            continue
        ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
        ax.set_title(title, weight='bold', size='small', position=(0.5, 1.1),
                     horizontalalignment='center', verticalalignment='center')
        ax.plot(theta, values, color=color)
        ax.fill(theta, values, facecolor=color, alpha=0.25)
        ax.set_varlabels(spoke_labels)
 
    # Hide any leftover empty axes if n_subplots doesn't fill the grid
    for ax in axs_flat[n_subplots:]:
        ax.axis('off')
 
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    return

def base_category(name):
    """Strip a leading 'L-' or 'R-' prefix so paired samples share an id."""
    if name.startswith('L-') or name.startswith('R-'):
        return name[2:]
    return name


def decluttered_spider(data, suptitle, path):
    spoke_labels = list(next(iter(data.values())).keys())
    N = len(spoke_labels)

    rows = []
    for i in range(1, 6):
        cur_cols={'Left': [], 'N/A': [], 'Right': []}
        for k, v in data.items():
            if f"{i}." in k:
                if k.startswith('L-'):
                    v['name']=k
                    cur_cols['Left'].append(v)
                elif k.startswith('R-'):
                    v['name']=k
                    cur_cols['Right'].append(v)
                else:
                    v['name']=k
                    cur_cols['N/A'].append(v)
        rows.append(cur_cols)
    theta = radar_factory(N, frame='polygon')
    # --- build one color per category, shared across L-/R- versions -----
    categories = sorted({base_category(v['name'])
                         for row in rows for col in row.values() for v in col})
    cmap = plt.get_cmap('tab20' if len(categories) <= 20 else 'hsv')
    n_colors = 20 if len(categories) <= 20 else len(categories)
    color_map = {cat: cmap(idx / max(n_colors - 1, 1))
                for idx, cat in enumerate(categories)}
 
    col_names = ['Left', 'N/A', 'Right']
 
    fig, axs = plt.subplots(figsize=(15, 23), nrows=5, ncols=3,
                            subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(wspace=0.5, hspace=0.6, top=0.94, bottom=0.03)
 
    for row_idx, row in enumerate(rows):
        for col_idx, col_name in enumerate(col_names):
            ax = axs[row_idx, col_idx]
            samples = row[col_name]
            ax.set_title(f"{col_name}",
                         weight='bold', size='small', position=(0.5, 1.15),
                         horizontalalignment='center', verticalalignment='center')
 
            if not samples:
                # Same opaque 'N/A' placeholder treatment as an all-zero panel
                ax.set_facecolor('#d9d9d9')
                ax.patch.set_alpha(1.0)
                ax.plot(theta, [0] * N, color='none')
                ax.set_rgrids([0.2, 0.4, 0.6, 0.8], labels=[])
                ax.set_varlabels([''] * N)
                ax.text(0.5, 0.5, 'N/A', transform=ax.transAxes, ha='center',
                        va='center', fontsize=14, weight='bold', color='dimgray')
                continue
 
            ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
            for v in samples:
                cat = base_category(v['name'])
                color = color_map[cat]
                values = [v[label] for label in spoke_labels]
                ax.plot(theta, values, color=color, label=v['name'])
                ax.fill(theta, values, facecolor=color, alpha=0.15)
            ax.set_varlabels(spoke_labels)
            ax.legend(loc='upper right', bbox_to_anchor=(1.4, 1.15),
                      fontsize='xx-small', frameon=False)
 
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    return


def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` Axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding Axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta