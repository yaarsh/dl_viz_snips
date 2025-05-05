import numpy as np

from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score
from matplotlib.colors import LinearSegmentedColormap , ListedColormap
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import seaborn as sns
from matplotlib import cm


def scatter2c(labels, scores, ax=None, title='', leglabs=('non_target', 'target'), auc=True, return_handle=False, filenames=None, images_on_scatter=False):
    if ax is None:
        ax = plt.gca()

    cmaprb = LinearSegmentedColormap.from_list("redbluecmap",["b","r"])
    h = ax.scatter(np.arange(len(scores)), scores, alpha=0.3, c=labels, cmap=cmaprb, linewidths=0.5, edgecolors='black')  # cmap='rainbow'
    # ax.legend(['innocent', 'target'])
    if images_on_scatter:
        url_paths = ['file:///' + str(images_path) for images_path in filenames]
        h.set_urls(url_paths)

    handels = h.legend_elements(prop='colors')[0]
#     h_thr_line = ax.axhline(y=blkdata.prod_thresh, label='Production Threshold', linestyle='--')  #  color='orange'
#     ax.legend(handels + [h_thr_line], ['non_target', 'target'])
    ax.legend(handels, leglabs)
    ax.grid()
    if auc:
        auc = roc_auc_score(labels, scores)
        title = title + f' [auc={auc:.3f}]'
    ax.set_title(title)
    if return_handle:
        return h

def scatter_multi(encoding, scores, ax=None, title='', leglabs=('non_target', 'target'), auc=True, return_handle=False, 
                  filenames=None, images_on_scatter=False, encoding_dict=None, colors=None):
    if ax is None:
        ax = plt.gca()
    new_encoding = []
    for e in encoding:
        new_e = ','.join(sorted(list(set(str(e).split(',')))))
        new_encoding.append(new_e)
    unique_encodings = list(set(new_encoding))
    n = len(unique_encodings)
    unique_encodings_int = sorted(unique_encodings)
    new_encoding_int = [unique_encodings_int.index(e) for e in new_encoding]
    if n > 2:
        leglabs = []
        for combo in sorted(unique_encodings):
            splitted_combo = combo.split(',')
            cats = []
            for e in splitted_combo:
                for cat, val in encoding_dict.items():
                    if int(float(e)) == val:
                        cats.append(cat)
            leglabs.append(','.join(cats))

    # assert len(colors) >= n, "Think of more colors for the plot!"
    # colors = colors[:encoding.nunique()]
    # cmap_costum = LinearSegmentedColormap.from_list("redbluecmap", colors)
    viridis = cm.get_cmap('tab10')
    if colors is None:
        colors = viridis(range(n))
    cmap = ListedColormap(colors)
    h = ax.scatter(np.arange(len(scores)), scores, alpha=0.3, c=new_encoding_int, cmap=cmap, linewidths=0.5, edgecolors='black')  # cmap='rainbow'
    # ax.legend(['innocent', 'target'])
    if images_on_scatter:
        url_paths = ['file:///' + str(images_path) for images_path in filenames]
        h.set_urls(url_paths)

    handels = h.legend_elements(prop='colors')[0]
#     h_thr_line = ax.axhline(y=blkdata.prod_thresh, label='Production Threshold', linestyle='--')  #  color='orange'
#     ax.legend(handels + [h_thr_line], ['non_target', 'target'])
    ax.legend(handels, leglabs)
    ax.grid()
    if auc:
        if type(encoding[0]) == str:
            labels = (~encoding.str.contains('0')).astype(int)
        else:
            labels = encoding.values
        auc = roc_auc_score(labels, scores)
        title = title + f' [auc_binary={auc:.3f}]'
    ax.set_title(title)
    if return_handle:
        return h


# ---- comfusion matrix ----
def plot_cm(labels, y_cova, cat_list, ax=None):
    # cm = np.array([tn, fp, fn, tp])
    cm = confusion_matrix(labels, y_cova)
    cmdisp = ConfusionMatrixDisplay(cm,  display_labels=cat_list)  # changed by yaar 20.10.22 to support sklearn 1.1.2
    cmdisp.plot(cmap=plt.cm.Blues, values_format='1d', ax=ax)
    

# ---- colors print ------
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
def c_err(s):
    "return string to be printed in bold and red"
    return bcolors.FAIL + bcolors.BOLD + s + bcolors.ENDC

def c_pass(s):
    "return string to be printed in bold and green"
    return bcolors.OKGREEN + bcolors.BOLD + s + bcolors.ENDC
# -------------------------

def plot_distplots(target_scores, nontarget_scores, n_bins=40, ax=None, title='Scores PDF'):
    """ seaborn distplots """
    #     ax = target_score.hist(density=True, alpha=0.3, bins=20)
    #     nontarget_score.hist(ax=ax, color='red', density=True, alpha=0.3, bins=20)
    bins = np.linspace(0., 1., num=n_bins+1)
    dff = pd.DataFrame({'score': target_scores, 'label': nontarget_scores})
    target_score = dff.loc[dff.label == 1, 'score']
    nontarget_score = dff.loc[dff.label == 0, 'score']
    ax = sns.distplot(target_score, hist=True, kde=True, bins=bins, color='blue', hist_kws={'edgecolor':'black', 'alpha': 0.3}, label='target', ax=ax)
    ax = sns.distplot(nontarget_score, hist=True, kde=True, bins=bins, color='red', hist_kws={'edgecolor':'black', 'alpha': 0.3}, label='innocent', ax=ax)
    ax.grid()
    ax.legend()
    ax.set_title(title)

def plot_histograms(encoding, scores, ax=None, density=False, title='', encoding_dict=None):
    if ax is None:
        ax = plt.gca()
    hist_bins = np.arange(0., 1.025, 0.025)
    # n, _ = np.histogram(predslist[0], hist_bins, density=density)
    n = encoding.nunique()
    colors = ['blue', 'red', 'green', 'yellow']
    viridis = cm.get_cmap('tab10')

    new_encoding = []
    for e in encoding:
        new_e = ','.join(sorted(list(set(e.split(',')))))
        new_encoding.append(new_e)
    unique_encodings = list(set(new_encoding))
    n = len(unique_encodings)
    unique_encodings_int = sorted(unique_encodings)
    new_encoding_int = [unique_encodings_int.index(e) for e in new_encoding]
    if n > 2:
        cats = []
        for combo in sorted(unique_encodings):
            splitted_combo = combo.split(',')
            curr_cats = []
            for e in splitted_combo:
                for cat, val in encoding_dict.items():
                    if e == val:
                        curr_cats.append(cat)
            cats.append(','.join(curr_cats))

    else:
        cats = ('non_target', 'target')
    # assert len(colors) >= len(cats), "Think of more colors for the plot!"
    colors = colors[:len(cats)]
    handles = []
    for encoding_value, color in zip(unique_encodings_int, colors):
        encoding_scores = scores.reindex(np.where(new_encoding_int == encoding_value)[0])
    # _, _, bar_container_0 = axes[1].hist(predslist[0], hist_bins, lw=1, alpha=0.3, fc='blue', ec='black', density=density)
        k, _, _ = ax.hist(encoding_scores, bins=hist_bins, lw=1, alpha=0.3, fc=color, ec='black', density=density)
        handles.append(k)
    # _, _, bar_container_1 = axes[1].hist(predslist[0][target_idxs], hist_bins, lw=1, alpha=0.3, fc='red', ec='black', density=density)
    ax.grid(True)
    ax.set_title(f'Scores Histogram : {title}')
    ax.legend(labels=cats)
    # axes.set_ylim((-0.05, 100))