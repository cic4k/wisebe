# Author: Carlos Gozález <carlos-emiliano.gonzalez-gallardo@sorbonne-universite.fr>
# Copyright (C) 2020 Carlos González <carlos-emiliano.gonzalez-gallardo@sorbonne-universite.fr>
# Cite as: González-Gallardo, C. E., & Torres-Moreno, J. M. (2018, October).
#          WiSeBE: Window-based Sentence Boundary Evaluation.
#          In Mexican International Conference on Artificial Intelligence (pp. 119-131). Springer, Cham.
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
    Window based Sentence Boundary Evaluation Toolkit (WiSeBETool).

    Usage
    --------
    # Display options and get general information of WiSeBE
    ~$ wisebetool -h

    # Simple execution without candidate segmentation and references in default folder
    ~$ wisebetool

    # Execution with candidate segmentation "~/Desktop/my_candidates/candidate_A.txt",
    #  references in "~/Desktop/my_references" folder, window borders size equal to 4,
    #  plotting and dumping options activated
    ~$ wisebetool -c ~/Desktop/my_candidates/candidate_A.txt -r ~/Desktop/my_references -m 4 -p -d
"""

from itertools import zip_longest
from argparse import ArgumentParser
import sys
import os
import glob
from collections import Counter
import pathlib
import matplotlib.pyplot as plt
import numpy as np
from wisebe.Segmentation import Segmentation
#from Segmentation import Segmentation


def check_params():
    """Parsing of comand line arguments

    Returns
    ----------
    :class:`argparse.Namespac`
    """

    description = "Window Based Sentence Boundary Evaluation Toolkit (WiSeBETool)."""
    parser = ArgumentParser(description=description)

    #here = pathlib.Path(__file__).parent.parent
    #references_path = os.path.join(here, "references")
    parser.add_argument("-r", "--references_dir", type=str,
                        help="Path to reference folder. Default: ./references",
                        default="./references")

    parser.add_argument("-m", "--min_window_limit", type=int,
                        help="Min distance between window borders. Default: 3",
                        default=3)

    parser.add_argument("-c", "--candidate_path", type=str,
                        help="Path to candidate segmentation. Default: None",
                        default=None)

    parser.add_argument("-p", "--plots", help="Whether or not to plot segmentations", action="store_true")

    parser.add_argument("-d", "--dump", help="Dump segments to file", action="store_true")

    return parser.parse_args()

def load_segmentation(segmentation_file):
    """Load a Segmentation object

    Parameters
    ----------
    segmentation_file : string, optional
        The location of the segmented file.

    Returns
    ----------
    :class:`~wisebe.Segmentation`
    """

    segmentation = Segmentation(segmentation_file=segmentation_file)
    segmentation.load()
    segmentation.create_borders()
    return segmentation

def load_references(references_dir):
    """Load all references from references_dir into `references`

    Parameters
    ----------
    references_dir : string
        The references directory

    Returns
    ----------
    list of :class:`~wisebe.Segmentation`
    """

    references = list()
    references_files = list(glob.iglob(os.path.join(references_dir, "*")))

    if not references_files:
        print("The directory {} has no references to load.".format(references_dir))
        sys.exit()

    for reference_file in references_files:
        reference = load_segmentation(reference_file)
        references.append(reference)

    return references

def show_diff(content_1, content_2):
    """

    Parameters
    ----------
    content_1 : list of string
        Content of first segmentation

    content_2 : list of string
        Content of second segmentation

    """

    equal = True
    tokens = zip_longest(content_1, content_2)

    while equal:
        token_1, token_2 = next(tokens)
        print("{} <-> {}".format(token_1, token_2))
        if token_1 != token_2:
            equal= False

def equal_references(references):
    """Compares the content of all references

    Returns
    ----------
    bool
        Whether or not all references are equal
    """

    _first = True

    for reference in references:
        content_noseg =  reference.get_noseg()

        if _first:
            prev_content_noseg = content_noseg
            _first = False
            continue

        if content_noseg == prev_content_noseg:
            prev_content_noseg = content_noseg
        else:
            print("Reference {} is different from previous ones.".format(reference.segmentation_file))
            show_diff(prev_content_noseg, content_noseg)
            return False

    print("All references look similar :)")
    return True

def equal_candidate_reference(candidate, gral_ref):
    """Compares the content of candidate and general reference

    Parameters
    ----------
    candidate : :class:`~wisebe.Segmentation`
        The candidate segmentation

    gral_ref : :class:`~wisebe.Segmentation`
        The general reference


    Returns
    ----------
    bool
        Whether or not the candidate and general references are equal
    """   

    candidate_content_noseg = candidate.get_noseg()
    gral_ref_content_noseg = gral_ref.get_noseg()

    if candidate_content_noseg != gral_ref_content_noseg:
        print("Candidate and general reference are different...")
        show_diff(candidate_content_noseg, gral_ref_content_noseg)
        return False
    
    print("Candidate and general reference are equal...")
    return True

def create_gral_ref(references, references_dir):
    """Creates a reference englobing all references from `references`

    Parameters
    ----------
    references_dir : string
        The references directory

    Returns
    ----------
    :class:`~wisebe.Segmentation`
    """

    gral_ref = Segmentation(os.path.join(references_dir, "gral_ref"))
    gral_ref.content = references[0].get_noseg()

    borders = [0] * len(gral_ref.content)

    for reference in references:
        borders = list(map(lambda i, j: i+j, borders, reference.borders))

    gral_ref.borders = borders

    return gral_ref

def create_win_borders(gral_ref, win_limit):
    """Creates the window borders from the general reference

    Parameters
    ----------
    gral_ref : :class:`~wisebe.Segmentation`
        The general reference

    win_limit : int
        The size of the window to consider


    Returns
    ----------

    list
    """

    win_borders = list()
    win_tmp = list()
    last_border_index = 0

    for index, border in enumerate(gral_ref.borders):
        if index - last_border_index > win_limit and win_tmp:
            win_borders.append(win_tmp)
            win_tmp = list()
        if border != 0:
            win_tmp.append(index)
            last_border_index = index

    if win_tmp:
        win_borders.append(win_tmp)

    return win_borders

def compute_fleiss_k(borders,  num_refs):
    """Computes the fleiss kappa of the borders from the general reference as
        defined in :class:`~wisebe.Segmentation`

    Parameters
    ----------
    borders : list of integers
            List of borders where 0 if word and >=1 if </S> depending of all references

    num_refs : int
        The number of available references


    Returns
    ----------
    float
        The computed Fleiss Kappa
    """

    def _create_fleiss_matrix():

        fleiss_matrix = []

        for border in borders:

            num_borders = 0 # by default considered as no border
            num_no_borders = num_refs # by default considered as no border

            if border != 0: # if a border
                num_borders = border
                num_no_borders = num_refs - num_borders

            fleiss_matrix.append([num_no_borders, num_borders])

        return np.array(fleiss_matrix)

    def _compute_p_k():
        sums = np.sum(fleiss_matrix, axis=0)
        return np.array([x/(N*num_refs) for x in sums])

    def _compute_P_N():

        P_N = np.zeros(([N]))

        for i in range(N):
            P_N[i] = (np.sum(pow(fleiss_matrix[i, :], 2)) - num_refs) / (num_refs*(num_refs-1))

        return P_N

    N = len(borders)
    P_mean = 0
    P_e = 0

    fleiss_matrix = _create_fleiss_matrix()
    p_k = _compute_p_k()
    P_N = _compute_P_N()

    P_mean = np.mean(P_N)
    P_e = np.sum(pow(p_k, 2))

    fleiss_k = (P_mean - P_e)/(1 - P_e)

    return fleiss_k

def compute_stats_refs(gral_ref, references):
    """Computes the stats of the references

    Parameters
    ----------
    gral_ref : :class:`~wisebe.Segmentation`
        The general reference

    references : list of :class:`~wisebe.Segmentation`
        List of references


    Returns
    ----------
    float
        The agreement borders ratio
    """

    common_borders = 0
    common_borders_ponderated = 0
    frecs_count = Counter(gral_ref.borders)
    num_refs = len(references)

    #print(frecs_count)

    for _ in frecs_count:
        if _ != 0: # Avoids no borders
            common_borders += frecs_count[_]
        if _ > 1: # Avoids disaccorded borders
            common_borders_ponderated += _ * frecs_count[_]
        #print("  %d -> %d" % (_, frecs_count[_]))

    max_borders = common_borders * num_refs
    agreement_borders_ratio = common_borders_ponderated / max_borders
    fleiss_k = compute_fleiss_k(gral_ref.borders, num_refs)
    fleiss_k_borders = compute_fleiss_k([_ for _ in gral_ref.borders if _ != 0], num_refs)

    print("**STATS**")
    print("- Common borders (cm): {}".format(common_borders))
    print("- Common borders ponderated (cmp): {}".format(common_borders_ponderated))
    print("- Agreement Ratio (AR) AR = cmp / mb : {:0.4f}".format(agreement_borders_ratio))
    print("- Max borders (max_b) max_b = cm x |references| : {}".format(max_borders))
    print("- Fleiss Kappa (K): {:0.4f}".format(fleiss_k))
    print("- Fleiss Kappa in borders (K_borders): {:0.4f}".format(fleiss_k_borders))

    return agreement_borders_ratio

def plot_window_candidate(candidate, gral_ref, references, win_borders):
    """Creates plots for candidate and window borders

    Parameters
    ----------
    candidate : :class:`~wisebe.Segmentation`
        The candidate segmentation

    gral_ref : :class:`~wisebe.Segmentation`
        The general reference

    references : list of :class:`~wisebe.Segmentation`
        List of references

    win_borders : list of int pairs [start, end]
        List of window borders

    """

    fig = plt.figure()
    x_axis = range(len(gral_ref.borders))

    ax = fig.add_subplot(3, 1, 1)
    y_axis = gral_ref.borders
    ax.bar(x_axis, y_axis, width=1, color='b',  align='center')
    ax.axis([0, len(x_axis)-1, 0, len(references)])
    ax.set_title('Borders of general reference')
    ax.set_yticks(range(len(references)+1))
    ax.set_ylabel('Frequence')
    ax.set_xlabel('Index of words')

    y_axis = [0]*len(x_axis)
    for win_border in win_borders:
        win_border_start = win_border[0]
        win_border_end = win_border[-1] + 1
        window = range(win_border_start, win_border_end)
        for index in window:
            y_axis[index] = 1

    ax = fig.add_subplot(3, 1, 2)
    ax.bar(x_axis, y_axis, width=1, color='b', align='center')
    ax.axis([0, len(x_axis)-1, 0, 1])
    ax.set_title('Window borders')
    ax.set_xticks([])
    ax.set_yticks([])

    y_axis = candidate.borders
    ax = fig.add_subplot(3, 1, 3)
    ax.bar(x_axis, y_axis, width=1, color='b', align='center')
    ax.axis([0, len(x_axis)-1, 0, 1])
    ax.set_yticks([])
    ax.set_xlabel('Words in transcription')
    ax.set_title('Borders of '+candidate.segmentation_file+' candidate')

    plt.show()

def plot_references(references, gral_ref, win_borders):
    """Creates plots for all references, general reference and window borders

    Parameters
    ----------
    references : list of :class:`~wisebe.Segmentation`
        List of references

    gral_ref : :class:`~wisebe.Segmentation`
        The general reference

    win_borders : list of int pairs [start, end]
        List of window borders

    """

    num_plots = len(references)+1
    if win_borders:
        num_plots += 1
    x_axis = range(len(gral_ref.borders))

    fig = plt.figure()

    for index, reference in enumerate(references, start=1):

        y_axis = reference.borders
        ax = fig.add_subplot(num_plots, 1, index)
        ax.bar(x_axis, y_axis, width=1, color='b', align='center')
        ax.axis([0, len(x_axis)-1, 0, 1])
        ax.set_title(reference.segmentation_file)
        ax.set_xticks([])
        ax.set_yticks([])

    if win_borders:
        y_axis = [0]*len(x_axis)
        for win_border in win_borders:
            win_border_start = win_border[0]
            win_border_end = win_border[-1] + 1
            window = range(win_border_start, win_border_end)
            for index in window:
                y_axis[index] = 1

        ax = fig.add_subplot(num_plots, 1, num_plots-1)
        ax.bar(x_axis, y_axis, width=1, color='b', align='center')
        ax.axis([0, len(x_axis)-1, 0, 1])
        ax.set_title('Window borders')
        ax.set_xticks([])
        ax.set_yticks([])


    y_axis = gral_ref.borders
    ax = fig.add_subplot(num_plots, 1, num_plots)
    ax.bar(x_axis, y_axis, width=1, color='b',  align='center')
    ax.axis([0, len(x_axis)-1, 0, len(references)])
    ax.set_title('Borders of general reference')
    ax.set_yticks(range(len(references)+1))
    ax.set_ylabel('Frequence')
    ax.set_xlabel('Index of words')

    plt.show()

def evaluate_candidate(candidate, win_borders, agreement_borders_ratio):
    """Calculates (smooth) precision, recall and F1 of the candidate segmentation

    Parameters
    ----------
    candidate : :class:`~wisebe.Segmentation`
        The candidate segmentation

    win_borders : list of int pairs [start, end]
        List of window borders

    agreement_borders_ratio : float
        The agreement borders ratio


    Returns
    ----------
    dict :
        A dictionary with the results
    """

    ws_bs_dict = dict() #dict that contains the windows with borders

    def _distance_to_nearest_window_border(_candidate_border_index):
        _distance = 0
        _last_distance = len(candidate.content)
        for win_border in win_borders:
            win_border_start = win_border[0]
            win_border_end = win_border[-1]

            if _candidate_border_index in range(win_border_start, win_border_end+1):
                ws_bs_dict.setdefault(tuple(win_border), 1)
                _distance = 0
                return _distance

            _distance = min([abs(_candidate_border_index-win_border_start),
                             abs(_candidate_border_index-win_border_end)])

            if _distance < _last_distance:
                _last_distance = _distance
            else:
                return _last_distance
        return _distance

    bs_ws = 0 #borders inside windows
    bs_nws = 0 #borders outside windows
    d_ws = 0 #distance to windows

    for index, border in enumerate(candidate.borders):
        if border != 0:
            dist = _distance_to_nearest_window_border(index)
            if dist == 0:
                bs_ws += 1
            else:
                bs_nws += 1
                d_ws += dist

    num_candidate_borders = candidate.borders.count(1)

    ws_bs = len(ws_bs_dict) #windows with borders
    ws_nbs = len(win_borders) - ws_bs #windows without borders

    smooth_precision = 0
    smooth_recall = 0

    if d_ws != 0:
        smooth_precision = bs_nws / d_ws
        smooth_recall = ws_nbs / d_ws

    s_precision = s_recall = s_f1 = precision = recall = f1 =0

    precision = bs_ws / num_candidate_borders
    recall = ws_bs / len(win_borders)
    f1 = 2 * (precision * recall) / (precision + recall)

    s_precision = (bs_ws + smooth_precision) / num_candidate_borders
    s_recall = (ws_bs + smooth_recall) / len(win_borders)
    s_f1 = 2 * (s_precision * s_recall) / (s_precision + s_recall)

    wisebe = s_wisebe = 0
    wisebe = f1 * agreement_borders_ratio #WiSeBE
    s_wisebe = s_f1 * agreement_borders_ratio #smooth WiSeBE

    results = dict()
    results["Precision"] = precision
    results["Recall"] = recall
    results["F1"] = f1
    results["s_Precision"] = s_precision
    results["s_Recall"] = s_recall
    results["s_F1"] = s_f1
    results["bs_ws"] = bs_ws
    results["bs_nws"] = bs_nws
    results["ws_bs"] = ws_bs
    results["ws_nbs"] = ws_nbs
    results["d_ws"] = d_ws
    results["candidate_borders"] = num_candidate_borders
    results["wisebe"] = wisebe
    results["s_wisebe"] = s_wisebe
    results["agreement_borders_ratio"] = agreement_borders_ratio

    return results

def print_eval(results):
    """Prints results into terminal

    Parameters
    ----------
    results : dict
        The results to print

    """

    print("+"*10)
    for key, value in results.items():
        print("{}\t{:0.3f}".format(key, value))
    print("+"*10)

def dump_segmentations(references, gral_ref, candidate):
    """Dumps segmentations into file

    Parameters
    ----------
    references : list of :class:`~wisebe.Segmentation`
        List of references

    gral_ref : :class:`~wisebe.Segmentation`
        The general reference

    candidate : :class:`~wisebe.Segmentation`
        The candidate segmentation

    """

    for reference in references:
        reference.dump_borders()

    gral_ref.dump_borders()

    if candidate:
        candidate.dump_borders()

def evaluate_vs_refs(references, candidate):
    """Calculates (smooth) precision, recall and F1 of the candidate segmentation vs
        each different reference

    Parameters
    ----------
    references : list of :class:`~wisebe.Segmentation`
        List of references

    candidate : :class:`~wisebe.Segmentation`
        The candidate segmentation

    """

    def _compute_refence_means():

        f1_mean = precision_mean = recall_mean = 0
        mean_results = dict()

        for references_result in references_results:
            f1_mean += references_result["F1"]
            precision_mean += references_result["Precision"]
            recall_mean += references_result["Recall"]

        mean_results["F1"] = f1_mean / len(references_results)
        mean_results["Precision"] = precision_mean / len(references_results)
        mean_results["Recall"] = recall_mean / len(references_results)

        return mean_results

    references_results = []

    for reference in references:
        print("- Reference: {}".format(reference.segmentation_file))
        window_tmp = [[_] for _, val in enumerate(reference.borders) if val == 1]
        results = evaluate_candidate(candidate, window_tmp, 0)
        references_results.append(results)
        print_eval(results)

    mean_results = _compute_refence_means()
    print("- Average performance vs. all references")
    print_eval(mean_results)

def main():
    """Entry to WiSeBE evaluation tool.

    """

    references = None
    gral_ref = None
    win_borders = None
    candidate = None
    params = check_params()
    references = load_references(params.references_dir)

    if not equal_references(references):
        sys.exit()

    gral_ref = create_gral_ref(references, params.references_dir)
    win_borders = create_win_borders(gral_ref, params.min_window_limit)
    agreement_borders_ratio = compute_stats_refs(gral_ref, references)

    if params.plots:
        plot_references(references, gral_ref, win_borders)

    if params.candidate_path:
        candidate = load_segmentation(params.candidate_path)

        if not equal_candidate_reference(candidate, gral_ref):
            sys.exit()

        print("Evaluating vs. each reference...")
        evaluate_vs_refs(references, candidate)


        print("Evaluating vs. general referece...")
        results = evaluate_candidate(candidate, win_borders, agreement_borders_ratio)
        print_eval(results)

        if params.plots:
            plot_window_candidate(candidate, gral_ref, references, win_borders)

    if params.dump:
        dump_segmentations(references, gral_ref, candidate)

if __name__ == '__main__':
    main()
