import matplotlib.pyplot as plt
import copy
import os
from datetime import datetime
from pylab import grid
import pickle
from tqdm import tqdm
import numpy as np
import copy
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors

# importing benchmarking routine
from benchmarking import *

def save_dict(target_folder, data_dict, timestamp):
    # Create a folder with current date and time
    folder_path = os.path.join(target_folder, timestamp)
    os.makedirs(folder_path, exist_ok=True)

    # Save dictionary to file within the created folder
    file_path = os.path.join(folder_path, "trajectories.pkl")
    with open(file_path, "wb") as f:
        pickle.dump(data_dict, f)

def ML4CE_uncon_eval(
    N_x_l,
    f_eval_l,
    functions_test,
    algorithms_test,
    reps,
    home_dir,
    SafeData=False,
):

    trajectories = {}
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    #############
    # Dimension #
    #############
    for i_run in range(len(N_x_l)):
        N_x_ = N_x_l[i_run]
        f_eval_ = f_eval_l[i_run]
        dim_S = "D" + str(N_x_)
        trajectories[dim_S] = {}
        bounds_ = np.array([[-7, 7] for i in range(N_x_)])

        #############
        # Functions #
        #############
        for i_function in functions_test:
            print("===================== ", i_function, "D" + str(N_x_))
            trajectories[dim_S][i_function] = {}
            trajectories[dim_S][i_function]["all means"] = {}
            trajectories[dim_S][i_function]["all 90"] = {}
            trajectories[dim_S][i_function]["all 10"] = {}
            trajectories[dim_S][i_function]["f_list"] = {}
            trajectories[dim_S][i_function]["x_list"] = {}
            # Removed: trajectories[dim_S][i_function]["PID_traj"] = {}
            all_f_ = []
            randShift_l = np.random.uniform(-3, 3, (reps, N_x_))

            # save random shift
            trajectories[dim_S][i_function]["rand_shift"] = randShift_l

            ##############
            # Algorithms #
            ##############
            info = []

            for i_algorithm in algorithms_test:
                print("== ", str(i_algorithm.__name__))
                trajectories[dim_S][i_function][str(i_algorithm.__name__)] = []
                trajectories[dim_S][i_function]["f_list"][
                    str(i_algorithm.__name__)
                ] = []
                trajectories[dim_S][i_function]["x_list"][
                    str(i_algorithm.__name__)
                ] = []
                # Removed: trajectories[dim_S][i_function]["PID_traj"][str(i_algorithm.__name__)] = []

                ###############
                # Repetitions #
                ###############
                for i_rep in tqdm(range(reps)):
                    x_shift_ = randShift_l[i_rep, :].reshape((N_x_, 1))
                    # test function
                    t_ = Test_function(i_function, N_x_, True, x_shift_, bounds_)
                    # algorithm
                    a, b, team_names, cids = i_algorithm(t_, N_x_, bounds_, f_eval_)
                    # post-processing
                    t_.best_f_list()  # List of best points so far
                    t_.pad_or_truncate(f_eval_)
                    # store result
                    trajectories[dim_S][i_function][str(i_algorithm.__name__)].append(
                        copy.deepcopy(t_.best_f_c)
                    )
                    trajectories[dim_S][i_function]["f_list"][
                        str(i_algorithm.__name__)
                    ].append(copy.deepcopy(t_.f_list))
                    trajectories[dim_S][i_function]["x_list"][
                        str(i_algorithm.__name__)
                    ].append(copy.deepcopy(t_.x_list))

                    # Removed block checking if i_function == "cstr_pid_f"

                    # safe data in an overwriting fashion
                    if SafeData:
                        save_dict(home_dir, trajectories, timestamp)

                # statistics from each algorithm for a function
                l_ = np.array(
                    trajectories[dim_S][i_function][str(i_algorithm.__name__)]
                )
                m_ = np.mean(l_, axis=0)
                q10_ = np.quantile(l_, 0.10, axis=0)
                q90_ = np.quantile(l_, 0.90, axis=0)
                trajectories[dim_S][i_function]["all means"][
                    str(i_algorithm.__name__)
                ] = copy.deepcopy(m_)
                trajectories[dim_S][i_function]["all 90"][str(i_algorithm.__name__)] = (
                    copy.deepcopy(q10_)
                )
                trajectories[dim_S][i_function]["all 10"][str(i_algorithm.__name__)] = (
                    copy.deepcopy(q90_)
                )
                all_f_.append(copy.deepcopy(l_))
                info.append(
                    {
                        "alg_name": str(i_algorithm.__name__),
                        "team names": team_names,
                        "CIDs": cids,
                    }
                )
                # safe data in an overwriting fashion
                if SafeData:
                    save_dict(home_dir, trajectories, timestamp)

            # statistics for all algorithms for a function
            trajectories[dim_S][i_function]["mean"] = np.mean(all_f_, axis=(0, 1))
            trajectories[dim_S][i_function]["median"] = np.median(all_f_, axis=(0, 1))
            trajectories[dim_S][i_function]["q 0"] = np.max(all_f_, axis=(0, 1))
            trajectories[dim_S][i_function]["q 100"] = np.min(all_f_, axis=(0, 1))

            # safe data in an overwriting fashion
            if SafeData:
                save_dict(home_dir, trajectories, timestamp)

    # over-write one last time
    if SafeData:
        save_dict(home_dir, trajectories, timestamp)
        return info, trajectories, timestamp
    else:
        return info, trajectories, None


def ML4CE_uncon_table(
    trajectories,
    algs_test,
    funcs_test,
    multim,
    N_x_l,
    start_,
):
    """
    This function calculates the test results based on the trajectories and puts them in a format ready to be plotted in tables.
    It also normalizes the results among all algorithms for a given dimension and test function.
    """
    # computing table of results
    alg_perf = {}
    n_f = len(funcs_test)

    # for every algorithm
    for i_dim in range(len(N_x_l)):
        dim_S = "D" + str(N_x_l[i_dim])
        alg_perf[dim_S] = {}
        for i_alg in algs_test:
            print("==  ", str(i_alg.__name__), " ==")
            alg_perf[dim_S][str(i_alg.__name__)] = {}
            # for every function
            for i_fun in range(n_f):
                # retrive performance
                medall_ = trajectories[dim_S][funcs_test[i_fun]]["mean"]
                trial_ = trajectories[dim_S][funcs_test[i_fun]]["all means"][
                    str(i_alg.__name__)
                ]
                lowall_ = trajectories[dim_S][funcs_test[i_fun]]["q 100"]
                higall_ = trajectories[dim_S][funcs_test[i_fun]]["q 0"]
                # score performance
                perf_ = (higall_[start_[i_dim] :] - trial_[start_[i_dim] :]) / (
                    higall_[start_[i_dim] :] - lowall_[start_[i_dim] :]
                )

                alg_perf[dim_S][str(i_alg.__name__)][funcs_test[i_fun]] = copy.deepcopy(
                    np.sum(perf_) / len(perf_)
                )

    cell_text_global = []

    # Build the data matrix for all algorithms and functions
    for i_alg in algs_test:
        # for different dimensions
        for row in range(len(N_x_l)):
            dim_S = "D" + str(N_x_l[row])
            r_ = [
                alg_perf[dim_S][str(i_alg.__name__)][funcs_test[i]] for i in range(n_f)
            ]
            cell_text_global.append(r_)

    # convert to np.array
    cell_text_global_array = np.array(cell_text_global)

    matrix_store_l = []

    # iteratively select all the performances for changing dimensions
    for i in range(len(N_x_l)):
        # select matrix for dimension i
        matrix = cell_text_global_array[i :: len(N_x_l)]
        # obtain column-wise the min-max values
        matrix_min = matrix.min(axis=0)
        matrix_max = matrix.max(axis=0)
        # normalize column-wise
        matrix_norm = (matrix - matrix_min) / (matrix_max - matrix_min)
        matrix_store_l.append(matrix_norm)

    # each matrix in matrix_store_l represents an input dimension
    combined_matrix_norm = np.stack(matrix_store_l, axis=0)

    # reshape so that each matrix is an algorithm, each row is dimension, each column is a test function
    combined_matrix_norm_reshape = np.transpose(combined_matrix_norm, axes=(1, 0, 2))

    # Build averages for the multimodal functions
    row_wise_averages_first = np.mean(
        combined_matrix_norm_reshape[:, :, : len(multim)], axis=2
    )
    row_wise_averages_first_expanded = row_wise_averages_first[:, :, np.newaxis]
    result1 = np.concatenate(
        (combined_matrix_norm_reshape, row_wise_averages_first_expanded), axis=2
    )

    row_wise_averages_second = np.mean(
        combined_matrix_norm_reshape[:, :, len(multim) :], axis=2
    )
    row_wise_averages_second_expanded = row_wise_averages_second[:, :, np.newaxis]
    result2 = np.concatenate((result1, row_wise_averages_second_expanded), axis=2)

    row_wise_averages_new = np.mean(result2[:, :, :6], axis=2)
    row_wise_averages_new_expanded = row_wise_averages_new[:, :, np.newaxis]
    result3 = np.concatenate((result2, row_wise_averages_new_expanded), axis=2)

    column_wise_averages = np.mean(result3, axis=1)
    arr_with_avg_row = np.concatenate(
        (result3, column_wise_averages[:, np.newaxis, :]), axis=1
    )

    return np.around(arr_with_avg_row, decimals=2)


def ML4CE_uncon_graph_abs(
    test_res, algs_test, funcs_test, N_x_l, home_dir, timestamp, SafeFig=False
):

    # Set the font properties globally
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.size": 28,
            "font.family": "lmr",
            "xtick.labelsize": 26,
            "ytick.labelsize": 26,
        }
    )

    colors = plt.cm.tab10(np.linspace(0, 1, len(algs_test)))
    line_styles = ["-", "--", "-.", ":"]
    alg_indices = {alg: i for i, alg in enumerate(algs_test)}

    n_f = len(funcs_test)

    for i_dim in range(len(N_x_l)):
        dim_S = "D" + str(N_x_l[i_dim])
        for i_fun in range(n_f):
            fig, ax = plt.subplots(figsize=(15, 15))
            for i_alg in algs_test:
                trial_ = test_res[dim_S][funcs_test[i_fun]]["all means"][
                    str(i_alg.__name__)
                ]
                up_ = test_res[dim_S][funcs_test[i_fun]]["all 90"][str(i_alg.__name__)]
                down_ = test_res[dim_S][funcs_test[i_fun]]["all 10"][
                    str(i_alg.__name__)
                ]
                alg_index = alg_indices[i_alg]
                color = colors[alg_index]
                line_style = line_styles[alg_index % len(line_styles)]
                plt.plot(
                    trial_,
                    color=color,
                    linestyle=line_style,
                    lw=4,
                    label=str(i_alg.__name__),
                )
                x_ax = np.linspace(0, len(down_), len(down_), endpoint=False)
                plt.gca().fill_between(x_ax, down_, up_, color=color, alpha=0.2)

                # Determine the tick interval
                if len(down_) < 100:
                    interval = 5
                else:
                    interval = 10

                tick_positions = np.arange(0, len(down_), interval)
                if (len(down_) - 1) % interval != 0:
                    tick_positions = np.append(tick_positions, len(down_) - 1)
                tick_labels = np.arange(0, len(down_), interval)
                if len(tick_labels) < len(tick_positions):
                    tick_labels = np.append(tick_labels, len(down_) - 1)

                plt.xticks(tick_positions, tick_labels)

            legend_handles = []
            legend_cust = [alg.__name__ for alg in algs_test]

            for alg, label in zip(algs_test, legend_cust):
                alg_index = alg_indices[alg]
                color = colors[alg_index]
                line_style = line_styles[alg_index % len(line_styles)]
                handle = Line2D(
                    [0], [0], color=color, linestyle=line_style, lw=4, label=label
                )
                legend_handles.append(handle)

            plt.ylabel("Objective Function Value")
            plt.xlabel("Iterations")
            plt.yscale("log")
            plt.legend(handles=legend_handles, loc="best", frameon=False)
            plt.grid(which="major")
            plt.grid(which="minor", alpha=0.4)

            # Example file save path
            plt.savefig("admin/results/trajectory_plots_1D.png", bbox_inches="tight")

            if SafeFig:
                def directory_exists(directory_name):
                    root_directory = os.getcwd()
                    directory_path = os.path.join(root_directory, directory_name)
                    return os.path.isdir(directory_path)

                directory_name = os.path.join(home_dir, timestamp, "trajectory_plots_1D")
                if directory_exists(directory_name):
                    plt.savefig(
                        directory_name + "/{}_{}_1D.png".format(dim_S, funcs_test[i_fun]),
                        bbox_inches="tight",
                    )
                    plt.savefig(
                        directory_name + "/{}_{}_1D.jpg".format(dim_S, funcs_test[i_fun]),
                        format="jpg",
                        bbox_inches="tight",
                        dpi=300,
                    )
                else:
                    print(
                        f"The directory '{directory_name}' does not exist in the root directory."
                    )
                    os.mkdir(directory_name)
                    plt.savefig(
                        directory_name + "/{}_{}_1D.png".format(dim_S, funcs_test[i_fun]),
                        bbox_inches="tight",
                    )
                    plt.savefig(
                        directory_name + "/{}_{}_1D.jpg".format(dim_S, funcs_test[i_fun]),
                        format="jpg",
                        bbox_inches="tight",
                        dpi=300,
                    )

def ML4CE_uncon_contour_allin1(
    functions_test,
    algorithms_test,
    N_x_,
    x_shift_origin,
    bounds_,
    bounds_plot,
    SafeFig=False,
):
    f_eval_ = 30
    track_x = True

    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.size": 28,
            "font.family": "lmr",
            "xtick.labelsize": 26,
            "ytick.labelsize": 26,
        }
    )

    colors = ["#1A73B2", "#D62627", "#E476C2", "#0BBBCD", "grey"]
    line_styles = ["-", "--", "-.", ":"]
    alg_indices = {alg: i for i, alg in enumerate(algorithms_test)}
    fun_n = 0

    for fun_ in functions_test:
        f_contour = Test_function(fun_, N_x_, track_x, x_shift_origin, bounds_)

        n_points = 200
        x1lb = bounds_plot[fun_n][0][0]
        x1ub = bounds_plot[fun_n][0][1]
        x1 = np.linspace(start=x1lb, stop=x1ub, num=n_points)
        x2lb = bounds_plot[fun_n][1][0]
        x2ub = bounds_plot[fun_n][1][1]
        x2 = np.linspace(start=x2lb, stop=x2ub, num=n_points)
        X1, X2 = np.meshgrid(x1, x2)

        fun_n += 1

        plt.figure(figsize=(15, 15))
        ax3 = plt.subplot()

        y = np.empty((n_points, n_points))
        print("Drawing Contours")
        for i in tqdm(range(n_points)):
            for j in range(n_points):
                y[i, j] = f_contour.fun_test(np.array([[X1[i, j], X2[i, j]]]))

        # Log-scale shift
        y = np.log2(y + 1)

        norm = mcolors.Normalize(vmin=np.min(y), vmax=np.max(y))
        contour = plt.contourf(
            X1,
            X2,
            y,
            levels=50,
            cmap="Spectral_r",
            norm=norm,
            alpha=0.5,
            linewidths=0.5,
        )

        ax3.axis([x1lb, x1ub, x2lb, x2ub])

        for alg_ in algorithms_test:
            print(alg_)

            alg_index = alg_indices[alg_]
            color = colors[alg_index]

            f_plot = Test_function(fun_, N_x_, track_x, x_shift_origin, bounds_)
            a, b, team_names, cids = alg_(f_plot, N_x_, bounds_, f_eval_, has_x0=True)

            X_all = np.array(f_plot.x_list)
            ax3.scatter(X_all[:, 0], X_all[:, 1], marker="o", color=color, alpha=0.4)

            f_feas = Test_function(fun_, N_x_, track_x, x_shift_origin, bounds_)
            X_feas = [x for x in X_all]
            for x in X_feas:
                f_feas.fun_test(x)

            f_feas.best_f_list()
            best_values_x1 = [point[0] for point in f_feas.best_x]
            best_values_x2 = [point[1] for point in f_feas.best_x]

            ax3.plot(
                best_values_x1,
                best_values_x2,
                colors[alg_index],
                linewidth=3,
                label=alg_.__name__,
                zorder=5,
            )

            # Start
            ax3.plot(
                X_all[0, 0],
                X_all[0, 1],
                marker="X",
                color="black",
                markersize=15,
                zorder=6,
            )
            # End
            ax3.plot(
                best_values_x1[-1],
                best_values_x2[-1],
                colors[alg_index],
                marker="H",
                markersize=15,
                zorder=10,
            )

            plt.ylabel("$x_2$")
            plt.xlabel("$x_1$")
            plt.xticks([])
            plt.yticks([])

        legend_cust_dict = {
            "LS_QM_v2": "LSQM",
            "opt_SRBF": "SRBF",
            "opt_DYCORS": "DYCORS",
            "opt_SOP": "SOP",
            "COBYQA": "COBYQA",
            "opt_SnobFit": "SNOBFIT",
            "opt_COBYLA": "COBYLA",
            "opt_CUATRO": "CUATRO",
            "BO_np_scipy": "BO",
            "opt_ENTMOOT": "ENTMOOT",
        }
        legend_cust = [legend_cust_dict[key.__name__] for key in algorithms_test]

        legend_handles = []
        for alg, label in zip(algorithms_test, legend_cust):
            alg_index = alg_indices[alg]
            color = colors[alg_index]
            handle = Line2D([0], [0], color=color, linestyle="-", lw=3, label=label)
            legend_handles.append(handle)

        start_handle = Line2D(
            [0], [0], marker="X", color="black", linestyle="None", markersize=15, label="Start"
        )
        end_handle = Line2D(
            [0], [0], marker="H", color="black", linestyle="None", markersize=15, label="End"
        )
        legend_handles.append(start_handle)
        legend_handles.append(end_handle)

        plt.legend(handles=legend_handles, loc="best")

        if SafeFig:
            def directory_exists(directory_name):
                root_directory = os.getcwd()
                directory_path = os.path.join(root_directory, directory_name)
                return os.path.isdir(directory_path)

            directory_name = "images/trajectory_plots_2D"
            if directory_exists(directory_name):
                if "LSQM" in legend_cust:
                    plt.savefig(
                        directory_name + "/{}_allin1_A.png".format(f_feas.func_type),
                        bbox_inches="tight",
                    )
                    plt.savefig(
                        directory_name + "/{}_allin1_A.jpg".format(f_feas.func_type),
                        format="jpg",
                        bbox_inches="tight",
                        dpi=300,
                    )
                else:
                    plt.savefig(
                        directory_name + "/{}_allin1_B.png".format(f_feas.func_type),
                        bbox_inches="tight",
                    )
                    plt.savefig(
                        directory_name + "/{}_allin1_B.jpg".format(f_feas.func_type),
                        format="jpg",
                        bbox_inches="tight",
                        dpi=300,
                    )
            else:
                print(
                    f"The directory '{directory_name}' does not exist in the root directory."
                )
            plt.close()
        else:
            plt.show()


def ML4CE_uncon_contour_allin1_smooth(
    functions_test,
    algorithms_test,
    N_x_,
    x_shift_origin,
    reps,
    bounds_,
    bounds_plot,
    SafeFig=False,
):
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.size": 28,
            "font.family": "lmr",
            "xtick.labelsize": 26,
            "ytick.labelsize": 26,
        }
    )

    f_eval_ = 30
    track_x = True
    colors = ["#1A73B2", "#D62627", "#E476C2", "#0BBBCD", "grey"]
    alg_indices = {alg: i for i, alg in enumerate(algorithms_test)}
    fun_n = 0

    for fun_ in functions_test:
        f_contour = Test_function(fun_, N_x_, track_x, x_shift_origin, bounds_)

        n_points = 100
        x1lb = bounds_plot[fun_n][0][0]
        x1ub = bounds_plot[fun_n][0][1]
        x1 = np.linspace(start=x1lb, stop=x1ub, num=n_points)
        x2lb = bounds_plot[fun_n][1][0]
        x2ub = bounds_plot[fun_n][1][1]
        x2 = np.linspace(start=x2lb, stop=x2ub, num=n_points)
        X1, X2 = np.meshgrid(x1, x2)

        fun_n += 1

        plt.figure(figsize=(15, 15))
        ax3 = plt.subplot()

        y = np.empty((n_points, n_points))
        print("drawing contour")
        for i in tqdm(range(n_points)):
            for j in range(n_points):
                y[i, j] = f_contour.fun_test(np.array([[X1[i, j], X2[i, j]]]).flatten())

        y = np.log2(y + 1)

        norm = mcolors.Normalize(vmin=np.min(y), vmax=np.max(y))
        contour = plt.contourf(
            X1,
            X2,
            y,
            levels=50,
            cmap="Spectral_r",
            norm=norm,
            alpha=0.5,
            linewidths=0.5,
        )

        ax3.axis([x1lb, x1ub, x2lb, x2ub])

        for alg_ in algorithms_test:
            alg_index = alg_indices[alg_]
            color = colors[alg_index]

            x_list = []
            for i_rep in range(reps):
                f_plot = Test_function(fun_, N_x_, track_x, x_shift_origin, bounds_)
                a, b, team_names, cids = alg_(
                    f_plot, N_x_, bounds_, f_eval_, has_x0=True
                )
                X_collect_pre = f_plot.x_list
                X_collect = []
                if len(X_collect_pre) < f_eval_:
                    x_last = X_collect_pre[-1]
                    X_collect = X_collect_pre + [x_last] * (f_eval_ - len(X_collect_pre))
                elif len(X_collect_pre) > f_eval_:
                    X_collect = X_collect_pre[:f_eval_]
                else:
                    X_collect = X_collect_pre
                x_list.append(X_collect)

            X_all = np.array(x_list)
            X_all_m = np.mean(X_all, axis=0).reshape((f_eval_, 2))

            f_feas = Test_function(fun_, N_x_, track_x, x_shift_origin, bounds_)
            X_feas = [x for x in X_all_m]
            for x in X_feas:
                f_feas.fun_test(x)

            f_feas.best_f_list()
            best_values_x1 = [point[0] for point in f_feas.best_x]
            best_values_x2 = [point[1] for point in f_feas.best_x]

            ax3.plot(
                best_values_x1,
                best_values_x2,
                colors[alg_index],
                linewidth=3,
                label=alg_.__name__,
                zorder=5,
            )

            ax3.plot(
                best_values_x1[0],
                best_values_x2[0],
                marker="X",
                color="black",
                markersize=15,
                zorder=6,
            )

            ax3.plot(
                best_values_x1[-1],
                best_values_x2[-1],
                colors[alg_index],
                marker="H",
                markersize=15,
                zorder=10,
            )

            plt.ylabel("$x_2$")
            plt.xlabel("$x_1$")
            plt.xticks([])
            plt.yticks([])

        legend_cust_dict = {
            "LS_QM_v2": "LSQM",
            "opt_SRBF": "SRBF",
            "opt_DYCORS": "DYCORS",
            "opt_SOP": "SOP",
            "COBYQA": "COBYQA",
            "opt_SnobFit": "SNOBFIT",
            "opt_COBYLA": "COBYLA",
            "opt_CUATRO": "CUATRO",
            "BO_np_scipy": "BO",
            "opt_ENTMOOT": "ENTMOOT",
        }

        legend_cust = [legend_cust_dict[key.__name__] for key in algorithms_test]
        legend_handles = []
        for alg, label in zip(algorithms_test, legend_cust):
            alg_index = alg_indices[alg]
            color = colors[alg_index]
            handle = Line2D([0], [0], color=color, linestyle="-", lw=3, label=label)
            legend_handles.append(handle)

        start_handle = Line2D(
            [0], [0], marker="X", color="black", linestyle="None", markersize=15, label="Start"
        )
        end_handle = Line2D(
            [0], [0], marker="H", color="black", linestyle="None", markersize=15, label="End"
        )
        legend_handles.append(start_handle)
        legend_handles.append(end_handle)

        plt.legend(handles=legend_handles, loc="best")

        if SafeFig:
            def directory_exists(directory_name):
                root_directory = os.getcwd()
                directory_path = os.path.join(root_directory, directory_name)
                return os.path.isdir(directory_path)

            directory_name = "images/trajectory_plots_2D"
            if directory_exists(directory_name):
                if "LSQM" in legend_cust:
                    plt.savefig(
                        directory_name + "/{}_allin1_A_smooth.png".format(f_feas.func_type),
                        bbox_inches="tight",
                    )
                    plt.savefig(
                        directory_name + "/{}_allin1_A_smooth.jpg".format(f_feas.func_type),
                        format="jpg",
                        bbox_inches="tight",
                        dpi=300,
                    )
                else:
                    plt.savefig(
                        directory_name + "/{}_allin1_B_smooth.png".format(f_feas.func_type),
                        bbox_inches="tight",
                    )
                    plt.savefig(
                        directory_name + "/{}_allin1_B_smooth.jpg".format(f_feas.func_type),
                        format="jpg",
                        bbox_inches="tight",
                        dpi=300,
                    )
            else:
                print(
                    f"The directory '{directory_name}' does not exist in the root directory."
                )

            plt.close()
        else:
            plt.show()


###################################################### UNUSED FUNCTIONS #############################################################
def ML4CE_uncon_graph_rel(test_res, algs_test, funcs_test, N_x_l):
    """
    Example of a function not referencing cstr_pid_f but not actively used.
    Shows relative performance but is not currently called.
    """

    n_f = len(funcs_test)
    n_p = len(funcs_test) * len(N_x_l)
    nrow = int(n_p / 2) if n_p % 2 == 0 else int(n_p / 2) + 1

    for i_alg in algs_test:
        plt.figure(figsize=(16, 25))
        for i_dim in range(len(N_x_l)):
            dim_S = "D" + str(N_x_l[i_dim])
            for i_fun in range(n_f):
                plt.subplot(nrow, 2, n_f * i_dim + i_fun + 1)
                trial_ = test_res[dim_S][funcs_test[i_fun]]["all means"][
                    str(i_alg.__name__)
                ]
                up_ = test_res[dim_S][funcs_test[i_fun]]["all 90"][str(i_alg.__name__)]
                down_ = test_res[dim_S][funcs_test[i_fun]]["all 10"][str(i_alg.__name__)]
                medall_ = test_res[dim_S][funcs_test[i_fun]]["mean"]
                lowall_ = test_res[dim_S][funcs_test[i_fun]]["q 100"]
                higall_ = test_res[dim_S][funcs_test[i_fun]]["q 0"]
                plt.plot(
                    trial_, color="C" + str(i_fun), lw=3, label=str(i_alg.__name__)
                )
                plt.plot(medall_, "--", lw=2, label="median alg")
                plt.plot(lowall_, "--", lw=2, label="best alg")
                plt.plot(higall_, "--", lw=2, label="worst alg")
                x_ax = np.linspace(0, len(down_), len(down_), endpoint=True)
                plt.gca().fill_between(
                    x_ax, down_, up_, color="C" + str(i_fun), alpha=0.2
                )
                plt.ylabel("obj value")
                plt.xlabel("iterations")
                plt.legend(loc="best")
                plt.title(funcs_test[i_fun] + " " + dim_S + " convergence plot")
                grid(True)


def ML4CE_uncon_contours(
    obj_func,
    i_algorithm,
    bounds_plot,
    X_opt,
    xnew,
    samples_number,
    func_type,
    Cons=False,
    TR_plot=False,
    TR_l=False,
    PlotArrows=False,
    Zoom=False,
    SafeFig=False,
):
    """
    A helper for 2D contour plotting of unconstrained routines.
    Shows a typical usage, not referencing cstr_pid_f either.
    """
    n_points = 100
    x1lb = bounds_plot[0][0]
    x1ub = bounds_plot[0][1]
    x1 = np.linspace(start=x1lb, stop=x1ub, num=n_points)
    x2lb = bounds_plot[1][0]
    x2ub = bounds_plot[1][1]
    x2 = np.linspace(start=x2lb, stop=x2ub, num=n_points)
    X1, X2 = np.meshgrid(x1, x2)

    plt.figure(figsize=(15, 15))
    ax3 = plt.subplot()

    y = np.empty((n_points, n_points))
    for i in range(n_points):
        for j in range(n_points):
            y[i, j] = obj_func.fun_test(np.array([[X1[i, j], X2[i, j]]]))

    ax3.contour(X1, X2, y, 50)

    if Cons:
        con_list = [obj_func.con_plot(x_i) for x_i in x2]
        if func_type == "Rosenbrock_f":
            ax3.plot(con_list, x2, "black", linewidth=3)
        else:
            ax3.plot(x1, con_list, "black", linewidth=3)

    # Plot points
    array_length = len(X_opt[samples_number:, 0])
    for i in range(array_length):
        ax3.plot(X_opt[samples_number + i, 0], X_opt[samples_number + i, 1], marker="o", color="grey")

    # Connect best so far
    best_points = []
    best_value = float("inf")
    for point in X_opt:
        valp = obj_func.fun_test(point)
        if valp < best_value:
            best_value = valp
            best_points.append(point)

    best_values_x1 = [point[0] for point in best_points]
    best_values_x2 = [point[1] for point in best_points]
    ax3.plot(best_values_x1, best_values_x2, marker="o", linestyle="-", color="black")

    if PlotArrows:
        threshold = 2
        x_coords = X_opt[:, 0, 0]
        y_coords = X_opt[:, 1, 0]
        for i in range(len(X_opt) - 1):
            plt.plot([x_coords[i], x_coords[i + 1]], [y_coords[i], y_coords[i + 1]], color="blue")
            distance = np.sqrt(
                (x_coords[i + 1] - x_coords[i]) ** 2 + (y_coords[i + 1] - y_coords[i]) ** 2
            )
            if distance > threshold:
                midpoint = (
                    (x_coords[i] + x_coords[i + 1]) / 2,
                    (y_coords[i] + y_coords[i + 1]) / 2,
                )
                direction = (
                    x_coords[i + 1] - x_coords[i],
                    y_coords[i + 1] - y_coords[i],
                )
                direction /= np.linalg.norm(direction)
                plt.arrow(
                    midpoint[0],
                    midpoint[1],
                    direction[0],
                    direction[1],
                    head_width=0.2,
                    head_length=0.2,
                    fc="blue",
                    ec="blue",
                )

    if TR_plot:
        for i in range(X_opt[samples_number:, :].shape[0]):
            x_pos = X_opt[samples_number + i, 0]
            y_pos = X_opt[samples_number + i, 1]
            circle1 = plt.Circle(
                (x_pos, y_pos),
                radius=TR_l[i],
                color="black",
                fill=False,
                linestyle="--",
            )
            ax3.add_artist(circle1)

    ax3.plot(X_opt[0, 0, 0], X_opt[0, 1, 0], marker="s", color="black", markersize=10)
    xnew = xnew.flatten()
    ax3.plot(xnew[0], xnew[1], marker="^", color="black", markersize=10)

    if Zoom:
        x_coords = X_opt[:, 0, 0]
        y_coords = X_opt[:, 1, 0]
        x1lb_zoom = min(x_coords) - 1
        x1ub_zoom = max(x_coords) + 1
        x2lb_zoom = min(y_coords) - 1
        x2ub_zoom = max(y_coords) + 1
        ax3.axis([x1lb_zoom, x1ub_zoom, x2lb_zoom, x2ub_zoom])
    else:
        ax3.axis([x1lb, x1ub, x2lb, x2ub])

    if SafeFig:
        def directory_exists(directory_name):
            root_directory = os.getcwd()
            directory_path = os.path.join(root_directory, directory_name)
            return os.path.isdir(directory_path)

        directory_name = "images/trajectory_plots_2D"
        if directory_exists(directory_name):
            plt.savefig(directory_name + f"/{i_algorithm.__name__}_{obj_func.func_type}.png")
        else:
            print(f"The directory '{directory_name}' does not exist in the root directory.")
        plt.close()
    else:
        plt.show()

def ML4CE_uncon_leaderboard(trajectories, as_html=False):
    """
    Generates leaderboards from trajectories data.

    Args:
        trajectories (dict): Nested dictionary with trajectory data.
        as_html (bool): If True, output is HTML string (otherwise printed).

    Returns:
        If as_html=True, returns HTML string for all leaderboards; otherwise, returns None.
    """
    html_parts = []

    # Add some basic styling
    html_parts.append("""
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #1e1e1e;
            color: #e0e0e0;
        }
        h2 {
            color: #ff4500;
            text-align: center;
        }
        table {
            width: 80%;
            margin: 20px auto;
            border-collapse: collapse;
            text-align: left;
        }
        th, td {
            border: 1px solid #e0e0e0;
            padding: 8px 12px;
        }
        th {
            background-color: #444;
            color: #fff;
        }
        tr:nth-child(even) {
            background-color: #333;
        }
        tr:nth-child(odd) {
            background-color: #444;
        }
        tr:hover {
            background-color: #555;
        }
    </style>
    """)

    skip_keys = {
        "all means", "all 90", "all 10", "f_list", "x_list",
        "mean", "median", "q 0", "q 100", "rand_shift"
    }

    for dim_key in trajectories:
        for func_key in trajectories[dim_key]:
            if func_key in skip_keys:
                continue

            scoreboard = {}
            for alg_key, runs in trajectories[dim_key][func_key].items():
                if alg_key in skip_keys:
                    continue
                arr = np.array(runs)
                final_vals = arr[:, -1]
                scoreboard[alg_key] = scoreboard.get(alg_key, []) + list(final_vals)

            if not scoreboard:
                continue

            ranking = {alg: np.mean(scoreboard[alg]) for alg in scoreboard}
            sorted_ranking = sorted(ranking.items(), key=lambda x: x[1])

            # Add leaderboard section
            html_parts.append("<table>")
            html_parts.append("<tr><th>Rank</th><th>Algorithm</th><th>Avg Final Value</th></tr>")
            for i, (alg, avg_val) in enumerate(sorted_ranking, start=1):
                html_parts.append(f"<tr><td>{i}</td><td>{alg}</td><td>{avg_val:.5f}</td></tr>")
            html_parts.append("</table>")

    if as_html:
        return "\n".join(html_parts)