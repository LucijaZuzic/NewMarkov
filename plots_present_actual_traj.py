from utilities import translate_var, translate_method_short, load_object, fill_gap, load_traj_name, scale_long_lat, process_time, preprocess_long_lat
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

all_subdirs = os.listdir()   
     
def str_convert(val):
    if val == False:
        return "0"
    if val == True:
        return "1"
    new_val = val
    power_to = 0
    while abs(new_val) < 1 and new_val != 0.0:
        new_val *= 10
        power_to += 1 
    if power_to == 1:  
        return str(np.round(new_val / 10, 3))
    rounded = str(np.round(new_val, 2))
    if rounded[-2:] == '.0':
        rounded = rounded[:-2]
    if power_to != 0:  
        rounded += " \\times 10^{-" + str(power_to) + "}"
    return rounded

def print_row(header_val, no_gap_val, int_val):
    middle_sec = translate_var[header_val.replace("predicted_", "").replace("_", " ").replace("alternative", "alt").replace("x speed alt", "x speed").replace("y speed alt", "y speed").replace("speed no abs alt", "speed no abs").capitalize()]
    if len(middle_sec) > 20:
        nh = middle_sec[:20]
    else:
        nh = "\\multirow{2}{*}{" + middle_sec + "}"
    nh += " & lanac" 
    for v in no_gap_val:
        nh += " & $" + str_convert(v) + "$"
    nh += " \\\\ \cline{2-" + str(len(no_gap_val) + 2) + "}\n" 
    if len(middle_sec) > 20:
        nh += middle_sec[20:] 
    nh += " & realno"  
    for v in int_val:
        nh += " & $" + str_convert(v) + "$"
    nh += " \\\\ \hline\n" 
    print(nh)
    return nh

def print_method(long, lat, newx):
    middle_sec = "Original"
    if long != "":
        middle_sec = translate_method_short(long + "-" + lat) 
    nh = middle_sec.replace("x", "$x$").replace("y", "$y$") 
    for v in newx:
        nh += " & $" + str_convert(v) + "$"
    nh += " \\\\ \hline\n"   
    print(nh)
    return nh
  
def plot_trapezoid(f, g, x, str_pr, filename):  
    l1 = [i for i in np.arange(min(min(f), min(g)) - 0.0001, min(f[0], g[0]) + 0.00001, 0.00001)]
    x1 = [x[0] for i in np.arange(min(min(f), min(g)) - 0.0001, min(f[0], g[0]) + 0.00001, 0.00001)] 
    l5 = [i for i in np.arange(min(min(f), min(g)) - 0.0001, f[-1] + 0.00001, 0.00001)]
    x5 = [x[-1] for i in np.arange(min(min(f), min(g)) - 0.0001, f[-1] + 0.00001, 0.00001)]
    l6 = [i for i in np.arange(min(min(f), min(g)) - 0.0001, g[-1] + 0.00001, 0.00001)]
    x6 = [x[-1] for i in np.arange(min(min(f), min(g)) - 0.0001, g[-1] + 0.00001, 0.00001)]
    x3 = [v for v in np.arange(x[0], x[-1] + 0.0001, 0.0001)]
    s2 = [0 for v in x3] 
    s = [min(min(f), min(g)) - 0.0001 for v in x3] 
    df = [abs(f[i] - g[i]) for i in range(len(x))]
    l4 = [i for i in np.arange(0, df[-1] + 0.00001, 0.00001)]
    x4 = [x[-1] for i in np.arange(0, df[-1] + 0.00001, 0.00001)] 
    diff = 1000
    for ixix in range(len(x)):
        if abs(x[ixix] - 25) < diff:
            diff = abs(x[ixix] - 25) 
            ix = ixix 
    plt.rcParams['font.size'] = 20
    plt.rcParams['font.family'] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.figure(figsize=(20, 7))
    plt.subplot(1, 2, 1)
    plt.xlabel("$t$")
    plt.ylabel('$' + str_pr[2] + '$')   
    plt.plot(x, f, label = "f", c = "r")
    if f[ix] > g[ix]:  
        plt.text(x[ix], f[ix] + 0.00031, '$f' + str_pr + '$', size=20, color='r')
    else:    
        plt.text(x[ix], f[ix] - 0.00042, '$f' + str_pr + '$', size=20, color='r')
    plt.ylim(min(min(f), min(g)), max(max(f), max(g)) + 0.0001)
    plt.plot(x, g, label = "g", c = "b")
    if g[ix] > f[ix]:  
        plt.text(x[ix], g[ix] + 0.00026, '$g' + str_pr + '$', size=20, color='b')
    else:    
        plt.text(x[ix], g[ix] - 0.00036, '$g' + str_pr + '$', size=20, color='b')
    plt.text(x[ix], (f[ix] + g[ix]) / 2, '$A$', size=20, color='k')
    if f[-1] < g[-1]:
        plt.plot(x1, l1, c = "r")
    else:
        plt.plot(x1, l1, c = "b")
    if f[-1] < g[-1]:
        plt.plot(x6, l6, c = "b")
        plt.plot(x5, l5, c = "r")
    else:
        plt.plot(x5, l5, c = "r")
        plt.plot(x6, l6, c = "b")
    if f[-1] < g[-1]:
        plt.plot(x3, s, c = "r") 
    else:
        plt.plot(x3, s, c = "b")
    plt.subplot(1, 2, 2) 
    plt.xlabel("$t$")
    plt.ylabel('$' + str_pr[2] + '$')  
    plt.plot(x, df, c = "magenta") 
    plt.ylim(- 0.0001, max(max(f), max(g)) + 0.0001 - min(min(f), min(g)) + 0.0001) 
    plt.plot(x4, l4, c = "magenta")
    plt.text(x[ix], df[ix] / 2, '$A$', size=20, color='k')
    plt.text(x[ix], df[ix] + 0.0005, '$|f' + str_pr + '-g' + str_pr + '|$', size=20, color='magenta')
    plt.plot(x3, s2, c = "magenta")
    plt.savefig("presentation_plots_" + name + "/" + filename + ".png", bbox_inches = "tight")
    plt.close()
    plot_trapezoid_right(f, g, x, str_pr, filename)
    plot_trapezoid_left(f, g, x, str_pr, filename)
 
def plot_trapezoid_right(f, g, x, str_pr, filename): 
    l1 = [i for i in np.arange(min(min(f), min(g)) - 0.0001, min(f[0], g[0]) + 0.00001, 0.00001)]
    x1 = [x[0] for i in np.arange(min(min(f), min(g)) - 0.0001, min(f[0], g[0]) + 0.00001, 0.00001)] 
    l5 = [i for i in np.arange(min(min(f), min(g)) - 0.0001, f[-1] + 0.00001, 0.00001)]
    x5 = [x[-1] for i in np.arange(min(min(f), min(g)) - 0.0001, f[-1] + 0.00001, 0.00001)]
    l6 = [i for i in np.arange(min(min(f), min(g)) - 0.0001, g[-1] + 0.00001, 0.00001)]
    x6 = [x[-1] for i in np.arange(min(min(f), min(g)) - 0.0001, g[-1] + 0.00001, 0.00001)]
    x3 = [v for v in np.arange(x[0], x[-1] + 0.0001, 0.0001)]
    s2 = [0 for v in x3] 
    s = [min(min(f), min(g)) - 0.0001 for v in x3] 
    df = [abs(f[i] - g[i]) for i in range(len(x))]
    l4 = [i for i in np.arange(0, df[-1] + 0.00001, 0.00001)]
    x4 = [x[-1] for i in np.arange(0, df[-1] + 0.00001, 0.00001)] 
    diff = 1000
    for ixix in range(len(x)):
        if abs(x[ixix] - 25) < diff:
            diff = abs(x[ixix] - 25) 
            ix = ixix 
    plt.rcParams['font.size'] = 20
    plt.rcParams['font.family'] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif" 
    plt.figure(figsize=(10, 7))
    plt.xlabel("$t$")
    plt.ylabel('$' + str_pr[2] + '$')  
    plt.plot(x, df, c = "magenta") 
    plt.ylim(- 0.0001, max(max(f), max(g)) + 0.0001 - min(min(f), min(g)) + 0.0001) 
    plt.plot(x4, l4, c = "magenta")
    plt.text(x[ix], df[ix] / 2, '$A$', size=20, color='k')
    plt.text(x[ix], df[ix] + 0.0005, '$|f' + str_pr + '-g' + str_pr + '|$', size=20, color='magenta')
    plt.plot(x3, s2, c = "magenta") 
    plt.savefig("presentation_plots_" + name + "/" + filename + "_right.png", bbox_inches = "tight")
    plt.close() 
    
def plot_trapezoid_left(f, g, x, str_pr, filename):  
    l1 = [i for i in np.arange(min(min(f), min(g)) - 0.0001, min(f[0], g[0]) + 0.00001, 0.00001)]
    x1 = [x[0] for i in np.arange(min(min(f), min(g)) - 0.0001, min(f[0], g[0]) + 0.00001, 0.00001)] 
    l5 = [i for i in np.arange(min(min(f), min(g)) - 0.0001, f[-1] + 0.00001, 0.00001)]
    x5 = [x[-1] for i in np.arange(min(min(f), min(g)) - 0.0001, f[-1] + 0.00001, 0.00001)]
    l6 = [i for i in np.arange(min(min(f), min(g)) - 0.0001, g[-1] + 0.00001, 0.00001)]
    x6 = [x[-1] for i in np.arange(min(min(f), min(g)) - 0.0001, g[-1] + 0.00001, 0.00001)]
    x3 = [v for v in np.arange(x[0], x[-1] + 0.0001, 0.0001)] 
    s = [min(min(f), min(g)) - 0.0001 for v in x3]  
    diff = 1000
    for ixix in range(len(x)):
        if abs(x[ixix] - 25) < diff:
            diff = abs(x[ixix] - 25) 
            ix = ixix 
    plt.rcParams['font.size'] = 20
    plt.rcParams['font.family'] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"  
    plt.figure(figsize=(10, 7))
    plt.xlabel("$t$")
    plt.ylabel('$' + str_pr[2] + '$')   
    plt.plot(x, f, label = "f", c = "r")
    if f[ix] > g[ix]:  
        plt.text(x[ix], f[ix] + 0.00041, '$f' + str_pr + '$', size=20, color='r')
    else:    
        plt.text(x[ix], f[ix] - 0.00052, '$f' + str_pr + '$', size=20, color='r')
    plt.ylim(min(min(f), min(g)), max(max(f), max(g)) + 0.0001)
    plt.plot(x, g, label = "g", c = "b")
    if g[ix] > f[ix]:  
        plt.text(x[ix], g[ix] + 0.00044, '$g' + str_pr + '$', size=20, color='b')
    else:    
        plt.text(x[ix], g[ix] - 0.00053, '$g' + str_pr + '$', size=20, color='b')
    plt.text(x[ix], (f[ix] + g[ix]) / 2, '$A$', size=20, color='k')
    if f[-1] < g[-1]:
        plt.plot(x1, l1, c = "r")
    else:
        plt.plot(x1, l1, c = "b")
    if f[-1] < g[-1]:
        plt.plot(x6, l6, c = "b")
        plt.plot(x5, l5, c = "r")
    else:
        plt.plot(x5, l5, c = "r")
        plt.plot(x6, l6, c = "b")
    if f[-1] < g[-1]:
        plt.plot(x3, s, c = "r") 
    else:
        plt.plot(x3, s, c = "b") 
    plt.savefig("presentation_plots_" + name + "/" + filename + "_left.png", bbox_inches = "tight")
    plt.close()
    
def plot_all(fx, fy, gx, gy, maxnum, filename):  
    plt.rcParams['font.size'] = 15 
    plt.figure(figsize=(10, 7))
    plt.axis('equal')
    for i in range(len(fx[:maxnum])):  
        plt.plot([fx[i], gx[i]], [fy[i], gy[i]], color='k')
        if i == 0:
            if fy[i] > gy[i]:
                plt.text(fx[i] - 0.0002, fy[i] + 0.00007,'$f_{' + str(i + 1) + '}$', size=15, color='r')
                plt.text(gx[i] - 0.0002, gy[i] - 0.00006, '$g_{' + str(i + 1) + '}$', size=15, color='b')
            else:
                plt.text(fx[i] - 0.0002, fy[i] - 0.0001,'$f_{' + str(i + 1) + '}$', size=15, color='r')
                plt.text(gx[i] - 0.0002, gy[i] + 0.00006, '$g_{' + str(i + 1) + '}$', size=15, color='b')
        else:
            if fy[i] > gy[i]:
                if i < 3:
                    if i % 2:
                        plt.text(fx[i] - 0.00007, fy[i] + 0.00007,'$f_{' + str(i + 1) + '}$', size=15, color='r')
                    else:
                        plt.text(fx[i] + 0.00007, fy[i] + 0.00007,'$f_{' + str(i + 1) + '}$', size=15, color='r')
                else:    
                    plt.text(fx[i], fy[i] + 0.00007,'$f_{' + str(i + 1) + '}$', size=15, color='r')
                plt.text(gx[i], gy[i] - 0.00006, '$g_{' + str(i + 1) + '}$', size=15, color='b')
            else:
                if i < 3:
                    if i % 2:
                        plt.text(fx[i] - 0.00007, fy[i] - 0.0001,'$f_{' + str(i + 1) + '}$', size=15, color='r')
                    else:
                        plt.text(fx[i] + 0.00007, fy[i] - 0.0001,'$f_{' + str(i + 1) + '}$', size=15, color='r') 
                else:    
                    plt.text(fx[i], fy[i] - 0.0001,'$f_{' + str(i + 1) + '}$', size=15, color='r')
                plt.text(gx[i] - 0.0001, gy[i] + 0.00006, '$g_{' + str(i + 1) + '}$', size=15, color='b')
    plt.plot(fx[:maxnum], fy[:maxnum], c = "r")
    plt.plot(gx[:maxnum], gy[:maxnum], c = "b")
    plt.xlabel("$x$")
    plt.ylabel('$y$')  
    plt.xlim(min(min(fx[:maxnum]), min(gx[:maxnum])) - 0.00033, max(max(fx[:maxnum]), max(gx[:maxnum])) + 0.00033)
    plt.ylim(min(min(fy[:maxnum]), min(gy[:maxnum])) - 0.00015, max(max(fy[:maxnum]), max(gy[:maxnum])) + 0.00015)
    plt.savefig("presentation_plots_" + name + "/" + filename + ".png", bbox_inches = "tight")
    plt.close()
 
def plot_original_file_method(lf, long):
    olong, olat, times = load_traj_name(lf)
    olong, olat = preprocess_long_lat(olong, olat)
    olong, olat = scale_long_lat(olong, olat, 0.1, 0.1, True)
    times_processed = [process_time(time_new) for time_new in times] 
    times_processed = [time_new - times_processed[0] for time_new in times_processed] 
    lat = long.replace("long", "lat").replace("x", "y") 
    filename = lf.replace("/cleaned_csv/events", "").replace(".csv", "") + "_" + long
    plot_trapezoid(long_dict[lf][long], olong, times_processed, "_{x}", filename + "_x_trapz")
    plot_trapezoid(lat_dict[lf][lat], olat, times_processed, "_{y}", filename + "_y_trapz")
    plot_all(long_dict[lf][long], lat_dict[lf][lat], olong, olat, 10, filename + "_euclid")

def print_cols(cols, a, b, start_tbl = ""):
    strpr = start_tbl + "\n\t\t"

    for colnum in range(len(cols)):
        strpr += "\multicolumn{2}{|c|}{" + cols[colnum][0] +"}"
        if colnum != len(cols) - 1:
            strpr += " & "
    strpr += " \\\\ \\hline\n\t\t"

    for colnum in range(len(cols)):
        strpr += a + " & " + b
        if colnum != len(cols) - 1:
            strpr += " & "
    strpr += " \\\\ \\hline\n\t\t"

    for rownum in range(len(cols[0][1])):
        for colnum in range(len(cols)):
            strpr += "$" + str_convert(cols[colnum][2][rownum]) + "$ & $" + str_convert(cols[colnum][1][rownum]) + "$"
            if colnum != len(cols) - 1:
                strpr += " & "
        strpr += " \\\\ \\hline\n\t\t"
    
    print(strpr[:-1] + "\\end{tabular}}\n\\end{table}\n")
    
def print_cols_shorter(cols, start_tbl = ""):
    strpr = start_tbl + "\n\t\t"

    for colnum in range(len(cols)):
        strpr += cols[colnum][0]
        if colnum != len(cols) - 1:
            strpr += " & "
    strpr += " \\\\ \\hline\n\t\t"
 
    for rownum in range(len(cols[0][1])):
        for colnum in range(len(cols)):
            strpr += "$" + str_convert(cols[colnum][1][rownum] * 10**5) + "$"
            if colnum != len(cols) - 1:
                strpr += " & "
        strpr += " \\\\ \\hline\n\t\t"
    
    print(strpr[:-1] + "\\end{tabular}}\n\\end{table}\n")

def get_all_for_name(lf, len_tr):  
    longitudes, latitudes, times = load_traj_name(lf)
    longitudes, latitudes = preprocess_long_lat(longitudes, latitudes)
    longitudes, latitudes = scale_long_lat(longitudes, latitudes, 0.1, 0.1, True)
    times_processed = [process_time(time_new) for time_new in times] 

    time_int = [np.round(times_processed[time_index + 1] - times_processed[time_index], 3) for time_index in range(len(times_processed) - 1)]  
    time_no_gap = fill_gap(predicted_time[lf])
    #print_row("predicted_time", time_no_gap[:len_tr], time_int[:len_tr])
 
    longitude_no_abs_int = [np.round(longitudes[longitude_index + 1] - longitudes[longitude_index], 10) for longitude_index in range(len(longitudes) - 1)]
    longitudes_no_abs_no_gap = fill_gap(predicted_longitude_no_abs[lf])
    #print_row("predicted_longitude_no_abs", longitudes_no_abs_no_gap[:len_tr], longitude_no_abs_int[:len_tr])
     
    latitude_no_abs_int = [np.round(latitudes[latitude_index + 1] - latitudes[latitude_index], 10) for latitude_index in range(len(latitudes) - 1)]
    latitudes_no_abs_no_gap = fill_gap(predicted_latitude_no_abs[lf])
    #print_row("predicted_latitude_no_abs", latitudes_no_abs_no_gap[:len_tr], latitude_no_abs_int[:len_tr])
    
    file_with_ride = pd.read_csv(lf)
    directions = list(file_with_ride["fields_direction"]) 
    direction_int = [np.round(direction, 0) for direction in directions]
    direction_no_gap = fill_gap(predicted_direction[lf])
    #print_row("predicted_direction", direction_no_gap[:len_tr], direction_int[:len_tr])
      
    speeds = list(file_with_ride["fields_speed"]) 
    speed_int = [np.round(speed, 0) for speed in speeds] 
    speed_no_gap = fill_gap(predicted_speed[lf])
    #print_row("predicted_speed", speed_no_gap[:len_tr], speed_int[:len_tr])

    list_print = [
                  ["$x$ offset", longitudes_no_abs_no_gap[:len_tr], longitude_no_abs_int[:len_tr]],
                  ["$y$ offset", latitudes_no_abs_no_gap[:len_tr], latitude_no_abs_int[:len_tr]],
                  ["Speed", speed_no_gap[:len_tr], speed_int[:len_tr]],
                  ["Heading", direction_no_gap[:len_tr], direction_int[:len_tr]],
                  ["Time", time_no_gap[:len_tr], time_int[:len_tr]]
                ]

    #print_cols(list_print, "actual", "predicted")

    start_tbl = "\\begin{table}[!t]\n\t\\centering"
    start_tbl += "\n\t\\caption{The first six values from the Markov chain and real values for $x$ and $y$ offset for the shortest test trajectory, using " + translate_name[name] + ".}"
    start_tbl += "\n\t\\label{tab:var_short_sim_long_lat_" + name + "}"
    start_tbl += "\n\t\\resizebox{\\linewidth}{!}{\n\t\\begin{tabular}{|c|c|c|c|}\n\t\t\\hline"

    list_print = [
                  ["$x$ offset", longitudes_no_abs_no_gap[:len_tr], longitude_no_abs_int[:len_tr]],
                  ["$y$ offset", latitudes_no_abs_no_gap[:len_tr], latitude_no_abs_int[:len_tr]],
                ]

    print_cols(list_print, "Actual", "Predicted", start_tbl)
 
    start_tbl = "\\begin{table}[!t]\n\t\\centering"
    start_tbl += "\n\t\\caption{The first six values from the Markov chain and real values for speed, heading, and time for the shortest test trajectory, using " + translate_name[name] + ".}"
    start_tbl += "\n\t\\label{tab:var_short_sim_speed_heading_time_" + name + "}"
    start_tbl += "\n\t\\resizebox{\\linewidth}{!}{\n\t\\begin{tabular}{|c|c|c|c|c|c|}\n\t\t\\hline"

    list_print = [
                  ["Speed", speed_no_gap[:len_tr], speed_int[:len_tr]],
                  ["Heading", direction_no_gap[:len_tr], direction_int[:len_tr]],
                  ["Time", time_no_gap[:len_tr], time_int[:len_tr]]
                ]

    print_cols(list_print, "Actual", "Pred.", start_tbl)

    start_tbl = "\\begin{table}[!t]\n\t\\centering"
    start_tbl += "\n\t\\caption{The first five values (given in $10^{-5}$ degrees longitude) obtained by combining estimated variables compared with the $x$ position for the shortest test trajectory, using " + translate_name[name] + ".}"
    start_tbl += "\n\t\\label{tab:var_long_sim_" + name + "}"
    start_tbl += "\n\t\\resizebox{\\linewidth}{!}{\n\t\\begin{tabular}{|c|c|c|c|c|}\n\t\t\\hline"

    list_print_shorter = [["Original", longitudes[1:len_tr]]]
    list_print_x = []
    #print_method("", "", longitudes[:len_tr])
    for long in long_dict[lf]:
        lat = long.replace("long", "lat").replace("x", "y") 
        list_print_shorter.append([translate_method_short(long + "-" + lat), long_dict[lf][long][1:len_tr]])
        list_print_x.append([translate_method_short(long + "-" + lat), long_dict[lf][long][1:len_tr], longitudes[1:len_tr]])
        #print_method(long, lat, long_dict[lf][long][:len_tr])
        #print_cols([list_print_x[-1]], "predicted x", "actual x")
    print_cols_shorter(list_print_shorter, start_tbl)
    #print_cols(list_print_x, "predicted x", "actual x")
        
    start_tbl = "\\begin{table}[!t]\n\t\\centering"
    start_tbl += "\n\t\\caption{The first five values (given in $10^{-5}$ degrees latitude) obtained by combining estimated variables compared with the $y$ position for the shortest test trajectory, using " + translate_name[name] + ".}"
    start_tbl += "\n\t\\label{tab:var_lat_sim_" + name + "}"
    start_tbl += "\n\t\\resizebox{\\linewidth}{!}{\n\t\\begin{tabular}{|c|c|c|c|c|}\n\t\t\\hline"

    list_print_shorter = [["Original", latitudes[1:len_tr]]]
    list_print = [["Original", longitudes[1:len_tr], latitudes[1:len_tr]]]
    list_print_y = []
    #print_method("", "", latitudes[:len_tr])
    for long in long_dict[lf]:
        lat = long.replace("long", "lat").replace("x", "y") 
        list_print_shorter.append([translate_method_short(long + "-" + lat), lat_dict[lf][lat][1:len_tr]])
        list_print.append([translate_method_short(long + "-" + lat), long_dict[lf][long][1:len_tr], lat_dict[lf][lat][1:len_tr]])
        list_print_y.append([translate_method_short(long + "-" + lat), lat_dict[lf][lat][1:len_tr], latitudes[1:len_tr]])
        #print_method(long, lat, lat_dict[lf][lat][:len_tr])
        #print_cols([list_print[0], list_print[-1]], "x", "y")
        #print_cols([list_print_y[-1]], "predicted y", "actual y")
    print_cols_shorter(list_print_shorter, start_tbl)
    #print_cols(list_print, "x", "y")
    #print_cols(list_print_y, "predicted y", "actual y")

translate_name = {"predicted": "predicted values and two previous states", "predicted_actual": "actual values and two previous states", "predicted_short": "predicted values and one previous state", "predicted_short_actual": "actual values and one previous state"}

for name in translate_name:
    predicted_time = load_object(name + "/predicted_time")   
    predicted_longitude_no_abs = load_object(name + "/predicted_longitude_no_abs")  
    predicted_latitude_no_abs = load_object(name + "/predicted_latitude_no_abs")  
    predicted_direction = load_object(name + "/predicted_direction")   
    predicted_speed = load_object(name + "/predicted_speed")   

    long_dict = load_object("markov_result_" + name + "/long_dict")
    lat_dict = load_object("markov_result_" + name + "/lat_dict")
    distance_predicted = load_object("markov_result_" + name + "/distance_predicted")
                
    min_len = 100000
    lf = "" 
    for long_filename in long_dict:
        for long in long_dict[long_filename]:
            if len(long_dict[long_filename][long]) < min_len:
                min_len = len(long_dict[long_filename][long])
                lf = long_filename
            break

    #print(name, min_len, lf)
    if not os.path.isdir("presentation_plots_" + name):
        os.makedirs("presentation_plots_" + name)
    plot_original_file_method(lf, "long no abs")
    get_all_for_name(lf, 6)