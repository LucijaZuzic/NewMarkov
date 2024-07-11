from utilities import load_object
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import numpy as np
from sklearn.metrics import r2_score

from sacrebleu.metrics import BLEU

def bleuval(varname, name): 
    all_x = load_object(name + "/predicted_" + varname)
    all_mine = load_object("actual/actual_" + varname)
 
    int_veh = sorted([int(v.split("/")[0].split("_")[1]) for v in all_mine.keys()])

    filenamessorted = []
    for i in set(int_veh):
        for filename in all_x:
            if "Vehicle_" + str(i) + "/cle" in filename:
                filenamessorted.append(filename)

    BLEU_all = []
    for filename in filenamessorted:
        bleu_params = dict(effective_order=True, tokenize=None, smooth_method="floor", smooth_value=0.01)
        bleu = BLEU(**bleu_params)
        pred_str = ""
        actual_str = "" 
        round_val = 10
        if "dir" in varname or "speed" in varname:
            round_val = 0
        if "time" in varname:
            round_val = 3
        for val_ix in range(len(all_mine[filename])):
            pred_str += str(np.round(float(all_x[filename][val_ix]), round_val)) + " "
            actual_str += str(np.round(float(all_mine[filename][val_ix]), round_val)) + " "
        pred_str = pred_str[:-1]
        actual_str = actual_str[:-1]
        blsc = bleu.sentence_score(hypothesis=pred_str, references=[actual_str]).score
        BLEU_all.append(blsc)
    print(varname, np.mean(BLEU_all))

all_subdirs = os.listdir() 

def flatten(all_x, all_mine):
 
    int_veh = sorted([int(v.split("/")[0].split("_")[1]) for v in all_mine.keys()])

    filenamessorted = []
    for i in set(int_veh):
        for filename in all_x:
            if "Vehicle_" + str(i) + "/cle" in filename:
                filenamessorted.append(filename)

    all_flat_x = []
    all_flat_mine = []
    filenames = []
    filenames_length = []
    for filename in filenamessorted:
        for val in all_x[filename]:
            all_flat_x.append(val)
        for val in all_mine[filename]:
            all_flat_mine.append(val)
        filenames.append(filename)
        filenames_length.append(len(all_mine[filename]))
    return all_flat_x, all_flat_mine, filenames, filenames_length

def read_var(varname, name): 
    all_x = load_object(name + "/predicted_" + varname)
    all_mine = load_object("actual/actual_" + varname)
    return flatten(all_x, all_mine)

def plot_rmse(file_extension, var_name, actual, predictions, filenames, filenames_length):
    if not os.path.isdir("rmse"):
        os.makedirs("rmse")
 
    plt.figure(figsize = (20, 6), dpi = 80)
    plt.rcParams.update({'font.size': 22}) 
    plt.plot(range(len(actual)), actual, color = "b") 
    plt.plot(range(len(predictions)), predictions, color = "orange") 
    plt.legend(['Actual', 'Predicted'], loc = "upper left", ncol = 2)
    plt.title("Actual and predicted values\n" + var_name )
    plt.xlabel("Point index")
    plt.ylabel(var_name)
    plt.savefig("rmse/" + name + "/" + file_extension + "_all.png", bbox_inches = "tight")
    plt.close()

    total_len_rides = 0 
    for ix_ride in range(len(filenames_length)):
        len_ride = filenames_length[ix_ride]
        actual_ride = actual[total_len_rides:total_len_rides + len_ride]
        predictions_ride = predictions[total_len_rides:total_len_rides + len_ride]
        total_len_rides += len_ride
        split_filename = filenames[ix_ride].split("/")
        vehicle = split_filename[0].replace("Vehicle_", "")
        ride = split_filename[-1].replace("events_", "").replace(".csv", "") 
        plt.figure(figsize = (20, 6), dpi = 80)
        plt.rcParams.update({'font.size': 22}) 
        plt.plot(range(len(actual_ride)), actual_ride, color = "b") 
        plt.plot(range(len(predictions_ride)), predictions_ride, color = "orange") 
        plt.legend(['Actual', 'Predicted'], loc = "upper left", ncol = 2)
        plt.title("Actual and predicted values\n" + var_name + "\nVehicle " + vehicle + " Ride " + ride)
        plt.xlabel("Point index")
        plt.ylabel(var_name)
        plt.savefig("rmse/" + name + "/" + file_extension + "_Vehicle_" + vehicle + "_Ride_" + ride + "_all.png", bbox_inches = "tight")
        plt.close()
  
translate_name = {"predicted": "predicted values and two previous states", "predicted_actual": "actual values and two previous states", "predicted_short": "predicted values and one previous state", "predicted_short_actual": "actual values and one previous state"}

varnames_title = {"direction": "Heading ($\degree$)", 
                  "latitude_no_abs": "x offset ($\degree$ long.)", 
                  "longitude_no_abs": "y offset ($\degree$ lat.)", 
                  "speed": "Speed (km/h)", 
                  "time": "Time (s)"}

varnames_name = {"direction": "heading", 
                  "latitude_no_abs": "latitude", 
                  "longitude_no_abs": "longitude", 
                  "speed": "speed", 
                  "time": "time"}

dicti_to_print = dict()
dicti_to_print_traj = dict()
for name in translate_name:
    dicti_to_print[name] = dict()
    dicti_to_print_traj[name] = dict()
    dicti_to_print[name]["MAE"] = dict()
    dicti_to_print[name]["MSE"] = dict()
    dicti_to_print[name]["RMSE"] = dict()
    dicti_to_print[name]["NRMSE"] = dict()
    dicti_to_print[name]["R2"] = dict()
    dicti_to_print_traj[name]["Euclid"] = dict()
    dicti_to_print_traj[name]["R2"] = dict()
    dicti_to_print_traj[name]["MAE"] = dict()
    dicti_to_print_traj[name]["MSE"] = dict()
    dicti_to_print_traj[name]["RMSE"] = dict()
    dicti_to_print_traj[name]["R2_wt"] = dict()
    dicti_to_print_traj[name]["MAE_wt"] = dict()
    dicti_to_print_traj[name]["MSE_wt"] = dict()
    dicti_to_print_traj[name]["RMSE_wt"] = dict()

for name in translate_name:
    for varname in varnames_name:
        all_x, all_mine, filenames, filenames_length = read_var(varname, name)
        #plot_rmse(varnames_name[varname], varnames_title[varname], all_mine, all_x, filenames, filenames_length)
        dicti_to_print[name]["MAE"][varname] = mean_absolute_error(all_mine, all_x)
        dicti_to_print[name]["MSE"][varname] = mean_squared_error(all_mine, all_x)
        dicti_to_print[name]["RMSE"][varname] = math.sqrt(dicti_to_print[name]["MSE"][varname])
        dicti_to_print[name]["NRMSE"][varname] = dicti_to_print[name]["RMSE"][varname] / (max(all_mine) - min(all_mine)) * 100
        dicti_to_print[name]["R2"][varname] = r2_score(all_mine, all_x) * 100
        #dicti_to_print[name]["BLEU"] = bleuval(varname, name)

    long_dict = load_object("markov_result_" + name + "/long_dict")
    lat_dict = load_object("markov_result_" + name + "/lat_dict") 
    distance_predicted = load_object("markov_result_" + name + "/distance_predicted")
    #print(distance_predicted["Vehicle_11"]['events_8354807.csv']['euclidean'].keys())
    #print(long_dict['Vehicle_17/cleaned_csv/events_9132790.csv'].keys())
    all_longlats = []

    for vehicle_event in long_dict:  
        for long in long_dict[vehicle_event]:
            lat = long.replace("long", "lat").replace("x", "y")
            all_longlats.append([long, lat])
        break 

    all_subdirs = os.listdir()

    actual_traj = load_object("actual/actual_traj")

    int_veh = sorted([int(v.split("_")[1]) for v in actual_traj.keys()])

    all_time = load_object(name + "/predicted_time")
    pred_time = load_object("actual/actual_time")

    for ix_longlat in range(len(all_longlats)):

        all_actual = []
        all_predicted = []

        actual_long_lat = []
        actual_long_lat_time = []
        predicted_long_lat = []
        predicted_long_lat_time = []

        vals_avg = []
        
        min_k = ""
        min_dist = 1000000
        max_k = ""
        max_dist = - 1000000

        for i in int_veh:  

            all_actual_vehicle = []
            all_predicted_vehicle = []

            subdir_name = "Vehicle_" + str(i)

            val_rides = set()
            if os.path.isfile(subdir_name + "/val_rides"):
                val_rides = load_object(subdir_name + "/val_rides")
                
            for some_file in actual_traj[subdir_name]: 
                
                longitudes, latitudes, is_test = actual_traj[subdir_name][some_file] 

                if is_test == "test":
                    long_pred = long_dict[subdir_name + "/cleaned_csv/" + some_file][all_longlats[ix_longlat][0]]
                    lat_pred = lat_dict[subdir_name + "/cleaned_csv/" + some_file][all_longlats[ix_longlat][1]]
                    time_actual = all_time[subdir_name + "/cleaned_csv/" + some_file]
                    time_pred = pred_time[subdir_name + "/cleaned_csv/" + some_file]
                    
                    vals_avg.append(distance_predicted[subdir_name][some_file]['euclidean'][all_longlats[ix_longlat][0] + "-" + all_longlats[ix_longlat][1]])
                    
                    if distance_predicted[subdir_name][some_file]['euclidean'][all_longlats[ix_longlat][0] + "-" + all_longlats[ix_longlat][1]] < min_dist:
                        min_k = subdir_name + "/cleaned_csv/" + some_file
                        min_dist = distance_predicted[subdir_name][some_file]['euclidean'][all_longlats[ix_longlat][0] + "-" + all_longlats[ix_longlat][1]]

                    if distance_predicted[subdir_name][some_file]['euclidean'][all_longlats[ix_longlat][0] + "-" + all_longlats[ix_longlat][1]] > max_dist:
                        max_k = subdir_name + "/cleaned_csv/" + some_file
                        max_dist = distance_predicted[subdir_name][some_file]['euclidean'][all_longlats[ix_longlat][0] + "-" + all_longlats[ix_longlat][1]]
                    
                    #print(i, some_file, all_longlats[ix_longlat][0], all_longlats[ix_longlat][1])
                    #print(distance_predicted[subdir_name][some_file]['euclidean'][all_longlats[ix_longlat][0] + "-" + all_longlats[ix_longlat][1]])

                    for ix_use_len in range(len(time_actual)):

                        actual_long_lat.append([longitudes[ix_use_len], latitudes[ix_use_len]])
                        actual_long_lat_time.append([longitudes[ix_use_len], latitudes[ix_use_len], time_actual[ix_use_len]])
                        
                        predicted_long_lat.append([long_pred[ix_use_len], lat_pred[ix_use_len]])
                        predicted_long_lat_time.append([long_pred[ix_use_len], lat_pred[ix_use_len], time_pred[ix_use_len]])

                    all_actual_vehicle.append({"long": longitudes, "lat": latitudes})
                    all_predicted_vehicle.append({"long": long_pred, "lat": lat_pred})

                    all_actual.append({"long": longitudes, "lat": latitudes})
                    all_predicted.append({"long": long_pred, "lat": lat_pred})
        
        #print(all_longlats[ix_longlat], np.round(np.average(vals_avg), 6))
        #print(min_k, min_dist)
        #print(max_k, max_dist)
        dicti_to_print_traj[name]["Euclid"][all_longlats[ix_longlat][1]] = np.average(vals_avg)

        r2_pred_wt = r2_score(actual_long_lat_time, predicted_long_lat_time)

        mae_pred_wt = mean_absolute_error(actual_long_lat_time, predicted_long_lat_time)

        mse_pred_wt = mean_squared_error(actual_long_lat_time, predicted_long_lat_time)

        rmse_pred_wt = math.sqrt(mse_pred_wt)

        r2_pred = r2_score(actual_long_lat, predicted_long_lat)

        mae_pred = mean_absolute_error(actual_long_lat, predicted_long_lat)

        mse_pred = mean_squared_error(actual_long_lat, predicted_long_lat)

        rmse_pred = math.sqrt(mse_pred)
        
        dicti_to_print_traj[name]["R2"][all_longlats[ix_longlat][1]] = r2_pred * 100
        dicti_to_print_traj[name]["MAE"][all_longlats[ix_longlat][1]] = mae_pred
        dicti_to_print_traj[name]["MSE"][all_longlats[ix_longlat][1]] = mse_pred
        dicti_to_print_traj[name]["RMSE"][all_longlats[ix_longlat][1]] = rmse_pred
        dicti_to_print_traj[name]["R2_wt"][all_longlats[ix_longlat][1]] = r2_pred_wt
        dicti_to_print_traj[name]["MAE_wt"][all_longlats[ix_longlat][1]] = mae_pred_wt
        dicti_to_print_traj[name]["MSE_wt"][all_longlats[ix_longlat][1]] = mse_pred_wt
        dicti_to_print_traj[name]["RMSE_wt"][all_longlats[ix_longlat][1]] = rmse_pred_wt
        #print("R2", np.round(r2_pred * 100, 2), "MAE", np.round(mae_pred, 6), "RMSE", np.round(rmse_pred, 6), "R2_wt", np.round(r2_pred_wt * 100, 2), "MAE_wt", np.round(mae_pred_wt, 6), "RMSE_wt", np.round(rmse_pred_wt, 6))

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

translate_var_new = {
             "direction": "Heading",  
             "latitude_no_abs": "$y$ offset",  
             "longitude_no_abs": "$x$ offset",   
             "time": "Time",
             "speed": "Speed", 
             }
ord_metric = {"NRMSE": "NRMSE (\\%)", "RMSE": "RMSE", "R2": "$R^{2}$", "MAE": "MAE"}
for name in dicti_to_print:
    strpr = "\\begin{table}[!t]\n\t\\centering"
    strpr += "\n\t\\caption{The NRMSE (\\%), RMSE, $R^{2}$ (\\%), and MAE for each estimated variable in the testing dataset, using " + translate_name[name] + ".}"
    strpr += "\n\t\\label{tab:varmetric_" + name + "}"
    strpr += "\n\t\\resizebox{\\linewidth}{!}{\n\t\\begin{tabular}{|c|c|c|c|c|}\n\t\t\\hline\n\t\tVariable"
    for metric in ord_metric:
        strpr += " & " + ord_metric[metric]
    strpr += " \\\\ \\hline"
    for varname in dicti_to_print[name]["R2"]:
        strpr += "\n\t\t" + translate_var_new[varname]
        for metric in ord_metric:
            strpr += " & $" + str_convert(dicti_to_print[name][metric][varname]) + "$"
        strpr += " \\\\ \\hline"
    strpr += "\n\t\\end{tabular}}\n\\end{table}\n"
    print(strpr)

translate_method_new = {
        "lat no abs": "$x$ and $y$",  
        "lat speed dir": "Time", 
        "lat speed ones dir": "1s", 
        "lat speed actual dir": "Actual time", 
    }
ord_metric = {"Euclid": "Euclid", "RMSE": "RMSE", "R2": "$R^{2}$", "MAE": "MAE"}
for name in dicti_to_print_traj:
    strpr = "\\begin{table}[!t]\n\t\\centering"
    strpr += "\n\t\\caption{The average Euclidean distance, RMSE, $R^{2}$ (\\%), and MAE for each trajectory estimation method, using " + translate_name[name] + ".}"
    strpr += "\n\t\\label{tab:trajmetric_" + name + "}"
    strpr += "\n\t\\resizebox{\\linewidth}{!}{\n\t\\begin{tabular}{|c|c|c|c|c|}\n\t\t\\hline\n\t\tMethod"
    for metric in ord_metric:
        strpr += " & " + ord_metric[metric]
    strpr += " \\\\ \\hline"
    for methodname in dicti_to_print_traj[name]["R2"]:
        strpr += "\n\t\t" + translate_method_new[methodname]
        for metric in ord_metric:
            strpr += " & $" + str_convert(dicti_to_print_traj[name][metric][methodname]) + "$"
        strpr += " \\\\ \\hline"
    strpr += "\n\t\\end{tabular}}\n\\end{table}\n"
    print(strpr)
    
ord_metric = {"RMSE_wt": "RMSE", "R2_wt": "$R^{2}$", "MAE_wt": "MAE"}
for name in dicti_to_print_traj:
    strpr = "\\begin{table}[!t]\n\t\\centering"
    strpr += "\n\t\\caption{The RMSE, $R^{2}$ (\\%), and MAE for each trajectory estimation method combined with timestamps, using " + translate_name[name] + ".}"
    strpr += "\n\t\\label{tab:trajmetric_wt_" + name + "}"
    strpr += "\n\t\\resizebox{\\linewidth}{!}{\n\t\\begin{tabular}{|c|c|c|c|c|}\n\t\t\\hline\n\t\tMethod"
    for metric in ord_metric:
        strpr += " & " + ord_metric[metric]
    strpr += " \\\\ \\hline"
    for methodname in dicti_to_print_traj[name]["R2"]:
        strpr += "\n\t\t" + translate_method_new[methodname]
        for metric in ord_metric:
            strpr += " & $" + str_convert(dicti_to_print_traj[name][metric][methodname]) + "$"
        strpr += " \\\\ \\hline"
    strpr += "\n\t\\end{tabular}}\n\\end{table}\n"
    print(strpr)