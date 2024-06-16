# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 17:31:39 2023

@author: micha
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 10:25:31 2023

@author: micha
"""
# import necesary packages


# label the columns in the data
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
from scipy.signal import find_peaks as fp
import glob
column_labels = ['years', 'months', 'days', 'precipitation']

data = glob.glob(
    "C:/Users/micha/OneDrive/Desktop/Rainfall Analysis Publication/zones/Sudan/*.txt")

datax = pd.DataFrame()
for file in data:
    df = pd.read_fwf(file, names=column_labels)
    df['date'] = pd.to_datetime(df[['years', 'months', 'days']])

    # Calculate the climatological daily mean
    climatology_daily_mean = cdm = df.groupby(df.date.dt.dayofyear)[
        'precipitation'].mean()

    # Calculate the climatological yearly mean
    climatological_year_mean = cym = climatology_daily_mean.mean()

    # Calculate the precipitation anomaly as the difference between the climatological daily mean precipitation
    # and the climatological-year mean precipitation
    climatological_anomaly = anomaly = cdm - cym

    # Find the cumulative of the the anomalies
    cumulative_anomaly = anomaly.cumsum()
    cumulative_anomaly.name = file

    if i == 0:
        datax = cumulative_anomaly
    else:
        datax = pd.concat([datax, cumulative_anomaly], axis=1)

#datax.to_csv("C:/Users/micha/OneDrive/Desktop/Rainfall Analysis Publication/text files/climatological cumulative/sudan.txt", sep='\t', index=True)


# info = np.genfromtxt(
#     "C:/Users/micha/OneDrive/Desktop/Rainfall Analysis Publication/text files/climatological cumulative/sudan.txt")

# days = info[:, 0]


# plot the chart

# create the figure

# fig = plt.figure()
# plt.minorticks_on()
# plt.grid(b=True, which='both', axis='both', color='#666666', linestyle='-')

# # labelling the diagram
# plt.title("SUDAN")
# plt.xlabel("Days")
# plt.ylabel("Cumulative anomaly")


# # making the plot
# ax = plt.subplot()
# ax.plot(cumulative_anomaly)
# # ax.plot(cumulative_anomaly.iloc[:190].idxmin(
# # ), cumulative_anomaly[cumulative_anomaly.iloc[:190].idxmin()], 'ro')
# # ax.plot(cumulative_anomaly.iloc[190:320].idxmin(
# # ), cumulative_anomaly[cumulative_anomaly.iloc[190:320].idxmin()], 'ro')

# anomaly = cumulative_anomaly.values
# dx = 28
# onset = []
# cessation = []
# for i in range(dx, len(anomaly)-dx):
#     if min(anomaly[i-dx:i+dx]) == anomaly[i]:
#         onset.append([i+1, anomaly[i+1]])
#     if max(anomaly[i-dx:i+dx]) == anomaly[i]:
#         cessation.append([i+1, anomaly[i+1]])
# onset = np.copy(onset).T
# cessation = np.copy(cessation).T

# ax.plot(onset[0], onset[1], 'ro')
# ax.plot(cessation[0], cessation[1], 'bo')


# max_index = cumulative_anomaly.idxmax()
# min_index = cumulative_anomaly.idxmin()

# #a['Zuarungu'] = cumulative_anomaly
# #print(onset, cessation)
# # writing the cumulative anomaly into a file


# # saving the images
# #plt.savefig("C:/Users/micha/OneDrive/Desktop/Rainfall Analysis Publication/images/climatological zones/Sudan.jpg")
