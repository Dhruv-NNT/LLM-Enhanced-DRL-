import os
import math
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely import geometry
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from haversine import haversine, Unit


def rescaling(r_min, r_max, m, t_min = 0, t_max = 100):
    """rescaling given the r_min and r_max (current range) and T_min and t_max of target range, and the measurement m """
    delta_r = r_max - r_min
    delta_t = t_max - t_min
    scaled = (m - r_min)*(delta_t)/delta_r + t_min
    return scaled

def generate_transformed_sector():
    df = gpd.read_file('test.geojson') 
    poly = df.geometry[5]
    x, y = poly.exterior.coords.xy
    xx = np.array(x)
    yy = np.array(y)

    wpfile = pd.read_csv('path_Waypoints22feb.csv')
    # print(wpfile.columns)
    wps = pd.DataFrame(wpfile[['NAME','LAT', 'LON']])
    lon = wps[['LON']].to_numpy()
    lat = wps[['LAT']].to_numpy()
    lon = lon.ravel()
    lat = lat.ravel()
    lon_array = np.concatenate((xx, lon))
    lat_array = np.concatenate((yy, lat))

    #setting min and max values for lat and lon myself, not from data references

    sector_X = []
    sector_Y = []
    for i in range(len(x)):
        # x_ = rescaling(min(lon_array), max(lon_array), x[i])
        # y_ = rescaling(min(lat_array), max(lat_array), y[i])
        x_ = rescaling(103, 108, x[i])
        y_ = rescaling(1, 5, y[i])
        sector_X.append(x_)
        sector_Y.append(y_)

    waypoints_X = []
    waypoints_Y = []
    for i in range(len(wps)):
        x_ = rescaling(103, 108, wps.iloc[i]['LON'])
        y_ = rescaling(1, 5,  wps.iloc[i]['LAT'])
        waypoints_X.append(x_)
        waypoints_Y.append(y_)

    return waypoints_X, waypoints_Y, sector_X, sector_Y


def plot_sector():

    waypoints_X, waypoints_Y, sector_X, sector_Y = generate_transformed_sector()
    pathfile = pd.read_csv('../22feb_paths')
    pathfile  = pathfile.drop(['ROUTE_NAME','SID', 'STAR', 'Unnamed: 10'], axis = 1).reset_index(drop = True)
    pathfile = pathfile.drop([])
    
    wpfile = pd.read_csv('path_Waypoints22feb.csv')
    wps = pd.DataFrame(wpfile[['NAME','LAT', 'LON']])
    w = wps['NAME'].to_list()
    namedata = pd.DataFrame(data = zip(w, waypoints_X, waypoints_Y), columns = ['name', 'X', 'Y'])
    namedata = namedata.round(2)

    pathlist = []
    for i in range(len(pathfile)):
        path = []
        for j in range(len(pathfile.loc[i])):
            for k in range(len(namedata)):
                if pathfile.loc[i][j] == namedata.loc[k]['name']:
                    path.append([namedata.loc[k]['X'], namedata.loc[k]['Y']])
        pathlist.append(path)

    fig,ax = plt.subplots(figsize = (12,8))
    ax.plot(sector_X, sector_Y, color = 'black')
    ax.scatter(waypoints_X, waypoints_Y, c='maroon',s =5, alpha = 1)

    for i in range(len(pathlist)):
        x = []
        y = []
        for j in range(len(pathlist[i])):
            x.append(pathlist[i][j][0])
            y.append(pathlist[i][j][1])
        ax.plot(x, y, color = 'blue', alpha  = 0.3)
    # plt.show()
    return fig



# creating individual filenames
def getfilenames(namestring):
    a_ = namestring.split('_')
    part1 = a_[0] +'_'+ a_[1]+ '_' + a_[2]
    part2 = a_[0] +'_'+ a_[1]+ '_' + a_[4]

    return part1, part2

def get_valid_scenario_index(featurefile):
    while True:
        _i = np.random.randint(0, len(featurefile))
        row = featurefile.iloc[_i].to_dict()
        _cond_1 = os.path.exists(f"../allresolvedtrajectories/{row['resolvedflight']}")
        _cond_2 = os.path.exists(f"../allflighttrajectories/{row['resolvedflight']}")
        if _cond_1 and _cond_2:
            return row['filenames'], _i
        


def get_conflict_scenario(featurefile):
    """select a random conflict scenario and related parameters from the featurefile"""
    _name, i = get_valid_scenario_index(featurefile)

    filepath = '../allresolvedtrajectories/'
    filepath2 = '../allflighttrajectories/'
    resolved_csv_list = [file for file in os.listdir(filepath) if file.endswith('.csv')]
    all_unresolved_csv_list = [file for file in os.listdir(filepath2) if file.endswith('.csv')]

    resflight, unresflight = getfilenames(featurefile.iloc[i]['filenames'])
    # print(resflight, unresflight)
    fileinformation = [resflight, unresflight]
 
    o_path_ = pd.read_csv(filepath + resflight)
    p1 = (o_path_.iloc[0]['latitude'], o_path_.iloc[0]['longitude'])
    p2 = (featurefile.iloc[i]['LAT'], featurefile.iloc[i]['LON'])
    conflict_distance_actual_o = haversine(p1, p2, unit = Unit.NAUTICAL_MILES)

    o_path = []
    for j in range(len(o_path_)):
        x_ = rescaling(103, 108, o_path_.iloc[j]['longitude'])
        y_ = rescaling(1, 5, o_path_.iloc[j]['latitude'])
        o_path.append([x_, y_])
    p3 = np.array([o_path[0][0], o_path[0][1]])
    p4 = np.array([featurefile.iloc[i]['lon_T'], featurefile.iloc[i]['lat_T']])
    conflict_distance_scaled_o = np.linalg.norm(p3 - p4)
    step_size_actual = 0.65 #from the data
    
    o_offset = featurefile.iloc[i]['resflight_offset']
    scaled_offset_distance_o = (step_size_actual * o_offset *conflict_distance_scaled_o)/conflict_distance_actual_o

    o_path_resampled = []
    for k in range(0, len(o_path), 6):
        o_path_resampled.append(o_path[k])
    
    if (np.round(o_path_resampled[-1][0], 2) < 20.5) and (np.round(o_path_resampled[-1][1], 2) < 20.5):
        # print('yes')
        o_path_resampled = o_path_resampled[:-13]   # 13 is based on the len of the entire trajectory, manually calculated
    o_startposition = o_path_resampled[0]
    o_offset = int((featurefile.iloc[i]['resflight_offset']*0.65)/7.5)
    o_destination = o_path_resampled[-1]
    o_heading = initial_heading(o_path_resampled[0], o_path_resampled[1])

    resflight_unrestraj_ = []
    resflight_unrestraj = pd.read_csv(filepath2 + resflight)
    for l in range(len(resflight_unrestraj)):
        x_ = rescaling(103, 108, resflight_unrestraj.iloc[l]['longitude'])
        y_ = rescaling(1, 5, resflight_unrestraj.iloc[l]['latitude'])
        resflight_unrestraj_.append([x_, y_])

    o_resflight_unrestraj_resampled = []
    for m in range(0, len(resflight_unrestraj_), 6):
        o_resflight_unrestraj_resampled.append(resflight_unrestraj_[m])
    if (np.round(o_path_resampled[-1][0], 2) < 20.5) and (np.round(o_path_resampled[-1][1], 2) < 20.5):
        o_resflight_unrestraj_resampled = o_resflight_unrestraj_resampled[:-13]
    
    i_path_ = pd.read_csv(filepath2 + unresflight)
    i_path = []
    for n in range(len(i_path_)):
        x_ = rescaling(103, 108, i_path_.iloc[n]['longitude'])
        y_ = rescaling(1, 5, i_path_.iloc[n]['latitude'])
        i_path.append([x_, y_])

    p11 = (i_path_.iloc[0]['latitude'], i_path_.iloc[0]['longitude'])
    p22 = (featurefile.iloc[i]['LAT'], featurefile.iloc[i]['LON']) # this is the conflict waypoint
    conflict_distance_actual_i = haversine(p11, p22, unit = Unit.NAUTICAL_MILES)

    p33 = np.array(i_path[0][0], i_path[0][1])
    p44 = np.array([featurefile.iloc[i]['lon_T'], featurefile.iloc[i]['lat_T']])
    conflict_distance_scaled_i = np.linalg.norm(p33 - p44)
    i_offset = featurefile.iloc[i]['unresflight_offset']

    scaled_offset_distance_i = (step_size_actual*i_offset*conflict_distance_scaled_i)/conflict_distance_actual_i


    #calculate conflict distance actual and scaled for the intruder also, and get the scaled_offset_distance_i. 
    # then , based on the speed we will calculate the offset step in the agent class.
    
    i_path_resampled = []
    for t in range(0, len(i_path), 6):
        i_path_resampled.append(i_path[t])
    if (np.round(i_path_resampled[-1][0], 2) < 20.5) and (np.round(i_path_resampled[-1][1], 2) < 20.5):
        # print('yes')
        i_path_resampled = i_path_resampled[:-13]
    
    i_headinglist = []
    for h in range(0,len(i_path_resampled)-1):
        head = initial_heading(i_path_resampled[h], i_path_resampled[h+1])
        i_headinglist.append(np.round(head, 3)) 
    #  initial_heading(i_path_resampled[0], i_path_resampled[1])
    i_headinglist.append(i_headinglist[-1])
    
    i_startposition = i_path_resampled[0]
    i_offset = int((featurefile.iloc[i]['unresflight_offset']*0.65)/7.5)
    i_destination = i_path_resampled[-1]


    # print('Offet distances', scaled_offset_distance_o, scaled_offset_distance_i)
    return o_path_resampled, o_resflight_unrestraj_resampled ,o_heading, o_startposition, scaled_offset_distance_o,\
          o_destination, i_path_resampled,i_headinglist, i_startposition, scaled_offset_distance_i,\
              i_destination, fileinformation




def get_points(line):
    x,y = [], []
    for i in range(len(line)):
        x.append(line[i][0])
        y.append(line[i][1])
    distance = np.cumsum(np.sqrt( np.ediff1d(x, to_begin=0)**2 + np.ediff1d(y, to_begin=0)**2 ))
    # print(distance)
    if distance[-1] < 1 :
        distance = distance / distance[-2]
        # print('distances from get points', distance)
    else:
        distance = distance/distance[-1]

    fx, fy = interp1d( distance, x ), interp1d( distance, y )

    alpha = np.linspace(0, 1, 500)
    x_regular, y_regular = fx(alpha), fy(alpha)
    transformed_line = []
    for i in range(len(x_regular)):
        transformed_line.append([x_regular[i], y_regular[i]])

    return transformed_line

def get_cumulated_distance(line1, line2):
    distlist = []
    for i in range(len(line1)):
        d = np.linalg.norm(np.array(line1[i]) - np.array(line2[i]))
        distlist.append(d)
    distance_sum = sum(distlist)
    return distance_sum



def initial_heading(location, next_location):
    x1, y1 = location
    x0, y0 = next_location
    heading = 0
    head = math.atan2(y1-y0, x1-x0) #radians
    # print('radians',head)

    if x1> x0 and y1> y0: # first quad 
        heading  = np.pi + head
    if x1 < x0 and y1 > y0 : # second quad
        heading =  np.pi + head
    if x1< x0 and y1< y0:
        heading = np.pi - (abs(head))
    if x1> x0 and y1< y0:
        heading = np.pi - abs(head)
    if x1 == x0 and y1> y0:
        heading = np.pi + head
    if x1 == x0 and y1 < y0:
        heading = abs(head)

    return heading