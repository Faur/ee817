
import numpy as np
import matplotlib
matplotlib.use('Agg')   # apparently this prevents figure() to pop up windows
import matplotlib.pyplot as plt
import os
import re

### np.vstack(): along first (0th) axis

# modifiable tuple
class XYTuple():
    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y

    def __getitem__(self, key):
        if key not in [0,1]:
            print("Error: This custom tuple is two-dimensional only.")
            raise NotImplementedError
        return self.x if key==0 else self.y

    def __setitem__(self, key, value):
        if key not in [0,1]:
            print("Error: This custom tuple is two-dimensional only.")
            raise NotImplementedError
        if key==0:
            self.x = value
        else:
            self.y = value




def store_plot(x_y_tuple, multi=False, title="default_title", y_name="y", step_name="steps"):
    x = x_y_tuple[0]
    y = x_y_tuple[1]
    if multi:
        assert len(y.shape) == 2 and y.shape[1]>1
    else:
        assert (len(y.shape) == 2 and y.shape[1]==1) or len(y.shape)==1

    plt.figure(1)
    plt.xlabel(step_name, fontsize=15)
    plt.ylabel(y_name, fontsize=15)
    if multi:
        for i_y in range(y.shape[1]):
            plt.plot(x, y[:,i_y], label=y_name+'_'+str(i_y))
    else:
        plt.plot(x, y, label=y_name)
    plt.legend(loc='best')
    plt.savefig(title+".png")
    plt.close()

# todo: rename "title" to "filepath", but only in the _plot functions
def store_stats_plot(x_y_tuple, percentiles, title="default_stat_title", y_name="y", step_name="steps"):
    x = x_y_tuple[0]
    y = x_y_tuple[1]
    assert len(y.shape) == 2 and y.shape[1] == len(percentiles) * 2 + 1

    plt.figure(1)
    plt.xlabel(step_name, fontsize=15)
    plt.ylabel(y_name, fontsize=15)
    for i_y, perc in enumerate(percentiles[::-1]):
        plt.plot(x, y[:, i_y], label=str(100-perc)+'th percentile')
    plt.plot(x, y[:, len(percentiles)], label=y_name)
    for i_y, perc in enumerate(percentiles):
        plt.plot(x, y[:, -i_y], label=str(perc)+'th percentile')
    plt.title(y_name)
    plt.legend(loc='best')
    plt.savefig(title + ".png")
    plt.close()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~+

class Logger():
    '''
    Fill arrays that are stored in a dictionary, and store them (for now, as plots only) in the end

    Three types of plots:
        single variables, multiple variables, (stored in self.dict, whether single or multi is stored in self.is_2d)
        and stats which have average, and value above which
        80% and 99% of all values lie, and value below which 80% and 99% of all values lie. Or that's the plan.
    '''

    def __init__(self, logdir):

        #self.dict = {}          # store tuples of 1-d arrays (or one 1-d, one 2-d) to be plotted in the end
        self.dict = {}     # same, but will be logged along shorter timespans
        #self.dict_collecting = {}
        self.dict_stats = {}
        self.dict_stats_collecting = {}

        self.is_multi_var = {}     # dict of key-boolean pairs, one for each plot to make in the end
        self.tracked_is_2d = {}

        self.logdir = logdir
        if not os.path.exists(logdir):
            os.makedirs(logdir)

        self.percentiles = [80, 99]
        #self.stats_names = ["99% are above this", "80% are above this", "average", "80% are below this", "99% are below this", "standard deviation"]


    def track(self, variable_dict, step_nr):

        for key in variable_dict:
            val = variable_dict[key]

            # if val is an array, and not a vector or scalar, track statistics instead
            if isinstance(val, np.ndarray) and (len(val.shape) > 1):
                self.track_stats({key: val}, step_nr=step_nr)
                continue

            if isinstance(val, list):
                val = np.array(val)
            if isinstance(val, np.ndarray):
                assert len(val.shape)==1
            else: # it's a scalar
                val = np.array([val])

            if not key in self.dict:
                ismulti = isinstance(val, np.ndarray) and len(val.flatten()) > 1
                self.is_multi_var[key] = ismulti
                val = val[np.newaxis, ...] if ismulti else val        # in case someone wants to create a plot with just one stored value, need to add an axis
                self.dict[key] = XYTuple([step_nr], val)

            else:
                try:
                    self.dict[key][1] = np.vstack((self.dict[key][1], np.array(val)))
                    self.dict[key][0].append(step_nr)
                except:
                    print("Error adding values to existing value array. Make sure it has the right size.")
                    raise



    def store_tracked(self, variable_names, title, step_name="steps"):



        for key in variable_names:
            if key in self.dict_stats:
                self.store_tracked_stats([key], title=title, step_name=step_name)
                continue
            assert key in self.dict and key in self.is_multi_var
            # store plot
            title = re.sub('[^a-zA-Z0-9_\n\-]', '-', title)
            key_ = re.sub('[^a-zA-Z0-9_\n\-]', '-', key)
            store_plot(self.dict[key],  title=self.logdir+"/"+title+"_"+key_, y_name=key, multi=self.is_multi_var[key], step_name=step_name)

            self.dict.pop(key, None)
            self.is_multi_var.pop(key, None)



    # collecting: stack values of many consecutive steps. Only log statistics of such a collection as one data point.
    # It's a sort of averaging, just with storing percentiles as well. Meant for logging gradients.
    def collect(self, variable_dict):

        for key in variable_dict:
            val = variable_dict[key]

            val = np.array(val)

            if not key in self.dict_stats_collecting:
                self.dict_stats_collecting[key] = np.array(val)

            else:
                try:
                    self.dict_stats_collecting[key] = np.vstack((self.dict_stats_collecting[key], np.array(val)))
                except:
                    print("Error adding values to existing value array. Make sure it has the right size.")
                    raise

    # take stats of collected arrays belonging to the variables in "variable_names":
    # calc average and some percentiles and add entries to to self.dict_stats
    def merge_collected(self, variable_names, big_step_nr):

        for key in variable_names:
            assert key in self.dict_stats_collecting
            val = self.dict_stats_collecting[key]

            self.track_stats({key: val}, big_step_nr)



    def track_stats(self, variable_dict, step_nr):

        for key in variable_dict:
            val = variable_dict[key]

            collected = np.zeros((1,len(self.percentiles ) * 2 + 1))
            for k, perc in enumerate(self.percentiles):
                collected[0,k] = np.percentile(val, 100 - perc)
                collected[0,-k] = np.percentile(val, perc)
            collected[0,len(self.percentiles)] = np.mean(val)

            if not key in self.dict_stats:
                self.dict_stats[key] = XYTuple([step_nr], collected)
            else:
                try:
                    self.dict_stats[key][1] = np.vstack((self.dict_stats[key][1], collected))
                    self.dict_stats[key][0].append(step_nr)
                except:
                    print("Error adding values to existing value array. Make sure it has the right size.")
                    raise



    def store_tracked_stats(self, variable_names, title, step_name="steps"):
        for key in variable_names:
            assert key in self.dict_stats
            val = self.dict_stats[key]

            # store plot
            title = re.sub('[^a-zA-Z0-9_\n\-]', '-', title)
            key_ = re.sub('[^a-zA-Z0-9_\n\-]', '-', key)
            store_stats_plot(val, self.percentiles, title=self.logdir+"/"+title+"_"+key_, y_name=key, step_name=step_name)

            self.dict_stats.pop(key, None)


    # create plots for all remaining
    def store_all(self, title, big_step_nr, step_name="steps"):
        #self.merge_collected(self.dict_collecting.keys(), title, step_name=step_name)
        self.merge_collected(self.dict_stats_collecting.keys(), big_step_nr)
        self.store_tracked(self.dict.keys(), title, step_name=step_name)
        self.store_tracked_stats(self.dict_stats.keys(), title=title, step_name=step_name)

    #def log(self, variable_dict, step_nr):
    #    pass








