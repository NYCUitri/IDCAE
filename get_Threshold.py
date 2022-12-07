# read from json
import json
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
import scipy.stats 


def GPD_cdf(x, zeta, sigma):
    return 1-(1+zeta/sigma*x)**(-1/zeta)

class MyRandomVariableClass(stats.rv_continuous):
    # def __init__(self,  zeta=0.1, sigma=1, seed=None):
    def __init__(self,  seed=None):
        super().__init__(a=0,  seed=seed)

    def _cdf(self, x, zeta, sigma):

        return GPD_cdf(x, zeta, sigma)

def F_hat(Nu, x, zeta, sigma):
    
    return (1-Nu)*GPD_cdf(x = x, zeta = zeta, sigma = sigma) + Nu



def get_threshold(n_path, an_path, name=""):
    with open(n_path) as jsonfile:
        loss_normal = np.array(json.load(jsonfile))

    with open(an_path) as jsonfile:
        loss_anomaly = np.array(json.load(jsonfile))

    loss_con=np.concatenate((loss_normal, loss_anomaly), axis=None)
    ''' plot loss_normal and loss_anomaly'''
    e=int(min(loss_con))
    d=2.5
    d=(int(max(loss_con))-int(min(loss_con)))/1000

    bins_list=[e]
    i=0
    # upper=int(max(loss_con))
    upper=max(loss_con)
    # upper=50000
    while e<upper:
        e+=d
        bins_list.append(e)
        # print(e)


    draw=0

    if draw:
        fig, ax = plt.subplots(figsize =(10, 7))
        ax.hist(loss_normal, bins = bins_list, color='r', alpha=0.5)
        ax.hist(loss_anomaly, bins = bins_list, color='g', alpha=0.5)
        plt.show()



    ''' GPD estimation '''
    loss_normal = np.sort(loss_normal)
    loss_anomaly = np.sort(loss_anomaly)
    # use GPD to estimate the last 15% of the distribution
    u_normal = loss_normal[int(len(loss_normal)*0.85)]
    u_anomaly = loss_anomaly[int(len(loss_anomaly)*0.15)]

    draw_u = 1
    if draw_u:
        fig, ax1 = plt.subplots()
        ax1.hist((loss_normal),bins=bins_list, color = 'blue', alpha=0.25)
        ax1.hist((loss_normal[loss_normal>=u_normal]),bins=bins_list, color = 'purple', alpha=0.25)
        ax1.hist(loss_anomaly,bins=list(np.array(bins_list)), color = 'orange', alpha=0.25) 
        ax1.hist(loss_anomaly[loss_anomaly<=u_anomaly],bins=list(np.array(bins_list)), color = 'red', alpha=0.25) 
        plt.title(name)
        plt.savefig("./loss_distr/" + name + "_u.png")
        plt.show()


    loss_normal_fit = loss_normal[loss_normal>=u_normal] - u_normal
    loss_anomaly_fit = -(loss_anomaly[loss_anomaly<=u_anomaly] - u_anomaly)
    fit_rv_n=MyRandomVariableClass()
    zeta_fit_n, sigma_fit_n, loc1, scale1=(fit_rv_n.fit(loss_normal_fit, floc=0, fscale=1)) #default method: MLE ,method='MLE'
    fit_rv_a=MyRandomVariableClass()
    zeta_fit_a, sigma_fit_a, loc1, scale1=(fit_rv_a.fit(loss_anomaly_fit, floc=0, fscale=1)) #default method: MLE ,method='MLE'

    ''' calculating threshold t '''
    p_unknown = 0.5
    # dt = (np.max(loss_normal)-np.min(loss_anomaly))/10000
    # t = np.min(loss_anomaly)
    # upper_t = np.max(loss_normal)
    dt = (u_anomaly - u_normal)/10000
    t = u_normal
    upper_t = u_anomaly

    t_star = t
    min_score = 1

    Nu_normal = len(loss_normal[loss_normal>u_normal])/len(loss_normal)
    Nu_anomaly = len(loss_anomaly[loss_anomaly<u_anomaly])/len(loss_anomaly)

    while t < upper_t:

        score = (1 - p_unknown) * (1-F_hat(Nu=Nu_normal, x=t-u_normal, zeta=zeta_fit_n, sigma=sigma_fit_n)) \
            + p_unknown * (1-F_hat(Nu=Nu_anomaly, x=u_anomaly-t, zeta=zeta_fit_a, sigma=sigma_fit_a)) 
        # score = (1 - p_unknown) * (len(loss_normal[loss_normal>t])/len(loss_normal)) \
        #     + p_unknown * (len(loss_anomaly[loss_anomaly<t])/len(loss_anomaly)) 
        # print(F_hat(Nu=Nu_normal, x=t-u_normal, zeta=zeta_fit_n, sigma=sigma_fit_n))
        if min_score > score:
            min_score = score
            t_star = t


        t+=dt


    draw_t = 1
    if draw_t:
        fig, ax1 = plt.subplots()
        ax1.hist((loss_normal),bins=bins_list, color = 'blue', alpha=0.5)
        # ax1.hist((loss_normal[loss_normal>=t_star]),bins=bins_list, color = 'purple', alpha=0.25)
        ax1.hist(loss_anomaly,bins=list(np.array(bins_list)), color = 'red', alpha=0.5) 
        # ax1.hist(loss_anomaly[loss_anomaly<=t_star],bins=list(np.array(bins_list)), color = 'red', alpha=0.25) 
        plt.vlines(x=t_star, ymin=0, ymax=1000)
        plt.title(name)
        plt.savefig("./loss_distr/" + name + "_t.png")
        plt.show()
    return t_star



