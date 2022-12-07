from random import sample
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import scipy.stats 

class MyRandomVariableClass(stats.rv_continuous):
    def __init__(self,  seed=None):
        super().__init__(a=0,  seed=seed)
    def _cdf(self, x, zeta, sigma):
        return 1-(1+zeta/sigma*x)**(-1/zeta)

def F_hat(sample, tau, G):
    nu = 0
    n = sample.size
    F = 0.0
    for x in sample:
        if x > tau:
            nu += 1
            # F += (1-((n - nu)/n)) * (1-(1+((zeta/sigma)*(x - tau)))**(-1/zeta)) + (n-nu)/n
    # print(F)
    F = (1-((n - nu)/n)) * G + (n-nu)/n

    return F
def get_threshold(Gm, Gnm, pts):
    tau = 1
    val = float("inf")
    print("size: ", len(Gm), len(Gnm), len(pts))
    for i in range(len(pts)):
        tmp = pts[i]
        val_tmp = 0.5 * (1-Gm[i]) + 0.5 * (1-Gnm[i])
        # val_tmp = 0.5 * (1-F_hat(pts, tau,)) + 0.5 * (1-Gnm[i])

        if val_tmp < val:
            val = val_tmp
            tau = tmp
    return tau
# if __name__ == "__main__":
def evt(sm, snm):
    # my_rv = MyRandomVariableClass()

    # # get sample_m from the given distribution
    # sample_size_m=700
    # zeta_sample_m=0.03
    # sigma_sample_m=0.05
    # samples_m = my_rv.rvs(size = sample_size_m, zeta=zeta_sample_m, sigma=sigma_sample_m)
    # samples_nm = my_rv.rvs(size = sample_size_m, zeta=0.5, sigma=0.165)
    # sm = samples_m
    # snm = samples_nm

    intersection = np.intersect1d(sm, snm)
    boundary_m = np.min(snm)
    boundary_nm = np.max(sm)
    print(boundary_m, boundary_nm)
    m_tail = np.empty([1,])
    is_empty = True
    for m in sm:
        if (m >= boundary_m):
            if (is_empty):
                m_tail = np.array(m)
                is_empty = False
            else:
                m_tail = np.append(m_tail, m)

    nm_tail = np.empty([1,])
    is_empty = True
    # Snm' = -Snm
    snm_inv = np.empty([1,])
    is_empty = True
    excess = boundary_nm - np.min(snm)
    for sample in snm:
        # print("sample", sample)
        sub = sample - boundary_nm
        if (is_empty):
            snm_inv = np.array(sample - (2 * sub) - excess)
            is_empty = False
        else:
            snm_inv = np.append(snm_inv, sample - (2 * sub) - excess)
    
    snm_inv.flatten()

    # # TODO: del!!
    # tt = snm_inv
    # sm = snm_inv
    # snm_inv = tt
    # for nm in snm_inv:
    for nm in snm:
        # if (nm >= boundary_m):
        if (nm <= boundary_nm):
            if (is_empty):
                nm_tail = np.array(nm)
                is_empty = False
            else:
                nm_tail = np.append(nm_tail, nm)

    fit_rv=MyRandomVariableClass()
    # print(fit_rv.fit(samples))
    # print(scipy.stats.fit(fit_rv, samples))
    # print(fit_rv.shapes)
    zeta_fit_m, sigma_fit_m, loc1, scale1=(fit_rv.fit(m_tail, floc=0, fscale=1)) #default method: MLE ,method='MLE'
    zeta_fit_nm, sigma_fit_nm, loc2, scale2=(fit_rv.fit(nm_tail, floc=0, fscale=1)) #default method: MLE ,method='MLE'
    print(zeta_fit_m, sigma_fit_m, loc1, scale1)
    print(zeta_fit_nm, sigma_fit_nm, loc2, scale2)


    # plot histogram of samples
    # fig, ax1 = plt.subplots()
    # plt.hist(snm_inv, bins=500, color="red", alpha = 0.3)

    
    pts = np.linspace(boundary_m, boundary_nm, 500)
    Gm = fit_rv.cdf(pts, zeta=zeta_fit_m, sigma=sigma_fit_m)
    Gnm = fit_rv.cdf(pts,zeta=zeta_fit_nm,sigma=sigma_fit_nm)
    tau = get_threshold(Gm, Gnm[::-1], pts)

    # plot PDF and CDF of distribution
    plt.legend(prop ={'size':10}) 
    plt.title("evt-modeling fig.")
    plt.hist(sm, bins=500, color="blue", alpha = 0.3)
    plt.hist(snm, bins=500, color="yellow", alpha = 0.3)

    plt.plot(pts, fit_rv.pdf(pts, zeta=zeta_fit_m, sigma=sigma_fit_m), color='red')
    plt.plot(pts, Gm, color='red', label = 'match')
    
    
    plt.plot(pts, fit_rv.pdf(pts,zeta=zeta_fit_nm,sigma=sigma_fit_nm), color='navy')
    plt.plot(pts, Gnm, color='navy', label = 'non-match')
    plt.plot(pts, Gnm[::-1], color='black', label = 'non-match-inv')
    plt.axvline(x=tau)
    

    plt.tight_layout()
    plt.savefig(f"evt.png")
    plt.show()
    print("threshold", tau)
    return tau
# evt(np.array([0.3, 0.55, 0.2, 0.1, 0.3, 0.55, 0.2, 0.15, 0.35, 0.55, 0.2, 0.15, 0.1]), np.array([0.73, 0.65, 0.9, 0.75, 0.5, 0.65, 0.9, 0.7, 0.3, 0.65, 0.25, 0.7, 0.2]))