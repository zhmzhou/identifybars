# /usr/local/bin/python3.8
import os
from astropy import table
from astropy.io import fits
import numpy as np
import glob
from scipy import signal
import ellipsefits

####################################
def read_ellfits(ellfits):
    # efit=table.Table.read(ellfits)
    dataDict = ellipsefits.ReadEllipse(ellfits)
    efit = table.Table(dataDict.data, names=dataDict.colNames)
    return efit

####################################
def getparcat(fn):
    parcat = table.Table.read('~/data/SAG/SGA-2020.fits',1)
    imname = os.path.split(fn)[-1]
    galaxyname = imname[0:imname.find('-largegalaxy')]
    galaxyid = np.argwhere(parcat['GALAXY'] == galaxyname)
    try:
        sma_sb225 = parcat['SMA_SB22.5'][galaxyid[0][0]]
    except:
        sma_sb225 = 100
    return sma_sb225

####################################
def detectbar_m2(efit, pixscale=0.262, zp=22.5, sma_sb225=30):
    ###### M2a
    barindex1 = 0
    isoindex = 1
    bar_ell1, bar_pa1, bar_sma1, paoff1, pavar1, eloff1 = -99, -99, -99, -99, -99, -99

    # select half of radius
    # sid2 = ~np.isnan(ellperr)
    sid2 = (efit['grad_err'] != 0.0) & (efit['ellip_err'] != 0.0)
    sid = sid2.nonzero()[0]
    # print('sid2.sum()= ',sid2.sum())
    if (np.size(sid) == 0):
        return isoindex, barindex1, bar_ell1, bar_pa1, bar_sma1, paoff1, pavar1, eloff1
    
    if sid[0]<5: isoindex = 0
    efit = efit[sid]
    sma = efit['sma'] * pixscale
    ellp = efit['ellip']
    pa = efit['pa']

    nsize = sid2.sum()
    k = range(nsize - 1)
    # did = [i+1 for i in k]
    dellp = [(ellp[i + 1] - ellp[i]) for i in k]
    dpa = [(pa[i + 1] - pa[i]) for i in k]

    ellp_end = np.median(ellp[-5:-1])
    if ellp_end > 0.66:
        return isoindex, barindex1, bar_ell1, bar_pa1, bar_sma1, paoff1, pavar1, eloff1

    # find peaks in ellip profile
    peakind = signal.find_peaks(ellp, height=0., prominence=0.005, distance=5)
    if np.size(peakind[0]) == 0:
        return isoindex, barindex1, bar_ell1, bar_pa1, bar_sma1, paoff1, pavar1, eloff1

    peaks = ((peakind[0] > 3) * (peakind[0] < np.size(efit['sma']) - 5))
    # peakse = peakind[1]['peak_heights'] == np.max(peakind[1]['peak_heights'])
    peakse = (peakind[1]['peak_heights'] == np.max(peakind[1]['peak_heights'])) & (peakind[0]>3)
    peaks = peaks + peakse

    if peakind[0][peaks].size <= 0:
        return isoindex, barindex1, bar_ell1, bar_pa1, bar_sma1, paoff1, pavar1, eloff1
        
    paoff=[]
    pavar=[]
    eloff=[]
    elvar=[]
    paend=[]
    elend=[]
    barsma=[]
    barindex=[]
    #for j,i in enumerate(peakind[0]):
    for j,i in zip(peaks.nonzero()[0], peakind[0][peaks]):
        index=1
        print(j,i)
        
        #i=peakind[0][j]
        barleft1 = (sma<=sma[i]/3).nonzero()[0][-1] if (sma<=sma[i]/3).sum()>0 else 0
        barleft2 = barleft1 if peakind[0][j]-peakind[1]['left_bases'][j]<5 else peakind[1]['left_bases'][j]#
        barleft = np.max([barleft1, barleft2, peakind[0][j]-30])
        # print(barleft1, barleft2)
        try:
            if peakind[1]['right_bases'][j] < peakind[0][j+1]:
                peakright = peakind[1]['right_bases'][j]
            else:
                peakright = np.max([peakind[0][j+1], peakind[0][j]+10])
        except: 
            peakright = peakind[1]['right_bases'][j]
        #barright = np.min([int(i*1.5+1),np.size(sma)-1]) if sma[i]<sma[-1]*2/3 else np.size(sma)-1
        barright = np.min([i+15, int(i*1.5+1),np.size(sma)-1])
        barright = barright if peakright-peakind[0][j]<=10 else int(peakright)
        #barright = np.min([i+15, peakright, barright])
        
        print('===>', i, barleft, barright, peakright)
        paoff0=np.abs(np.array(pa[i+1:barright+1]-pa[i-3:i+1][:, None])).max()
        pavar0=np.abs(np.array(pa[barleft:i-3])-pa[i-3]).max()
        elloff0=np.array(ellp[i]-ellp[i+1:barright+1]).max()
        ellvar0=np.array(ellp[i]-ellp[barleft:i]).max()
        num_dellp = (np.array(dellp[barleft:i]) < -0.001).sum()
        
        paend0 = pa[i]
        elend0 = ellp[i]
        barsma0 = sma[i]
        paoff.extend([paoff0])
        pavar.extend([pavar0])
        eloff.extend([elloff0])
        elvar.extend([ellvar0])
        paend.extend([paend0])
        elend.extend([elend0])
        barsma.extend([barsma0])
        
        el1 = elloff0 > 0.05
        el2 = (elend0 >= 0.25)\
            & (elend0 - peakind[1]['peak_heights'].max()<0.1)
        el3 = (ellvar0 >= 0.01)\
            & (num_dellp<5)\
            | ((elend0>0.5) & (elloff0>0.2))
        if (np.array(dellp[np.max([i-3,0]):i]).max()>0.2): index -= 2
    
        # pa1 = (paoff0 >= 10) & (paoff0 <= 170) & ((paoff0 > pavar0*1.5)|(pavar0>170))
        pa1 = (paoff0 >= 10) & (paoff0 <= 170) & ((paoff0 > pavar0*1.5)|(pavar0>170))
        pa1 = pa1 | ((paoff0 > pavar0*1.5) & (elloff0>0.2)) |((elend0>0.5)&(elloff0>0.2))
        pa2 = (pavar0 <= 21) |(pavar0>170) | (paoff0 > 2.5*pavar0) |((elend0>0.5)&(elloff0>0.2))
        sma1 = (barsma0 > 2.0)  #& (barsma0 < sma[-1]*2/3)
        if (elend0<0.36) & (barsma0>sma_sb225): index -= 2
        if (peakind[1]['prominences'][j] == peakind[1]['prominences'].max()) & (elend0>0.36): index += 1
        #if elend0 >= 0.25
        
        # index += np.array([(el1 & el2 & el3)*3 , (pa1 & pa2)*2 , sma1*1]).sum()
        index += np.array([el1, el2, el3, pa1, pa2, sma1]).sum()
        print(el1,el2, el3, pa1, pa2, sma1)
    
        barindex.extend([index])
    # pid = (barindex == np.max(barindex)).nonzero()[0][-1] 
    pid_maxi = (barindex == np.max(barindex)).nonzero()[0]
    if np.size(pid_maxi)>1:
        #pid_sma = np.array(barsma)[pid_maxi] < sma[-1]/2.
        pid_maxh = (peakind[1]['prominences'][pid_maxi] == np.max(peakind[1]['prominences'][pid_maxi])).nonzero()[0][0]
        pid = pid_maxi[pid_maxh]
        if (barsma[pid] > sma[-1]*0.66) & (np.max(peakind[1]['prominences'][pid_maxi]) < 0.1):
            pid = pid_maxi[0]
    else:
        pid = pid_maxi[0]
    barindex1 = barindex[pid]
    bar_ell1 = elend[pid]
    bar_pa1 = paend[pid]
    bar_sma1 = barsma[pid]
    paoff1 = paoff[pid]
    pavar1 = pavar[pid]
    eloff1 = eloff[pid]
    elvar1 = elvar[pid]
    return isoindex, barindex1, bar_ell1, bar_pa1, bar_sma1, paoff1, pavar1, eloff1

# bar ##############################
def bar_collect(filenames, cat='fitresult_barp.fits', ifplot=True):
    checking_path = '/nfsdata/users/zmzhou/SAG/checking/'
    # parcat = table.Table.read('/nfsdata/users/zmzhou/SAG/SGA-2020.fits',1)
    # color_gr = parcat['G_MAG_SB22.5'] - parcat['R_MAG_SB22.5']
    filekeys = ['GALAXY']
    headk3 = ['isoindex', 'barindex1', 'bar_ell1', 'bar_pa1', 'bar_sma1', 'paoff1', 'pavar1', 'eloff1']
    allkeys3 = filekeys + headk3
    vals3 = dict([(k, []) for k in allkeys3])

    for i, fn in enumerate(filenames):
        print('Reading', (i + 1), 'of', len(filenames), ':', fn)
        imname = os.path.split(fn)[-1]
        galaxyname = imname[0:imname.find('-largegalaxy')]
        vals3['GALAXY'].append(galaxyname)
        efit = read_ellfits(fn)
        # select criterions
        sma_sb225 = getparcat(fn)
        result = detectbar_m2(efit, sma_sb225=sma_sb225)

        for i, k in enumerate(headk3):
            vals3[k].append(result[i])
        # plot
        if ifplot:
            radir = os.path.split(os.path.split(fn)[0])[-1]
            figfile = imname[0:imname.rfind('.')] + '.png'
            workdir = os.path.join(checking_path, radir)
            if not os.path.exists(workdir):
                os.mkdir(workdir)
            figname = os.path.join(workdir, figfile)
            if not os.path.isfile(figname):
                plot = plotellipse(efit, figname=figname)

    T3 = table.Table(vals3)
    for k in allkeys3:
        T3.rename_column(k, k.lower().replace('-', '_'))
    T3.write(cat, format='fits', overwrite=True)


###########################
def plotellipse(efit, pix=0.262, zp=22.5, figname='figure.png'):
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    # figname = ellfits[0:ellfits.rfind('.fits')+1]+figform
    if os.path.isfile(figname):
        print(figname, ': File exists')
        return

    # efit=table.Table.read(ellfits)
    sma = efit['sma'] * pix
    intens = efit['intens']
    intmag = -2.5 * np.log10(intens / pix / pix) + zp
    intmagerr = 2.5 / np.log(10) * efit['int_err'] / intens
    intmagerr = np.abs(intmagerr)
    ellp = efit['ellip']
    ellperr = efit['ellip_err']
    pa = efit['pa']
    paerr = efit['pa_err']
    a4 = efit['a4']
    a4err = efit['a4_err']
    b4 = efit['b4']
    b4err = efit['b4_err']
    x0 = efit['x0']
    x0err = efit['x0_err']
    y0 = efit['y0']
    y0err = efit['y0_err']

    fig, ax = plt.subplots(ncols=1, nrows=5, figsize=(4, 6), dpi=120, sharex=True)
    plt.subplots_adjust(left=0.2, right=0.9, bottom=0.1, top=0.95, hspace=0)  # wspace=0
    ax[0].errorbar(sma, intmag, yerr=intmagerr, fmt='o', markeredgecolor='k', markerfacecolor='none', markersize=2,
                   ecolor='k')
    ax[0].set_ylabel(r'$\mathrm{\mu\ (mag/arcsec^{2})}$', fontsize=12)
    ax[0].invert_yaxis()
    ax[1].errorbar(sma, ellp, yerr=ellperr, fmt='o', markeredgecolor='k', markerfacecolor='none', markersize=2,
                   ecolor='k')
    ax[1].set_ylabel(r'$\mathrm{\epsilon}$', fontsize=12)
    ax[2].errorbar(sma, pa, yerr=paerr, fmt='o', markeredgecolor='k', markerfacecolor='none', markersize=2, ecolor='k')
    ax[2].set_ylabel(r'$\mathrm{PA}$', fontsize=12)
    ax[2].set_ylim([np.min(pa), np.max(pa)])
    ax[3].errorbar(sma, a4, yerr=a4err, fmt='o', markeredgecolor='r', markerfacecolor='none', markersize=2, label='A4',
                   ecolor='r')
    ax[3].errorbar(sma, b4, yerr=b4err, fmt='o', markeredgecolor='b', markerfacecolor='none', markersize=2, label='B4',
                   ecolor='b')
    ax[3].set_ylabel(r'$\mathrm{A4,B4}$', fontsize=12)
    ax[3].set_xlabel(r'$\mathrm{R (arcsec)}$', fontsize=12)
    ax[3].set_ylim([-0.5, 0.5])
    ax[3].legend(ncol=2)
    ax[4].plot(sma, x0, 'ro', markersize=2, label='X0')
    ax[4].plot(sma, y0, 'bo', markersize=2, label='Y0')
    ax[4].set_ylabel(r'$\mathrm{X0,Y0}$', fontsize=12)
    ax[4].set_xlabel(r'$\mathrm{R (arcsec)}$', fontsize=12)
    # ax[4].set_ylim([-0.5,0.5])
    ax[4].legend(ncol=2)

    plt.savefig(figname)
    plt.clf()

#################################
def main3():
    fns = '../wangww/result_r/???/*ell_tdump.txt'
    filenames = sorted(glob.glob(fns))
    catname = '../checking/fitresult_barp_m2_1025.fits'
    p = bar_collect(filenames, cat=catname, ifplot=False)

##################################
pip3 = main3()
