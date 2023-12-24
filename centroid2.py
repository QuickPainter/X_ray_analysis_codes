from ciao_contrib.runtool import *
import os
import glob
import numpy as np
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import ICRS, Galactic, FK4, FK5
import sys
from matplotlib.patches import Ellipse


def main(evt,radius):
    ra = '17:50:46.8624'
    dec = '-31:16:28.848'
    obsid = 0

    if '23444' in evt:
        obsid = '23444'
        bounds = (1000,14000,29000)
        region = "/Users/caleb/AstroPythonPrograms/Regions/PaperRegion23444_2.txt"

    if '21218' in evt:
        obsid = '21218'
        bounds = (8000,14000,1000)
        region = "/Users/caleb/AstroPythonPrograms/Regions/PaperRegion21218_2.txt"

    if '23443' in evt:
        obsid = '23443'

        bounds = (12000,19000,29000)
        region = "/Users/caleb/AstroPythonPrograms/Regions/PaperBkgRegion23443.txt"

    if '23441' in evt:
        obsid = '23441'
        bounds = (10000,11000,10000)
        region = "/Users/caleb/AstroPythonPrograms/Regions/PaperRegion23441.txt"

    command = f"circle({ra},{dec},{radius}\")"
    with open('temp_reg.txt', 'w') as f:
        f.write(command)  

    if obsid == '23444':
        counts = 100
    if obsid == '21218':
        counts = 313

    dmlist.punlearn()
    dmlist.infile = f'/Users/caleb/astropythonprograms/codes/hr/{obsid}/primary/out1.fits[energy=500:7000][ccd_id=7,sky=region({region})][cols time, energy,ra,dec]'
    dmlist.opt = 'data,clean'
    out = dmlist().split('\n')
    out_np = np.genfromtxt(out,dtype='str').astype('float64')
    start_time = out_np[0,0]
    src_times = out_np[::,0] - start_time
    src_energies = out_np[::,1]
    ras = out_np[::,2]
    decs = out_np[::,3]

    print(ras,decs)
    length = len(ras)
    print(f'len {obsid}',len(src_times))

    centroids = []

    current_time = 0
    i = 0
    for i in range(0,30):
        try:
            print(i)
            ra_batch = ras[i*counts:(i+1)*counts]
            dec_batch = decs[i*counts:(i+1)*counts]
            current_time = src_times[(i+1)*counts]
            print(dec_batch)
            print(current_time)
            centroids.append((np.mean(ra_batch),np.mean(dec_batch)))
        except:
            print("index out of bounds")
            break


    # # closest time after eclipse
    # index = src_times.tolist().index(src_times[min(range(len(src_times)), key = lambda i: abs(src_times[i]-bounds[1]))])
    # print("index",index)
    # current_time = bounds[1]
    # i= 0
    # while current_time < bounds[2]:
    #     try:
    #         ra_batch = ras[index+i*225:index+(i+1)*225]
    #         print(ra_batch)
    #         dec_batch = decs[index+i*225:index+(i+1)*225]
    #         current_time = src_times[index+(i+1)*225]
    #         print(current_time)
    #         centroids.append((np.mean(ra_batch),np.mean(dec_batch)))
    #         i +=1
    #     except:
    #         print("index out of bounds")
    #         current_time =  bounds[0]


    # plt.scatter(x,y)
    # plt.show()
    # print(len(src_energies))

    x = [i[0] for i in centroids]
    y= [i[1] for i in centroids]
    return(x,y)



if __name__ == '__main__':
    evt23443 = sys.argv[1]
    evt23444 = sys.argv[2]
    evt21218 = sys.argv[3]
    evt23441 = sys.argv[4]

    radius = 5

    # x,y = main(evt23443,radius)
    # xmean = np.mean(x)
    # ymean = np.mean(y)
    # plt.scatter(x,y,color='blue',label='23443')
    # plt.scatter(xmean,ymean,color='blue',s=400,label='23443')
    plt.rcParams["figure.figsize"] = (15,12)
    plt.tight_layout()

    plt.rcParams.update({'font.size': 40,
                        'xtick.labelsize' :40 ,
                        'ytick.labelsize' : 40,
                        'figure.figsize' : (15,12),
                        'font.sans-serif': 'Times New Roman',
                        'font.family': 'serif'

})
    plt.figure()

    ax = plt.gca()


    x,y = main(evt21218,radius)

    ra_21218 = 267.6952103
    dec_21218 = -31.2748316

    x = [(i - ra_21218)*3600 for i in x]
    y= [(i - dec_21218)*3600 for i in y]

    ax.scatter(x,y,color='green',label='21218',marker='square',s=150)
    print("21218 len:",len(x))




    xmean = np.mean(x)
    ymean = np.mean(y)

    x_std = np.std(x)
    y_std = np.std(y)


    # ax.scatter(xmean,ymean,color='green',s=300)
    # ellipse = Ellipse(xy=(xmean, ymean), width=3*x_std, height=3*y_std, 
    #                     edgecolor='g', ls='--',fc='None', lw=2)

    ax.errorbar(xmean,ymean,yerr=y_std,xerr=x_std,color='g',linewidth=4,capsize=5)
    # ax.add_patch(ellipse)



    
    x,y = main(evt23444,radius)

    print("23443 len:",len(x))


    ra_23443 = 267.6956190
    dec_23443 =-31.2747685

    x = [(i - ra_23443)*3600 for i in x]
    y= [(i - dec_23443)*3600 for i in y]



    ax.scatter(x,y,color='red',label='23444',s=150)
    xmean = np.mean(x)
    ymean = np.mean(y)

    x_std = np.std(x)
    y_std = np.std(y)
    ax.errorbar(xmean,ymean,yerr=y_std,xerr=x_std,color='r',linewidth=4,capsize=5)


    # ax.scatter(xmean,ymean,color='red',s=300)
    # ellipse = Ellipse(xy=(xmean, ymean), width=3*x_std, height=3*y_std, 
    #                     edgecolor='r',ls='--', fc='None', lw=2)
    # ax.add_patch(ellipse)


    # x,y = main(evt23441,radius)
    # xmean = np.mean(x)
    # ymean = np.mean(y)
    # plt.scatter(x,y,color='black',label='black')

    # plt.scatter(xmean,ymean,color='black',s=400,label='black')


    # if radius == 15:
    # #21218
    #     ra = 46.8599
    #     dec = 29.570
    #     plt.scatter(ra,dec,color='green',marker='*',s=500)

    # # 23443

    #     ra = 46.9651
    #     dec =29.338
    #     plt.scatter(ra,dec,color='blue',marker='*',s=500)

    # # 23444

    #     ra = 46.9479
    #     dec = 29.309
    #     plt.scatter(ra,dec,color='red',marker='*',s=500)

    # if radius == 30:
    #         #21218
    #     ra = (267.6950801 - 267.69526)*3600
    #     dec = (-31.2749221 +  31.27468)*3600
    #     ax.scatter(ra,dec,color='green',marker='*',s=500)

    # # 23443

    #     # ra = 46.9651
    #     # dec =29.338
    #     # plt.scatter(ra,dec,color='blue',marker='*',s=500)

    # # 23444

    #     ra =(267.6955841- 267.69526)*3600
    #     dec = (-31.2748337 +  31.27468)*3600
    #     ax.scatter(ra,dec,color='red',marker='*',s=500)

    plt.rcParams.update({
            'font.sans-serif': 'Times New Roman',
            'font.family': 'serif'
            })

    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    plt.grid()



    ax.set_xlabel(r'$\delta$ RA [arcsec]',fontsize=40)
    ax.set_ylabel(r'$\delta$ DEC [arcsec]',size=40)
    plt.axvline(x=0,ls='--',color='black',linewidth=3)
    plt.axhline(y=0,ls='--',color='black',linewidth=3)
    plt.legend(fontsize=35)
    plt.tick_params(axis='both', which='both',direction='inout',length=15)

    plt.tick_params(which='minor', width=0.75, length=7)
    plt.minorticks_on()

    plt.savefig('better_centroid.png',dpi=199)
    plt.show()