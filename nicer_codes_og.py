import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import math
import scipy
from scipy import stats
import pandas as pd
import random
import warnings
import statistics
import gzip
import subprocess

import shutil
import re

# from HRSig import *

warnings.simplefilter(action='ignore', category=FutureWarning)

BEHR_DIR = '/Users/caleb/AstroPrograms/BEHR_Container/BEHR'

BINSIZE = 5

def main():
    # set up directory and grab the lcs
    working_directory = os.getcwd()
    path = "/Users/caleb/Downloads/caleb_delivery"
    listOfFiles = getListOfFiles(path)
    cleaned_files = cleaning(listOfFiles)
    # check it is correct amount of files
    print(len(cleaned_files))   
    
    egress_list = ['/Users/caleb/Downloads/egress_NICER_grs1747/2050440106/jspipe/js_ni2050440106_0mpu7_silver_GTI0-v3-bands.lc', '/Users/caleb/Downloads/egress_NICER_grs1747/2050440107/jspipe/js_ni2050440107_0mpu7_silver_GTI2-v3-bands.lc', '/Users/caleb/Downloads/egress_NICER_grs1747/2050440107/jspipe/js_ni2050440107_0mpu7_silver_GTI0-v3-bands.lc', '/Users/caleb/Downloads/egress_NICER_grs1747/2050440105/jspipe/js_ni2050440105_0mpu7_silver_GTI0-v3-bands.lc', '/Users/caleb/Downloads/egress_NICER_grs1747/2050440103/jspipe/js_ni2050440103_0mpu7_radium_GTI0-v3-bands.lc', '/Users/caleb/Downloads/egress_NICER_grs1747/2050440103/jspipe/js_ni2050440103_0mpu7_silver_GTI2-v3-bands.lc', '/Users/caleb/Downloads/egress_NICER_grs1747/2050440104/jspipe/js_ni2050440104_0mpu7_silver_GTI0-v3-bands.lc']
    shortened_list = ['/Users/caleb/Downloads/egress_NICER_grs1747/2050440106/jspipe/js_ni2050440106_0mpu7_silver_GTI0-v3-bands.lc', '/Users/caleb/Downloads/egress_NICER_grs1747/2050440107/jspipe/js_ni2050440107_0mpu7_silver_GTI2-v3-bands.lc', '/Users/caleb/Downloads/egress_NICER_grs1747/2050440107/jspipe/js_ni2050440107_0mpu7_silver_GTI0-v3-bands.lc', '/Users/caleb/Downloads/egress_NICER_grs1747/2050440105/jspipe/js_ni2050440105_0mpu7_silver_GTI0-v3-bands.lc', '/Users/caleb/Downloads/egress_NICER_grs1747/2050440103/jspipe/js_ni2050440103_0mpu7_silver_GTI2-v3-bands.lc']
    shortened_list_bkgs = ['/Users/caleb/Downloads/egress_NICER_grs1747/2050440106/jspipe/js_ni2050440106_0mpu7_silver_GTI0-v3-bands.bg-lc', '/Users/caleb/Downloads/egress_NICER_grs1747/2050440107/jspipe/js_ni2050440107_0mpu7_silver_GTI2-v3-bands.bg-lc', '/Users/caleb/Downloads/egress_NICER_grs1747/2050440107/jspipe/js_ni2050440107_0mpu7_silver_GTI0-v3-bands.bg-lc', '/Users/caleb/Downloads/egress_NICER_grs1747/2050440105/jspipe/js_ni2050440105_0mpu7_silver_GTI0-v3-bands.bg-lc', '/Users/caleb/Downloads/egress_NICER_grs1747/2050440103/jspipe/js_ni2050440103_0mpu7_silver_GTI2-v3-bands.bg-lc']

    # for i in egress_list[1:2]:
    #     passer(i,False)
        
    stacked_times, stacked_counts, stacked_softs, stacked_hards = stacker(shortened_list)
    stacked_times, bkg_stacked_counts, bkg_stacked_softs, bkg_stacked_hards = stacker(shortened_list_bkgs)
    

    stacked_times = [(i*BINSIZE)/1000 for i in stacked_times]


    # plt.step(stacked_times,stacked_counts)
    plt.step(stacked_times,stacked_softs,label='Soft')
    plt.step(stacked_times,stacked_hards,label='Hard')
    plt.xlabel("Time (s)")
    plt.ylabel("Count Rate (counts/s)")

    plt.legend()
    plt.show()

    soft_mean = np.mean(stacked_softs[int(np.ceil(len(stacked_softs)/2)):])
    norm_softs = [i/soft_mean for i in stacked_softs]
    hard_mean = np.mean(stacked_hards[int(np.ceil(len(stacked_hards)/2)):])
    norm_hards = [i/hard_mean for i in stacked_hards]


    plt.step(stacked_times,norm_softs,label='softs')
    plt.step(stacked_times,norm_hards,label='hards')
    plt.xlabel("Time (ks)")
    plt.xlim(160,300)
    plt.ylabel("Normalized Flux")

    plt.legend()
    plt.show()



    plt.rcParams.update({'font.size': 40,
                        'xtick.labelsize' :40 ,
                        'ytick.labelsize' : 40,
                        'figure.figsize' : (15,12),
                        'font.sans-serif': 'Times New Roman',
                        'font.family': 'serif'

    })
    plt.tight_layout()

    plt.figure()


    medians, uppers, lowers = BEHR_passer(stacked_softs, stacked_hards, bkg_stacked_softs, bkg_stacked_hards, "STACKED")
        
        
    pd_medians = pd.DataFrame(medians)
    rolling_average = pd_medians.rolling(window=3).mean()
    rolling_average = rolling_average.values.tolist()

    
    lc_errors_upper = abs(np.array(uppers) - np.array(medians))
    lc_errors_lower = abs(np.array(lowers) - np.array(medians))

    errors = np.column_stack((lc_errors_lower,lc_errors_upper))
    print(errors)
    lc_errors = np.transpose(errors)
    textsize = 40
    
    
    plt.scatter(stacked_times,medians,color='purple')
    plt.plot(stacked_times[2:-2],rolling_average[2:-2],color='black')
    plt.xlabel("Time [ks]",fontsize=textsize)
    plt.ylabel("HR",fontsize=textsize)
    plt.tick_params(axis='both', which='major', labelsize=textsize)

    plt.rcParams.update({
            'font.sans-serif': 'Times New Roman',
            'font.family': 'serif'
            })
    plt.errorbar(stacked_times,medians,
                yerr = lc_errors,
                xerr = None,
                fmt ='.',
                color = "black",
                linewidth = .5,
                capsize = 1)
    
    plt.tick_params(axis='both', which='both',direction='inout',length=15)

    plt.tick_params(which='minor', width=0.75, length=7)
    plt.minorticks_on()

    plt.legend(fontsize=30)
    plt.tight_layout()

    plt.savefig("NICER_HR_stacked")
    plt.show()
    
    

    

def normalizer(l, index):
    pass
        
def stacker(egress_list):
    
    first_times = []
    second_times = []
    
    first_counts = []
    second_counts = []
    
    first_softs = []
    second_softs = []
    
    first_hards = []
    second_hards = []
    
    
    for i in egress_list:
        src_bin_times, src_binned_list,initial_time,end_time, soft_binned_list, medium_binned_list, hard_binned_list, very_hard_binned_list = counts_grabber(BINSIZE,i)

        softs = np.array(soft_binned_list) + np.array(medium_binned_list)
        softs  = softs.tolist()
        hards = hard_binned_list
        
        egress = find_egress(initial_time,end_time)
        src_bin_times = src_bin_times.tolist()
        # softs = softs.tolist()
        # hards = hards.tolist()

        closest_in = min(src_bin_times, key=lambda x:abs(x-egress))
        index_in = src_bin_times.index(closest_in)
        
        first_half_times = src_bin_times[:index_in] 
        second_half_times = src_bin_times[index_in:]
        first_times.append(first_half_times)
        second_times.append(second_half_times)
        
        first_count = src_binned_list[:index_in] 
        second_count = src_binned_list[index_in:]
        first_counts.append(first_count)
        second_counts.append(second_count)
        
        first_soft = softs[:index_in] 
        second_soft = softs[index_in:]
        first_softs.append(first_soft)
        second_softs.append(second_soft)

        first_hard = hards[:index_in] 
        second_hard = hards[index_in:]
        first_hards.append(first_hard)
        second_hards.append(second_hard)

        
        
    shortest = min([len(i) for i in first_counts])
    shortest2 = min([len(i) for i in second_counts])
    
    print(len(first_counts))
    print(shortest)
    first_counts = [i[-shortest:] for i in first_counts]        
    second_counts = [i[:shortest2] for i in second_counts]  
          
    first_softs = [i[-shortest:] for i in first_softs]        
    second_softs = [i[:shortest2] for i in second_softs]        

    first_hards = [i[-shortest:] for i in first_hards]        
    second_hards = [i[:shortest2] for i in second_hards]        
    
    first = np.array(first_counts)
    second = np.array(second_counts)
    
    print(first_softs)
    
    first_softs = np.sum(np.array(first_softs),0)
    second_softs = np.sum(np.array(second_softs),0)
    
    first_hards = np.sum(np.array(first_hards),0)
    second_hards = np.sum(np.array(second_hards),0)


    first_sum = np.sum(first,0)
    second_sum = np.sum(second,0)
    

    combined_sum = np.concatenate((first_sum,second_sum))
    combined_softs = np.concatenate((first_softs,second_softs))
    combined_hards = np.concatenate((first_hards,second_hards))

    times = np.arange(0,len(combined_sum),1)
    print(combined_softs)
    return times,combined_sum, combined_softs, combined_hards

def plotter(bin_times,binned_list,binsize,flag,specific_ingress,specific_egress):
#     print(initial_time)
    fig,(ax1) = plt.subplots(1,1)
    plt.rcParams["figure.figsize"] = (25,10)
    if flag==True:
        print(specific_egress)
        fig.patch.set_facecolor('xkcd:mint green')
#         ax1.axvline(x=specific_ingress,color = 'red',linestyle='dashed')
#         ax1.axvline(x=specific_egress,color = 'red',linestyle='dashed')
    if flag == "HR":
        norm_soft = np.array(binned_list[0])
        norm_medium = np.array(binned_list[1])
        norm_hard = np.array(binned_list[2])
        norm_very_hard = np.array(binned_list[3])
        
        hr_binned = (norm_soft+norm_medium)/norm_hard
        
        # ax1.step(bin_times[:-1],norm_soft,color='gold')
        # ax1.step(bin_times[:-1],norm_medium,color='red')
        # ax1.step(bin_times[:-1],norm_hard,color='purple')
        # ax1.step(bin_times[:-1],norm_very_hard,color='blue')
        ax1.step(bin_times[:-1],hr_binned,color='black')

        ax1.set_title(f"{binsize}s binned lightcurve",fontsize = 10)
        ax1.set_xlabel("Time [ks]")
        ax1.set_ylabel(f"Counts per Bin")
        plt.show()
        
        binned_HR = (norm_soft+norm_medium)/(norm_very_hard+norm_hard)
        plt.step(bin_times[:-1],binned_HR)
        plt.show()
        return

    ax1.step(bin_times[:-1],binned_list)
    ax1.set_title(f"{binsize}s binned lightcurve",fontsize = 10)
    ax1.set_xlabel("Time [ks]")
    ax1.set_ylabel(f"Counts per Bin")
#     ax1.set_xlim(.6,.7)
#     ax1.set_ylim(0,100)

    plt.show()
    
    
def passer(i,override):
    print(i)
    file_name = i
    binsize = 10
    bkg_lightcurve = re.sub(r'lc$', 'bg-lc', i)
    src_bin_times, src_binned_list,initial_time,end_time, soft_binned_list, medium_binned_list, hard_binned_list, very_hard_binned_list = counts_grabber(binsize,i)
    bkg_bin_times, bkg_binned_list,initial_time,end_time, bkg_soft_binned_list, bkg_medium_binned_list, bkg_hard_binned_list, bkg_very_hard_binned_list = counts_grabber(binsize,bkg_lightcurve)
    arr_src = np.array(src_binned_list)
    arr_bkg = np.array(bkg_binned_list)
    arr_diff = np.subtract(arr_src, arr_bkg)
    binned_list = arr_diff.tolist()
    
    arr_diff_soft = np.array(soft_binned_list) - np.array(bkg_soft_binned_list)
    arr_diff_medium = np.array(medium_binned_list) - np.array(bkg_medium_binned_list)
    arr_diff_hard = np.array(hard_binned_list) - np.array(bkg_hard_binned_list)
    arr_diff_very_hard = np.array(very_hard_binned_list) - np.array(bkg_very_hard_binned_list)

#     print(initial_time)


    # BEHR STUFF
    print("BEHR")
    soft_band_src = np.array(soft_binned_list) + np.array(medium_binned_list)
    soft_band_bkg = np.array(bkg_soft_binned_list) + np.array(bkg_medium_binned_list)
    hard_band_src = hard_binned_list
    hard_band_bkg = bkg_hard_binned_list
    
        
    
    obsid_GTI = i.split("/")[-1]
    obsid_GTI = obsid_GTI[0:-3]
    
    
    medians, uppers, lowers = BEHR_passer(soft_band_src, hard_band_src, soft_band_bkg, hard_band_bkg, obsid_GTI)
        
        
    pd_medians = pd.DataFrame(medians)
    rolling_average = pd_medians.rolling(window=3).mean()
    rolling_average = rolling_average.values.tolist()

    
    lc_errors_upper = abs(np.array(uppers) - np.array(medians))
    lc_errors_lower = abs(np.array(lowers) - np.array(medians))

    errors = np.column_stack((lc_errors_lower,lc_errors_upper))
    print(errors)
    lc_errors = np.transpose(errors)
    textsize = 30
    
    
    plt.scatter(src_bin_times[1:],medians,color='purple')
    plt.plot(src_bin_times[3:-2],rolling_average[2:-2],color='black')
    plt.xlabel("Time (ks)",fontsize=textsize)
    plt.ylabel("HR",fontsize=textsize)
    plt.tick_params(axis='both', which='major', labelsize=textsize)

    plt.rcParams.update({
            'font.sans-serif': 'Times New Roman',
            'font.family': 'sans-serif'
            })
    plt.title(f"HR -- {obsid_GTI}")
    plt.errorbar(src_bin_times[1:],medians,
                yerr = lc_errors,
                xerr = None,
                fmt ='.',
                color = "black",
                linewidth = .5,
                capsize = 1)
    plt.show()
    
    
    bkg_errors = [.25*x for x in arr_bkg]
#     print(bkg_errors)
    src_errors = [np.sqrt(x) for x in arr_src]
#     print(src_errors)
    tot_errors = []
    for i in range(0,len(src_errors)):
        tot = np.sqrt(src_errors[i]**2+bkg_errors[i]**2)
        tot_errors.append(tot)

    
#     plt.step(src_bin_times[1:],binned_list,color='black')
#     plt.errorbar([x-(.5*binsize)/1000 for x in src_bin_times[1:]],binned_list, yerr=tot_errors, fmt ='.',color = "black",linewidth = 1,capsize = 1)
#     plt.xlabel("Time [ks]",size=15,labelpad=20)
#     plt.ylabel("Counts/Bin",size=15,labelpad=20)
#     plt.xticks(size=15)
#     plt.yticks(size=15)
#     obs=file_name.split("/")[5]
#     gti=file_name.split("/")[-1].split("_")[-1][:4]
#     plt.title(f"Observation {obs} -- {gti}",size=15)
#     plt.show()
    
#     return src_bin_times[1:], binned_list, tot_errors
    
    flag = False
    specific_ingress = 0
    specific_egress = 0
    for n in range(0,500):
        ingress = 168351205.343 + 44494.2981792*n - 600
        egress = 168353768.693 + 44494.29576*n - 600
        if (initial_time < ingress < end_time) or (initial_time < egress < end_time) or ingress<initial_time<egress:
            flag = True
            print("N value", n)
            specific_ingress = (ingress - initial_time)/1000
            specific_egress = (egress - initial_time)/1000
        
            plotter(src_bin_times,binned_list,binsize,flag,specific_ingress,specific_egress)
            flag = "HR"
            all_hr = (arr_diff_soft,arr_diff_medium,arr_diff_hard,arr_diff_very_hard)
            # plotter(src_bin_times,all_hr,binsize,flag,specific_ingress,specific_egress)

            break
    
#     plotter(src_bin_times,binned_list,binsize,flag,specific_ingress,specific_egress)
    
    return src_bin_times, binned_list,tot_errors

def BEHR_passer(soft_band_src, hard_band_src, soft_band_bkg, hard_band_bkg,obsid_GTI):
    
    BEHR_outdir = f'{BEHR_DIR}/{obsid_GTI}'
    WORKING_DIR = f'/Users/caleb/Downloads/caleb_delivery/2050440105/jspipe/{obsid_GTI}'
    try:
        os.mkdir(WORKING_DIR)
    except:
        pass

    outfile = f'{WORKING_DIR}/BEHR_bash.txt'


    medians = []
    uppers = []
    lowers = []
    for i in range(0,len(soft_band_src)):
        med,upper,lower = block_med(BEHR_DIR,soft_band_src[i],hard_band_src[i],soft_band_bkg[i],hard_band_bkg[i],outfile,BEHR_outdir)
        medians.append(med)
        uppers.append(upper)
        lowers.append(lower)
        
    return medians, uppers, lowers

def find_egress(initial_time,end_time):
    specific_egress = 0
    for n in range(0,500):
        ingress = 168351205.343 + 44494.2981792*n - 600
        egress = 168353768.693 + 44494.29576*n - 600
        if (initial_time < ingress < end_time) or (initial_time < egress < end_time) or ingress<initial_time<egress:
            specific_egress = (egress - initial_time)/1000
    
    return specific_egress
    
    
def block_med(BEHR_DIR,soft_src,hard_src,soft_bkg,hard_bkg,outfile,BEHR_outdir):
    confidence = '68.0'
    
    hard_area = 1
    soft_area = 1
    
    subprocess.run(f'rm -rf {BEHR_outdir}',shell=True)
    os.makedirs(BEHR_outdir)


    
    print('blabla')
    print(f'softsrc={soft_src} hardsrc={hard_src}   softbkg={soft_bkg}   hardbkg={hard_bkg}    softarea={soft_area} hardarea={hard_area} output={BEHR_outdir}/block_BEHRresults level={confidence}')
    with open(outfile,'w') as writeto:
        writeto.write(f'cd {BEHR_DIR}')
        writeto.write(f'\n echo "softsrc={soft_src} hardsrc={hard_src}   softbkg={soft_bkg}   hardbkg={hard_bkg}"')
        writeto.write(f'\n./BEHR softsrc={soft_src} hardsrc={hard_src}   softbkg={soft_bkg}   hardbkg={hard_bkg}   softarea={soft_area} hardarea={hard_area} output={BEHR_outdir}/block_BEHRresults level={confidence}')

    sig_run_BEHR(outfile)
    
    file = f'{BEHR_outdir}/block_BEHRresults.txt'
    med,upper,lower = readfile(file)
    
    return round(float(med),3),round(float(upper),3),round(float(lower),3)

def sig_run_BEHR(bash_file):
    subprocess.run(f'bash {bash_file}', shell = True)

def readfile(file):
    with open(file,'r') as data:
        contents = data.read()
        line = contents.splitlines()[3].split()
        
        print("Option: ", line[0])

        med = line[3]

        lower = line[4]
        upper = line[5]
        return med,upper,lower
    
def counts_grabber(binsize,lightcurve):
    info = pd.read_table(lightcurve, delim_whitespace=True, skiprows=20)
    info.columns =['TIME(rel)', 'TIME(abs)',"DEFAULT-RATE", 'NDETS',"0.3-1 keV","1-2 keV","2-4 keV","4-6 keV","6-12 keV","extra1","extra2","extra3","extra4","extra5","extra6","extra7"]
    info = info.drop(columns=['extra1', 'extra2','extra3', 'extra4','extra5', 'extra6','extra7'])
    times = info["TIME(rel)"].tolist()
    abs_times = info["TIME(abs)"].tolist()

    hard_binned_list = []
    medium_binned_list = []
    soft_binned_list = []
    very_hard_binned_list = []
    tot_binned_list = []

    rate = info["DEFAULT-RATE"].tolist()
    energy1 = info["0.3-1 keV"].tolist()
    energy2 = info["1-2 keV"].tolist()
    energy3 = info["2-4 keV"].tolist()
    energy4 = info["4-6 keV"].tolist()
    energy5 = info["6-12 keV"].tolist()
    
    ndets = info["NDETS"].tolist()
    binsize = int(binsize)
    print(binsize)
    
    bin_times = np.arange(0,times[-1],binsize)
    bin_times = bin_times/1000

    hard = 0
    very_hard = 0 
    soft = 0
    tot = 0
    medium = 0

    for row,i in enumerate(info["TIME(rel)"]):
        soft += (energy1[row])
        medium += energy2[row]
        hard += (energy3[row] + energy4[row])
        very_hard += energy5[row]
        tot += rate[row]
        if i % binsize == 0:
            # print("time: ",i)
            # print(soft,hard)
            hard_binned_list.append(hard)
            medium_binned_list.append(medium)
            soft_binned_list.append(soft)
            very_hard_binned_list.append(very_hard)
            tot_binned_list.append(tot)
            hard = 0
            very_hard = 0
            soft = 0
            tot =0 
            medium = 0
            
    hard_binned_list = [np.round(x/8) for x in hard_binned_list]
    very_hard_binned_list = [np.round(x/8) for x in very_hard_binned_list]
    medium_binned_list = [np.round(x/8) for x in medium_binned_list]
    soft_binned_list = [np.round(x/8) for x in soft_binned_list]
    tot_binned_list = [x/8 for x in tot_binned_list]
    scale = ndets[0]/52
    

    return bin_times, tot_binned_list, abs_times[0], abs_times[-1], soft_binned_list, medium_binned_list, hard_binned_list,very_hard_binned_list

def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles 


def cleaning(file_list):
    new_file_list = []
    # print(goods)
    for i in file_list:
        if i.split(".")[-1] == "lc":
            new_file_list.append(i)
    return new_file_list


if __name__ == '__main__':
    main()    