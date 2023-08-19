from crypt import methods
from logging.handlers import BaseRotatingHandler
from operator import truediv
from pyexpat import ErrorString
from tkinter import N
from tokenize import PlainToken
from ciao_contrib.runtool import *
import os
import glob
import numpy as np
from extract_counts import *
from BEHR_countbins import *
import sys
from re import sub
import subprocess
import matplotlib.pyplot as plt
from scipy.stats import bootstrap
import numpy as np
import random
from scipy.stats import entropy
from astropy.stats import bayesian_blocks
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from matplotlib import axes
from astropy.stats import histogram
from statistics import mean, median, median_high
from scipy.stats import norm
from scipy import interpolate
from numpy import trapz
from matplotlib import gridspec
from matplotlib import rc
import math
from matplotlib.ticker import FormatStrFormatter


BEHR_DIR = '/Users/caleb/AstroPrograms/BEHR_Container/BEHR'


def main(obsid,position,downloaded,bin_size,eclipse_time,focus_time,algo,blocking,src_region,bkg_region):
    
    '''
    This is the overarching function which contains all the subprograms
    The steps are as follows:
    1) Download the data from the region files the user inputs. Grab the events list for the source and the background
    3) Make a lightcurve of source and background region, along with bayesian blocks (BBs) superimposed, for the user to check. 
    4) Define which blocking schema to use. Either total (looking at total events list), bands (doing a union of change points for S/M/H bands), dip (taking the dip duration as the bin width), or x (integer value for bins outside dip)
    5) Get the energies of each bayesian block
    6) Generate bootstrap samples from photons within the dip. Create a color distribution for each sample. 
    7) Generate color distributions for dip BB and flanking BBs
    8) Format all the distributions the same way so that they can be passed into Kullback Leibler Divergence test. 
        This includes normalizing the plots to have area of 1, interpolating the distributions onto a uniform grid, and extending them to the same mininimum and maximum value.
    9) Calculate KL Distance for Flanks and bootstrap distributions
    10) Plot the results
    11) Also make a Time Series Plot
    '''
    
    
    
    '''
    STEP 1 -- DOWNLOAD THE DATA:
    '''
    
    ## Downloads the needed obsid if not present already
    if downloaded == False:
        print("Downloading...")
        download_obsid(obsid)
        
    #labels the needed directories
    primary = f'./{obsid}/primary'
    POSITION_BASIC = sub('\:','',position)
    WORKING_DIR = f'{primary}/{POSITION_BASIC}'
    try:
        os.mkdir(WORKING_DIR)
    except:
        pass
    
    #how many photons ahead and behind the BEHR running average will go 
    bin_size = int(bin_size)
    print('Enter the src and background region:')
    # path to region files
    
    # src_region = input('Enter path to src region file: ').strip()
    # bkg_region = input('Enter path to bkg region file: ').strip()
    
    # grabs event file
    evt = unglob(glob.glob(f'{primary}/acisf*evt2*'))


    # grabs the areas of the region, ratio of bkg to src region
    try:
        print("getting area...")
        area = region_area(evt,bkg_region,1000)/region_area(evt,src_region,1000)
        print(f"The area is {area}")
    except Exception as e:
        print('If OSError because the virtual file for a region could not be opened')
        print('Remember to save the bkg region in ciao format instead of ds9')
        print('Talk me if you need help')
        
    
    #getting events (time+energy) from bkg_region
    dmlist.punlearn()
    dmlist.infile = f'{evt}[energy=500:7000][sky=region({bkg_region})][cols time, energy]'
    dmlist.opt = 'data,clean'
    bkg_out = dmlist().split('\n')
    bkg_out_np = np.genfromtxt(bkg_out,dtype='str').astype('float64')
    bkg_times = bkg_out_np[::,0]-bkg_out_np[0,0]
    bkg_energies = bkg_out_np[::,1]
    
    
    #getting events (time+energy) from src_region
    dmlist.punlearn()
    dmlist.infile = f'{evt}[energy=500:7000][sky=region({src_region})][cols time, energy]'
    dmlist.opt = 'data,clean'
    out = dmlist().split('\n')
    out_np = np.genfromtxt(out,dtype='str').astype('float64')
    start_time = out_np[0,0]
    src_times = out_np[::,0] - start_time
    src_energies = out_np[::,1]
    
    print('start_time:',start_time)
    soft_quantile = np.quantile(src_energies,1/3)
    medium_quantile = np.quantile(src_energies,2/3)
    hard_quantile = max(src_energies)
    
    print('quantiles',soft_quantile,medium_quantile,hard_quantile)

    
    
    print('total source counts', len(src_energies))
    
    print("photon distribution:")
    print('soft',len([x for x in src_energies if x<1200]))
    print('med',len([x for x in src_energies if (x < 2000 and x > 1200)]))
    print('hard',len([x for x in src_energies if x > 2000]))
    print('very_hard',len([x for x in src_energies if x > 7000]))

    
    
    src_times = src_times.tolist()
    
    
    
    
    
    '''
    STEP 2: PLOT THE LIGHTCURVE AND BAYESIAN BLOCKS'''
    
    
    # creating a binned lightcurve of a certain bin size as well as a bayesian binned lightcurve
    bin_time = 1000
    
    # # lets also make some soft and hard lightcurves
    # soft_lc = []
    # hard_lc = []
    # for i in range(0,len(src_energies)):
    #     if src_energies[i] > 1200:
    #         hard_lc.append((src_times[i],src_energies[i]))
    #     else:
    #         soft_lc.append((src_times[i],src_energies[i]))

    
    
    src_lc_times, counts_list = make_lightcurve(src_region,evt,obsid,position,WORKING_DIR)
    
    
    binned_times, binned_counts,errors = bin(src_lc_times,counts_list,bin_time)
    bkg_lc_times, bkg_counts_list = make_lightcurve(bkg_region,evt,obsid,position,WORKING_DIR)
    bkg_binned_times, bkg_binned_counts,bkg_errors = bin(bkg_lc_times,bkg_counts_list,bin_time)

    
    # bayesian lightcurve
    # set your prior here. You can plot it to see what it looks like.

    prior = 1
    #src bb
    times_and_energies = np.column_stack((src_times,src_energies))
    edges, rates, src_in_dip, dip_value_index = make_blocks(src_times,prior,eclipse_time)
    
                
    # plt.scatter(src_times,src_energies)
    # for i in edges:
    #     print(i)
    #     plt.axvline(x=i)
    # plt.xlabel("Time (s)")
    # plt.ylabel("Energy (eV)")
    # # plt.xlim(10000,21000)
    # plt.show()
    # plt.scatter(bkg_times,bkg_energies,color='orange')
    # for i in edges:
    #     print(i)
    #     plt.axvline(x=i)
    # plt.xlabel("Time (s)")
    # plt.ylabel("Energy (eV)")
    # # plt.xlim(10000,21000)
    # plt.show()
            
    bkg_edges, bkg_rates, bkg_in_dip, bkg_dip_value_index = make_blocks(bkg_times,prior,eclipse_time)
    edges_ks = [x/1000 for x in edges]
    bkg_edges_ks = [x/1000 for x in bkg_edges]
    # plots superimposed bb on lightcurve
    bayesian_block_plotter(edges_ks,rates,binned_times,binned_counts,errors,bkg_edges_ks,bkg_rates,bkg_binned_times,bkg_binned_counts,bkg_errors,area,prior,bin_time)

    dip_value_index = get_dip_value_index(focus_time,edges)
        # make bayesian blocks for soft/medium/hard
        
    
    
    '''
    STEP 3: Choose the blocking schema
    '''
    #split into soft/medium/hard bands, and do bayesian blocks on each band. Then do union of all the change points. Do a union with the total BB as well.
    if blocking == "bands":
        
        # define soft/medium/hard bands
        soft_filter = times_and_energies[:,1] < 1200
        medium_filter = (times_and_energies[:,1] < 2000) *(times_and_energies[:,1] > 1200)
        hard_filter = times_and_energies[:,1] > 2000
        
        # generate lightcurves for each band
        soft_lc = times_and_energies[soft_filter]
        medium_lc = times_and_energies[medium_filter]
        hard_lc = times_and_energies[hard_filter]
        
        # find the bayesian block edges for each lightcurve
        soft_edges, soft_rates, soft_src_in_dip, soft_dip_value_index = make_blocks(soft_lc[:,0],prior,eclipse_time)
        medium_edges, medium_rates, medium_src_in_dip, medium_dip_value_index = make_blocks(medium_lc[:,0],prior,eclipse_time)
        hard_edges, hard_rates, hard_src_in_dip, hard_dip_value_index = make_blocks(hard_lc[:,0],prior,eclipse_time)
        
        # convert these to ks
        soft_edges_ks = [x/1000 for x in soft_edges]
        medium_edges_ks = [x/1000 for x in medium_edges]
        hard_edges_ks = [x/1000 for x in hard_edges]

        # union them, and remove outliers
        edges_union = np.concatenate((soft_edges,medium_edges,hard_edges))
        sorted_edges_union = np.sort(edges_union)
        print("sorted",sorted_edges_union)
        sorted_edges_union = remove_outlier_blocks(sorted_edges_union)
        sorted_edges_union_ks = [x/1000 for x in sorted_edges_union]


        # plot them
        all_bands_edges = (soft_edges_ks,medium_edges_ks,hard_edges_ks,sorted_edges_union_ks)
        all_rates, all_values = get_rates(sorted_edges_union,src_times)
        all_bands_rates = (soft_rates,medium_rates,hard_rates,all_rates)
        multiple_bands_blocks_plot(all_bands_edges,all_bands_rates,binned_times,binned_counts,errors,prior,bin_time)

        # combine them with total 'edges' and set the new union as 'edges' variable which will be used later on
        edges = np.concatenate((sorted_edges_union,edges))
        edges = np.sort(edges)
        edges = remove_outlier_blocks(edges)
        edges_ks = [x/1000 for x in edges]
        
        # grab a new dip_value_index, which is the index of the block associated with the eclipse
        dip_value_index = get_dip_value_index(focus_time,edges)
        
        
    # fix blocking outside the dip as a set value
    new_edges = []
    new_dip_value_index = 0
    if blocking == "dip" or blocking.isdigit() or blocking == "total": 
        delta_t = 0
        
        # put in the eclipse start/end times as the first edges
        for i in range(0,len(edges)):
            if edges[i] > eclipse_time: 
                new_edges.append(edges[i-1])
                new_edges.append(edges[i])
                delta_t = (edges[i]-edges[i-1])
                break
            
        if blocking.isdigit():
            delta_t = int(blocking)*1000
        if blocking == "total":
            delta_t = 1000
        # first add times below dip
        time1 = new_edges[0] - delta_t
        time2 = new_edges[1] + delta_t

        while time1 > 0:
            new_edges.insert(0,time1)
            time1 -= delta_t
            
        # now add times above dip
        while time2 < src_times[-1]:
            new_edges.append(time2)
            time2 += delta_t
                
        # add first and final bin
        new_edges.insert(0,0)
        new_edges.append(src_times[-1])
        
        # define edges as new_edges
        new_edges = remove_outlier_blocks(new_edges)

        # changing this to focus one: 
        new_dip_value_index = get_dip_value_index(focus_time,new_edges)
    

    # defines eclipse start and end time as the edges of the bayesian block associated with the dip
    left_flank_start = edges[dip_value_index-1]

    eclipse_start = edges[dip_value_index]
    eclipse_end = edges[dip_value_index+1]
    right_flank_end = edges[dip_value_index+2]

    print('start',eclipse_start)
    print('end',eclipse_end)
    
    plt.rcParams['text.usetex'] = False
    # This is code to recreate the Zand plot of just a simple HR for each bin. 
    # Comment it to true if you want the plot.
    Zand_plot = True
    if Zand_plot:
        textsize = 40
        ones = np.full(shape=len(src_times),fill_value=1,dtype=np.int)
        
        # here we get the bins
        # out_x, out_y,errors = bin(src_times,ones,size)    
        out_x = [x*1000 for x in binned_times]
        
        src_counts_split = tricolor_split(evt,src_region,bin_time)
        softs = src_counts_split['soft']
        hards = src_counts_split['hard']
        
        print(softs)
        print(hards)

        # divide the energies of each bin, for background and source
        lc_src_energies, lc_bkg_energies = block_energy_divider(src_times,src_energies,bkg_times,bkg_energies,out_x)
        print('counts,',binned_counts)
        # print(out_x)
        # print('bined',[x*1000 for x in binned_times])
        
        lc_HR_meds = []
        lc_HR_uppers = []
        lc_HR_lowers = []

        lc_energy_infos = []
        
        # run the individual bins through BEHR to get median values and 2 sigma uncertainties
        for i in range(0,len(lc_src_energies)):
            med, energy_info, upper,lower = hard_soft_divider(lc_src_energies[i], lc_bkg_energies[i],area,WORKING_DIR,POSITION_BASIC,True,src_region,bkg_region,evt,algo)
            print("med",med)
            lc_HR_meds.append(round(float(med),3))
            lc_HR_uppers.append(round(float(upper),3))
            lc_HR_lowers.append(round(float(lower),3))
            lc_energy_infos.append(energy_info)
        
        lc_HR_meds.insert(0,lc_HR_meds[0])
        lc_HR_lowers.insert(0,lc_HR_lowers[0])
        lc_HR_uppers.insert(0,lc_HR_uppers[0])

        
        print('errrors', lc_HR_uppers)
        print('med',lc_HR_meds)
        print('lowerrrors', lc_HR_lowers)

        lc_errors_upper = abs(np.array(lc_HR_uppers) - np.array(lc_HR_meds))
        lc_errors_lower = abs(np.array(lc_HR_lowers) - np.array(lc_HR_meds))

        errors = np.column_stack((lc_errors_lower,lc_errors_upper))
        print(errors)
        lc_errors = np.transpose(errors)

        # fig,(ax1,ax2) = plt.subplots(2,1)
        
        fig = plt.figure()
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1]) 
        plt.rcParams.update({
            'font.sans-serif': 'Times New Roman',
            'font.family': 'sans-serif'
            })

        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1], sharex = ax0)

        
        
        out_x_err = [(i - .5*(bin_time/1000)) for i in binned_times]

        ax1.step(binned_times,lc_HR_meds,where='mid',color='black')
        new_edges = [i*1000 for i in binned_times]

        dip_indexes = []
        left_indexes = []
        right_indexes = []

        for index, time in enumerate(new_edges):
            if time > left_flank_start and time < eclipse_start:
                left_indexes.append(index)
            if time > eclipse_start and time < eclipse_end:
                dip_indexes.append(index)
            if time > eclipse_end and time < right_flank_end:
                right_indexes.append(index)

        print('indexes are:',dip_indexes,left_indexes,right_indexes)


        ax1.errorbar(binned_times, lc_HR_meds,
                yerr = lc_errors,
                xerr = None,
                fmt ='.',
                color = "black",
                linewidth = .5,
                capsize = 1)    
        
        
        
        
        split_times = np.arange(0,len(softs)*(bin_time/1000),(bin_time/1000))

        softs_rate = [i/bin_time for i in softs]
        hards_rate = [i/bin_time for i in hards]


        ax0.step(split_times,softs_rate,where='mid',color='red',label='.5-2 keV')
        ax0.step(split_times,hards_rate,where='mid',color='orange',label='2-7 keV')
        
        soft_bin_errs = [math.sqrt(x) for x in softs]
        hard_bin_errs = [math.sqrt(x) for x in hards]
        out_x_err_split = [(i - .5*(bin_time/1000)) for i in split_times]
        soft_rate_errs = [i/bin_time for i in soft_bin_errs]
        hard_rate_errs = [i/bin_time for i in hard_bin_errs]


        ax0.errorbar(split_times, softs_rate,
                yerr = soft_rate_errs,
                xerr = None,
                fmt ='.',
                linewidth = .5,
                color='red',
                capsize = 1)    

        ax0.errorbar(split_times, hards_rate,
                yerr = hard_rate_errs,
                xerr = None,
                fmt ='.',
                linewidth = .5,
                color='orange',
                capsize = 1)    
        
        ax0.set_ylabel(r"Net Rate [ct s$^{-1}$]",fontsize=textsize)
        plt.rcParams["figure.figsize"] = (15,12)

        ax1.set_xlabel("Time [ks]",fontsize=textsize)
        ax1.set_ylabel("HR=Log(S/H)",fontsize=textsize)
        plt.subplots_adjust(hspace=.0)
        ax0.tick_params(axis='both', which='major', labelsize=textsize)
        ax1.tick_params(axis='both', which='major', labelsize=textsize)



        ax0.legend(fontsize=25)
        ax0.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        
        # ax1.tick_params(axis='both', which='both', labelsize=size,direction='inout',length=15)

        # ax1.tick_params(which='minor', width=0.75, length=7)
        # plt.tight_layout()

        plt.savefig(f'Zand - {bin_time}s')
        plt.show()  


    
     
    '''
    STEP 5: Get energies of each bayesian block, for source and background.
    '''
    # gets the src and background events for during the eclipse and the flanking sides    
    all_src_energies,all_bkg_energies = block_energy_divider(src_times,src_energies,bkg_times,bkg_energies,edges)
    print('NEW EDGES',new_edges)
    second_all_src_energies,second_all_bkg_energies = block_energy_divider(src_times,src_energies,bkg_times,bkg_energies,new_edges)

        
    
    '''
    STEP 6: Create Bootstrap samples for the photon energies during the dip. We only bootstrap the source photons, not the background ones'''
    all_bootstrap_distrubitions = []    
    src_energies_during_eclipse = all_src_energies[dip_value_index]
    bkg_energies_during_eclipse = all_bkg_energies[dip_value_index]
    # print('all,',src_energies_during_eclipse,bkg_energies_during_eclipse)
    
    # define the number of bootstrap samples you want. A good number is 500 or 1000. Can take a few minutes to run.
    number_bootstrap = 500
    for i in range(0,number_bootstrap):   
        print(i) 
        # this creates a bootstrap distribution
        boot_dist = bootstrap(src_energies_during_eclipse,bkg_energies_during_eclipse,area,WORKING_DIR,POSITION_BASIC,src_region,bkg_region,evt,algo)
        # We store all the bootstrap distributions in this list
        all_bootstrap_distrubitions.append(boot_dist)
    
    # We save the boostrap distributions here if we want to look at them later
    with open("bootstrapprob.txt", 'w') as output:
        for row in all_bootstrap_distrubitions:
            output.write(str(row) + '\n')
          
    
    '''
    STEP 7: Generate Color distributions for dip, as well as flanks
    '''
    # now we are going to divide the photons into soft and hard
    # we are creating color distribitions for the dip, as well as the flanks
    # We first get the dip distribution, labeled original_eclipse_distribution
    print('energies for BEHR:')
    print(src_energies_during_eclipse, bkg_energies_during_eclipse)
    original_eclipse_distribution = hard_soft_divider(src_energies_during_eclipse, bkg_energies_during_eclipse,area,WORKING_DIR,POSITION_BASIC,False,src_region,bkg_region,evt,algo)
    
    
    ## FOR BAYESIAN BLOCKS
    # now we get the left flank distribution, defining the flank photons as the ones in the bayesian block immediately before the dip bayesian block

    src_left_flank = all_src_energies[dip_value_index-1]
    bkg_left_flank = all_bkg_energies[dip_value_index-1]
    left_flank_distribution = hard_soft_divider(src_left_flank, bkg_left_flank,area,WORKING_DIR,POSITION_BASIC,False,src_region,bkg_region,evt,algo)
    print('energies for BEHR -- left:')
    print(src_left_flank, bkg_left_flank)

    # now we get the right flank distribution, defining the flank photons as the ones in the bayesian block immediately after the dip bayesian block

    src_right_flank = all_src_energies[dip_value_index+1]
    # test = [1200] * 100
    # new_right_flank = np.concatenate((src_right_flank,test))
    # new_right_flank = new_right_flank.tolist()
    bkg_right_flank = all_bkg_energies[dip_value_index+1]
    right_flank_distribution = hard_soft_divider(src_right_flank, bkg_right_flank,area,WORKING_DIR,POSITION_BASIC,False,src_region,bkg_region,evt,algo)
    print('energies for BEHR -- right:')
    print(src_right_flank, bkg_right_flank)

    # print('original_distribition:',original_eclipse_distribution)


    # FOR 1KS BINS
    

    # we split up the returned distribution into the respective color values (x) and associated probabilities (prob)
    original_xs = np.array([x[0] for x in original_eclipse_distribution])
    left_xs = np.array([x[0] for x in left_flank_distribution])
    right_xs = np.array([x[0] for x in right_flank_distribution])

    original_prob = np.array([x[1] for x in original_eclipse_distribution])
    left_flank_prob = np.array([x[1] for x in left_flank_distribution])
    right_flank_prob = np.array([x[1] for x in right_flank_distribution])
    

    
    # save distributions to text file, in case you want to look at them later
    with open('left_flank_distribution.txt', 'w') as f:
        f.write(f"{left_flank_distribution}\n")
    with open('right_flank_distribution.txt', 'w') as f:
        f.write(f"{right_flank_distribution}\n")
    with open('original_distribution.txt', 'w') as f:
        f.write(f"{original_eclipse_distribution}\n")
        
        
    ''' STEP 8: Normalizing the Distributions.'''
    
    # first we find the lowest value out of all the bootstrap and flanking distributions. This will be the lowest values that we will extend all of our plots too.
    # We do the same for the maximum value
    bootstrap_mins = []
    bootstrap_maxs = []
    for y in all_bootstrap_distrubitions:
        bootstrap_mins.append(min([x[0] for x in y]))
        bootstrap_maxs.append(max([x[0] for x in y]))


    lowest = np.floor(min(min(original_xs),min(right_xs),min(left_xs),min(bootstrap_mins)))
    highest = np.ceil(max(max(left_xs),max(right_xs),max(original_xs),max(bootstrap_maxs)))
    
    # this is just the min of original and flanking, to set limits on the final plot
    secondary_lowest = np.floor(min(min(original_xs),min(right_xs),min(left_xs)))
    secondary_highest = np.ceil(max(max(left_xs),max(right_xs),max(original_xs)))

    # Now we convert them to standard spacing for the entropy test.
    final_left_x, final_left_y = dist_to_pdf(left_xs, left_flank_prob,lowest,highest)
    final_right_x, final_right_y = dist_to_pdf(right_xs, right_flank_prob,lowest,highest)
    print('middle smooth')
    final_middle_x, final_middle_y = dist_to_pdf(original_xs, original_prob,lowest,highest)
    
    
    print(min(bootstrap_mins))
    print("leftx:",left_xs)
    print("rightx:",right_xs)
    print("middle_x:",original_xs)

    print("NEW")

    print("leftx:",final_left_x)
    print("rightx:",final_right_x)
    print("middle_x:",final_middle_x)


    ''' 
    STEP 9: CALCULATE K-L DISTANCE FOR FLANKS AND BOOTSTRAPS'''

    # calculate the kl distance of the flanks. 
    left_k = round(kl_distance(final_middle_y,final_left_y),3)
    right_k = round(kl_distance(final_middle_y,final_right_y),3)
    
    
    # now calculate the KL distance of the bootsrap distributions. We store the  in the variable LK_values. 
    LK_values = []
    for y in all_bootstrap_distrubitions:
        boot_prob = np.array([x[1] for x in y])
        boot_xs = np.array([x[0] for x in y])
        # normalize the bootstrap distributions 
        final_boot_x, final_boot_y = dist_to_pdf(boot_xs, boot_prob,lowest,highest)
        # plt.plot(final_boot_x,final_boot_y,color='red')
        LK_values.append(round(kl_distance(final_middle_y,final_boot_y),4))
        
    
    # plt.plot(final_middle_x,final_middle_y,color='green',label="Original Eclipse")

    
    # plt.show()
    
    '''
    STEP 10: PLOT THESE RESULTS
    '''
    
    ### PLOTTING

    fig,((ax1, ax2, ax4)) = plt.subplots(3,1)
    plt.rcParams["figure.figsize"] = (10,1)   
    plt.rc('font', family='serif')

    plt.subplots_adjust(hspace=0.3)
    plt.suptitle(f'Obs. {obsid}')


    
    textsize= 15
    ## AX2 --> Posterior Density Distributions for eclipse and flanks
    ax2.plot(final_middle_x,final_middle_y,color='green',label="2.6ks Dip")
    ax2.plot(final_left_x,final_left_y,color='orange',label="Left Flank")
    ax2.plot(final_right_x,final_right_y,color='red',label="Right Flank")
    # ax2.set_title(f'Posterior Probability Distribution of HR')
    ax2.set_xlabel('Color Value',fontsize=textsize)
    ax2.set_ylabel("p(HR|Data)",style='italic',fontsize=textsize)
    ax2.set_xlim(-2,1)

    ax2.legend()


    ## AX4 --> Cummulative Distributionm, based on bootsrap CDF
    
    greater_left = 0
    greater_right = 0
    ecdf_y = []
    ecdf_x = []
    LK_values.sort()
    # create a cdf based on bootsrap values. It appears stepped because of a fixed number of possible divisions between H/S that get entered into BEHR.
    for count,h in enumerate(LK_values):
        ecdf_x.append(h)
        ecdf_y.append((count+1)/number_bootstrap)
        
        # also calculate the percentage of bins > left and right KL distances, for a bootstrap p value
        if h > left_k:
            greater_left += 1
        if h > right_k:
            print(h)
            greater_right += 1 
            
    greater_left = greater_left/number_bootstrap 
    greater_right = greater_right/number_bootstrap
    
    
    
    
    
    
    ax4.plot(ecdf_x, ecdf_y)
    # ax4.set_title(f'CDF')
    ax4.set_xlabel('K-L Divergence Distance',fontsize=textsize)
    ax4.set_ylabel("cdf",style='italic',fontsize=textsize)
    ax4.axvline(x=left_k,color = 'orange',linestyle='dashed', label = r"Left Flank: $D_{KL}$"+f"={left_k}" + " (p-value ="+f"{greater_left})")
    ax4.axvline(x=right_k,color = 'red',linestyle='dashed', label = r"Right Flank: $D_{KL}$"+f"={right_k}" + " (p-value ="+f"{greater_right})")

    ax4.legend()
    
    ### AX3 --> Frequency Distribution, histogram of bootstrap distances and dashed lines for flanking ones.
    
    print("LK",LK_values)
    # ax3.hist(LK_values, bins = 10)
    # ax3.set_title(f'Histogram of K-L Divergence Distances')
    # ax3.set_xlabel('K-L Divergence Distance')
    # line_left = ax3.axvline(x=left_k,color = 'orange',linestyle='dashed', label = f'Left Flank: {left_k} (% greater = {greater_left})')
    # line_right = ax3.axvline(x=right_k,color = 'red',linestyle='dashed', label = f'Right Flank {right_k} (% greater = {greater_right})')
    # ax3.legend(handles=[line_left,line_right])
    # # ax3.text(3, 12.5, f'> left = {greater_left}', fontsize = 12)
    # # ax3.text(3, 10.5, f'> right = {greater_right}', fontsize = 12)
    # ax3.set_ylabel('Frequency')
    
    
    
    ## AX1 --> Running Color Lightcurve
    
    # generate a running lightcurve using BEHR
    outfile = f'{WORKING_DIR}/BEHR_bash.txt'
    BEHR_outdir = f'{BEHR_DIR}/{obsid}/{POSITION_BASIC}'

    '''
    sig_make_behr_running_avg(evt,src_region,bkg_region,BEHR_DIR,outfile,BEHR_outdir,bin_size,confidence='68.00')
    print('Running averages BEHR...')

    sig_run_BEHR(outfile)
    medians = []
    uppers = []
    lowers = []

    # we want all the uncertaintinities for the running average 
    for i in range(0,len(src_times)):
        file = f'{BEHR_outdir}/{i}_BEHRresults.txt'
        med,upper,lower = readfile(file)
        uppers.append(upper)
        medians.append(med)
        lowers.append(lower)

    uppers = np.array(uppers).astype('float64')
    lowers = np.array(lowers).astype('float64')
    medians = np.array(medians).astype('float64')
    '''
    meds = []
    block_uppers = []
    block_lowers = []
    # but we only want black lines for the median of each block distribution
    energy_infos = []
    for i in range(0,len(all_src_energies)):
        med, energy_info,upper,lower = hard_soft_divider(all_src_energies[i], all_bkg_energies[i],area,WORKING_DIR,POSITION_BASIC,True,src_region,bkg_region,evt,algo)
        meds.append(round(float(med),3))
        
        block_uppers.append(upper)
        block_lowers.append(lower)

        energy_infos.append(energy_info)
    
    second_meds = []
    second_block_uppers = []
    second_block_lowers = []

    second_energy_infos = []
    print(second_all_src_energies)
    for i in range(0,len(second_all_src_energies)):
        second_med, second_energy_info,second_upper,second_lower = hard_soft_divider(second_all_src_energies[i], second_all_bkg_energies[i],area,WORKING_DIR,POSITION_BASIC,True,src_region,bkg_region,evt,algo)
        print(second_med)
        second_meds.append(round(float(second_med),3))
        
        second_block_uppers.append(second_upper)
        second_block_lowers.append(second_lower)

        second_energy_infos.append(second_energy_info)

    
        
    meds.insert(0,meds[0])
    block_uppers.insert(0,block_uppers[0])
    block_lowers.insert(0,block_lowers[0])

    second_meds.insert(0,second_meds[0])
    second_block_uppers.insert(0,second_block_uppers[0])
    second_block_lowers.insert(0,second_block_lowers[0])


    # make errors for the 1ks bin
    second_errors_up = abs(np.array(second_block_uppers) - np.array(second_meds))
    second_errors_low = abs(np.array(second_block_lowers) - np.array(second_meds))

    second_errors = np.column_stack((second_errors_low,second_errors_up))
    second_HR_errors = np.transpose(second_errors)

    
    # now plot the medians of each block, with the running averages in the background.
    new_edges = [i/1000 for i in new_edges]

    ax1.step(new_edges,second_meds,color='black')
    # make error bars in right place:
    new_edges_errors = []
    for i in range(0,len(new_edges)):
        if i == 0 :
            middle = .5*new_edges[i]
        else:
            middle = .5*(new_edges[i] - new_edges[i-1])+new_edges[i-1]
        new_edges_errors.append(middle)
            
    # new_edges_errors = [i/1000 for i in new_edges_errors]

    ax1.errorbar(new_edges_errors, second_meds,
                yerr = second_HR_errors,
                xerr = None,
                fmt ='.',
                color = "black",
                linewidth = .5,
                capsize = 1)    
    
    edges = [i/1000 for i in edges]
    ax1.fill_between(edges,block_lowers,block_uppers,step='pre')

    ax1.set_ylabel("HR=log(S/H)",style='italic',fontsize=textsize)
    ax1.set_xlabel('Time [ks]',fontsize=textsize)
    # ax1.set_title(f'Bayesian Blocked Color Curve')



    ax1.axvspan(edges[dip_value_index],edges[dip_value_index+1], alpha=0.2, color='green',label="2.6ks Dip")
    ax1.axvspan(edges[dip_value_index-1],edges[dip_value_index], alpha=0.2, color='orange',label="Left Flank")
    ax1.axvspan(edges[dip_value_index+1],edges[dip_value_index+2], alpha=0.2, color='red',label="Right Flank")

    ax1.legend()

    

    # Show plot and save it
    ax1.tick_params(axis='both', which='major', labelsize=15)
    ax2.tick_params(axis='both', which='major', labelsize=15)
    ax4.tick_params(axis='both', which='major', labelsize=15)

    plt.show()
    plt.savefig(f'{position}_{obsid}_HR_SIG.png')

    
    photons = []
    bkg_photons = []
    for i in range(0,len(src_times)):
        photons.append(i)
        
    for i in range(0,len(bkg_times)):
        bkg_photons.append(i)
        
        
    # This is seperate plot just for HR and Lightcurve
    '''
    fig,(ax2, ax1) = plt.subplots(2,1,sharex=True)

    medians = [1/x for x in medians]
    uppers = [1/x for x in uppers]
    lowers = [1/x for x in lowers]
    ax1.plot(src_times[:-1], medians[:-1],color='black')
    ax1.fill_between(src_times[:-1],lowers[:-1],uppers[:-1],step='pre',color='grey')
    ax1.set_ylabel('(H/S)')
    ax1.set_xlabel('Time [s]')
    # ax1.set_title(f'Running Color Lightcurve (± {bin_size})')
    
    binned_times = [x*1000 for x in binned_times]
    ax2.step(binned_times, binned_counts,color='black')
    err_times = [x - (.5*float(bin_size))/1000 for x in binned_times]
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Counts per Bin')
    # ax2.set_title('')
    ax2.errorbar(err_times, binned_counts,
            yerr = errors,
            xerr = None,
            fmt ='.',
            color = "black",
            linewidth = .5,
            capsize = 1)
    
    plt.suptitle(f'Running {bin_time}s Lightcurve and Running Hardness Ratio +/- {bin_size} photons')
    plt.subplots_adjust(hspace=0)
    plt.show()
    '''
    
    '''
    STEP 10: TIME SERIES PLOT.
    '''
    ## Time Series PLOT. Each bin gets a point on log(S/M) vs log(M/H).##
    
    # energy infos stores soft/medium/hard counts for each block as a tuple (soft, medium, hard)
    print(energy_infos)
    
    print("blocks number:", len(energy_infos[0]))
    
    # stores median value and uncertainties for S/M and M/H for each block
    low_med_high_blocks_SM = []
    low_med_high_blocks_MH = []
    
    

    # iterate through each block in energy infos, grabbing respective number of phootns in each band and calculating median and uncertainty color values
    # stores it in low_med_high_blocks array as a tuple (med, upper uncertainity, lower uncertainity)
    for i in second_energy_infos:   
        med_SM,upper_SM,lower_SM = block_med(evt,BEHR_DIR,i[0][0],i[0][1],i[1][0],i[1][1],area,outfile,BEHR_outdir,src_region,bkg_region)
        med_MH,upper_MH,lower_MH = block_med(evt,BEHR_DIR,i[0][1],i[0][2],i[1][1],i[1][2],area,outfile,BEHR_outdir,src_region,bkg_region)
        low_med_high_blocks_SM.append((med_SM,upper_SM,lower_SM))
        low_med_high_blocks_MH.append((med_MH,upper_MH,lower_MH))
    
    print('SM')
    print(low_med_high_blocks_SM)
    print("MH")
    print(low_med_high_blocks_MH)
    n = np.arange(len(energy_infos))
    plt.rcParams["figure.figsize"] = (15,12)

    fig = plt.figure()
    gs = gridspec.GridSpec(1, 1) 
    plt.rcParams.update({
        'font.sans-serif': 'Times New Roman',
        'font.family': 'serif'
        })

    fig, ax = plt.subplots()
    textsize = 40

    # grab median values for the scatter plot
    log_SM = [x[0] for x in low_med_high_blocks_SM]
    log_MH = [x[0] for x in low_med_high_blocks_MH]
    # ax.scatter(log_SM,log_MH,facecolors='none', edgecolors='black')
    # dip point

    lc_errors_upper_SM = abs(np.array([x[1] for x in low_med_high_blocks_SM]) - np.array([x[0] for x in low_med_high_blocks_SM]))
    lc_errors_lower_SM = abs(np.array([x[2] for x in low_med_high_blocks_SM]) - np.array([x[0] for x in low_med_high_blocks_SM]))

    errors_SM = np.column_stack((lc_errors_lower_SM,lc_errors_upper_SM))
    lc_errors_SM = np.transpose(errors_SM)

    lc_errors_upper_MH = abs(np.array([x[1] for x in low_med_high_blocks_MH]) - np.array([x[0] for x in low_med_high_blocks_MH]))
    lc_errors_lower_MH = abs(np.array([x[2] for x in low_med_high_blocks_MH]) - np.array([x[0] for x in low_med_high_blocks_MH]))

    errors_MH = np.column_stack((lc_errors_lower_MH,lc_errors_upper_MH))
    lc_errors_MH = np.transpose(errors_MH)


    ax.errorbar(log_SM, log_MH, xerr=lc_errors_SM, fmt='o',capsize = 10,color='black',alpha=.3)
    ax.errorbar(log_SM, log_MH, yerr=lc_errors_MH, fmt='o',capsize = 10,color='black',alpha=.3)
    # ax.scatter(log_SM,log_MH,facecolors='none', edgecolors='black',s=100)


    for index in dip_indexes:
        ax.scatter(log_SM[index],log_MH[index],facecolors='green', edgecolors='black',s=500)
    for index in left_indexes:
        ax.scatter(log_SM[index],log_MH[index],facecolors='orange', edgecolors='black',s=500)
    for index in right_indexes:
        ax.scatter(log_SM[index],log_MH[index],facecolors='red', edgecolors='black',s=500)

    # ax.scatter(log_SM[new_dip_value_index],log_MH[new_dip_value_index],facecolors='green', edgecolors='black',s=500)
    # ax.scatter(log_SM[new_dip_value_index-1],log_MH[new_dip_value_index-1],facecolors='orange', edgecolors='black',s=500)
    # ax.scatter(log_SM[new_dip_value_index+1],log_MH[new_dip_value_index+1],facecolors='red', edgecolors='black',s=500)    


    ax.set_xlabel("Log(S/M)",fontsize=textsize)
    ax.set_ylabel("Log(M/H)",fontsize=textsize)

    # ax.set_title(f"Time-Series Plot of HR of Bayesian Blocks Obsid {obsid}")


    # draw the arrows between each block, represented by a point.
    arrows_SM = log_SM.copy()
    del arrows_SM[0]   
    arrows_SM.append(log_SM[-1])

    arrows_MH = log_MH.copy()
    del arrows_MH[0]   
    arrows_MH.append(log_MH[-1])

    # print(arrows_SM, log_SM,  (np.array(arrows_SM)-np.array(log_SM)))
    # for i, txt in enumerate(n):
    #     ax.quiver(log_SM, log_MH, (np.array(arrows_SM)-np.array(log_SM)), (np.array(arrows_MH)-np.array(log_MH)), angles='xy', scale_units='xy', scale=1,width=0.001,headlength=40,headaxislength=35,headwidth=20,color='tab:blue')

    plt.rcParams["figure.figsize"] = (15,12)   
    plt.rcParams.update({
            'font.sans-serif': 'Times New Roman',
            'font.family': 'serif'
            })
    plt.tick_params(axis='both', which='major', labelsize=40)
    plt.tick_params(axis='both', which='both',direction='inout',length=15,labelsize=40)

    plt.tick_params(which='minor', width=0.75, length=7)
    plt.minorticks_on()

    # plt.legend(fontsize=30)
    plt.tight_layout()
    plt.savefig("color_color_time_series")

    plt.show()

    

    
def make_lightcurve(src_region,evt,obsid,position,WORKING_DIR):
    '''
    This function grabs the times and count list for a lightcurve.
    '''

    dmextract.punlearn()
    # grabs lightcurve from chandra with bin time of 3.24104 seconds
    dmextract.infile = f'{evt}[sky=region({src_region})][bin time=::3.24104]'
    dmextract.outfile = f'{WORKING_DIR}/{position}_{obsid}_lc.fits'
    dmextract.opt = 'ltc1'
    dmextract.clobber = 'yes'
    dmextract()
    # list of columns
    cols = "TIME_BIN TIME_MIN TIME TIME_MAX COUNTS STAT_ERR AREA EXPOSURE COUNT_RATE COUNT_RATE_ERR"
    cols = cols.split(" ")


    # list of columns

    # accessing fits data
    hdu_list = fits.open(f'{WORKING_DIR}/{position}_{obsid}_lc.fits', memmap=True)
    evt_data = Table(hdu_list[1].data)

    # initialising DataFrame
    df = pd.DataFrame()

    # writing to dataframe
    for col in cols:
        df[col] = list(evt_data[col])
        
    count_list = df["COUNTS"].tolist()
    print("count list sum",sum(count_list))
    times = df["TIME"].tolist()
    still = True
    start_skip = 0
    end_skip = 0
    #     print(count_list)
    #     print("SHSHSH",len(count_list),len(times))
    if len(count_list) != 0:
        for b in range(0,len(count_list)):
            try:
                if count_list[0] == 0:
                    if still == True:
                        del count_list[0]
                        start_skip +=1 
                if count_list[0] != 0:
                    break
            except:
                print("an error occured")
                pass
        end = True
        for b in range(0,len(count_list)):
            if count_list[-1] == 0:
                if end == True:
                    del count_list[-1]
                    end_skip +=1
            if count_list[-1] != 0:
                break

        if start_skip != 0:
            times = times[start_skip:]
        if end_skip != 0:
            times = times[:-end_skip]

    low = times[0]
    times = [i - low for i in times]

    return(times,count_list)

def tricolor_split(evt,reg,bin_size):


    energies = {'soft':'300:2000',
    'hard':'2000:10000'
    }

    counts_dict = {}

    for key,val in energies.items():

        dmextract.punlearn()
        dmextract.infile = f'{evt}[energy={val}][sky=region({reg})][bin time=::{bin_size}]'
        dmextract.clobber = 'yes'
        dmextract.opt = 'generic'
        dmextract.outfile = 'temp.fits'
        dmextract()

        dmlist.punlearn()
        dmlist.infile = 'temp.fits[cols counts]'
        dmlist.opt = 'data, clean'

        out = np.asarray(dmlist().split('\n')[1:]).astype('int')

        counts_dict.update({key:out})

    return counts_dict


def hard_soft_divider(src_energies,bkg_energies,area,working_dir,position_basic,blocks,src_region,bkg_region,evt,algo):
    '''
    This function takes a list of energies and divides them by energy.
    '''
    
    # hard is > 1.2kev, Soft < 1.2kev
    
    hard_cutoff = 2000
    med_cutoff = 1200
    
    hard_src = []
    soft_src = []
    
    # if we want to do S/M/H, we define Medium as 1.2-2kev, and specific_soft as <2kev
    medium_src = []   
    specific_soft_src = []

    
    hard_bkg = []
    soft_bkg = []
    medium_bkg = []
    specific_soft_bkg =[]
    
    
    
    
    for energy in src_energies:
        if energy > hard_cutoff:
            hard_src.append(energy)
        else:
            soft_src.append(energy)
            if energy > med_cutoff:
                medium_src.append(energy)
            else:
                specific_soft_src.append(energy)
            
    for energy in bkg_energies:
        if energy > hard_cutoff:
            hard_bkg.append(energy)
        else:
            soft_bkg.append(energy)
            if energy > med_cutoff:
                    medium_bkg.append(energy)
            else:
                specific_soft_bkg.append(energy)
    
    
    number_hard_src = len(hard_src)
    number_soft_src = len(soft_src)
    number_specific_soft_src = len(specific_soft_src)
    number_medium_src = len(medium_src)
    
    energy_info_src = (number_specific_soft_src,number_medium_src,number_hard_src)
    energy_info_bkg = (len(specific_soft_bkg),len(medium_bkg),len(hard_bkg))
    
    energy_info = (energy_info_src,energy_info_bkg)
    number_hard_bkg = len(hard_bkg)
    number_soft_bkg = len(soft_bkg)
    number_specific_soft_bkg = len(specific_soft_bkg)
    number_medium_bkg = len(medium_bkg)
   
    

    
    print('Making BEHR bash f   ile...')
    outfile = f'{working_dir}/BEHR_bash.txt'
    BEHR_outdir = f'{BEHR_DIR}/{obsid}/{position_basic}'
    
    subprocess.run(f'rm -rf {BEHR_outdir}',shell=True)
    os.makedirs(BEHR_outdir)
    
    
    
    if blocks == True:
        print('blocking...')
        med,upper,lower = block_med(evt,BEHR_DIR,number_soft_src,number_hard_src,number_soft_bkg,number_hard_bkg,area,outfile,BEHR_outdir,src_region,bkg_region)
        return med, energy_info, upper,lower
    
    if algo == 'quad':
        return sig_make_behr_file(BEHR_DIR,number_soft_src,number_hard_src,number_soft_bkg,number_hard_bkg,area,outfile,BEHR_outdir)
    
    if algo == 'gibbs':
        return sig_make_behr_file_gibbs(BEHR_DIR,number_soft_src,number_hard_src,number_soft_bkg,number_hard_bkg,area,outfile,BEHR_outdir)

    
    

    
def block_med(evt,BEHR_DIR,soft_src,hard_src,soft_bkg,hard_bkg,area,outfile,BEHR_outdir,srcreg,bkgreg):
    confidence = '68.0'
    
    hard_area = region_area(evt,bkgreg,3000)/region_area(evt,srcreg,3000)
    soft_area = region_area(evt,bkgreg,1000)/region_area(evt,srcreg,1000)

    
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
    


def block_energy_divider(src_times,src_energies,bkg_times,bkg_energies,edges):
    print("Dividing energies by following bayesian blocks...")
    print("Edges: ",edges)
    
    all_src_energies = []
    
    all_bkg_energies = []

    for i in range(0,len(edges)-1):
        temp = []
        for p in range(0,len(src_times)):
            if src_times[p] > edges[i] and src_times[p] < edges[i+1]:
                temp.append(src_energies[p])
        all_src_energies.append(temp)
    
    for i in range(0,len(edges)-1):
        temp = []
        for p in range(0,len(bkg_times)):
            if bkg_times[p] > edges[i] and bkg_times[p] < edges[i+1]:
                temp.append(bkg_energies[p])      
        all_bkg_energies.append(temp)
    
    
    # check
    tot = 0
    for i in all_src_energies:
        tot += len(i)
        
    print(len(all_src_energies))
    
    return all_src_energies, all_bkg_energies
    
def get_counts(src_times,src_energies,bkg_times,bkg_energies,eclipse_start,eclipse_end,edges,dip_value_index):
    
    eclipse_times = []
    # these are all the photons from source during this time
    src_energies_during_eclipse = []
    # these are all the photons from background during this time
    bkg_energies_during_eclipse = []
    
    src_left_flank = []
    bkg_left_flank = []
    
    src_right_flank = []
    bkg_right_flank = []
    
    
    left_flank_start = edges[dip_value_index-1]
    right_flank_end= edges[dip_value_index+2]

    print('left flank',left_flank_start)
    print('right flank',right_flank_end)

    # eclipse photons
    for i in range (0,len(src_times)):
        if src_times[i] > eclipse_start and src_times[i] < eclipse_end:
            eclipse_times.append(src_times[i])
            src_energies_during_eclipse.append(src_energies[i])
    
    # left flank photons
    for i in range (0,len(src_times)):
        if src_times[i] > eclipse_end and src_times[i] < right_flank_end:
            eclipse_times.append(src_times[i])
            src_energies_during_eclipse.append(src_energies[i])        
        
            
    # this grabs the N flanking photons, where N is photons during eclipse
    # eclipse_index_start = src_times.index(eclipse_times[0])
    # eclipse_index_end = src_times.index(eclipse_times[-1])
    # # buffer of a certain amount of photons between eclipse and flanks
    # buffer = 1
    # left_flank_start = eclipse_index_start - len(eclipse_times) - buffer
    # left_flank_end = eclipse_index_start - buffer
    # right_flank_start = eclipse_index_end + buffer
    # right_flank_end = eclipse_index_end + len(eclipse_times) + buffer
    # src_left_flank = src_energies[left_flank_start:left_flank_end]
    # src_right_flank = src_energies[right_flank_start:right_flank_end]
    
    # left_flank_start_time = src_times[left_flank_start]
    # left_flank_end_time = src_times[left_flank_end]
    # right_flank_start_time = src_times[right_flank_start]
    # right_flank_end_time = src_times[right_flank_end]


    # That seemed wrong, we just want all the photons in the bayesian bl




    bkg_left_flank = []
    bkg_right_flank = []


    # print("Time orders:")
    # print(left_flank_start)
    # print(left_flank_end)
    # print(eclipse_index_start)
    # print(eclipse_index_end)
    # print(right_flank_start)
    # print(right_flank_end)
    
    # print(np.ceil((eclipse_index_end-eclipse_index_start)/2) - 1)
    # central_index = eclipse_index_start + np.ceil((eclipse_index_end-eclipse_index_start)/2) - 1
    # print(central_index)

    # this grabs the photons during the eclipse from the background
    # for i in range(0,len(bkg_times)):
    #     if bkg_times[i] > eclipse_start and bkg_times[i] < eclipse_end:            
    #         # print('adding eclipse backgrounds')
    #         bkg_energies_during_eclipse.append(bkg_energies[i])
    #     if bkg_times[i] > left_flank_start_time and bkg_times[i] < left_flank_end_time:
    #         # print('adding left flank backgrounds')
    #         bkg_left_flank.append(bkg_energies[i])
    #     if bkg_times[i] > right_flank_start_time and bkg_times[i] < right_flank_end_time:
    #         # print('adding right flank backgrounds')
    #         bkg_right_flank.append(bkg_energies[i])
           
        
    
    return src_energies_during_eclipse, bkg_energies_during_eclipse, src_left_flank, bkg_left_flank, src_right_flank, bkg_right_flank, left_flank_start_time, left_flank_end_time,right_flank_start_time, right_flank_end_time, central_index
    
    
def bootstrap(data,bkg_energies_during_eclipse,area,WORKING_DIR,POSITION_BASIC,src_region,bkg_region,evt,algo):
    
    number = len(data)

    print(f"bootstrapping {number} photons...")
    # print(data)

    bootstrap_data = random.choices(data, k=number)
    bootstrap_data.sort()
    print(bootstrap_data)
    
    # print(bootstrap_data)

    return hard_soft_divider(bootstrap_data, bkg_energies_during_eclipse,area,WORKING_DIR,POSITION_BASIC,False,src_region,bkg_region,evt,algo)

    
    
    
def sig_make_behr_file(BEHR_DIR,soft_src,hard_src,soft_bkg,hard_bkg,area,outfile,BEHR_outdir):
    print("./BEHR softsrc={soft_src} hardsrc={hard_src} softbkg={soft_bkg} hardbkg={hard_bkg} softarea={area} outputPr=true algo=quad output={BEHR_outdir}/BEHRresults")
    with open(outfile,'w') as writeto:
        writeto.write(f'cd {BEHR_DIR}')
        writeto.write(f'\n echo "softsrc={soft_src} hardsrc={hard_src} softbkg={soft_bkg} hardbkg={hard_bkg} softarea={area} outputPr=True algo=quad"')
        writeto.write(f'\n./BEHR softsrc={soft_src} hardsrc={hard_src} softbkg={soft_bkg} hardbkg={hard_bkg} softarea={area} outputPr=true algo=quad output={BEHR_outdir}/BEHRresults')
    
    
    print('Running quad BEHR...')

    sig_run_BEHR(outfile)
    
    print("Reading BEHR...")
    
    data = np.loadtxt(f'{BEHR_outdir}/BEHRresults_prob.txt')
    c_value = data[::,4]
    prob = data[::,5]
    # plt.scatter(c_value,prob)
    # plt.show()
    
    return list(zip(c_value, prob))

def sig_make_behr_file_gibbs(BEHR_DIR,soft_src,hard_src,soft_bkg,hard_bkg,area,outfile,BEHR_outdir):
    print("./BEHR softsrc={soft_src} hardsrc={hard_src} softbkg={soft_bkg} hardbkg={hard_bkg} softarea={area} nsim=40000 algo=gibbs outputMC=true output={BEHR_outdir}/BEHRresults")
    with open(outfile,'w') as writeto:
        writeto.write(f'cd {BEHR_DIR}')
        writeto.write(f'\n echo "softsrc={soft_src} hardsrc={hard_src} softbkg={soft_bkg} hardbkg={hard_bkg} softarea={area} nsim=40000 algo=gibbs outputMC=True"')
        writeto.write(f'\n./BEHR softsrc={soft_src} hardsrc={hard_src} softbkg={soft_bkg} hardbkg={hard_bkg} softarea={area} nsim=40000 algo=gibbs outputMC=true output={BEHR_outdir}/BEHRresults')

    print('Running gibbs BEHR...')

    sig_run_BEHR(outfile)
        
    print("Reading BEHR...")
    
    data = np.loadtxt(f'{BEHR_outdir}/BEHRresults_draws.txt')
    softs = data[::,0]
    hards = data[::,1]
    vals = []
    
    for i in range(0,len(hards)):
        val = np.log10(softs[i]/hards[i])
        vals.append(val)    
    
    sample_mean = np.mean(vals)
    sample_std = np.std(vals)
    low = min(vals)
    high = max(vals)
    lowest = sample_mean - 5*sample_std
    highest = sample_mean + 5*sample_std
    print('Mean=%.3f, Standard Deviation=%.3f' % (sample_mean, sample_std))
    dist = norm(sample_mean, sample_std)
    x = np.linspace(lowest,highest,100)
    probabilities = [dist.pdf(value) for value in x]
    # plt.hist(vals, bins=100, density=True)
    # plt.plot(x, probabilities)
    # plt.show()    
    # curve_area = trapz(probabilities, dx=.01)
    # print("area =", curve_area)
    tot = sum(probabilities)
    probabilities = [x/tot for x in probabilities]

    return list(zip(x,probabilities))

def constant_count_split(evt,reg,bkgreg,divide_energy,N):

    dmlist.punlearn()
    dmlist.infile = f'{evt}[sky=region({reg})][cols time, energy]'
    dmlist.opt = 'data,clean'

    raw_out = np.genfromtxt(dmlist().split('\n'),dtype='str').astype('float64')
    times = raw_out[::,0]
    energies = raw_out[::,1]

    dmlist.punlearn()
    dmlist.infile = f'{evt}[sky=region({bkgreg})][cols time, energy]'
    dmlist.opt = 'data,clean'

    raw_out_bkg = np.genfromtxt(dmlist().split('\n'),dtype='str').astype('float64')
    bkg_times = raw_out_bkg[::,0]
    bkg_energies = raw_out_bkg[::,1]

    softs = []
    hards = []
    bkg_softs = []
    bkg_hards = []

    soft_count = 0
    hard_count = 0

    soft_bkg_count = 0
    hard_bkg_count = 0

    for i in range(len(times)):
        low_bound = max((0,i-N))
        high_bound = min((len(times)-1,i+N))
        energy_slice = energies[low_bound:high_bound]

        #the first time, we have to count them all
        if i == 0:
            for count in energy_slice:
                if count <= divide_energy:
                    soft_count += 1
                else:
                    hard_count += 1

        else:
            #general case, both ends moving
            if i >= N and i+N <= len(times)-1:
                new_count = energy_slice[-1]
                old_count = energies[low_bound - 1]

                if old_count <= divide_energy:
                    soft_count -= 1
                else:
                    hard_count -= 1

                if new_count <= divide_energy:
                    soft_count += 1
                else:
                    hard_count += 1

            #left edge: adding a new count but not losing any
            elif i < N:
                new_count = energy_slice[-1]

                if new_count <= divide_energy:
                    soft_count += 1
                else:
                    hard_count += 1

            #right edge: losing an old count but not gaining any
            #elif i+N <= len(times)-1:
            else:
                old_count = energies[low_bound - 1]

                if old_count <= divide_energy:
                    soft_count -= 1
                else:
                    hard_count -= 1

        #now to deal with the background:
        #need to take only the background counts from inside the time range
        #that the source counts are taken from
        bkg_energies_bool = [1 if en < divide_energy else 0 for en in bkg_energies]

        low_time = times[low_bound]
        high_time = times[high_bound]

        bkg_low = low_bound
        bkg_high = high_bound

        for i,time in enumerate(bkg_times):
            if time < low_time:
                bkg_low = i + 1

            elif time == low_time:
                bkg_low = i

            elif time == high_time:
                bkg_high = i
                break

            elif time > high_time:
                bkg_high = i - 1
                break

        bkg_energy_slice = bkg_energies_bool[bkg_low:bkg_high]

        bkg_softs.append(max(sum(bkg_energy_slice),0))
        bkg_hards.append(max(0,len(bkg_energy_slice)-sum(bkg_energy_slice)))


        softs.append(max(soft_count,0))
        hards.append(max(hard_count,0))

    softs = np.array(softs).astype('int')
    hards = np.array(hards).astype('int')
    bkg_softs = np.array(bkg_softs).astype('int')
    bkg_hards = np.array(bkg_hards).astype('int')

    bkg_soft_nonzero = [i for i in bkg_softs if i != 0]
    bkg_hard_nonzero = [i for i in bkg_hards if i != 0]

    assert len(bkg_soft_nonzero) != 0
    assert len(bkg_hard_nonzero) != 0

    all = np.column_stack((softs,hards,bkg_softs,bkg_hards))

    np.save('debugging.npy',all)
    np.savetxt('debugging.txt',all,fmt='%s',header='#softs,hards,bkg_softs,bkg_hards',delimiter=',')

    return softs,hards,bkg_softs,bkg_hards



def sig_make_behr_running_avg(evt,srcreg,bkgreg,BEHR_DIR,outfile,BEHR_outdir,N,confidence='68.00'):
    soft_src,hard_src,soft_bkg,hard_bkg = constant_count_split(evt,srcreg,bkgreg,2000,N)

    hard_area = region_area(evt,bkgreg,3000)/region_area(evt,srcreg,3000)
    soft_area = region_area(evt,bkgreg,1000)/region_area(evt,srcreg,1000)
    
    ## need list of 2N+1 
    

    with open(outfile,'w') as writeto:
        writeto.write(f'cd {BEHR_DIR}')
        for i in range(len(soft_src)):
            writeto.write(f'\n echo "softsrc={soft_src[i]} hardsrc={hard_src[i]}   softbkg={soft_bkg[i]}   hardbkg={hard_bkg[i]}"')
            writeto.write(f'\n./BEHR softsrc={soft_src[i]} hardsrc={hard_src[i]}   softbkg={soft_bkg[i]}   hardbkg={hard_bkg[i]}   softarea={soft_area} hardarea={hard_area} output={BEHR_outdir}/{i}_BEHRresults level={confidence}')

    

def sig_run_BEHR(bash_file):
    subprocess.run(f'bash {bash_file}', shell = True)


# this is where you choose log(S/h), s/h, (h-s)/(h+s)
def readfile(file):
    with open(file,'r') as data:
        contents = data.read()
        line = contents.splitlines()[3].split()
        
        print("Option: ", line[0])

        med = line[3]

        lower = line[4]
        upper = line[5]
        return med,upper,lower


def bin(xs,ys,binsize):
    out_y =  []
    out_x = []
    y_sum = 0
    x_sum = 0
    for i in range(len(xs)):
        x = xs[i]
        y = ys[i]
        if i == 0:
            last_x = 0
        else:
            last_x = xs[i-1]
        x_sum += (x-last_x)
        y_sum += y
        if x_sum >= binsize:
            out_y.append(y_sum)
            out_x.append(x)
            x_sum = 0
            y_sum = 0
    if y_sum != 0:
        out_y.append(y_sum)
        out_x.append(x)
    
    out_x = [x/1000 for x in out_x]
    errors = [np.sqrt(x) for x in out_y]
    
    return out_x,out_y, errors

def get_events():
    pass

def multiple_bands_blocks_plot(all_bands_edges,all_bands_rates,binned_times,binned_counts,errors,prior,bin_time):

    fig, (ax1)= plt.subplots(1,1)

    soft_edges = all_bands_edges[0]
    medium_edges = all_bands_edges[1]
    hard_edges = all_bands_edges[2]
    all_edges = all_bands_edges[3]

    
    soft_rates = all_bands_rates[0]
    medium_rates= all_bands_rates[1]
    hard_rates = all_bands_rates[2]
    all_rates = all_bands_rates[3]


    ax1.step(soft_edges,soft_rates,where='pre',linewidth=3,color='green')
    ax1.step(medium_edges,medium_rates,where='pre',linewidth=3,color='blue')
    ax1.step(hard_edges,hard_rates,where='pre',linewidth=3,color='red')
    ax1.step(all_edges,all_rates,where='pre',linewidth=3,color='black')

    ax1.set_ylabel(r"Net Rate [ct s$^{-1}$]",fontsize=textsize)
    ax1.set_xlabel("Time [ks]")
    ax3 = ax1.twinx()  
    color = 'tab:orange'

    ax3.step(binned_times,binned_counts,color=color)
    ax3.errorbar(binned_times, binned_counts,
            yerr = errors,
            xerr = None,
            fmt ='.',
            color = "black",
            linewidth = .5,
            capsize = 1)

    ax3.set_ylabel(f"Photon Count per {bin_time}s bin",color=color)
    ax1.set_title(f"Source Lightcurve with Bayesian Blocks, ncp_prior of {prior}")
    plt.show()
    
def bayesian_block_plotter(edges,rates,binned_times,binned_counts,errors,bkg_edges,bkg_rates,bkg_binned_times,bkg_binned_counts,bkg_errors,area,prior,bin_time):
    plt.rcParams["figure.figsize"] = (15,7)

    fig = plt.figure()
    gs = gridspec.GridSpec(1, 1) 
    plt.rcParams.update({
        'font.sans-serif': 'Times New Roman',
        'font.family': 'serif'
        })

    ax1 = plt.subplot(gs[0])
    # ax2 = plt.subplot(gs[1], sharex = ax1)

    textsize = 40
    print("CHUNKS")
    color = 'tab:blue'


    # rescale to per sec intstead of per ks
    rates = [i/1000 for i in rates]
    
    
    # put bayesian blocks on binned rates -- adjust this manually
    # ax1.set_ylim([0, .12])

    
    ax1.step(edges,rates,where='pre',linewidth=3,color=color)
    ax1.set_ylabel(r"Net Rate [ct s$^{-1}$]",fontsize=textsize)
    # ax1.set_xlabel("Time [ks]")
    color = 'tab:orange'

    binned_rates = []
    for i in range(0,len(binned_times)):
        binned_rates.append(binned_counts[i]/bin_time)
    binned_rates_errors = []
    for i in range(0,len(binned_times)):
        binned_rates_errors.append(errors[i]/bin_time)
    
    
    ax1.step(binned_times,binned_rates,where='pre',color=color)
    
    binned_times_errs = [x-.5*(bin_time/1000) for x in binned_times]
    ax1.errorbar(binned_times_errs, binned_rates,
            yerr = binned_rates_errors,
            xerr = None,
            fmt ='.',
            color = "black",
            linewidth = .5,
            capsize = 1)

    # ax1.set_ylabel(f"Photon Count per {bin_time}s bin",color=color)
    # ax1.set_title(f"Source Lightcurve with Bayesian Blocks, p0={prior}")
    
    color = 'tab:blue'

    ## scaling by area of bkg region
    bkg_rates = [x/area for x in bkg_rates]
    bkg_binned_counts = [x/area for x in bkg_binned_counts]
    bkg_errors = [x/area for x in bkg_errors]
    
    fig.subplots_adjust(hspace=0.3)


    bkg_rates = [i/bin_time for i in bkg_rates]

    
    # ax2.step(bkg_edges,bkg_rates,where='pre',linewidth=3,color=color)
    # ax2.set_ylabel("Count Rate (counts/sec)",fontsize=textsize)
    ax1.set_xlabel("Time [ks]",fontsize=textsize)

    # ax4 = ax2.twinx()
    color = 'tab:orange'
    print("ERRORS")
    print(bkg_binned_counts)
    print(bkg_errors)

    bkg_binned_rates = []
    for i in range(0,len(bkg_binned_times)):
        bkg_binned_rates.append(bkg_binned_counts[i]/bin_time)
    bkg_binned_rates_errors = []
    for i in range(0,len(bkg_binned_times)):
        bkg_binned_rates_errors.append(bkg_errors[i]/bin_time)

    bkg_rates = [x/1000 for x in bkg_rates]

    bkg_binned_times_errs = [x-.5*(bin_time/1000) for x in bkg_binned_times]
    # ax2.step(bkg_binned_times[:-1],bkg_binned_rates[:-1],where='pre',color=color)
    # ax2.errorbar(bkg_binned_times_errs[:-1], bkg_binned_rates[:-1],
    #         yerr = bkg_binned_rates_errors[:-1],
    #         xerr = None,
    #         fmt ='.',
    #         color = "black",
    #         linewidth = .5,
    #         capsize = 1)
    # ax2.set_ylabel(f"Photon Count per {bin_time}s bin",color=color)

    # ax2.set_title(f"Background Lightcurve with Bayesian Blocks, p0={prior}")
     
    plt.subplots_adjust(hspace=.0)
    # ax1.tick_params(axis='both', which='major', labelsize=40)
    # ax2.tick_params(axis='both', which='major', labelsize=20)
    # ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax1.tick_params(axis='both', which='both',direction='inout',length=15,labelsize=40)

    ax1.tick_params(which='minor', width=0.75, length=7)
    ax1.minorticks_on()

    # plt.legend(fontsize=30)
    plt.tight_layout()
    plt.savefig("bayesian_23443")
    plt.show()

## this function will take the distribution that BEHR gives and give it a defined spacing, and normalize it
def dist_to_pdf(dist_x,dist_y,lowest,highest):
    binsize = .01
    area = trapz(dist_y,dist_x)
    dist_y = (1/area)*dist_y
    cdf_left_y = []
    sum = 0
    for i in dist_y:
        sum+=i
        cdf_left_y.append(sum)
        
    # plt.scatter(dist_x,dist_y,color='purple',s=1)
    # now we interpolate pdf
    temp = interpolate.interp1d(dist_x, dist_y)
    # print('dist_x',dist_x)
    x_inter = np.arange((np.ceil(100*min(dist_x)))/100,(np.floor(100*max(dist_x)))/100,binsize)
    ynew = temp(x_inter)
    # plt.scatter(x_inter,ynew,color='pink',s=1)
    # plt.show()

    # pdf_new = []
    # for i,y in enumerate(ynew):
    #     if i == 0:
    #         pdf_new.append(y)
    #     else:
    #         pdf_new.append(y-ynew[i-1])
  
  ## add zeroes to each end
    print(len(x_inter), len(ynew))

    extended_x, extended_y = add_zeroes(x_inter,ynew,binsize,lowest,highest)

    print(len(extended_y), len(extended_x))
  ## then normalize so area = 1
    area = trapz(extended_y,extended_x)
    extended_y = (1/area) * extended_y

    
    # plt.step(extended_x,extended_y)
    return extended_x, extended_y

# this will extend the distribution to a fixed low and high point
def add_zeroes(x,y,binsize,lowest,highest):
    lower_tail = np.arange(lowest,x[0],binsize)
    lower_tail_ys = [0] * len(lower_tail)
    try:
        if lower_tail[-1] == x[0]:
            lower_tail = lower_tail[:-1]
            lower_tail_ys = lower_tail_ys[:-1]
    except:
        print('smoothing error?')
        pass
    upper_tail = np.arange(x[-1]+binsize,highest+binsize,binsize)
    upper_tail_ys = [0] * len(upper_tail)
    try:
        if upper_tail[0] == x[-1]:
            upper_tail = lower_tail[1:]
            upper_tail_ys = upper_tail_ys[1:]

    except:
        print('smoothing error?')
        pass
    

    extended_x = np.concatenate((lower_tail,x,upper_tail))
    extended_y = np.concatenate((lower_tail_ys,y,upper_tail_ys))
    return extended_x, extended_y

# define entropy test: KL Divergence, where we discard elements with zeroes
def kl_distance(p, q):
    binsize = .01
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)
    sum = 0
    for i in range(0,len(p)):
        if p[i] != 0 and q[i] != 0:
            sum += (p[i] * np.log(p[i] / q[i]))*binsize
    # print("pq:",p[i],q[i])
    # print(sum)
    return sum
    
def remove_outlier_blocks(edges):
    # print('stuff',len(edges),len(values))
    new_chunk_times = [0]
    old = 0
    # iterate through all edges after 0
    for i in edges:
        # check that time is greater than a certain amount
        if (i - old) > (10*3.24):
            new_chunk_times.append(i)
            old = i
        else:
            new_chunk_times.pop()
            new_chunk_times.append(i)
            old = i


    return new_chunk_times

def get_rates(chunk_edges_times,times):
    print('chunks',chunk_edges_times)
     #convert the chunk_edges from time space to index space
    chunk_values = []
    for p in chunk_edges_times[1:]: 
        val = 0
        for i in times:
            if i < p:
                val +=1
        try: 
            val = val - sum(chunk_values)
        except:
            pass
        chunk_values.append(val)
    
    
    chunk_edges_times_ks = [x/1000 for x in chunk_edges_times]
    time_diffs = [x - chunk_edges_times_ks[i - 1] for i, x in enumerate(chunk_edges_times_ks)][1:]
    chunk_count_rates = []
    
    # plt.scatter(time_diffs,chunk_values)
    # plt.xlabel("Block Duration (ks)")
    # plt.ylabel("Block Counts")

    # plt.show()

    #gets count rate within each block
    for i in range(0,len(chunk_values)):
        chunk_count_rates.append(chunk_values[i]/time_diffs[i])
    

    chunk_count_rates.insert(0,chunk_count_rates[0])
    print("RATEs,",chunk_count_rates)

    return chunk_count_rates, chunk_values

def get_dip_value_index(eclipse_time,chunk_edges_times):
    print('chunkedgestimes',chunk_edges_times)
    for i in range(0,len(chunk_edges_times)): 
        if eclipse_time < chunk_edges_times[i]:
            dip_value_index = i-1
            return dip_value_index

def make_blocks(times,prior,eclipse_time):
    dip_start_23443 = 13075.9 
    dip_end_23443 = 18229.16
    buffer = 5
    #grab chunk edges based on time list
    chunk_edges_times = bayesian_blocks(times,fitness='events',p0=prior)
    
    # remove blocks < 100s
    print(chunk_edges_times)
    edges_temp = []
    
    # obsid = 23443
    # if obsid == 23443:
    #     for i in chunk_edges_times:
    #         if i < (dip_start_23443+buffer) or i > (dip_end_23443-buffer):
    #             edges_temp.append(i)
    
    # chunk_edges_times = edges_temp
    # og_chunk_count_rates, og_chunk_values = get_rates(chunk_edges_times,times)

    chunk_edges_times = remove_outlier_blocks(chunk_edges_times)

    print(chunk_edges_times)

    chunk_count_rates, chunk_values = get_rates(chunk_edges_times,times)
    # get number of counts during eclipse
    for i in range(0,len(chunk_edges_times)): 
        if eclipse_time < chunk_edges_times[i]:
            num_in_eclipse = chunk_values[i-1]
            dip_value_index = i-1
            break
        

    return (chunk_edges_times,chunk_count_rates,num_in_eclipse,dip_value_index)
    
    
if __name__ == '__main__':
    obsid = sys.argv[1]
    position = sys.argv[2]
    downloaded = sys.argv[3]
    bin_size = sys.argv[4]
    algo = sys.argv[5] ## gibbs or quad (quad default) --> gibbs better for high count rate sources
    blocking = sys.argv[6] # either total, bands, eclipse (for set eclipse time for each bin), or integer value for #ks, with dip as its own bayesian block
    source_region = sys.argv[7]
    bkg_region = sys.argv[8]

    if 'T' in downloaded:
        downloaded = True
    else:
        downloaded = False
        
    # eclipse_start = 1000*float(input('Enter the beginning of the dip in ks: '))
    # eclipse_end = 1000*float(input('Enter the end of the dip in ks: '))
    eclipse_time = 1000*float(input('Enter some time near the middle of the dip in ks: '))
    
    # maybe we want to look at a specific block, after its binned a certain way
    # focus_time = 1000*float(input('Enter time of block to focus on: '))
    focus_time = eclipse_time
    
    main(obsid,position,downloaded,bin_size,eclipse_time,focus_time,algo,blocking,source_region,bkg_region)