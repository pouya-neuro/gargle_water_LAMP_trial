"""
Created: December 2021
Author: Pouya Babakhani
"""

import os
import random
import json
import numpy as np
import pandas as pd
import scipy.stats as st
import plotly.graph_objects as go
import plotly.io
from plotly.subplots import make_subplots
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from math import floor,ceil,atan2,sqrt,log,cos,sin,exp,pi
from scipy import fft

# path=input('Path to raw data directory:')
path = os.path.dirname(os.path.realpath(__file__))

raw_df=pd.read_csv(filepath_or_buffer=os.path.join(path,'raw_data.csv'))
processed_df=pd.DataFrame(
    data={'STT':raw_df['Days since first symptoms on sampling date']})

# The section of code below processes the raw_data to extract a single Ct 
# value as the result of a particular RT-PCR test:

def get_ct_value(df,row,gene_order,call_number=0):   
    if call_number == len(gene_order):
        return (41.0,'negative_result')
    for gene in gene_order:
        try:
            return (float(df.loc[row,gene_order[call_number]].replace(',','.')
                          ),gene_order[call_number])
        except:
            return get_ct_value(df,row,gene_order,call_number+1)

def find_ct_value(RT_PCR_type,gene_order,df,row):   
    assert type(row) == int 
    assert RT_PCR_type in ['gargle_water_indirect_RT_PCR',
                           'gargle_water_direct_RT_PCR',
                           'oro_nasopharyngeal_indirect_RT_PCR']
    if RT_PCR_type == 'oro_nasopharyngeal_indirect_RT_PCR':
        assert (type(gene_order) == list and
                set(gene_order) == {0,1,2,3,4} and
                len(gene_order) == 5)
    else:
        assert (type(gene_order) == list and
                set(gene_order) == {0,1,2} and
                len(gene_order) == 3)    
    if RT_PCR_type == 'oro_nasopharyngeal_indirect_RT_PCR':
        columns=['E gene Ct value (Oro-nasopharyngeal)',
                   'RdRp/S gene Ct value (Oro-nasopharyngeal)',
                   'N gene Ct value (Oro-nasopharyngeal)',
                   'ORF1ab gene Ct value (Oro-nasopharyngeal)',
                   'S gene Ct value (Oro-nasopharyngeal)']
        return  get_ct_value(df,row,[columns[i] for i in gene_order])    
    if RT_PCR_type == 'gargle_water_indirect_RT_PCR':
        columns=['E gene Ct value (indirect gargle-water)',
                   'RdRP/S gene Ct value (indirect gargle-water)',
                   'N gene Ct value (indirect gargle-water)']
        return  get_ct_value(df,row,[columns[i] for i in gene_order])
    if RT_PCR_type == 'gargle_water_direct_RT_PCR':
        columns=['E gene Ct value (direct gargle-water)',
                   'RdRP/S gene Ct value (direct gargle-water)',
                   'N gene Ct value (direct gargle-water)']
        return  get_ct_value(df,row,[columns[i] for i in gene_order])    
    
def shorten_gene_names(gene_name): 
    if gene_name in ['E gene Ct value (Oro-nasopharyngeal)',
                     'E gene Ct value (indirect gargle-water)',
                     'E gene Ct value (direct gargle-water)']:
        return 'E gene'
    if gene_name in ['RdRp/S gene Ct value (Oro-nasopharyngeal)',
                     'RdRp/S gene Ct value (indirect gargle-water)',
                     'RdRp/S gene Ct value (direct gargle-water)']:
        return 'RdRp/S gene'    
    if gene_name in ['N gene Ct value (Oro-nasopharyngeal)',
                     'N gene Ct value (indirect gargle-water)',
                     'N gene Ct value (direct gargle-water)']:
        return 'N gene' 
    if gene_name in ['ORF1ab gene Ct value (Oro-nasopharyngeal)',
                     'ORF1ab gene Ct value (indirect gargle-water)',
                     'ORF1ab gene Ct value (direct gargle-water)']:
        return 'ORF1ab gene'     
    if gene_name in ['S gene Ct value (Oro-nasopharyngeal)',
                     'S gene Ct value (indirect gargle-water)',
                     'S gene Ct value (direct gargle-water)']:
        return 'S gene'   
    else:
        return 'negative result'

# The order of genes searched for Ct values in the oro-nasopharyngeal indirect
# RT-PCR results was: N, ORF1ab, E, S, RdRp/S  

processed_df[['oro_nasopharyngeal_indirect_RT_PCR_Ct_value',
                   'oro_nasopharyngeal_indirect_RT_PCR_Ct_gene']]=[
                       find_ct_value('oro_nasopharyngeal_indirect_RT_PCR',
                                     [2,3,0,4,1],raw_df,row) for row in
                                             range(len(raw_df))]
                       
processed_df['oro_nasopharyngeal_indirect_RT_PCR_Ct_gene']=(
    [shorten_gene_names(gene_name) for gene_name in processed_df[
        'oro_nasopharyngeal_indirect_RT_PCR_Ct_gene']])

# The order of genes searched for Ct values in the gargle water indirect
# RT-PCR results was: N, E, RdRP/S    
                    
processed_df[['gargle_water_indirect_RT_PCR_Ct_value',
                   'gargle_water_indirect_RT_PCR_Ct_gene']]=[
                       find_ct_value('gargle_water_indirect_RT_PCR',
                                     [2,0,1],raw_df,row) for row in
                                             range(len(raw_df))]
processed_df['gargle_water_indirect_RT_PCR_Ct_gene']=(
    [shorten_gene_names(gene_name) for gene_name in processed_df[
        'gargle_water_indirect_RT_PCR_Ct_gene']])                       
                       
# The order of genes searched for Ct values in the gargle water direct
# RT-PCR results was: N, E, RdRP/S    
                    
processed_df[['gargle_water_direct_RT_PCR_Ct_value',
                   'gargle_water_direct_RT_PCR_Ct_gene']]=[
                       find_ct_value('gargle_water_direct_RT_PCR',
                                     [2,0,1],raw_df,row) for row in
                                             range(len(raw_df))]
processed_df['gargle_water_direct_RT_PCR_Ct_gene']=(
    [shorten_gene_names(gene_name) for gene_name in processed_df[
        'gargle_water_direct_RT_PCR_Ct_gene']])                                                   
processed_df['positive_gargle_water_direct_LAMP_result']=[result ==
    'positive' for result in raw_df['Gargle-water LAMP result']]

# Below is the code for the sensitivity, specificity, PPV and NPV calculations:
    
analysis_results={}

def wilson_score_interval_estimator(
        postive_lamp_results,
        postive_RT_PCR_results,alpha=0.05):
    z=st.norm.ppf(1-alpha/2)
    p_middle=(postive_lamp_results+0.5*z*z)/(postive_RT_PCR_results+z*z)
    p_upper=(p_middle+(z/(postive_RT_PCR_results+z*z))*np.sqrt(
        (postive_lamp_results*(postive_RT_PCR_results-postive_lamp_results)
         /postive_RT_PCR_results)+(z*z/4)))
    if p_upper > 0.999:
        p_upper=0.999
    p_lower=(p_middle - (z/(postive_RT_PCR_results+z*z))*np.sqrt(
        (postive_lamp_results*(postive_RT_PCR_results-postive_lamp_results)
         /postive_RT_PCR_results)+(z*z/4)))
    return (p_lower,p_middle,p_upper)

def sensitivity_calculator(RT_PCR_type,ct_value_cutoff):
    assert RT_PCR_type in ['gargle_water_indirect_RT_PCR',
                           'gargle_water_direct_RT_PCR',
                           'oro_nasopharyngeal_indirect_RT_PCR']
    
    if RT_PCR_type == 'gargle_water_indirect_RT_PCR':
        positive_RT_PCR_results=processed_df[
            'gargle_water_indirect_RT_PCR_Ct_value'] < ct_value_cutoff
        positive_lamp_results=processed_df.loc[positive_RT_PCR_results,
            'positive_gargle_water_direct_LAMP_result']
        sum_1=sum(positive_lamp_results)
        sum_2=sum(positive_RT_PCR_results)
        return (wilson_score_interval_estimator(sum_1,sum_2),sum_1,sum_2)

    if RT_PCR_type == 'gargle_water_direct_RT_PCR':
        positive_RT_PCR_results=processed_df[
            'gargle_water_direct_RT_PCR_Ct_value'] < ct_value_cutoff
        positive_lamp_results=processed_df.loc[positive_RT_PCR_results,
            'positive_gargle_water_direct_LAMP_result']
        sum_1=sum(positive_lamp_results)
        sum_2=sum(positive_RT_PCR_results)
        return (wilson_score_interval_estimator(sum_1,sum_2),sum_1,sum_2)
    
    
    if RT_PCR_type == 'oro_nasopharyngeal_indirect_RT_PCR':
        positive_RT_PCR_results=processed_df[
            'oro_nasopharyngeal_indirect_RT_PCR_Ct_value'] < ct_value_cutoff
        positive_lamp_results=processed_df.loc[positive_RT_PCR_results,
            'positive_gargle_water_direct_LAMP_result']
        sum_1=sum(positive_lamp_results)
        sum_2=sum(positive_RT_PCR_results)
        return (wilson_score_interval_estimator(sum_1,sum_2),sum_1,sum_2)
    
analysis_results['sub_40_Ct_sensitivity_oro_nasopharyngeal_indirect_RT_PCR']=(
    ('Proportion: {1}/{2}, Wilson score interval: {0[1]:.1%}'
    ' (95% CI: {0[0]:.1%} - {0[2]:.1%})')
    .format(*sensitivity_calculator('oro_nasopharyngeal_indirect_RT_PCR',40)))
    
analysis_results['sub_30_Ct_sensitivity_oro_nasopharyngeal_indirect_RT_PCR']=(
    ('Proportion: {1}/{2}, Wilson score interval: {0[1]:.1%}'
    ' (95% CI: {0[0]:.1%} - {0[2]:.1%})')
    .format(*sensitivity_calculator('oro_nasopharyngeal_indirect_RT_PCR',30)))  

analysis_results['sub_25_Ct_sensitivity_oro_nasopharyngeal_indirect_RT_PCR']=(
    ('Proportion: {1}/{2}, Wilson score interval: {0[1]:.1%}'
    ' (95% CI: {0[0]:.1%} - {0[2]:.1%})')
    .format(*sensitivity_calculator('oro_nasopharyngeal_indirect_RT_PCR',25)))          

analysis_results['sub_40_Ct_sensitivity_gargle_water_indirect_RT_PCR']=(
    ('Proportion: {1}/{2}, Wilson score interval: {0[1]:.1%}'
    ' (95% CI: {0[0]:.1%} - {0[2]:.1%})')
    .format(*sensitivity_calculator('gargle_water_indirect_RT_PCR',40)))  

analysis_results['sub_30_Ct_sensitivity_gargle_water_indirect_RT_PCR']=(
    ('Proportion: {1}/{2}, Wilson score interval: {0[1]:.1%}'
    ' (95% CI: {0[0]:.1%} - {0[2]:.1%})')
    .format(*sensitivity_calculator('gargle_water_indirect_RT_PCR',30))) 

analysis_results['sub_25_Ct_sensitivity_gargle_water_indirect_RT_PCR']=(
    ('Proportion: {1}/{2}, Wilson score interval: {0[1]:.1%}'
    ' (95% CI: {0[0]:.1%} - {0[2]:.1%})')
    .format(*sensitivity_calculator('gargle_water_indirect_RT_PCR',25)))

def specificity_calculator(RT_PCR_type,ct_value_cutoff):
    assert RT_PCR_type in ['gargle_water_indirect_RT_PCR',
                           'gargle_water_direct_RT_PCR',
                           'oro_nasopharyngeal_indirect_RT_PCR']
    
    if RT_PCR_type == 'gargle_water_indirect_RT_PCR':
        negative_RT_PCR_results=processed_df[
            'gargle_water_indirect_RT_PCR_Ct_value'] > ct_value_cutoff
        negative_lamp_results=~processed_df.loc[negative_RT_PCR_results,
            'positive_gargle_water_direct_LAMP_result']
        sum_1=sum(negative_lamp_results)
        sum_2=sum(negative_RT_PCR_results)
        return (wilson_score_interval_estimator(sum_1,sum_2),sum_1,sum_2)

    if RT_PCR_type == 'gargle_water_direct_RT_PCR':
        negative_RT_PCR_results=processed_df[
            'gargle_water_direct_RT_PCR_Ct_value'] > ct_value_cutoff
        negative_lamp_results=~processed_df.loc[negative_RT_PCR_results,
            'positive_gargle_water_direct_LAMP_result']
        sum_1=sum(negative_lamp_results)
        sum_2=sum(negative_RT_PCR_results)
        return (wilson_score_interval_estimator(sum_1,sum_2),sum_1,sum_2) 

    if RT_PCR_type == 'oro_nasopharyngeal_indirect_RT_PCR':
        negative_RT_PCR_results=processed_df[
            'oro_nasopharyngeal_indirect_RT_PCR_Ct_value'] > ct_value_cutoff
        negative_lamp_results=~processed_df.loc[negative_RT_PCR_results,
            'positive_gargle_water_direct_LAMP_result']
        sum_1=sum(negative_lamp_results)
        sum_2=sum(negative_RT_PCR_results)
        return (wilson_score_interval_estimator(sum_1,sum_2),sum_1,sum_2)

analysis_results['40_Ct_specificity_oro_nasopharyngeal_indirect_RT_PCR']=(
    ('Proportion: {1}/{2}, Wilson score interval: {0[1]:.1%}'
    ' (95% CI: {0[0]:.1%} - {0[2]:.1%})')
    .format(*specificity_calculator('oro_nasopharyngeal_indirect_RT_PCR',40))) 

analysis_results['40_Ct_specificity_gargle_water_indirect_RT_PCR']=(
    ('Proportion: {1}/{2}, Wilson score interval: {0[1]:.1%}'
    ' (95% CI: {0[0]:.1%} - {0[2]:.1%})')
    .format(*specificity_calculator('gargle_water_indirect_RT_PCR',40))) 

def ppv_calculator(RT_PCR_type,ct_value_cutoff):
    assert RT_PCR_type in ['gargle_water_indirect_RT_PCR',
                           'gargle_water_direct_RT_PCR',
                           'oro_nasopharyngeal_indirect_RT_PCR']
    
    if RT_PCR_type == 'gargle_water_indirect_RT_PCR':
        positive_RT_PCR_results=processed_df[
            'gargle_water_indirect_RT_PCR_Ct_value'] < ct_value_cutoff
        positive_lamp_results=processed_df[
            'positive_gargle_water_direct_LAMP_result']
        sum_1=sum((positive_lamp_results)&(positive_RT_PCR_results))
        sum_2=sum(positive_lamp_results)
        return (wilson_score_interval_estimator(sum_1,sum_2),sum_1,sum_2)

    if RT_PCR_type == 'gargle_water_direct_RT_PCR':
        positive_RT_PCR_results=processed_df[
            'gargle_water_direct_RT_PCR_Ct_value'] < ct_value_cutoff
        positive_lamp_results=processed_df[
            'positive_gargle_water_direct_LAMP_result']
        sum_1=sum((positive_lamp_results)&(positive_RT_PCR_results))
        sum_2=sum(positive_lamp_results)
        return (wilson_score_interval_estimator(sum_1,sum_2),sum_1,sum_2)

    if RT_PCR_type == 'oro_nasopharyngeal_indirect_RT_PCR':
        positive_RT_PCR_results=processed_df[
            'oro_nasopharyngeal_indirect_RT_PCR_Ct_value'] < ct_value_cutoff
        positive_lamp_results=processed_df[
            'positive_gargle_water_direct_LAMP_result']
        sum_1=sum((positive_lamp_results)&(positive_RT_PCR_results))
        sum_2=sum(positive_lamp_results)
        return (wilson_score_interval_estimator(sum_1,sum_2),sum_1,sum_2)

analysis_results['40_Ct_PPV_oro_nasopharyngeal_indirect_RT_PCR']=(
    ('Proportion: {1}/{2}, Wilson score interval: {0[1]:.1%}'
    ' (95% CI: {0[0]:.1%} - {0[2]:.1%})')
    .format(*ppv_calculator('oro_nasopharyngeal_indirect_RT_PCR',40))) 

analysis_results['40_Ct_PPV_gargle_water_indirect_RT_PCR']=(
    ('Proportion: {1}/{2}, Wilson score interval: {0[1]:.1%}'
    ' (95% CI: {0[0]:.1%} - {0[2]:.1%})')
    .format(*ppv_calculator('gargle_water_indirect_RT_PCR',40))) 

def npv_calculator(RT_PCR_type,ct_value_cutoff):
    assert RT_PCR_type in ['gargle_water_indirect_RT_PCR',
                           'gargle_water_direct_RT_PCR',
                           'oro_nasopharyngeal_indirect_RT_PCR']
    
    if RT_PCR_type == 'gargle_water_indirect_RT_PCR':
        positive_RT_PCR_results=processed_df[
            'gargle_water_indirect_RT_PCR_Ct_value'] < ct_value_cutoff
        positive_lamp_results=processed_df[
            'positive_gargle_water_direct_LAMP_result']
        sum_1=sum((~positive_lamp_results)&(~positive_RT_PCR_results))
        sum_2=sum(~positive_lamp_results)
        return (wilson_score_interval_estimator(sum_1,sum_2),sum_1,sum_2)        

    if RT_PCR_type == 'gargle_water_direct_RT_PCR':
        positive_RT_PCR_results=processed_df[
            'gargle_water_direct_RT_PCR_Ct_value'] < ct_value_cutoff
        positive_lamp_results=processed_df[
            'positive_gargle_water_direct_LAMP_result']
        sum_1=sum((~positive_lamp_results)&(~positive_RT_PCR_results))
        sum_2=sum(~positive_lamp_results)
        return (wilson_score_interval_estimator(sum_1,sum_2),sum_1,sum_2)  

    if RT_PCR_type == 'oro_nasopharyngeal_indirect_RT_PCR':
        positive_RT_PCR_results=processed_df[
            'oro_nasopharyngeal_indirect_RT_PCR_Ct_value'] < ct_value_cutoff
        positive_lamp_results=processed_df[
            'positive_gargle_water_direct_LAMP_result']
        sum_1=sum((~positive_lamp_results)&(~positive_RT_PCR_results))
        sum_2=sum(~positive_lamp_results)
        return (wilson_score_interval_estimator(sum_1,sum_2),sum_1,sum_2)  

analysis_results['40_Ct_NPV_oro_nasopharyngeal_indirect_RT_PCR']=(
    ('Proportion: {1}/{2}, Wilson score interval: {0[1]:.1%}'
    ' (95% CI: {0[0]:.1%} - {0[2]:.1%})')
    .format(*npv_calculator('oro_nasopharyngeal_indirect_RT_PCR',40))) 

analysis_results['40_Ct_NPV_gargle_water_indirect_RT_PCR']=(
    ('Proportion: {1}/{2}, Wilson score interval: {0[1]:.1%}'
    ' (95% CI: {0[0]:.1%} - {0[2]:.1%})')
    .format(*npv_calculator('gargle_water_indirect_RT_PCR',40))) 

def sensitivity_calculator_STT(RT_PCR_type,ct_value_cutoff,
                               STT_value_cutoff,bound):
    assert RT_PCR_type in ['gargle_water_indirect_RT_PCR',
                           'gargle_water_direct_RT_PCR',
                           'oro_nasopharyngeal_indirect_RT_PCR']
    assert bound in ['lower','upper']
    
    if bound == 'upper':
    
        if RT_PCR_type == 'gargle_water_indirect_RT_PCR':
            positive_RT_PCR_results=((processed_df[
                'gargle_water_indirect_RT_PCR_Ct_value'] < ct_value_cutoff)&
                (processed_df['STT'] < STT_value_cutoff))
            
            positive_lamp_results=((processed_df.loc[
                positive_RT_PCR_results,
                'positive_gargle_water_direct_LAMP_result'])&
                (processed_df['STT'] < STT_value_cutoff))                          
            sum_1=sum(positive_lamp_results)
            sum_2=sum(positive_RT_PCR_results)
            return (wilson_score_interval_estimator(sum_1,sum_2),sum_1,sum_2)
    
        if RT_PCR_type == 'gargle_water_direct_RT_PCR':
            positive_RT_PCR_results=((processed_df[
                'gargle_water_direct_RT_PCR_Ct_value'] < ct_value_cutoff)&
                (processed_df['STT'] < STT_value_cutoff))
            positive_lamp_results=((processed_df.loc[
                positive_RT_PCR_results,
                'positive_gargle_water_direct_LAMP_result'])&
                (processed_df['STT'] < STT_value_cutoff))           
            sum_1=sum(positive_lamp_results)
            sum_2=sum(positive_RT_PCR_results)
            return (wilson_score_interval_estimator(sum_1,sum_2),sum_1,sum_2)
              
        if RT_PCR_type == 'oro_nasopharyngeal_indirect_RT_PCR':
            positive_RT_PCR_results=((processed_df[
                'oro_nasopharyngeal_indirect_RT_PCR_Ct_value'] < 
            ct_value_cutoff)&(processed_df['STT'] < STT_value_cutoff)) 
            positive_lamp_results=((processed_df.loc[
                positive_RT_PCR_results,
                'positive_gargle_water_direct_LAMP_result'])&
                (processed_df['STT'] < STT_value_cutoff))            
            sum_1=sum(positive_lamp_results)
            sum_2=sum(positive_RT_PCR_results)
            return (wilson_score_interval_estimator(sum_1,sum_2),sum_1,sum_2)

    if bound == 'lower':
    
        if RT_PCR_type == 'gargle_water_indirect_RT_PCR':
            positive_RT_PCR_results=((processed_df[
                'gargle_water_indirect_RT_PCR_Ct_value'] < ct_value_cutoff)&
                (processed_df['STT'] > STT_value_cutoff))
            
            positive_lamp_results=((processed_df.loc[
                positive_RT_PCR_results,
                'positive_gargle_water_direct_LAMP_result'])&
                (processed_df['STT'] > STT_value_cutoff))                          
            sum_1=sum(positive_lamp_results)
            sum_2=sum(positive_RT_PCR_results)
            return (wilson_score_interval_estimator(sum_1,sum_2),sum_1,sum_2)
    
        if RT_PCR_type == 'gargle_water_direct_RT_PCR':
            positive_RT_PCR_results=((processed_df[
                'gargle_water_direct_RT_PCR_Ct_value'] < ct_value_cutoff)&
                (processed_df['STT'] > STT_value_cutoff))
            positive_lamp_results=((processed_df.loc[
                positive_RT_PCR_results,
                'positive_gargle_water_direct_LAMP_result'])&
                (processed_df['STT'] > STT_value_cutoff))           
            sum_1=sum(positive_lamp_results)
            sum_2=sum(positive_RT_PCR_results)
            return (wilson_score_interval_estimator(sum_1,sum_2),sum_1,sum_2)
              
        if RT_PCR_type == 'oro_nasopharyngeal_indirect_RT_PCR':
            positive_RT_PCR_results=((processed_df[
                'oro_nasopharyngeal_indirect_RT_PCR_Ct_value'] < 
            ct_value_cutoff)&(processed_df['STT'] > STT_value_cutoff)) 
            positive_lamp_results=((processed_df.loc[
                positive_RT_PCR_results,
                'positive_gargle_water_direct_LAMP_result'])&
                (processed_df['STT'] > STT_value_cutoff))            
            sum_1=sum(positive_lamp_results)
            sum_2=sum(positive_RT_PCR_results)
            return (wilson_score_interval_estimator(sum_1,sum_2),sum_1,sum_2)        

analysis_results['40_Ct_sub_7_STT_sensitivity_oro_nasopharyngeal_indirect_'
    'RT_PCR']=(('Proportion: {1}/{2}, Wilson score interval: {0[1]:.1%}'
    ' (95% CI: {0[0]:.1%} - {0[2]:.1%})')
    .format(*sensitivity_calculator_STT(
        'oro_nasopharyngeal_indirect_RT_PCR',40,7,'upper'))) 

analysis_results['40_Ct_above_14_STT_sensitivity_gargle_water_indirect_'
    'RT_PCR']=(('Proportion: {1}/{2}, Wilson score interval: {0[1]:.1%}'
    ' (95% CI: {0[0]:.1%} - {0[2]:.1%})')
    .format(*sensitivity_calculator_STT(
        'gargle_water_indirect_RT_PCR',40,14,'lower'))) 

analysis_results['sub_30_Ct_sub_7_STT_sensitivity_oro_nasopharyngeal_indirect'
    '_RT_PCR']=(('Proportion: {1}/{2}, Wilson score interval: {0[1]:.1%}'
    ' (95% CI: {0[0]:.1%} - {0[2]:.1%})')
    .format(*sensitivity_calculator_STT(
        'oro_nasopharyngeal_indirect_RT_PCR',30,7,'upper'))) 

analysis_results['sub_30_Ct_above_14_STT_sensitivity_gargle_water_indirect'
    '_RT_PCR']=(('Proportion: {1}/{2}, Wilson score interval: {0[1]:.1%}'
    ' (95% CI: {0[0]:.1%} - {0[2]:.1%})')
    .format(*sensitivity_calculator_STT(
        'gargle_water_indirect_RT_PCR',30,14,'lower'))) 

analysis_results['sub_25_Ct_sub_7_STT_sensitivity_oro_nasopharyngeal_indirect'
    '_RT_PCR']=(('Proportion: {1}/{2}, Wilson score interval: {0[1]:.1%}'
    ' (95% CI: {0[0]:.1%} - {0[2]:.1%})')
    .format(*sensitivity_calculator_STT(
        'oro_nasopharyngeal_indirect_RT_PCR',25,7,'upper'))) 

analysis_results['sub_25_Ct_above_14_STT_sensitivity_gargle_water_indirect'
    '_RT_PCR']=(('Proportion: {1}/{2}, Wilson score interval: {0[1]:.1%}'
    ' (95% CI: {0[0]:.1%} - {0[2]:.1%})')
    .format(*sensitivity_calculator_STT(
        'gargle_water_indirect_RT_PCR',25,14,'lower'))) 

# Now we compare the two sampling methods by fixing the sample processing 
# method to indirect RT-PCR:

contingency_table_indirect_RT_PCR_results=pd.DataFrame(index=
    ['Total','Positive_gargle_water_RT_PCR','Negative_gargle_water_RT_PCR'],
    columns=['Positive_oro_nasopharyngeal_RT_PCR','Negative_oro_nasopharyngeal'
             '_RT_PCR'])

contingency_table_indirect_RT_PCR_results.loc['Total',
    'Positive_oro_nasopharyngeal_RT_PCR']=sum(
        processed_df['oro_nasopharyngeal_indirect_RT_PCR_Ct_value'] < 40)
contingency_table_indirect_RT_PCR_results.loc['Total',
    'Negative_oro_nasopharyngeal_RT_PCR']=sum(
        processed_df['oro_nasopharyngeal_indirect_RT_PCR_Ct_value'] > 40)

contingency_table_indirect_RT_PCR_results.loc['Positive_gargle_water_RT_PCR',
    'Positive_oro_nasopharyngeal_RT_PCR']=sum((
        processed_df['oro_nasopharyngeal_indirect_RT_PCR_Ct_value'] < 40)&
        (processed_df['gargle_water_indirect_RT_PCR_Ct_value'] < 40))
        
contingency_table_indirect_RT_PCR_results.loc['Positive_gargle_water_RT_PCR',
    'Negative_oro_nasopharyngeal_RT_PCR']=sum((
        processed_df['oro_nasopharyngeal_indirect_RT_PCR_Ct_value'] > 40)&
        (processed_df['gargle_water_indirect_RT_PCR_Ct_value'] < 40))

contingency_table_indirect_RT_PCR_results.loc['Negative_gargle_water_RT_PCR',
    'Positive_oro_nasopharyngeal_RT_PCR']=sum((
        processed_df['oro_nasopharyngeal_indirect_RT_PCR_Ct_value'] < 40)&
        (processed_df['gargle_water_indirect_RT_PCR_Ct_value'] > 40))
contingency_table_indirect_RT_PCR_results.loc['Negative_gargle_water_RT_PCR',
    'Negative_oro_nasopharyngeal_RT_PCR']=sum((
        processed_df['oro_nasopharyngeal_indirect_RT_PCR_Ct_value'] > 40)&
        (processed_df['gargle_water_indirect_RT_PCR_Ct_value'] > 40))


# Code for producing figure 1: 

# graph_path=input('Path for the directory to save the figure SVG files in: ')
graph_path=path  
 
cut_offs=np.arange(20,40.5,0.5)
percentage_of_positives=np.zeros(len(cut_offs))
percentage_of_positives_lower=np.zeros(len(cut_offs))
percentage_of_positives_upper=np.zeros(len(cut_offs))
patients_with_results=np.empty(len(cut_offs))
patients_with_results[:]=np.nan

for index, value in enumerate(cut_offs):
    patients_under_cut_off=sum(
        processed_df['oro_nasopharyngeal_indirect_RT_PCR_Ct_value']<value)
    (percentage_of_positives_lower[index],percentage_of_positives[index],
    percentage_of_positives_upper[index])=( 
    wilson_score_interval_estimator(sum(processed_df.loc[(
        processed_df['oro_nasopharyngeal_indirect_RT_PCR_Ct_value']<value)
        ,'positive_gargle_water_direct_LAMP_result']),patients_under_cut_off))
    if index % 2 == 0:
        patients_with_results[index]=patients_under_cut_off 

 
fig_1=go.Figure()
fig_1.add_traces(data=go.Scatter(x=cut_offs,y=percentage_of_positives,
                                 text=patients_with_results, 
                                 mode='lines+markers+text',
                                 marker=dict(color='black'),textfont_size=65))
fig_1.add_traces(data=go.Scatter(x=cut_offs,y=percentage_of_positives_lower,
                                 mode= 'lines+markers',fill=None,
                                 line_color='#FF5733'))
fig_1.add_traces(data=go.Scatter(x=cut_offs,y=percentage_of_positives_upper,
                                 mode='lines+markers',fill='tonexty',
                                 line_color='#FF5733'))
fig_1.update_traces(textposition='top center')
fig_1.add_annotation(ax=30.5,ay=0.95,x=30,y=0.65,xref='x',yref='y',axref='x',
  ayref='y',text=('Cumulative number of sample pairs with oro-nasopharyngeal'
                  ' RT-PCR results under a given Ct value'),font_size=70,
  showarrow= True,arrowhead=3,arrowsize=3,arrowwidth=2,arrowcolor='black')
fig_1.update_layout(showlegend=False,xaxis=dict(gridcolor='#DF5B3F',
    tickmode='linear',range=[19.75,40.35],tick0=20,dtick=2),
                    yaxis=dict(gridcolor='#DF5B3F',tickmode='linear',
    range=[0.30,1.04],tickformat= '%',dtick=0.05))
fig_1.update_layout(xaxis=dict(title=dict(text='Ct value cut-off',
                                font_size=70),tickfont_size=70))
fig_1.update_layout(yaxis=dict(title=dict(text='Sensitivity',font_size=70),
                                 tickfont_size=70))
fig_1.update_layout(legend_font_size=70,font_family="Myriad")
plotly.io.write_image(fig=fig_1,file=os.path.join(graph_path,'fig_1.svg'),
                      width=3600,height=1800,engine='kaleido')
        
# Code for producing figure 2:   


fig_2=make_subplots(rows=2, cols=2,row_heights=[0.1, 0.9],
                    column_widths=[0.9, 0.1],horizontal_spacing=0.02,
                    vertical_spacing=0.02)
fig_2.add_trace(go.Scatter(
    x=processed_df.loc[
        ((processed_df['oro_nasopharyngeal_indirect_RT_PCR_Ct_value']<40)
        &(processed_df['gargle_water_indirect_RT_PCR_Ct_value']>40)
        &(processed_df['positive_gargle_water_direct_LAMP_result']==True))
        ,'oro_nasopharyngeal_indirect_RT_PCR_Ct_value'],
    y=[0]*((processed_df['oro_nasopharyngeal_indirect_RT_PCR_Ct_value']<40)
        &(processed_df['gargle_water_indirect_RT_PCR_Ct_value']>40)
        &(processed_df['positive_gargle_water_direct_LAMP_result']==True)),
    mode='markers', marker_size=15,marker_color='#DECA02',
    legendgroup="positive LAMP results",showlegend=False),row=1,col=1)
fig_2.add_trace(go.Scatter(
    x=processed_df.loc[
        ((processed_df['oro_nasopharyngeal_indirect_RT_PCR_Ct_value']<40)
        &(processed_df['gargle_water_indirect_RT_PCR_Ct_value']>40)
        &(processed_df['positive_gargle_water_direct_LAMP_result']==False))
        ,'oro_nasopharyngeal_indirect_RT_PCR_Ct_value'],
    y=[0]*((processed_df['oro_nasopharyngeal_indirect_RT_PCR_Ct_value']<40)
        &(processed_df['gargle_water_indirect_RT_PCR_Ct_value']>40)
        &(processed_df['positive_gargle_water_direct_LAMP_result']==False)),
    mode='markers', marker_size=15,marker_color='#EA03B2',
    legendgroup="negative LAMP results",showlegend=False),row=1,col=1)
fig_2.update_xaxes(range=[0,40],tickfont_size=24,row=1,col=1)
fig_2.update_yaxes(showgrid=False,
                 zerolinecolor='black', zerolinewidth=3,
                 showticklabels=False,row=1,col=1)
fig_2.add_trace(go.Scatter(
    x=[0,40],y=[0,40],name='identity line',
    mode='lines',marker_size=15,marker_color='red',line_width=5),row=2, col=1)
fig_2.add_trace(go.Scatter(
    x=processed_df.loc[
        ((processed_df['oro_nasopharyngeal_indirect_RT_PCR_Ct_value']<40)
        &(processed_df['gargle_water_indirect_RT_PCR_Ct_value']<40)
        &(processed_df['positive_gargle_water_direct_LAMP_result']==True))
        ,'oro_nasopharyngeal_indirect_RT_PCR_Ct_value'], 
    y=processed_df.loc[
        ((processed_df['oro_nasopharyngeal_indirect_RT_PCR_Ct_value']<40)
         &(processed_df['gargle_water_indirect_RT_PCR_Ct_value']<40)
         &(processed_df['positive_gargle_water_direct_LAMP_result']==True))
        ,'gargle_water_indirect_RT_PCR_Ct_value'],
    mode='markers',marker_size=15,marker_color='#DECA02',
    legendgroup="positive LAMP results",name="positive LAMP result"),
    row=2, col=1)
fig_2.add_trace(go.Scatter(
    x=processed_df.loc[
        ((processed_df['oro_nasopharyngeal_indirect_RT_PCR_Ct_value']<40)
        &(processed_df['gargle_water_indirect_RT_PCR_Ct_value']<40)
        &(processed_df['positive_gargle_water_direct_LAMP_result']==False))
        ,'oro_nasopharyngeal_indirect_RT_PCR_Ct_value'], 
    y=processed_df.loc[
        ((processed_df['oro_nasopharyngeal_indirect_RT_PCR_Ct_value']<40)
         &(processed_df['gargle_water_indirect_RT_PCR_Ct_value']<40)
         &(processed_df['positive_gargle_water_direct_LAMP_result']==False))
        ,'gargle_water_indirect_RT_PCR_Ct_value'],
    mode='markers',marker_size=15,marker_color='#EA03B2',
    legendgroup="negative LAMP results",name="negative LAMP result")
    ,row=2, col=1)
fig_2.update_xaxes(range=[0,40],scaleanchor='y',
                 title=dict(text='Oro-nasopharyngeal RT-PCR Ct value',
                           font_size=45),tickfont_size=24,row=2,col=1)
fig_2.update_yaxes(range=[0,40],
                 title=dict(text='Gargle water RT-PCR Ct value',
                            font_size=45),tickfont_size=24,row=2,col=1)
fig_2.add_trace(go.Scatter(
    x=[0]*((processed_df['oro_nasopharyngeal_indirect_RT_PCR_Ct_value']>40)
        &(processed_df['gargle_water_indirect_RT_PCR_Ct_value']<40)
        &(processed_df['positive_gargle_water_direct_LAMP_result']==True)),
    y=processed_df.loc[
        ((processed_df['oro_nasopharyngeal_indirect_RT_PCR_Ct_value']>40)
        &(processed_df['gargle_water_indirect_RT_PCR_Ct_value']<40)
        &(processed_df['positive_gargle_water_direct_LAMP_result']==True))
        ,'gargle_water_indirect_RT_PCR_Ct_value'],           
    mode='markers', marker_size=15,marker_color='#DECA02',
    legendgroup="positive_LAMP_results",showlegend=False),row=2,col=2)
fig_2.add_trace(go.Scatter(
    x=[0]*((processed_df['oro_nasopharyngeal_indirect_RT_PCR_Ct_value']>40)
        &(processed_df['gargle_water_indirect_RT_PCR_Ct_value']<40)
        &(processed_df['positive_gargle_water_direct_LAMP_result']==False)),
    y=processed_df.loc[
        ((processed_df['oro_nasopharyngeal_indirect_RT_PCR_Ct_value']>40)
        &(processed_df['gargle_water_indirect_RT_PCR_Ct_value']<40)
        &(processed_df['positive_gargle_water_direct_LAMP_result']==False))
        ,'gargle_water_indirect_RT_PCR_Ct_value'],           
    mode='markers', marker_size=15,marker_color='#EA03B2',
    legendgroup="negative_LAMP_results",showlegend=False),row=2,col=2)
fig_2.update_xaxes(showgrid=False,
                 zeroline=True, zerolinecolor='black', zerolinewidth=3,
                 showticklabels=False,row=2,col=2)
fig_2.update_yaxes(range=[0,40],tickfont_size=24,row=2,col=2)
fig_2.update_layout(legend=dict(yanchor="top",y=0.85,xanchor="left",x=0.01,
                              font_size=30))
plotly.io.write_image(fig=fig_2,file=os.path.join(graph_path,'fig_2.svg'),
                      width=1800,height=1800,engine='kaleido')

# The poisson-binomial 0.0.1 package was not compatible with the version of 
# scipy (v. 1.6.1) being used. scipy.fft was previously a method, but now  is
# a class in v. 1.6.1 and so scipy.fft.fft must be called. Hence the source
# code of the module was copied here and changed such that it calls on 
# fft.fft(xs) instead of fft(xs)

class PoissonBinomial:
    
    def __init__(self,prob_array):
        self.p=np.array(prob_array)
        self.pmf=self.get_poisson_binomial()
        self.cdf=np.cumsum(self.pmf)
        
    def x_or_less(self,x):
        return self.cdf[x]
    def x_or_more(self,x):
        return 1-self.cdf[x]+self.pmf[x]

    def get_poisson_binomial(self):

        """This version of the poisson_binomial is implemented 
        from the fast fourier transform method described in 
        'On computing the distribution function for the 
        Poisson binomial distribution'by Yili Hong 2013."""

        real=np.vectorize(lambda x: x.real)

        def x(w,l):
            v_atan2=np.vectorize(atan2)
            v_sqrt=np.vectorize(sqrt)
            v_log=np.vectorize(log)

            if l==0:
                return complex(1,0)
            else:

                wl=w*l
                real=1+self.p*(cos(wl)-1)
                imag=self.p*sin(wl)
                mod=v_sqrt(imag**2+real**2)
                arg=v_atan2(imag,real)
                d=exp((v_log(mod)).sum())
                arg_sum=arg.sum()
                a=d*cos(arg_sum)
                b=d*sin(arg_sum)
                return complex(a,b)

        n=self.p.size 
        w=2*pi/(1+n)

        xs=[x(w,i) for i in range((n+1)//2+1)]
        for i in range((n+1)//2+1,n+1):
            c=xs[n+1-i]
            xs.append(c.conjugate())

        return real(fft.fft(xs))/(n+1)
    

X=np.array(processed_df.loc[
    processed_df['gargle_water_indirect_RT_PCR_Ct_value']<40,
                 'gargle_water_indirect_RT_PCR_Ct_value'].copy()).reshape(-1,1)
Y=np.array([1 if x else 0 for x in processed_df.loc[
    processed_df['gargle_water_indirect_RT_PCR_Ct_value']<40,
    'positive_gargle_water_direct_LAMP_result']])

validation_traning_split=np.array([True]*(len(X)-21)+[False]*21)
random.seed(1)
accuracy_scores=[]

for i in range(10000):
    random.shuffle(validation_traning_split)
    X_training, Y_training=(X[validation_traning_split,:],
     Y[validation_traning_split])
    X_testing, Y_testing=(X[~validation_traning_split,:],
     Y[~validation_traning_split])
    model_temp=LogisticRegression().fit(X_training,Y_training)
    accuracy_scores+= [model_temp.score(X_testing,Y_testing)]

analysis_results['logistic_model_monte_carlo_cross_validation_mean_accuracy']=(
    ('{0:.1%}').format(np.mean(accuracy_scores))) 

auroc_scores=[]
for i in range(10000):
    random.shuffle(validation_traning_split)
    X_training, Y_training=(X[validation_traning_split,:],
     Y[validation_traning_split])
    X_testing, Y_testing=(X[~validation_traning_split,:],
     Y[~validation_traning_split])
    model_temp=LogisticRegression().fit(X_training,Y_training)
    auroc_scores+= [roc_auc_score(Y_testing,model_temp.predict_proba(
        X_testing)[:, 1])]

analysis_results['logistic_model_monte_carlo_cross_validation_mean_auroc_'
                 'score']=('{0:.1%}').format(np.mean(auroc_scores))

logistic_model=LogisticRegression().fit(X,Y)
probs_pos_neg_LAMP=logistic_model.predict_proba(X)
design_matrix=np.hstack([np.ones((X.shape[0], 1)), X])
V=np.diagflat(np.product(probs_pos_neg_LAMP, axis=1))
cov_of_logit=np.linalg.inv(np.dot(np.dot(design_matrix.T, V), design_matrix))


sub_7_STT_positive_PCR_gargle_water_Ct_values=(np.array(processed_df.loc[
        (processed_df['STT']<=7)&
        (processed_df['gargle_water_indirect_RT_PCR_Ct_value']<40)
        ,'gargle_water_indirect_RT_PCR_Ct_value']))
sub_7_STT_positive_PCR_gargle_water_LAMP_results=(np.array(processed_df.loc[
        (processed_df['STT']<=7)&
        (processed_df['gargle_water_indirect_RT_PCR_Ct_value']<40)
        ,'positive_gargle_water_direct_LAMP_result']))
sup_14_STT_positive_PCR_gargle_water_Ct_values=(np.array(processed_df.loc[
        (processed_df['STT']>=14)&
        (processed_df['gargle_water_indirect_RT_PCR_Ct_value']<40)
        ,'gargle_water_indirect_RT_PCR_Ct_value']))
sup_14_STT_positive_PCR_gargle_water_LAMP_results=(np.array(processed_df.loc[
        (processed_df['STT']>=14)&
        (processed_df['gargle_water_indirect_RT_PCR_Ct_value']<40)
        ,'positive_gargle_water_direct_LAMP_result']))

def two_tailed_p_value_calculator(logistic_model,Ct_values,LAMP_results,
                                  high_or_low_STT,error='no'):
    assert high_or_low_STT in ['low','high']
    assert error in ['lower_bound','no','upper_bound']
    logit_terms=(logistic_model.intercept_[0]+
                 logistic_model.coef_[0][0]*Ct_values)  
    if ((error == 'lower_bound' and high_or_low_STT =='low') or 
        (error == 'upper_bound' and high_or_low_STT == 'high')):     
        logit_terms=(logit_terms-np.sqrt(cov_of_logit[0][0]+
            (cov_of_logit[1][1]*Ct_values*Ct_values)+
            (2*cov_of_logit[0][1]*Ct_values))*st.norm.ppf(1-0.05))
    if ((error == 'upper_bound' and high_or_low_STT =='low') or 
        (error == 'lower_bound' and high_or_low_STT == 'high')):
        logit_terms=(logit_terms+np.sqrt(cov_of_logit[0][0]+
            (cov_of_logit[1][1]*Ct_values*Ct_values)+
            (2*cov_of_logit[0][1]*Ct_values))*st.norm.ppf(1-0.05))  
    pb_distribution=PoissonBinomial(np.exp(logit_terms)
                                    /(1+np.exp(logit_terms)))
    if high_or_low_STT == 'low':
        first_tail=pb_distribution.x_or_more(sum(LAMP_results))
        second_tail=pb_distribution.x_or_less(
            ceil(2*sum(pb_distribution.p) - sum(LAMP_results)))
        return (first_tail+second_tail,sum(pb_distribution.p),
                sum(LAMP_results),len(LAMP_results))
    if high_or_low_STT == 'high':
        first_tail=pb_distribution.x_or_less(sum(LAMP_results))
        second_tail=pb_distribution.x_or_more(
            floor(2*sum(pb_distribution.p) - sum(LAMP_results)))
        return (first_tail+second_tail,sum(pb_distribution.p),
                sum(LAMP_results),len(LAMP_results))
           
sub_7_STT_p_value=two_tailed_p_value_calculator(
    logistic_model,sub_7_STT_positive_PCR_gargle_water_Ct_values,
    sub_7_STT_positive_PCR_gargle_water_LAMP_results,high_or_low_STT='low',
    error='no')       
sub_7_STT_p_value_lower=two_tailed_p_value_calculator(
    logistic_model,sub_7_STT_positive_PCR_gargle_water_Ct_values,
    sub_7_STT_positive_PCR_gargle_water_LAMP_results,high_or_low_STT='low',
    error='lower_bound')               
sub_7_STT_p_value_upper=two_tailed_p_value_calculator(
    logistic_model,sub_7_STT_positive_PCR_gargle_water_Ct_values,
    sub_7_STT_positive_PCR_gargle_water_LAMP_results,high_or_low_STT='low',
    error='upper_bound')   

analysis_results['sub_7_STT_two_tailed_p_value_with_95%_CI']=(
    ('Expected proportion: {0[1]:.1f}/{0[3]}, Empirical proportion: ' 
     '{0[2]}/{0[3]}, P-value interval: {0[0]:.2} (95% CI: {1[0]:.2} - '
     '{2[0]:.2%})').format(sub_7_STT_p_value,sub_7_STT_p_value_lower,
                           sub_7_STT_p_value_upper)) 

sup_14_STT_p_value=two_tailed_p_value_calculator(
    logistic_model,sup_14_STT_positive_PCR_gargle_water_Ct_values,
    sup_14_STT_positive_PCR_gargle_water_LAMP_results,high_or_low_STT='high',
    error='no')       
sup_14_STT_p_value_lower=two_tailed_p_value_calculator(
    logistic_model,sup_14_STT_positive_PCR_gargle_water_Ct_values,
   sup_14_STT_positive_PCR_gargle_water_LAMP_results,high_or_low_STT='high',
    error='lower_bound')               
sup_14_STT_p_value_upper=two_tailed_p_value_calculator(
    logistic_model,sup_14_STT_positive_PCR_gargle_water_Ct_values,
    sup_14_STT_positive_PCR_gargle_water_LAMP_results,high_or_low_STT='high',
    error='upper_bound')     
     
analysis_results['sup_14_STT_two_tailed_p_value_with_95%_CI']=(
    ('Expected proportion: {0[1]:.1f}/{0[3]}, Empirical proportion: ' 
     '{0[2]}/{0[3]}, P-value interval: {0[0]:.2} (95% CI: {1[0]:.2} - '
     '{2[0]:.2%})').format(sup_14_STT_p_value,sup_14_STT_p_value_lower,
                           sup_14_STT_p_value_upper)) 

# analysis_results_path=input('Path for the directory to save an '
# '"analysis results" text file in: ')
analysis_results_path=path  
with open(os.path.join(graph_path,analysis_results_path,
                       'analysis_results.txt'), 'w') as file:
    file.write(json.dumps(analysis_results,indent=4))