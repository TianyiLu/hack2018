
# coding: utf-8

# In[435]:


import os, sys, time
import pandas as pd
import numpy as np
from sklearn import linear_model
import numpy as np
import scipy.optimize
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import xlrd
import datetime


# In[436]:


def files_to_timestamp(path):
    files = [os.path.join(path, f) for f in os.listdir(path)]
    return dict ([(f, os.path.getmtime(f)) for f in files])

def F(y,x):
  # y is fix band
  # x is total band
   costperunit = 1.3
   costperGB = 0.009
   multipler = 37.5
   band_buffer = 1

   x_band = y   
   band_cost = x_band*costperunit/30*band_buffer

   x_flow = x - y
   for i in x_flow.index:
     if x_flow.values[i] < 0:
        x_flow.values[i] = 0

   flow_cost = sum(x_flow)/1024*costperGB*multipler

   #return_value = sum(x_flow)
   costfinal = band_cost+flow_cost
   return costfinal

def constraint1(y):
    return y-216520


def opti_cost(df): 
    df = pd.DataFrame(df)
    x = df["bandwidthInMbps"]
    bnds = (0,max(x))
    con1 = {'type': 'ineq', 'fun': constraint1} 
    cons = ([con1])
    
    x0 = np.zeros(1)
    x0= max(x)*0.5

    solution = minimize(F,x0,args = x,method='Nelder-Mead',tol=1e-6)
    bandwidth = solution.x[0]
    cost = solution.fun
    
    return bandwidth, cost
      
        


# In[437]:


def model_output(opts,df):
  # y is fix band
  # x is total band
   x = df["bandwidthInMbps"]
   y = opts[0]

   costperunit = 1.3
   costperGB = 0.009
   multipler = 37.5
   band_buffer = 1

   x_band = y   
   band_cost = x_band*costperunit/30*band_buffer

   x_flow = x - y
   for i in x_flow.index:
     if x_flow.values[i] < 0:
        x_flow.values[i] = 0

   flow_cost = sum(x_flow)/1024*costperGB*multipler
   
   #x_flow_df = pd.DataFrame(x_flow)
   #x_flow_df.columns = ["flow_bandwidth"]
   return x_flow


def get_latest_file(path,filter):
    #path = "data/input/"
    aList = []
    for i in os.listdir(path):
        if os.path.isfile(os.path.join(path,i)) and filter in i:
           filename = os.path.join(path,i)
           aList.append(filename)
    if not aList:
        return
    else: 
        latest_file = max(aList, key=os.path.getctime)
        return latest_file


# In[438]:


def xlsx_data_load(filename):
    workbook = xlrd.open_workbook(filename)
    worksheet = workbook.sheet_by_index(0)

    # Change this depending on how many header rows are present
    # Set to 0 if you want to include the header data.
    offset = 1
 
    rows = []
    for i, row in enumerate(range(worksheet.nrows)):
        if i <= offset:  # (Optionally) skip headers
           continue
        r = []
        for j, col in enumerate(range(worksheet.ncols)):
           r.append(worksheet.cell_value(i, j))
        rows.append(r)

    df = pd.DataFrame(rows)
    df.columns = ['timestamp', 'bandwidthInMbps']
    #workbook.close()
    return df


# In[439]:


def xldate_to_datetime(xldate):
  tempDate = datetime.datetime(1900, 1, 1)
  deltaDays = datetime.timedelta(days=int(xldate)-2)
  secs = (int((xldate%1)*86400)-60)
  detlaSeconds = datetime.timedelta(seconds=secs)
  TheTime = (tempDate + deltaDays + detlaSeconds )
  return TheTime.strftime("%Y-%m-%d %H:%M:%S")


# In[440]:


def datetime_convert(file,colname):
  file_new = file
  for i in file.index:
     file_new[colname][i] = xldate_to_datetime(file[colname][i])
  return file_new


# In[441]:


if __name__ == "__main__":

    #path_to_watch = sys.argv[1]
    path = "data/input/"
    outfilename = "data/output/Output_para.txt"
    
    print("Watching ", path)
    
    before = files_to_timestamp(path)

    while 1:
        time.sleep (2)
        after = files_to_timestamp(path)

        added = [f for f in after.keys() if not f in before.keys()]
        removed = [f for f in before.keys() if not f in after.keys()]
        modified = []

        for f in before.keys():
            if not f in removed:
                if os.path.getmtime(f) != before.get(f):
                    modified.append(f)

        if added:
                latest_base_file = get_latest_file(path,"Baseline_Bandwidths")
                latest_new_file = get_latest_file(path,"Customer_Bandwidths")
                
                base_data = xlsx_data_load(latest_base_file)
                new_customer_data = xlsx_data_load(latest_new_file)   
                
                costperunit = 1.3
                costperGB = 0.009
                multipler = 37.5
                band_buffer = 1   
                
                
                ##
                ## baseline results:
                cost_output = opti_cost(base_data)         
                
                ## output function for baseline
                x = base_data["bandwidthInMbps"]
                y = cost_output[0]

                x_band = y   
                band_cost = x_band*costperunit/30*band_buffer

                x_flow = x - y
                for i in x_flow.index:
                   if x_flow.values[i] < 0:
                      x_flow.values[i] = 0

                flow_cost = sum(x_flow)/1024*costperGB*multipler
                x_flow_df = pd.DataFrame(x_flow)
                x_flow_df.columns = ["flow_bandwidth"]
                
                base_data_output = base_data
                base_data_output["flow"] = x_flow_df
                base_data_output["band"] = base_data_output["bandwidthInMbps"] - base_data_output["flow"]
                
                #base_data_output_new = base_data_output
                #base_data_output_new = datetime_convert(base_data_output,"timestamp")
                
                outfilename_1 = "data/output/Base_output.csv"
                base_data_output.to_csv(outfilename_1,sep=',', index= False)
                
                
                columns = ["flow_cost","fix_band_cost", "mix_total_cost","flow_only_cost","band_only_cost","fix_band"]
                data = np.array([np.arange(6)]*1)
                base_para = pd.DataFrame(data, columns=columns)
                base_para = base_para.fillna(0) # with 0s rather than NaNs
                base_para["flow_cost"] = flow_cost
                base_para["fix_band_cost"] = band_cost
                base_para["mix_total_cost"] = band_cost+flow_cost
                base_para["flow_only_cost"] =  sum(base_data["bandwidthInMbps"])/1024*costperGB*multipler
                base_para["band_only_cost"] =  max(base_data["bandwidthInMbps"])*costperunit/30*band_buffer
                base_para["fix_band"] = y
                
                outfilename_2 = "data/output/Base_para.csv"
                base_para.to_csv(outfilename_2,sep=',', index= False)
                
                ## new customer
                combined_data = base_data
                combined_data["bandwidthInMbps"] = base_data["bandwidthInMbps"] + new_customer_data["bandwidthInMbps"]
                cost_output = opti_cost(combined_data)         
                
                ## output function for baseline
                print(combined_data["bandwidthInMbps"][1:5])
                x = combined_data["bandwidthInMbps"]
                y = cost_output[0]

                x_band = y   
                band_cost = x_band*costperunit/30*band_buffer

                x_flow = x - y
                for i in x_flow.index:
                   if x_flow.values[i] < 0:
                      x_flow.values[i] = 0

                flow_cost = sum(x_flow)/1024*costperGB*multipler
                x_flow_df = pd.DataFrame(x_flow)
                x_flow_df.columns = ["flow_bandwidth"]
                
                combined_data_output = combined_data
                combined_data_output["flow"] = x_flow_df
                combined_data_output["band"] = combined_data_output["bandwidthInMbps"] - combined_data_output["flow"]
                
                combined_data_output_new = combined_data_output
                #combined_data_output_new = datetime_convert(combined_data_output,"timestamp")
                
                outfilename_3 = "data/output/New_output.csv"
                combined_data_output_new.to_csv(outfilename_3,sep=',', index= False)
                
                
                columns = ["flow_cost","fix_band_cost", "mix_total_cost","flow_only_cost","band_only_cost","fix_band"]
                data = np.array([np.arange(6)]*1)
                new_para = pd.DataFrame(data, columns=columns)
                new_para = new_para.fillna(0) # with 0s rather than NaNs
                new_para["flow_cost"] = flow_cost
                new_para["fix_band_cost"] = band_cost
                new_para["mix_total_cost"] = band_cost+flow_cost
                new_para["flow_only_cost"] =  sum(combined_data_output["bandwidthInMbps"])/1024*costperGB*multipler
                new_para["band_only_cost"] =  max(combined_data_output["bandwidthInMbps"])*costperunit/30*band_buffer
                new_para["fix_band"] = y
                
                outfilename_4 = "data/output/New_para.csv"
                new_para.to_csv(outfilename_4,sep=',', index= False)
                
                ## incremental
                incremental_para = new_para - base_para
                outfilename_5 = "data/output/incremental_para.csv"
                incremental_para.to_csv(outfilename_5,sep=',', index= False)
                print(incremental_para)
                
        if removed: print("Removed: ", ", ".join(removed))
        if modified: print("modified: ", ", ".join(modified))
                   
        before = after

