
# coding: utf-8

# In[48]:


import os, sys, time
import pandas as pd
import numpy as np
from sklearn import linear_model
import numpy as np
import scipy.optimize
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import xlrd


# In[49]:


def files_to_timestamp(path):
    files = [os.path.join(path, f) for f in os.listdir(path)]
    return dict ([(f, os.path.getmtime(f)) for f in files])

def F(y,x):
   costperunit = 1.3
   costperGB = 0.009
   multipler = 37.5
   band_buffer = 1.1

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
    return solution.fun
      
        


# In[ ]:


if __name__ == "__main__":

    #path_to_watch = sys.argv[1]
    path = "data/input/"
    outfilename = "data/output/Output_para.txt"
    
    print("Watching ", path)
    
   # for i in os.listdir(path):
   #    if os.path.isfile(os.path.join(path,i)) and 'Baseline_Bandwidths' in i:
   #        filename = os.path.join(path,i)
    
    #df = pd.pandas.read_csv(filename, index_col=False, header=0)
    #cost_output = opti_cost(df)
    #print("Initial ", cost_output)
    #print(df[1:5])

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
                print(added[0])
                #df=pd.pandas.read_csv(added[0], index_col=False, header=0)
                #path = 'data/input/Baseline_Bandwidths_1532328714832.xlsx'

                workbook = xlrd.open_workbook(added[0])
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
                
                
                
                cost_output = opti_cost(df)
                print("New baseline", cost_output)
                print(df[1:5])
                
                text_file = open(outfilename, "w")
                writetext = str(cost_output)
                text_file.write(writetext)
                text_file.close()

               
        if removed: print("Removed: ", ", ".join(removed))
        if modified: print("modified: ", ", ".join(modified))
                   
        before = after


# In[ ]:


# bandwidth, flow split, optimal cost, lastest new customer, total cost, incremental cost, bandwidth flow incremental. total bandwidth cost and flow cost.


# In[38]:


get_ipython().system('pip install xlrd')


# In[47]:


import xlrd

path = 'data/input/Baseline_Bandwidths_1532328714832.xlsx'

workbook = xlrd.open_workbook(path)
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

rows_pd = pd.DataFrame(rows)
rows_pd.columns = ['timestamp', 'bandwidthInMbps']

#print 'Got %d rows' % len(rows) - offset
#print (rows_pd)  # Print column headings
#print (rows[offset])  # Print first data row sample

