#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# import all the required modules
import os, numpy as np,pandas as pd, math,matplotlib as mpl,matplotlib.pyplot as plt

from numpy import exp

from lmfit import minimize, Parameters, Parameter, report_fit



# Modify global settings
mpl.rcParams['font.family'] = 'sans'  # Set the font family to a generic serif font
mpl.rcParams['font.sans-serif'] = 'Times New Roman'  # Specify Times New Roman as the serif font
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['font.size'] = 12
mpl.rcParams['figure.figsize'] = 7.2, 8
mpl.rcParams['axes.titlesize'] = 13
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['lines.linewidth'] = 2
mpl.rcParams['lines.markersize'] = 6
mpl.rcParams['legend.fontsize'] = 12
mpl.rcParams['mathtext.fontset'] = 'stix'


# In[2]:


## Reads processed data and convert to dataframe
def read_data(data):
    if not os.path.exists(data):
        print( f" Output file '{data}' does not exist. Use process_covid_data.py to create file")
    else:
        return pd.DataFrame(pd.read_excel(data))


# In[3]:


# Defined the structure (ordinary differential equation )of model 
def Model(x, params):
    beta = params['betasymptomatic']
    delta = params['delta']
    gamma = params['gamma']
    lambd = params['lambd']
    N = 360404816


    
    xdot = np.array([
        - (beta * x[0] * (x[2] / N)),
        (beta * x[0] * (x[2] / N)) - lambd * x[1],
        lambd * x[1] - (gamma + delta) * x[2],
        (gamma * x[2]),
        delta * x[2],
        lambd * x[1],
        gamma * x[2],
        delta * x[2]
    ])
    return xdot


# In[4]:


# Runge-Kutta 4th order method for integration
def RungeKutta4(f, x0, t0, tf, dt):
    t = np.arange(t0, tf, dt)
    nt = t.size
    nx = x0.size
    x = np.zeros((nx, nt))
    x[:, 0] = x0
    
    for k in range(nt-1):
        k1 = dt * f(t[k], x[:, k])
        k2 = dt * f(t[k] + dt / 2, x[:, k] + k1 / 2)
        k3 = dt * f(t[k] + dt / 2, x[:, k] + k2 / 2)
        k4 = dt * f(t[k] + dt, x[:, k] + k3)
        dx = (k1 + 2 * k2 + 2 * k3 + k4) / 6
        x[:, k+1] = x[:, k] + dx
    return x, t


# In[5]:


# Calculate error between model and data
def error(params, x0, t0, tf, dt, data):
    f = lambda t, x: Model(x, params)
    x, t = RungeKutta4(f, x0, t0, tf, dt)
    error_indices = [5, 7]
    model_data = x.T[:, error_indices]
    error = (model_data - data)*(0.7,1)
    return error.ravel()


# In[6]:


# Set model parameters and the range of value for parameters for optimization process
def set_parameters(t):
    beta = 0.20
    delta = (1/60)/365
    gamma_ = [0.03333333, 0.03333333, 0.03333333, 0.04736315, 0.06534143,
                    0.09213795, 0.10268256, 0.09396827, 0.10130317, 0.10758116,
                    0.10584565]
    lambd_= [0.2       , 0.5       , 0.2       , 0.27286684, 0.2       ,
       0.2201717 , 0.23415593, 0.26252763, 0.24820297, 0.20178826,
       0.29873344]
    thresholds = [(11, 0),
        (37, 1),
        (51, 2),
        (79, 3),
        (140, 4),
        (191, 5),
        (201, 6),
        (215, 7),
        (227, 8),
        (253, 9)
    ] 
  
    #print (beta,delta,gamma,lambd)
    gamma = gamma_[-1]  # Default value
    lambd = lambd_[-1]  # Default value
    for threshold, i in thresholds:
        if t == threshold:
            
            gamma = gamma_[i]
            lambd = lambd_[i]
            break  # Exit loop once a match is found
    params = Parameters()
    params.add("betasymptomatic", value=beta, vary=True, min=0, max=1)
    params.add("lambd", value=lambd, vary=False)
    params.add("delta", value=delta, vary=True, min=0, max=0.1)
    params.add('gamma', value=gamma, vary=False)
    return params


# In[7]:


def main(df):
    
    
    
    x0 = np.array([(360404816 - 5 - 13 - 0 - 0), 5, 13, 0, 0, 13, 0, 0])
    tf = [11, 37, 51, 79, 140, 191, 201, 215, 227, 253, 293]
    t0 = 0
    delta1 = []
    gamma1 = []
    lambd1 = []
    beta1 = []

    for a in tf:
        tf = a
        dt = 1
        data = df.loc[t0: (tf - 1), ['Cinfected',  'Cdeath']].values
        result = minimize(error, set_parameters(tf), args=(x0, t0, tf, dt, data), method='least_squares')
        beta = result.params['betasymptomatic'].value
        delta = result.params['delta'].value
        gamma = result.params['gamma'].value
        lambd = result.params['lambd'].value
        
       
        
        params1 = set_parameters(tf)
        
        params1.add('lambd', value=lambd, vary=False)
        params1.add('delta', value=delta, vary=False)
        params1.add('gamma', value=gamma, vary=False)
        params1.add('betasymptomatic', value=beta, vary=False)
        
        delta1.append(delta)
        gamma1.append(gamma)
        lambd1.append(lambd)
        beta1.append(beta)
        seir = lambda t, x: Model(x, params1)
        x, t = RungeKutta4(seir, x0, t0, tf, dt)
        num = tf - t0 - 1
        t0 = tf - 1
        x0 = np.array([x[0, num], x[1, num], x[2, num], x[3, num], x[4, num], x[5, num], x[6, num], x[7, num]])
        
    b = np.array(beta1).reshape(1, 11)
    c = np.array(gamma1).reshape(1, 11)
    d = np.array(delta1).reshape(1, 11)
    e = np.array(lambd1).reshape(1, 11)
    par = np.concatenate([b, c, d, e], axis=0).T
    return par
if __name__ == "__main__":
    df = read_data("CentralCOVID-rolling-avereage.xlsx")
    par = main(df)

    


# In[8]:


#Function to pick different parameters based on the thresholds
def params(t):
    thresholds = [(11, 0),
        (37, 1),
        (51, 2),
        (79, 3),
        (140, 4),
        (191, 5),
        (201, 6),
        (215, 7),
        (227, 8),
        (253, 9)
    ]
    
    for threshold, i in thresholds:
        if t < threshold:
            params = {'betasymptomatic': par[i, 0], 'gamma': par[i, 1], 'delta': par[i, 2], 'lambd': par[i, 3]}
            return params
            
    params = {'betasymptomatic': par[10, 0], 'gamma': par[10, 1], 'delta': par[10, 2], 'lambd': par[10, 3]}
    return params


# In[9]:


#run the model with optimized parameter
f =lambda t , x : Model(x,params(t))
x0 = np.array([(360404816 - 5 - 13 - 0 - 0), 5, 13, 0, 0, 13, 0, 0])
t0 = 0
tf =293
dt =1
tesko,t = RungeKutta4 (f,x0,t0,tf,dt)


# In[10]:



#(pd.DataFrame(par)).to_excel("India_parameter.xlsx")
        


# In[11]:


def latentsamplehypercube(n):
    c = []
    for i in range(11):
        b = []
        
        for j in range(4):
            lower_limits = par[i,j]- 0.075 * par[i,j]
            upper_limits = par[i,j]+ 0.075 * par[i,j]
            np.random.seed(101)
            points = np.random.uniform(low=lower_limits,high = upper_limits, size = [1,n]).T
            #print(points)
            np.random.shuffle(points)
            b.append(points)   
        c.append(b)
    #e=np.array(p)
            
    return np.array(c).flatten().reshape(11,4*n)


# In[12]:


n = 1000
parameter_sample =latentsamplehypercube(n)


# In[13]:


def params2(t):
    global iter
    if t < 11:
        i =0
    elif t < 37:
        i =1
    elif t < 51:
        i =2
    elif t< 79:
        i =3
    elif t < 140:
        i=4 
    elif t <191:
        i=5 
    elif t< 201:
        i=6 
    elif t< 215:
        i =7
    elif t< 227:
        i = 8
    elif t< 253:
        i =9
    else:
        i =10
        
    params2 = {'betasymptomatic':parameter_sample[i,iter],'gamma':parameter_sample[i,iter+n],'delta':parameter_sample[i,iter+2*n],'lambd':parameter_sample[i,iter+3*n]}
    return params2


# In[14]:


#solve the diff equation
g =lambda t , x : Model(x,params2(t))
y = []
for iter in range (n):
    t0 = 0
    tf =293
    dt =1
    x,t = RungeKutta4 (g,x0,t0,tf,dt)
    y.append(x)


# In[15]:


a = []
for i in range(n):
    if np.array(a).size < 1: 
        #print ("non")
        a = y[i]
        #print(a)
    else:
        a = a+y[i]

avg = a/n
avg_T = avg.T
avg_T = pd.DataFrame(avg_T)


# In[16]:


#calculate the percentile
low_percentile_infected= []
upper_percentile_infected =[]

q1_values_infected = []
q3_values_infected = []

low_percentile_deceased= []
upper_percentile_deceased =[]

q1_values_deceased = []
q3_values_deceased = []
for i in range (0,293):
    a1 =[]
    a2 =[]
    for j in range(0,n):
        a1.append(y[j][5][i])
        a2.append(y[j][7][i])
    low_percentile_infected.append(np.percentile(a1, 2.5))
    upper_percentile_infected.append(np.percentile(a1, 97.5))
    q1_values_infected.append(np.percentile(a1, 25))
    q3_values_infected.append(np.percentile(a1, 75))
    
    # Calculate percentiles for the deceased data
    low_percentile_deceased.append(np.percentile(a2, 2.5))
    upper_percentile_deceased.append(np.percentile(a2, 97.5))
    q1_values_deceased.append(np.percentile(a2, 25))
    q3_values_deceased.append(np.percentile(a2, 75))
df_time = read_data("India_COVID_rolling_average.xlsx")   


# In[17]:


fig,(ax1,ax2) = plt.subplots(2,1)
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)

#plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.4)
ax1.set_title('C',fontsize =14,loc ="right")
ax2.set_title('D',fontsize =14,loc ="right")

ax1.text(0.5, 0.9, 'Central zone', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
#ax1.set_xlabel("Days since Mar 14 2020",fontsize =12)
ax1.set_ylabel("Cummulative Number of Infected",fontsize = 12)
ax2.set_xlabel("Days since Mar 14 2020*",fontsize =12)
ax2.set_ylabel("Cummulative Number of Deceased",fontsize = 12)
ax1.fill_between(df_time["Time"],low_percentile_infected, upper_percentile_infected,color ="lightgrey")
ax1.fill_between(df_time["Time"], q1_values_infected, q3_values_infected, color="lightcoral")
ax1.plot(df_time["Time"],avg_T[5],linestyle="--",color ="black", label ="Average", alpha = None)
ax1.plot(df_time["Time"],tesko[5],color ="black", label ="Model fit",alpha = None)
ax1.scatter(df_time["Time"][::5],df["Cinfected"][::5],marker = "o",facecolors="None", label = "Reported",edgecolor = "blue")
#ax.plot(df["Time"],avg_T[6],linestyle="--",color ="black", label ="average")


#ax1.plot(df["Time"],df["Average infected"],linestyle = None, marker = "o",markersize =5,markerfacecolor='None',label = "Reported",markeredgecolor = "blue",markevery=5)



ax1.legend(loc='upper left',fontsize = 12)
ax2.fill_between(df_time["Time"],low_percentile_deceased, upper_percentile_deceased,color = "lightgrey")
ax2.fill_between(df_time["Time"], q1_values_deceased, q3_values_deceased, color="lightcoral")

#ax.plot(df["Time"],avg_T[6],linestyle="--",color ="black", label ="average")

ax2.plot(df_time["Time"],avg_T[7],linestyle="--",color ="black", label ="Average",alpha = None)
ax2.plot(df_time["Time"],tesko[7],color ="black", label ="Model fit",alpha = None)
ax2.scatter(df_time["Time"][::5],df["Cdeath"][::5],label = "Reported", marker = "o",facecolors="None",edgecolor = "blue")
ax2.ticklabel_format(axis = 'y',style ='sci',scilimits =(0,0))

ax2.sharex(ax1)
#plt.xticks(fontsize=14)
#plt.yticks(fontsize=14)
ax1.legend(loc='upper left',fontsize = 12)
ax2.legend(loc='upper left',fontsize = 12)
fig.tight_layout()
fig.savefig(os.path.join('Output','Central_fit6.png'),dpi = 300)
plt.close()


# In[18]:


par


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




