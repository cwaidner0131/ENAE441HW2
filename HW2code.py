import numpy as np
import math
import matplotlib.pyplot as plt
import scipy
import scipy.integrate
import cartopy.crs as ccrs

def convert_to_rv(oe,mu): #converts orbital elements to r and v
    #intaializations
    r=[0,0,0]
    v=[0,0,0]
    
    a=oe[0]
    e=oe[1]
    i=oe[2]*math.pi/180
    omega=oe[3]*math.pi/180
    OMEGA=oe[4]*math.pi/180
    nu=oe[5]*math.pi/180
    p=a*(1-e**2)
    ri=p/(1+e*math.cos(nu))
    h=math.sqrt(mu*a*p)
    # equations
################
    #PQW frame
    r=np.array([[ri*math.cos((nu))],
                [ri*math.sin(nu)],
                [0]])
    v=math.sqrt(mu/p)*np.array([[-math.sin(nu)],
                                [(e+math.cos(nu))],
                                [0]])
    #313 rotation
    Rw=rotate(2,omega)
    Ri=rotate(0,i)
    RO=rotate(2,OMEGA)

    
    Rtot=np.transpose(Rw@Ri@RO)
    r_vec=Rtot@r
    v_vec=Rtot@v
    return r_vec,v_vec

def find_period(oe,mu): #finds period of orbit
    a=oe[0]
    return 2*math.pi*math.sqrt(a**3/mu)

def rotate(val,angle): #rotation matrix
    # value 0 corresponds to first column or x and so on
    if val==0:
        R=np.array([
        [1, 0, 0],
        [0, np.cos(angle), np.sin(angle)],
        [0, -np.sin(angle), np.cos(angle)]
        ])
    elif(val==1):
        R=np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
        ])
    elif(val==2):
        R=np.array([
        [np.cos(angle), np.sin(angle), 0],
        [-np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
        ])
    return R

def fun_def1(t,y): #numerical integration function
    #function format passed to scipy
    #initializations
    mu=3.986004418*math.pow(10,5)
    r=y[0:3]
    v=y[3:6]
    r_norm=np.linalg.norm(r)
    mu=int(mu)
    a1=-mu*r[0]/r_norm**3
    a2=-mu*r[1]/r_norm**3
    a3=-mu*r[2]/r_norm**3
    a=([a1,a2,a3])
    A=np.concatenate([v,a])
    return A

def convert_frame_ECI_to_Peri(r,v): #converts ECI to Perifocal frame
    #Converting to Perifocal frame
    R1=rotate(2,-oe[4]) # oe 4 is OMEGA
    R2=rotate(1,-oe[2]) # oe 2 is i
    R3=rotate(2,-oe[3]) # oe 3 is omega
    r=np.transpose(R3@R2@R1)@r #transpose is used to convert from ECI to Perifocal
    v=np.transpose(R3@R2@R1)@v 
    return r,v

def convert_frame_ECI_to_ECEF(r,v,tstep,rotation_rate): #converts ECI to ECEF frame
    delta_g=rotation_rate*tstep#+delta_go # initial angle zero
    R3=rotate(2,delta_g) 
    r=R3@r
    v=R3@v
    return r,v

def convert_frame_ECEF_to_AnglesOnly(ECEF_r): #converts ECEF to Topocentric frame
    lambda1 = math.atan2(ECEF_r[1] , ECEF_r[0])  # use atan2 to get the correct quadrant, NOTE DONT USE atan
    norm_r = math.sqrt(ECEF_r[0] ** 2 + ECEF_r[1] ** 2 + ECEF_r[2] ** 2)
    phi = math.asin(ECEF_r[2] / norm_r)  # Corrected from y.y[2] to y[2]
    return phi, lambda1

def propogate(r,v,tspan): #propogates orbit  using scipy
    y0=np.concatenate([r.flatten(),v.flatten()])
    t=np.linspace(0,tspan,1000)
    y=scipy.integrate.solve_ivp(fun_def1,(0,tspan),y0,method='DOP853',t_eval=t)
    return y

def plot_orbit(y,title,xaxis=None,yaxis=None,zaxis=None,color=None): #plots orbit, simple 3d plot after propogation
    fig=plt.figure()
    ax=plt.axes(projection='3d')
    ax.plot3D(y.y[0],y.y[1],y.y[2],color)
    plt.title(title)
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)  
    ax.set_zlabel(zaxis)
    plt.show()

def plot_orbit_non_ECI(x,y,z,title,xaxis=None,yaxis=None,zaxis=None,color=None): #plots orbit w/r to x y z coordinates
    fig=plt.figure()
    ax=plt.axes(projection='3d')
    ax.plot3D(x,y,z,color)
    plt.title(title)
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    ax.set_zlabel(zaxis)
    plt.show()

def plot_groundtrack(phi,lambda1,title,xaxis=None,yaxis=None,color=None): #plots groundtrack
    fig=plt.figure(figsize=(10,5))
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0))  # Use Cartopy's PlateCarree projection for lat/lon

    ax.set_global()
    ax.stock_img()
    ax.coastlines()
    ax.scatter(lambda1*180/math.pi,phi*180/math.pi) # Ground track of satellite
    ax.plot(-116.9141, 35, 'x', transform=ccrs.PlateCarree()) # Observer location
    
    plt.title(title)
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    plt.show()

def ECEF_vec_from_angles(phi,lambda1): #converts phi and lambda to ECEF unit vector
    r=np.zeros(3)
    r[0]=math.cos(phi)*math.cos(lambda1) # x
    r[1]=math.cos(phi)*math.sin(lambda1)# y
    r[2]=math.sin(phi) # z
    return r

def ECEF_to_Topo(r,phi,lambda1): #converts ECEF to Topocentric frame
    R1=rotate(2,-(90+lambda1))
    R2=rotate(0,-(90-phi))
    Rtot=np.transpose(R1@R2)
    r=Rtot@r
    return r

def polarplot(elevation,azimuth): #takes elevation and azimuth lists and then plots polar plot of Ground track w/r to observer
    # ngl no idea how this function works
    # Generate a polar projection plot
    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)

    # Mask out any negative elevations (below horizon)
    mask = elevation < 0
    el_plot = elevation.copy()
    el_plot[mask] = np.nan

    # NOTE:Polar plots accept angle and radius values.
    # To plot the elevation, we need to think of defining the radius as 90 - elevation. This is so
    # that the plot places 0 degrees elevation far away from the center, and 90 degrees at the center. 
    r = 90 - el_plot

    # WARNING: Azimuth is in radians, elevation is in degrees
    ax.plot(azimuth, r)

    # set the r ticks to be 90 at center and increment down by 15
    ax.set_rticks([0, 15, 30, 45, 60, 75, 90])  # Define the tick positions
    ax.set_yticklabels(['90°', '75°', '60°', '45°', '30°', '15°', '0°'])  # Customize the labels

    
    ax.set_theta_zero_location('N')  # Azimuth 0 points to North
    ax.set_theta_direction(-1)       # Clockwise azimuth
    plt.show()

if __name__=="__main__":

    ################ Problem 1 ################

    mu=398600
    oe=[7000,.05,45,30,60,0] #In order a e i w W nu
    r,v=convert_to_rv(oe,mu)  # ECI frame
    rotation_rate=7.2911*math.pow(10,-5)
    tspan=24*3600 #24 hours
    
    # ECI frame orbit
    y=propogate(r,v,tspan)

    # Initialize lists to store r and v values
    t=y.y.shape[1]
    ECEF_r=np.zeros(t)
    ECEF_v=np.zeros(t)
    x1=np.zeros(t)
    y1=np.zeros(t)
    z1=np.zeros(t)
    x2=np.zeros(t)
    y2=np.zeros(t)
    z2=np.zeros(t)
    PERIFOCAL_r=np.zeros(t)
    PERIFOCAL_v=np.zeros(t)

    #Converting to ECEF and Perifocal frame
    for i in range(y.y.shape[1]):
        tstep = i * (tspan / t)
        ECEF_r,ECEF_v=convert_frame_ECI_to_ECEF(y.y[0:3,i],y.y[3:6,i],tstep,rotation_rate)
        x1[i]=ECEF_r[0]
        y1[i]=ECEF_r[1]
        z1[i]=ECEF_r[2]
        PERIFOCAL_r,PERIFOCAL_v=convert_frame_ECI_to_Peri(y.y[0:3,i],y.y[3:6,i])
        x2[i]=PERIFOCAL_r[0]
        y2[i]=PERIFOCAL_r[1]
        z2[i]=PERIFOCAL_r[2]

    #Plotting orbits
    plot_orbit(y,'ECI frame orbit','x','y','z','red')
    plot_orbit_non_ECI(x1,y1,z1,'ECEF frame orbit','i1','i2','i3','blue')
    plot_orbit_non_ECI(x2,y2,z2,'Perifocal frame orbit','i','p ','h','green')



    ################ Problem 2 ################
    oe1=[6798,.007,51.6*2, 0 ,215,0] #In order a e i w W nu
    oe2=[26560,.02,55,0,215,0] #In order a e i w W nu
    oe3=[26600,.74,63.4,270,80,0]#In order a e i w W nu
    oe4=[42164,0.02,0,0,35,0] #In order a e i w W nu

    #Converting to r and v from OE
    r1,v1=convert_to_rv(oe1,mu)
    r2,v2=convert_to_rv(oe2,mu)
    r3,v3=convert_to_rv(oe3,mu)
    r4,v4=convert_to_rv(oe4,mu)

    # ECI frame orbit propogation
    y1=propogate(r1,v1,tspan)
    y2=propogate(r2,v2,tspan)
    y3=propogate(r3,v3,tspan)
    y4=propogate(r4,v4,tspan)
    
    # Initialize lists to store phi and lambda values
    t=y1.y.shape[1]
    phi_values1=np.zeros(t)
    phi_values2=np.zeros(t)
    phi_values3=np.zeros(t)
    phi_values4 = np.zeros(t)
    lambda_values1=np.zeros(t)
    lambda_values2= np.zeros(t)
    lambda_values3=np.zeros(t)
    lambda_values4 = np.zeros(t)

    #Converting to Anglesonly values for ECEF frame from ECI frame
    for i in range(y1.y.shape[1]):
        
        tstep = i * (tspan / t)

        ECEF_r1,ECEF_v1=convert_frame_ECI_to_ECEF(y1.y[0:3,i],y1.y[3:6,i],tspan,rotation_rate)
        phi1,lambda1=convert_frame_ECEF_to_AnglesOnly(ECEF_r1) # Convert ECEF to angles
        phi_values1[i]=phi1
        lambda_values1[i]=lambda1

        ECEF_r2,ECEF_v2=convert_frame_ECI_to_ECEF(y2.y[0:3,i],y2.y[3:6,i],tstep,rotation_rate)
        phi2,lambda2=convert_frame_ECEF_to_AnglesOnly(ECEF_r2) # Convert ECEF to angles
        phi_values2[i]=phi2
        lambda_values2[i]=lambda2

        ECEF_r3,ECEF_v3=convert_frame_ECI_to_ECEF(y3.y[0:3,i],y3.y[3:6,i],tstep,rotation_rate)
        phi3,lambda3=convert_frame_ECEF_to_AnglesOnly(ECEF_r3) # Convert ECEF to angles
        phi_values3[i]=phi3
        lambda_values3[i]=lambda3

        ECEF_r4,ECEF_v4=convert_frame_ECI_to_ECEF(y4.y[0:3,i],y4.y[3:6,i],tstep,rotation_rate)
        phi4,lambda4=convert_frame_ECEF_to_AnglesOnly(ECEF_r4) # Convert ECEF to angles
        phi_values4[i]=phi4
        lambda_values4[i]=lambda4
    
    
    #Plotting topocentric frame
    plot_groundtrack(phi_values1,lambda_values1,'Topocentric frame orbit','phi','lambda','red')
    plot_groundtrack(phi_values2,lambda_values2,'Topocentric frame orbit','phi','lambda','blue')
    plot_groundtrack(phi_values3,lambda_values3,'Topocentric frame orbit','phi','lambda','green')
    plot_groundtrack(phi_values4,lambda_values4,'Topocentric frame orbit','phi','lambda','black')

    ################ Problem 3 ################

    #Phi,lambda values of observer
    phi=35.2967*math.pi/180
    lambda1=-116.9141*math.pi/180

    #Spacecraft position
    rmag_earthrad=6378
    R1=rotate(2,-(math.pi/2+lambda1))
    R2=rotate(0,-(math.pi/2-phi))
    Rtot=np.transpose(R2@R1)

    #Range vector w/r to observer
    R_vec=ECEF_vec_from_angles(phi,lambda1)*rmag_earthrad

    # Initialize lists to store azimuth and elevation values
    range_vec1=np.zeros(t)
    azimuth1=np.zeros(t)
    elevation1=np.zeros(t)
    azimuth2=np.zeros(t)
    elevation2=np.zeros(t)
    azimuth3=np.zeros(t)
    elevation3=np.zeros(t)
    azimuth4=np.zeros(t)
    elevation4=np.zeros(t)

    #Converting to ECEF frame and then to Topocentric frame to get azimuth and elevation
    for i in range(y1.y.shape[1]):
        tstep = i * (tspan / t)

        ECEF_r1,ECEF_v1=convert_frame_ECI_to_ECEF(y1.y[0:3,i],y1.y[3:6,i],tstep,rotation_rate) # ECEF frame from ECI frame
        range_vec1=Rtot@(ECEF_r1-R_vec) # Range vector w/r to observer
        azimuth1[i]=math.atan2(range_vec1[1],range_vec1[0]) # Azimuth NOTE in radians
        elevation1[i]=math.asin(range_vec1[2]/np.linalg.norm(range_vec1))*180/math.pi # Elevation NOTE in degress
        #Repeat below...

        ECEF_r2,ECEF_v2=convert_frame_ECI_to_ECEF(y2.y[0:3,i],y2.y[3:6,i],tstep,rotation_rate)
        range_vec2=Rtot@(ECEF_r2-R_vec)
        azimuth2[i]=math.atan2(range_vec2[1],range_vec2[0])
        elevation2[i]=math.asin(range_vec2[2]/np.linalg.norm(range_vec2))*180/math.pi

        ECEF_r3,ECEF_v3=convert_frame_ECI_to_ECEF(y3.y[0:3,i],y3.y[3:6,i],tstep,rotation_rate)
        range_vec3=Rtot@(ECEF_r3-R_vec)
        azimuth3[i]=math.atan2(range_vec3[1],range_vec3[0])
        elevation3[i]=math.asin(range_vec3[2]/np.linalg.norm(range_vec3))*180/math.pi

        ECEF_r4,ECEF_v4=convert_frame_ECI_to_ECEF(y4.y[0:3,i],y4.y[3:6,i],tstep,rotation_rate)
        range_vec4=Rtot@(ECEF_r4-R_vec)
        azimuth4[i]=math.atan2(range_vec4[1],range_vec4[0])
        elevation4[i]=math.asin(range_vec4[2]/np.linalg.norm(range_vec4))*180/math.pi

    #Plotting polar plots
    polarplot(elevation1,azimuth1)
    polarplot(elevation2,azimuth2)
    polarplot(elevation3,azimuth3)
    polarplot(elevation4,azimuth4)
        
        



        


