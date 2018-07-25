#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate, scipy.integrate
import scipy.io as sio
import def_radius
from scipy.optimize import curve_fit

def func(x, a, b, c):
  return 5.5 * np.exp(-6.0 * (x/1500)**1.0) + 13.6

plt.ion()

binprec = '>f4'

flag_plot = 0

# # physical constants
rho_const = 999.8
alphaK = 2.0e-4
g0 = 9.8
f0 = 6.5e-5
# options
barotrop_mod = 0 # to turn on barotopic eddy mode
sponge = 1 # 1 with sponge, 0 no sponge

#% ================== NEW GRID =====================================
si_x = 250
si_y = 250
si_z = 75
si_x1 = si_x + 1
si_y1 = si_y + 1
si_z1 = si_z + 1

# in m
Lx = 300.0e3
Ly = 300.0e3
Lz = 1500

dx = Lx/si_x;
dy = Ly/si_y;

# == Gradual horizontal grid ==
# To create dissipation near the boundaries due to unresolved fields caused by increasing grid 

if sponge == 1:
  dxs = [dx*16,dx*8,dx*4,dx*2,dx,dx*2,dx*4,dx*8,dx*16]
  si_xs = [10,10,10,10,si_x,10,10,10,10]
  si_xssum = np.cumsum(si_xs)
#
  xx1=np.linspace(0,si_xssum[-1],si_xssum[-1]+1)
  xx1[0]=0.0
  for ix in np.arange(1,si_xssum[0]+1):
    xx1[ix]=xx1[ix-1]+dxs[0]
  for ii in np.arange(1,si_xssum.size):
    for ix in np.arange(si_xssum[ii-1]+1,si_xssum[ii]+1):
      xx1[ix]=xx1[ix-1]+dxs[ii]
#
  yy1 = xx1 # ok for square domain
#
  dx1 = np.diff(xx1)
  dy1 = np.diff(yy1)
#
  xx = xx1[0:-1] + 0.5*dx1
  yy = yy1[0:-1] + 0.5*dy1
  si_x = xx.size
  si_y = yy.size
#
else:
  xx = Lx*(np.arange(0,si_x) + 0.5)/(1.0*si_x)
  yy = Ly*(np.arange(0,si_y) + 0.5)/(1.0*si_y)
  #zz = Lz*(np.arange(0,si_z) + 0.5)/(1.0*si_z)
#
  xx1 = Lx*(np.arange(0,si_x+1) )/(1.0*si_x)
  yy1 = Ly*(np.arange(0,si_y+1) )/(1.0*si_y)
#
  dx1 = dx*np.ones((si_x))
  dy1 = dy*np.ones((si_y))

#
xg,yg = np.meshgrid(xx,yy) 
xu,yu = np.meshgrid(xx1[:-1],yy) 
xv,yv = np.meshgrid(xx,yy1[:-1]) 
xc,yc = np.meshgrid(xx1,yy1) 

# Vertical grid 
if barotrop_mod ==1:
#create uniform vertical grid
  # xf is % of grid points
  xf = [0, 0.5, 1]
  # yf is % of thickness
  yf = [0, 0.5, 1]
else:
#create gradual vertical grid
  # xf is % of grid points
  xf = [0, 0.4, 0.6, 0.8, 0.9, 1]
  # yf is % of thickness
  yf = [0, 0.04, 0.10, 0.21, 0.4, 1]

hh = np.linspace(0,1,si_z1)
zf = Lz*np.interp(hh,xf,yf)

# smooth
nc = int(si_z/10)
if nc % 2 == 0:
  nc = nc + 1
zz2 = np.convolve(zf, np.ones((nc,))/nc, mode='valid')

zf[int((nc-1)/2):int(-(nc-1)/2)] = zz2

if flag_plot:
  plt.figure()
  plt.plot(hh,zz/Lz,'k')
  plt.plot(hh,hh,'k--')
  plt.plot(xf,yf,'.')
  # plt.savefig(outputdir2 + 'vert_res.png')
  plt.savefig('vert_res.png')
  plt.close()

dz1 = np.diff(zf)

iz = np.argmin(np.abs(zf-500.0))

print ('dx= ', dx)
print ('min dz: ', np.min(dz1))
print ('max dz: ', np.max(dz1))
print ('nb layers above 500m:', iz, '/', si_z)

if np.sum(dz1 < 0) > 0:
  print ('you need you change the polynomial fit!')

zc = zf[0:-1] + 0.5*dz1


dx1.astype(binprec).tofile('dx.box')
dy1.astype(binprec).tofile('dy.box')
dz1.astype(binprec).tofile('dz.box')


#% ============== background density profile ===================
if barotrop_mod == 1:
  stratification = 0
else:
  stratification = 1 # choose stratification here
#overwrite strat
#stratification = 0
# Constant stratification
if stratification == 0:
  N2 = 1e-5#0.0 #3e-5 0 for barotopic eddy
  temp_i = -N2/g0/alphaK*zc
  temp_i = temp_i - temp_i[-1]
elif stratification == 1:
  temp_ref_argo=sio.loadmat('temp_ref_argo.mat')
  temp_ref = temp_ref_argo['temp']
  temp_ref_art=sio.loadmat('art_temp_ref.mat')
  temp_ref1 = temp_ref_art['mPtempout']
#  temp_ref_argo=sio.loadmat('wintertemp.mat')
#  temp_ref = temp_ref_argo['winter_mtemp']
#
#  N2 = 3e-5#0.0 #3e-5 0 for barotopic eddy
#  temp_i = -N2/g0/alphaK*zc
#  temp_i = temp_i - temp_i[-1]
#
  rho_dataie=sio.loadmat('rhodata.mat')
  rho_data = rho_dataie['ewq']
  r_rho = np.linspace(-100,100,201)
  z_ref=np.linspace(5,2000,400)
# Make smooth
  tref_opt, tref_cov = curve_fit(func, z_ref, temp_ref1[:,0])
  tref_smooth = func(zc, *tref_opt)
  tref_art= np.interp(zc,z_ref,temp_ref1[:,0])
#Choose the background profile 
  tref_aut= np.interp(zc,z_ref,temp_ref[:,0])
  temp_i = tref_smooth #tref_aut # tref_smooth
  rho_anomd = np.interp(zc,z_ref,rho_data[:,100])

temp_i = temp_i.reshape((si_z,1,1))
rho_anomd = rho_anomd.reshape((si_z,1,1))

#%==================== SST - LAND ===================================

landh  = np.zeros((si_y,si_x));
theta  = np.zeros((si_z,si_y,si_x));
uvel   = np.zeros((si_z,si_y,si_x));
vvel   = np.zeros((si_z,si_y,si_x));

H = dz1.cumsum()[-1]
landh = -H + landh

#gaussian eddy
x_c = xx[int(si_x/2)] # x center
y_c = yy[int(si_y/2)] # y center
Rmax = 25e3    # radius of max velocity of eddy

DeltaT = 2.0   # SST anomaly
z0 = 200.0     # characteristic depth (m)
vmax = -0.26     # m/s

vel_profile = 1 # Horizontal velocity profile function (1: exponential or gaussian ; 2:Hyperbolic)
alpha = 2 # alpha = 2 for gaussian

###====== Vertical profile =================
if barotrop_mod == 1:
# vertical profile
  FZ = 1+0.0*zc
# vertical derivative
  FpZ = 0.0*zc
else:
# vertical profile
#  FZ = 1-scipy.special.erf(zc/z0)
  FZ = np.exp(-zc**2/z0**2/2)
# vertical derivative
#  FpZ = -1/z0*2/np.sqrt(np.pi)*np.exp(-zc**2/z0**2)
  FpZ = -2*zc/(2*z0**2)*np.exp(-zc**2/z0**2/2)


FZ = FZ.reshape(si_z,1,1)
FpZ = FpZ.reshape(si_z,1,1)

###====== Horizontal velocity profile ======
def vel_hor(rr, vel_profile):
  if vel_profile == 1: # exponential profile
    v = -(vmax*(rr/Rmax))*np.exp((1-(rr/Rmax)**alpha)/alpha)
  elif vel_profile == 2: # hyperbolic vortex (~rankine)
#    v = -vmax*np.tanh(rr/R0)/(np.cosh(rr/R0))**2/(np.tanh(1.0)/(np.cosh(1.0))**2)
    beta = 2.0/3.0 # 2/3 as constant to make vmax and Rmax the max value
    v = -vmax*np.tanh(beta*rr/Rmax)/(np.cosh(beta*rr/Rmax))**2/(np.tanh(beta)/(np.cosh(beta))**2)
  v = np.where(rr == 0, 0.0,v)
  return v

# grid at U,V,T points
rad_gg = np.sqrt((xg-x_c)**2 + (yg-y_c)**2)
rad_cc = np.sqrt((xc-x_c)**2 + (yc-y_c)**2)
rad_gu = np.sqrt((xu-x_c)**2 + (yu-y_c)**2)
rad_gv = np.sqrt((xv-x_c)**2 + (yv-y_c)**2)

theta_gg = np.arctan2(yg-y_c,xg-x_c)
theta_cc = np.arctan2(yc-y_c,xc-x_c)
theta_gu = np.arctan2(yu-y_c,xu-x_c)
theta_gv = np.arctan2(yv-y_c,xv-x_c)

u_out = vel_hor(rad_gu,vel_profile)*np.sin(-theta_gu)
v_out = vel_hor(rad_gv,vel_profile)*np.cos(theta_gv)

# 3D velocity field
uvel = FZ*np.tile(u_out,[si_z,1,1])
vvel = FZ*np.tile(v_out,[si_z,1,1])

# compute pressure field
def geostrophic_part(rr):
  v = vel_hor(rr,vel_profile)
  res = f0*v
  res = np.where(rr == 0, 0.0,res)
  return res

def cyclogeo_part(rr):
  v = vel_hor(rr,vel_profile)
  res = v**2/rr 
  res = np.where(rr == 0, 0.0,res)
  return res

def comp_p(x,func):
  
  if x ==0:
    xx = 1e-12
  else:
    xx = 1.0*x

  a,b = scipy.integrate.quad(func,0,xx)
  return a

rr = np.linspace(0.0,4*Lx, 10*si_x)
p1 = [ comp_p(x, geostrophic_part) for x in rr.flatten() ]
p2 = [ comp_p(x, cyclogeo_part) for x in rr.flatten() ]#trying to compute a correction term

fint1 = scipy.interpolate.interp1d(rr, p1)
fint2 = scipy.interpolate.interp1d(rr, p2) #correction term

p_out1 = rho_const*fint1(rad_gg)
p_out2 = rho_const*fint2(rad_gg) #correction term

# remove const at infinity
p_out1 = p_out1 - p_out1[0,0]
p_out2 = p_out2 - p_out2[0,0]

dpdz = FpZ*np.tile(p_out1,[si_z,1,1]) + 2*FZ*FpZ*np.tile(p_out2,[si_z,1,1])
rhop= dpdz/g0
rho_bck = rho_const*(1-alphaK*temp_i)
# convert to temperature
theta_a = -rhop/(rho_const*alphaK) 
theta = theta_a + temp_i

# free surface
pres = FZ*np.tile(p_out1,[si_z,1,1]) + FZ**2*np.tile(p_out2,[si_z,1,1])
eta = pres[0,:,:]/rho_const/g0 - rhop[0,:,:]/rho_const*0.5*dz1[0]
#eta = (p_out1+p_out2)/rho_const/g0

uvel.astype (binprec).tofile ('uinit.box')
vvel.astype(binprec).tofile('vinit.box')
theta.astype(binprec).tofile('tinit.box')

eta.astype(binprec).tofile('einit.box')

temp_i.astype(binprec).tofile('tref.box')
#sref.astype(binprec).tofile('sref.box')

landh.astype(binprec).tofile('topog.box')

########## OBCS files ================

# East
u_e = uvel[:,:,-1]
v_e = vvel[:,:,-1]
t_e = theta[:,:,-1]

u_e.astype(binprec).tofile('u_E.box')
v_e.astype(binprec).tofile('v_E.box')
t_e.astype(binprec).tofile('t_E.box')

# East
u_w = uvel[:,:,0]
v_w = vvel[:,:,0]
t_w = theta[:,:,0]

u_w.astype(binprec).tofile('u_W.box')
v_w.astype(binprec).tofile('v_W.box')
t_w.astype(binprec).tofile('t_W.box')

# North
u_n = uvel[:,-1,:]
v_n = vvel[:,-1,:]
t_n = theta[:,-1,:]

u_n.astype(binprec).tofile('u_N.box')
v_n.astype(binprec).tofile('v_N.box')
t_n.astype(binprec).tofile('t_N.box')

# South
u_s = uvel[:,0,:]
v_s = vvel[:,0,:]
t_s = theta[:,0,:]

u_s.astype(binprec).tofile('u_S.box')
v_s.astype(binprec).tofile('v_S.box')
t_s.astype(binprec).tofile('t_S.box')

# OBCS mask file

obcsmask = 0*landh + 1
obcsmask[0,:]  = 0
obcsmask[:,0]  = 0
obcsmask[:,-1] = 0
obcsmask[-1:]  = 0

obcsmask.astype(binprec).tofile('obcsmask.box')

# # === RBCS files
#tmask  = np.zeros((si_z,si_y,si_x));
tmask  = np.zeros((si_z,yy.size,xx.size));
tmask[-1,:,:] = 1.0
trelax = 1.0*theta

tmask.astype(binprec).tofile('tmask.box')
trelax.astype(binprec).tofile('trelax.box')

# ==== atmospheric files
u0 = 10  # m/s
v0 = 0  # m/s
s0 = 240 # watt/m2
t2 = 15 #C
q2 = 1e-3 #humidite kg/kg

uatm  = u0*np.ones((si_y,si_x));
vatm  = v0*np.ones((si_y,si_x));
solar = s0*np.ones((si_y,si_x));
tair  = t2*np.ones((si_y,si_x));
qair  = q2*np.ones((si_y,si_x));

uatm.astype(binprec).tofile('u10.box')
vatm.astype(binprec).tofile('v10.box')
solar.astype(binprec).tofile('ssrd.box')
tair.astype(binprec).tofile('t2.box')
qair.astype(binprec).tofile('d2.box')

#====== Compute deformation radius ======

#H = np.array([500, 3000]) # Layer thickness (m)
H = np.diff(zc) 
#gp = np.array([1e-2])     # reduced gravity (m/s^2) -- density jumps between layers
gp = g0*np.diff(rho_bck, axis=0)/rho_bck[:-1]

rd = def_radius.cal_rad(H,gp,f0)#[2:-3]

print("first deformation radius: {0:.1f} km".format(rd[1]*1e-3))

instability = np.diff((rho_bck+rhop), axis=0)

#====== figures ========
vmin = np.min(vvel)
vcont = np.linspace(vmin,-vmin,6)

v_gauss=vel_hor(rad_gu,1)
v_hyprb=vel_hor(rad_gu,2)

if barotrop_mod == 0:
  plt.figure()
  plt.contourf(xx[40:290]*1e-3,-zc[:70],theta[:70,int(si_x/2),40:290],50)
  plt.colorbar(label="Theta (K)")
  plt.contour(xx[40:290]*1e-3,-zc[:70],vvel[:70,int(si_x/2),40:290],vcont,colors='k')
  plt.xlabel("x (km)")
  plt.ylabel("z (m)")
  plt.title("Temperature (color) and velocity (contours)")
  plt.tight_layout(pad=1)
  plt.savefig("surf_eddy_c.png")

plt.figure()
plt.contourf(xx[40:290]*1e-3,-zc,vvel[:,int(si_x/2),40:290],50)
plt.colorbar(label="V [m/s]")
plt.contour(xx[40:290]*1e-3,-zc,vvel[:,int(si_x/2),40:290],vcont,colors='0.6')
plt.xlabel("x (km)")
plt.ylabel("z (m)")
plt.title("Velocity")
plt.tight_layout(pad=1)
plt.savefig("v_eddy.png")
plt.close()

fig=plt.figure(figsize=(10,4))
ax5=fig.add_subplot(1, 2, 1)
#plt.plot(rad_gu[int(si_y/2),:]*1e-3, v_hyprb[int(si_x/2),:], 'r', label='Hyperbolic')
plt.plot(rad_gu[int(si_y/2),:]*1e-3, v_gauss[int(si_x/2),:], 'b', label='alpha='+str(alpha))
plt.xlabel("radius (km)")
plt.ylabel(r'$G(r) \ [m/s]$')
plt.xlim(0,100)
#plt.legend(loc='lower right')
plt.text(40,0.27,r'$V_\theta = G(r)F(z)$', fontsize=12)
plt.text(40,0.24,r'$G(r)=-\frac{r V_{max}}{R_{max}} \exp \left[\frac{1}{2}(1-(\frac{r}{R_{max}})^2\right]$', fontsize=12)
#plt.text(60,-0.3,r'$G(r)=-V_{max}\frac{r}{R_{max}} \exp \left[\frac{1}{\alpha}(1-(\frac{r}{R_{max}})^\alpha\right]$', fontsize=14)
plt.title("Horizontal velocity profile")
ax6=fig.add_subplot(1, 2, 2)
ax6.set_aspect(0.0012)
plt.plot(FZ.squeeze(),-zc)
plt.xlabel(r'$F(z)$')
plt.ylabel(r'$z(m)$')
#plt.text(0.05,-1450,r'$F(z)=1-\frac{2}{\sqrt{\pi}}\int \exp \left[-\left(\frac{z}{z_0}\right)^2\right]$', fontsize=14)
plt.text(0.05,-900,r'$F(z)= \exp \left[-\frac{1}{2}\left(\frac{z}{z_0}\right)^2\right]$', fontsize=14)
plt.xlim(-0.05,1.05)
plt.ylim(-1000,0)
plt.title("Vertical velocity profile")
plt.tight_layout(pad=1)
plt.savefig("velocity_profile.png")
plt.close()

fig=plt.figure(figsize=(8,6))
ax7=fig.add_subplot(1, 2, 1)
plt.plot(temp_i[:,0,0], -zc, label='background temperature')
#plt.plot(tref_smooth, -zc)
plt.plot(tref_art.squeeze(), -zc, label='winter')
plt.plot(tref_aut.squeeze(), -zc, label='summer')
ax7.xaxis.tick_top()
ax7.xaxis.set_ticks_position('both')
ax7.xaxis.set_major_locator(plt.MaxNLocator(5))
ax7.legend(loc='lower right', fontsize=10)
plt.xlabel("Background Temperature [°C]")
plt.ylabel("Depth [m]")
#plt.title("Background temperature")
ax8=fig.add_subplot(1, 2, 2)
ax8.plot(rho_bck[:,0,0], -zc, label='background')
ax8.plot(rho_bck[:,0,0]+rhop[:,165,165], -zc, label='core cyclone')
ax8.plot(rho_bck[:,0,0]-rhop[:,165,165], -zc, label='core anticyclone')
#plt.plot(rho_bck[:,0,0]+rhop[:,165,165], -zc, label='core')
ax8.legend(loc='lower left', fontsize=10)
ax8.xaxis.tick_top()
ax8.xaxis.set_ticks_position('both')
ax8.xaxis.set_major_locator(plt.MaxNLocator(5))
plt.xlabel("Background Density [kg/m^3]")
#plt.ylabel("Depth [m]")
#plt.title("Background density")
plt.tight_layout(pad=1)
plt.savefig("Background_state.png")

plt.figure(figsize=(3.5,6))
plt.plot(rho_bck[:,0,0], -zc, label='background')
plt.plot(rho_bck[:,0,0]+rhop[:,165,165], -zc, label='core cyclone')
plt.plot(rho_bck[:,0,0]-rhop[:,165,165], -zc, label='core anticyclone')
plt.xlabel("Density")
plt.ylabel("Depth [m]")
plt.title("Density profile")
plt.tight_layout(pad=1)
plt.legend(loc='lower right')
plt.savefig("coredens_profile.png")
#plt.close()

plt.figure(figsize=(4.5,6))
plt.plot(rho_anomd[20:,0,0], -zc[20:], label='core 3D-cons')
plt.plot(rhop[:,165,165], -zc, label='core Initial Cond')
#plt.plot(rho_anomd[49:,0,0], -zc[49:], label='@Rmax 3D-cons')
#plt.plot(rhop[:,165,165], -zc, label='@Rmax Initial Cond')
plt.xlabel("Density anomaly [kg^3/m]")
plt.ylabel("Depth [m]")
plt.title("Density anomaly")
plt.tight_layout(pad=1)
plt.legend(loc='best')
plt.savefig("rho_anom.png")


fig=plt.figure(figsize=(7.5,9))
ax=fig.add_subplot(2, 1, 1)
ax.set_aspect(1)
plt.contourf(xx[40:290]*1e-3,yy[40:290]*1e-3,eta[40:290,40:290],100)
cb = plt.colorbar(label="Eta (m)")
#cb.set_ticklabels([r'$<10^{0}$', 1, 2, r'$10^{14}$', r'$10^{14}+12345678$'])
cb.set_label(r'$\eta$ (m)', labelpad=-40, y=1.1, rotation=0)
plt.xlabel("x (km)")
plt.ylabel("y (km)")
#plt.title("SSH Anomaly")
plt.suptitle('SSH Anomaly', fontsize=13)

#
ax2 = fig.add_subplot(2, 1, 2)
ax2.set_aspect(8)
plt.plot(xx[40:290]*1e-3,eta[int(si_x/2),40:290]*100)
plt.xlabel("x (km)")
plt.ylabel(r'$\eta \ (cm)$')
plt.xlim(0,xx[-1]*1e-3)
plt.tight_layout(pad=1)
plt.savefig("ssh_anomaly.png")
plt.close()

"""
fig=plt.figure(figsize=(5,4))
#ax5=fig.add_subplot(1, 2, 1)
plt.plot(FZ.squeeze(),-zc)
plt.xlabel(r'$F(z)$')
plt.ylabel(r'$\eta \ (cm)$')
plt.xlim(0,xx[-1]*1e-3)
plt.tight_layout(pad=1)
plt.title("Temperature profile")
plt.savefig("ssh_anomaly.png")
#plt.plot(FZg.squeeze(),-zc)

ax6=fig.add_subplot(1, 2, 2)
plt.plot(FpZ.squeeze(),-zc)
#plt.plot(FpZg.squeeze(),-zc)
"""
