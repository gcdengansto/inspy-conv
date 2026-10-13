
import numpy as np
from inspy import TripleAxisSpectr


def SMADemo2D(H,K,L,p):

    # Extract the three parameters contained in "p":
    Deltax=p[0]                    #Gap at the AF zone-center in meV for x-axis mode
    Deltay=p[1]                    #Gap at the AF zone-center in meV for y-axis mode
    Deltaz=p[2]                    #Gap at the AF zone-center in meV for z-axis mode
    cc    =p[3]                    #Bandwidth in meV 
    Gamma =p[4]                    #Intrinsic HWHM of excitation in meV
    #I    =p[5]                    #Intensity prefactor. This parameter is used only in PrefDemo.m
    #bgr  =p[6]                    #A flat background. This parameter is used only in PrefDemo.m
    
    # Calculate the dispersion
    omegax = np.sqrt(cc**2*(np.sin(4*np.pi*H))**2+Deltax**2)
    omegay = np.sqrt(cc**2*(np.sin(4*np.pi*H))**2+Deltay**2)
    omegaz = np.sqrt(cc**2*(np.sin(4*np.pi*H))**2+Deltaz**2)
    w0     = np.array([omegax, omegay, omegaz]).T

    # Intensity scales as (1-cos(2*pi*H))/omega0
    S1   = (1-np.cos(1*np.pi*H))/omegax/2
    S2   = (1-np.cos(1*np.pi*H))/omegay/2
    S3   = (1-np.cos(1*np.pi*H))/omegaz/2
    S    = np.array([S1, S2, S3]).T
    # Now set all energy widths of all branches to Gamma
    HWHM = np.ones(S.shape)*Gamma
    #print(w0.shape)
    
    return [w0, S, HWHM]

def main():
    #this function is to fit the inelastic experimental data by convoluting with
    # the instrument resolution.
    # two other functions should be defined before to use this function. 
    
    #Setup the instrument configuration for resolution calculation
    Exp = TripleAxisSpectr()

    Exp.method = 1  # 1 for Popovici, 0 for Cooper-Nathans
    Exp.moncor=1

    Exp.efixed = 14.87
    Exp.infin=-1    #const-Ef
    
    Exp.mono.dir =-1
    Exp.ana.dir =-1
    
    Exp.mono.tau = 'PG(002)'
    Exp.mono.mosaic = 30
    Exp.mono.vmosaic = 30
    Exp.mono.height=10            #no need for /sqrt(12)
    Exp.mono.width=10
    Exp.mono.depth=0.2
    Exp.mono.rh=100
    Exp.mono.rv=100
    
    Exp.ana.tau = 'PG(002)'
    Exp.ana.mosaic = 30
    Exp.ana.vmosaic = 30
    Exp.ana.height = 10
    Exp.ana.width = 10
    Exp.ana.depth = 0.2
    Exp.ana.rh=100
    Exp.ana.rv=100
    
    #Put the sample information below
    Exp.sample.a = 4
    Exp.sample.b = 5
    Exp.sample.c = 6
    Exp.sample.alpha = 90
    Exp.sample.beta = 90
    Exp.sample.gamma = 90
    Exp.sample.mosaic=60
    Exp.sample.vmosaic=60
    Exp.sample.u=np.array([1, 0, 0])
    Exp.sample.v=np.array([0, 0, 1])
    Exp.sample.shape_type='rectangular'
    Exp.sample.shape = np.diag([0.6, 0.6, 10])**2

    Exp.hcol = [60, 60, 60, 60]
    Exp.vcol = [120, 120, 120, 120]
    Exp.arms = [300, 300, 300, 300,160]


    Exp.orient1 = np.array([1, 0, 0])
    Exp.orient2 = np.array([0, 0, 1])
    
    Exp.guide.height=15
    Exp.guide.width=5
    
    Exp.detector.height=15
    Exp.detector.width=2.5
    
    #Exp.horifoc=1  # -1 is the default value
    Deltax=2                   # Gap at the AF zone-center in meV for x-axis mode
    Deltay=4                    # Gap at the AF zone-center in meV for y-axis mode
    Deltaz=4                    # Gap at the AF zone-center in meV for z-axis mode
    cc    =20                   # Bandwidth in meV 
    Gamma =0.2                  # Intrinsic HWHM of excitation in meV
    I    =100                   # Intensity prefactor. This parameter is used only in PrefDemo.m
    bgr  =0                     # A flat background. This parameter is used only in PrefDemo.m

    p = [Deltax, Deltay, Deltaz, cc, Gamma, I, bgr]


    q = np.array([1.5, 0, 0.2, 0])    #hklw
    Exp.CalcResMatHKL(q)
    
    hklw=np.array([[1.5,0,0.2,0],[1.5,0,0.2,2],[1.5,0,0.2,4],[1.5,0,0.2,6],[1.5,0,0.2,8]])
    
    [qx_low,qy_low,qz_low,q1] = Exp.R2S(1.47, 0, 0.15)  #H-0.1, K, L-0.1
    [qx_up ,qy_up ,qz_up ,q2] = Exp.R2S(1.53, 0, 0.25)  #H+0.1, K, L+0.1
    SX=np.linspace(qx_low, qx_up, 11)
    SY=np.linspace(qy_low, qy_up, 11)
    [SXg, SYg]=np.meshgrid(SX,SY)

    [H,K,L,W]=hklw.T
    w0, S, hwhm = SMADemo2D(H,K,L,p)
    
    Exp.ResolutionPlot(hklw.T) 
    
    # this is a 3D test
    Exp.ResolutionPlot3D(hkle=hklw.T, RANGE=[2.3, 2.55,0.15, 0.30, -2, 10], SMA=SMADemo2D, SMAp=p, SXg=SXg, SYg=SYg)
    print("3D resolution test finished.")
    
    return






if __name__ == '__main__':
    main()