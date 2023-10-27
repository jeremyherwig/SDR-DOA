"""
Linear Phased Array DOA Estimator - Simulations
Written by Jeremy Herwig
Methods simluated -
- Delay and Sum
- Minimum Variance Distortionless Response (MVDR) - Capon's Method
- MUltiple SIgnal Classification Algorithm (MUSIC)
"""
'Load Required Libraries'
import numpy as np
import matplotlib.pyplot as plt
'User Defined Simulation Parameters'
N = 8               # Number of elements in beamforming array
d = 0.5             # Antenna separation relative to one wavelength
Fc = 10e6           # Carrier frequency (Hz)
Fb = 200e3          # Baseband frequency (Hz)
Fs = 500e6          # Sampling rate (Hz)
Nsamp = 2**14       # Number of samples
scan_range = 90     # +/- angles (in deg) to scan from broadside
SNR = 15             # Noise power (dB)
simDOA = 20         # simulated direction of arrival (deg) from broadside - +ve is clockwise
save_plots = False
'Useful Functions------------------------------'
def array_factor(N,d,theta):
    # takes number of antennas N, separation (relative to wavelength) d, number of incident signals M,
    # and direction of arrival theta as inputs
    # returns an Nx1 array containg the phase shifts of the signal arriving at each antenna
    n = np.arange(0,N,1)
    array_fac = np.exp(-2j*np.pi*n*d*np.sin(np.deg2rad(theta)))
    array_fac = np.asmatrix(array_fac)
    return array_fac
def awgn(s,SNR):
    # takes a signal s and signal-noise-ratio SNR as input
    # returns s with added AWGN    
    signal_power = np.mean(np.abs(s)**2)
    noise_power = signal_power/(10**(SNR/10))
    noise = np.random.normal(0,noise_power,size=s.shape) + 1j*np.random.normal(0,noise_power,size=s.shape)
    signal_noise = s + noise
    return signal_noise
def sig_pwr(s):    
    p = np.mean(np.abs(s)**2)
    p = 10*np.log10(p)
    return p
'Create baseband signal'
Ts = 1/Fs                        # Sampling period (s)
t = np.arange(0,Nsamp*Ts,Ts)    # Construct time vector
tx_baseband = np.cos(2*np.pi*Fb*t) + 1j*np.sin(2*np.pi*Fb*t) # generate Tx I-Q samples
'Plot baseband signal in time domain'
plt.plot(t*1e6,tx_baseband)
plt.xlim(0,(2/Fb)*1e6)
plt.title('Baseband Tx waveform')
plt.xlabel('Time [us]')
plt.ylabel('s(t) [V]')
if save_plots:
    string = "SIM_Tx_Baseband_Time_Domain.png"
    plt.savefig(string,dpi=300)
    plt.show()
    plt.show()
else:
    plt.show()    
'Simulate upconverting of baseband singal'
tx = tx_baseband*np.exp(2j*np.pi*Fc*t)
'Plot upconverted signal'
plt.plot(t*1e6,tx)
plt.xlim(0,(2/(Fc))*1e6)
plt.title('Tx waveform')
plt.xlabel('Time [us]')
plt.ylabel('s(t) [V]')
if save_plots:
    string = "SIM_Tx_Time_Domain.png"
    plt.savefig(string,dpi=300)
    plt.show()
    plt.show()
else:
    plt.show()
'Add phase shifts to signal to simulate arriving at array with a non-zero DOA'
array_manifold = array_factor(N,d,simDOA)
S = np.fft.fft(tx)
S = np.asmatrix(S)
X = array_manifold.T @ S
x = np.zeros([N,Nsamp],dtype="complex")
for i in range(N):
    x[i,:] = np.fft.ifft(X[i,:])
'Add noise to the signal at each antenna'
x = awgn(x,SNR)
for i in range(N):    
    # plot received signals in time domain
    string = "Rx" + str(i)
    plt.plot(t*1e6,x[i,:],label=string)    
'Plot received signals for each antenna'
plt.xlim(0,(4/(Fc))*1e6)
plt.title('Received Signal at Each Element')
plt.xlabel('Time [us]')
plt.ylabel('s(t) [V]')
plt.legend(loc="upper right")
if save_plots:
    string = "SIM_x(t)_time_domain.png"
    plt.savefig(string,dpi=300)
    plt.show()
    plt.show()
else:
    plt.show()
'Simulate DOA Estimators -----------------------------------------------------'
'DAS Method ---'
for i in range(N):
    X[i,:] = np.fft.fft(x[i,:])
angles = np.arange(-scan_range,scan_range,1)    # array containing scanning angles
signal_power_DAS = []       # initialise variable to record measured signal strength
for angle in angles:    
    w = array_factor(N,d,angle).T     # array weight vector
    Y = w.H @ X                     # apply array weights to find summed signal y(t)
    Y = np.asarray(Y,dtype="complex").squeeze()    
    y = np.fft.ifft(Y)  # convert to time domain
    y_pwr = sig_pwr(y)  # calculate signal power    
    signal_power_DAS.append(y_pwr)    
signal_power_DAS = signal_power_DAS - np.max(signal_power_DAS)
peak_index = np.where(signal_power_DAS == np.max(signal_power_DAS))
DOA_angle_DAS = angles[peak_index]    
'MVDR Method ---'
'convert Rx data to frequency domain for processing'
x = np.asmatrix(x)
Rxx = x @ x.H                    # covariance matrix of received data at every antenna
Rxx_inv = np.linalg.pinv(Rxx)    # inverse of covariance matrix
signal_power_MVDR = []  # initialise variable to record MVDR spectrum
angles = np.arange(-scan_range,scan_range,1)
for angle in angles:    
    steer_vec = array_factor(N,d,angle).T    
    P = 1/(steer_vec.H @ Rxx_inv @ steer_vec)
    P = np.asarray(P,dtype="complex").squeeze()
    P = 10*np.log10(np.abs(P))
    signal_power_MVDR.append(P)    
signal_power_MVDR = signal_power_MVDR - np.max(signal_power_MVDR)
peak_index = np.where(signal_power_MVDR == np.max(signal_power_MVDR))
DOA_angle_MVDR = angles[peak_index]  
'MUSIC Algorithm ---'
Rxx = x @ x.H   # calculate covariance matrix
eigenval, eigenvec = np.linalg.eig(Rxx) # find eigenvalues and eigenvectors
order = np.argsort(np.abs(eigenval))    # find indices for sorting
eigenvec = eigenvec[:,order]            # reorder eigenvectors accordingly.                                   
# separate noise subspace, assuming only 1 incident signal (max for 2 element array)
noise_subspace = np.zeros((N,N-1),dtype="complex")
noise_subspace = np.asmatrix(noise_subspace)
for i in range(N-1):
    noise_subspace = eigenvec[:,i]  
signal_power_MUSIC = []  # initialise variable to record MVDR spectrum
angles = np.arange(-scan_range,scan_range,1)
for angle in angles:
    steer_vec = np.asmatrix(np.exp(-2j * np.pi * d * np.arange(N) * np.sin(np.deg2rad(angle))))
    steer_vec = steer_vec.T
    #steer_vec = array_factor(N,d,angle).T   # steering vector
    P = 1/(steer_vec.H @ noise_subspace @ noise_subspace.H @ steer_vec) # compute signal power
    P = 10*np.log10(np.abs(P[0,0]))
    signal_power_MUSIC.append(P)    
signal_power_MUSIC = signal_power_MUSIC - np.max(signal_power_MUSIC)
peak_index = np.where(signal_power_MUSIC == np.max(signal_power_MUSIC))
DOA_angle_MUSIC = angles[peak_index]
# sort eigenvalues for plotting
eigenval = np.flip(eigenval[order])
plt.plot(10*np.log10(np.abs(eigenval)),'.-')
plt.plot(10*np.log10(np.abs(eigenval)),'o')
plt.xticks(np.arange(N))
plt.xlabel('Eigenvalue Index')
plt.ylabel('Eigenvalue Magnitude [dB]')
if save_plots:
    string = "MUSIC_Eigenvalues.png"
    plt.savefig(string,dpi=300)
    plt.show()
    plt.show()
else:
    plt.show()
plt.plot(angles,signal_power_DAS)
plt.xlabel('Steering Angle [deg]')
plt.xlim(-90,90)
plt.xticks(np.arange(-90,90,30))
plt.grid()
plt.ylabel('Relative Signal Strength [dB]')
plt.ylim(-50,10)
if save_plots:
    string = "SIM_DAS_rect.png"
    plt.savefig(string,dpi=300)
    plt.show()
    plt.show()
else:
    plt.show()    
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(np.deg2rad(angles), signal_power_DAS) 
ax.set_theta_zero_location('N') 
ax.set_theta_direction(-1) 
ax.set_rlim(-30,10)
ax.set_rgrids(np.arange(-30,10,10))
ax.set_rlabel_position(0)  
ax.set_title('Relative Signal Strength [dB]',fontsize="9", va='bottom')
if save_plots:
    string = "SIM_DAS_polar.png"
    plt.savefig(string,dpi=300)
    plt.show()
    plt.show()
else:
    plt.show()
plt.plot(angles,signal_power_MVDR)
plt.xlabel('Steering Angle [deg]')
plt.xlim(-90,90)
plt.xticks(np.arange(-90,90,30))
plt.grid()
plt.ylabel('Relative Signal Strength [dB]')
plt.ylim(-50,10)
if save_plots:
    string = "SIM_MVDR_rect.png"
    plt.savefig(string,dpi=300)
    plt.show()
    plt.show()
else:
    plt.show()
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(np.deg2rad(angles), signal_power_MVDR)
ax.set_theta_zero_location('N') 
ax.set_theta_direction(-1) 
ax.set_rlim(-30,10)
ax.set_rgrids(np.arange(-30,10,10))
ax.set_rlabel_position(0)  
ax.set_title('Relative Signal Strength [dB]',fontsize="9", va='bottom')
if save_plots:
    string = "SIM_MVDR_polar.png"
    plt.savefig(string,dpi=300)
    plt.show()
    plt.show()
else:
    plt.show()
plt.plot(angles,signal_power_MUSIC)
plt.xlabel('Steering Angle [deg]')
plt.xlim(-90,90)
plt.xticks(np.arange(-90,90,30))
plt.grid()
plt.ylabel('Relative Signal Strength [dB]')
plt.ylim(-80,10)
if save_plots:
    string = "SIM_MUSIC_rect.png"
    plt.savefig(string,dpi=300)
    plt.show()
    plt.show()
else:
    plt.show()
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(np.deg2rad(angles), signal_power_MVDR)
ax.set_theta_zero_location('N') 
ax.set_theta_direction(-1) 
ax.set_rlim(-30,5)
ax.set_rgrids(np.arange(-30,10,10))
ax.set_rlabel_position(0)  
ax.set_title('Relative Signal Strength [dB]',fontsize="9", va='bottom')
if save_plots:
    string = "SIM_MUSIC_polar.png"
    plt.savefig(string,dpi=300)
    plt.show()
    plt.show()
else:
    plt.show()
plt.plot(angles,signal_power_DAS,label="DAS")
plt.plot(angles,signal_power_MVDR,label="MVDR")    
plt.plot(angles,signal_power_MUSIC,label="MUSIC")
plt.xlabel('Steering Angle [deg]')
plt.xlim(-90,90)
plt.xticks(np.arange(-90,90,30))
plt.grid()
plt.ylabel('Relative Signal Strength [dB]')
plt.ylim(-80,10)
plt.legend(loc="upper right")
if save_plots:
    string = "SIM_COMP_rect.png"
    plt.savefig(string,dpi=300)
    plt.show()
    plt.show()
else:
    plt.show()
