"""

Linear Phased Array DOA Estimator - 2-Element Tests

Written by Jeremy Herwig

Methods used -

- Delay and Sum
- Minimum Variance Distortionless Response (MVDR) - Capon's Method
- MUltiple SIgnal Classification Algorithm (MUSIC)

Written by Jeremy Herwig

"""

'Load Required Libraries'
import numpy as np
import matplotlib.pyplot as plt
import adi
import keyboard
import time


print("\nPluto Beamformer")
print("\nTest Code")

'Test configuration - user input----------------------------------------------'

# baseband frequency input in kHz
Fb = float(input("\nEnter baseband frequency in kHz: "))
print("\nBaseband frequency set to " + str(Fb) + " kHz")
Fb *= 1e3

# carrier frequency input in MHz
Fc = float(input("\nEnter carrier frequency in MHz: "))
print("\nCarrier frequency set to " + str(Fc) + " MHz")
Fc *= 1e6

# carrier frequency input in MHz
save_plots = bool(int(input("\nSave set-up/cal plots (1/0): ")))
print("\nSave Set-up/Cal plots set to " + str(save_plots))


'Non configurable properties--------------------------------------------------'

N = 2;          # number of antennas
Fs = 5e6        # sampling rate (Hz)
Ntx = 2**18     # number of samples (Tx)
Nrx = 2**12     # number of samples (Rx)

rx_gain = 0     # Rx gain (dB)
tx_gain = -3    # Tx gain (dB)

scan_range = 90 # +/- scan range (deg)

angles = np.arange(-scan_range,scan_range,1)    # scan angles

d = 0.5     # antenna separation (in terms of wavelength, i.e. d=0.5 means 0.5*lambda separation)

'Create SDR Radios------------------------------------------------------------'

sdr = adi.ad9361(uri='ip:192.168.2.1')

'Configure Rx Properties------------------------------------------------------'

sdr.rx_enabled_channels = [0, 1]                # enable both rx channels
sdr.sample_rate = int(Fs)                       # set sample rate
sdr.rx_rf_bandwidth = int(Fb*3)                 # set Rx RF bandwidth (needs to be greater than the baseband frequency to accommodate for carrier + baseband offset)
sdr.rx_lo = int(Fc)                             # set Rx local oscillator frequency (common to both Rx channels)
sdr.gain_control_mode = "manual"                # set Rx gain control mode - "manual" 
sdr.rx_hardwaregain_chan0 = int(rx_gain)        # set Rx0 gain
sdr.rx_hardwaregain_chan1 = int(rx_gain)        # set Rx1 gain
sdr.rx_buffer_size = int(Nrx)                   # set rx buffer (number of samples taken by ADC)
sdr._rxadc.set_kernel_buffers_count(1)          # set number of buffers to 1 - this should prevent stale data as only 1 buffer will be stored, so sampling a new buffer overwrites any saved data.

'Configure Tx Properties------------------------------------------------------'

sdr.tx_rf_bandwidth = int(Fb*3)             # set Tx bandwidth
sdr.tx_lo = int(Fc)                         # set Tx transmission frequency
sdr.tx_cyclic_buffer = True                 # enable continuous transmission
sdr.tx_hardwaregain_chan0 = int(tx_gain)    # set Tx0 gain
sdr.tx_hardwaregain_chan1 = int(-88)        # set Tx1 gain to minimum (Tx1 is unused)
sdr.tx_buffer_size = int(Ntx)               # set Tx buffer size - make it larger than rx buffer size. 

'Useful Functions-------------------------------------------------------------'

'Calculate array factor (array weights/steering vectors)'
def array_factor(N,d,theta):
    
    # takes number of antennas N, separation (relative to wavelength) d, number of incident signals M,
    # and direction of arrival theta as inputs
    # returns an Nx1 array containg the phase shifts of the signal arriving at each antenna
    n = np.arange(0,N,1)
    array_fac = np.exp(-2j*np.pi*n*d*np.sin(np.deg2rad(theta)))
    array_fac = np.asmatrix(array_fac)
    return array_fac

'Calculate signal power'
def sig_pwr(s):
    
    p = np.mean((np.abs(s)/2**11)**2)
    p = 10*np.log10(p)
    return p

'Fourier Transform of array, treating each row as a separate signal'
def FT_rows(data,N,Nrx):
    
    data_fft = np.zeros([N,Nrx],dtype="complex")
    
    for row in range(N):
        data_fft[row][:] = np.fft.fft(data[row][:])
    
    return(data_fft)

    
'Generate Baseband Waveform---------------------------------------------------'

Ts = 1/Fs                           # Sampling period (Hz)
t = np.arange(0,Ntx * Ts, Ts)       # Construct time vector
tx_baseband = np.cos(2*np.pi*Fb*t) + 1j*np.sin(2*np.pi*Fb*t)    # generate Tx samples

plt.plot(t*1e6,tx_baseband)
plt.xlim(0,4e6/Fb)
plt.xlabel("Time [us]")
plt.ylabel("s(t) [V]")
#plt.title("Baseband Signal Before Scaling")
if save_plots:
    string = "TEST_Tx_Baseband_Time_Domain.png"
    plt.savefig(string,dpi=300)
    plt.show()
else:
    plt.show()

tx_baseband *= 2**14    # scale by multiplying by 2^14 (pluto expects signal betwen -2^14 and 2^14)

'Receive side time vector-----------------------------------------------------'
t_rx = np.arange(0,Nrx*Ts,Ts)

'Transmit data'
print("test")
input("\nStart continuous transmission? press enter to continue\n")

sdr.tx([tx_baseband,tx_baseband])      # transmit data continuously - sdr.tx_destroy_buffer() stops transmission

'Before beginning calibration, sample buffers --------------------------------'

print('\nSampling Rx antennas. Hold q to begin calibration\n')

# initialise rx_data
rx_data = []
rx_data = np.zeros([N,Nrx],dtype="clongdouble")

test = 0

while True:
    
    # Sample 1 buffer of data from the Rx antennas
    rx_data = sdr.rx()
  
        
    for n in range(N):
       string = "Rx"+str(n)
       plt.plot(t_rx*1e6,rx_data[n][:],label=string)
    
    plt.xlim(0,4e6/(Fb))
    plt.xlabel("Time [us]")
    plt.ylabel("Signal [V]")
    plt.title("Baseband Rx Signal")
    plt.legend(loc="upper right")
    plt.show()
    


    # if q is preseed break while loop.
    if keyboard.is_pressed("q"):
        print("q pressed, ending loop\n")
        time.sleep(0.5)
        
        # if print setup/cals is true, plot & print last set of data:
        if save_plots:
            for n in range(N):
               string = "Rx"+str(n)
               plt.plot(t_rx*1e6,rx_data[n][:],label=string)
            
            plt.xlim(0,4e6/(Fb))
            plt.xlabel("Time [us]")
            plt.ylabel("Signal")
            plt.legend(loc="upper right")
            string = "TEST_Rx_Baseband_Time_Domain.png"
            plt.savefig(string,dpi=300)
            plt.show()
        
        break
    
        
    
'Calibration------------------------------------------------------------------'

print("\nConfigure Tx antenna so that it is 0 degrees from broadside\n")
input("Press enter to continue\n")

'Clear buffer'

rx_data = []

# receive 20 frames to ensure no chance of stale data on the buffer
for i in range(20):
       
    # Sample 1 buffer of data from the Rx antennas
    rx_data = sdr.rx()

    
'Sample 1 buffer of Rx antenna baseband signal'
rx_data = sdr.rx()
X = FT_rows(rx_data,N,Nrx)    # fourier transform

'Work out phase calibrations'
# Rx0 and Rx1 channels do not have the same transmission line lengths. Additionally, additional connectors are used.
# These cause phase shifts because of the hardware, which need to be corrected.
# The calibration process aims to do this. The process is as follows:
    # 1 - When Tx is at boresight, sample Rx antennas
    # 2 - iteratively apply phase shifts between -180 and 180 degrees
    # 3 - find which phase shift results in the highest signal power of Rx0 + Rx1

phases = np.arange(-180,180,1)
phase_cals = np.zeros((N,1))


for i in range(N):
    
    signal_power = []   # initialise variable to store power of summed signal
    
    for phase in phases:
        
        X_DELAY = X[i,:]*np.exp(1j*np.deg2rad(phase))   # apply a phase shift
        
        rx_sum = rx_data[0][:] + np.fft.ifft(X_DELAY)   # sum with Rx0 in time domain
                
        rx_sum_av_pwr = np.mean((np.abs(rx_sum)/2**11)**2)          # average power of the signal, relative to full scale of Pluto ADC (12 bit)
        rx_sum_av_pwr = 10*np.log10(rx_sum_av_pwr)                  # in dB
        
        signal_power.append(rx_sum_av_pwr)
    
    peak_index = signal_power.index(np.max(signal_power))
    phase_cal = phases[peak_index]
    phase_cals[i][0] = phase_cal
    string = "\nPhase calibration value for Rx" + str(i) + " set to " + str(phase_cal) + " deg\n"
    print(string)    
    
    string = "Phase calibration plot for Rx" + str(i)
    plt.plot(phases,signal_power)
    plt.xlabel('Phase Difference Between Rx0 and Rx1 [deg]')
    plt.xlim(-180,180) 
    plt.xticks(np.arange(-180,181,60))
    plt.grid()
    plt.ylabel('Signal Strength [dBfs]')
    plt.title(string)
    plt.ylim(-50,0)
    plt.axvline(x=phase_cal, color='r', linestyle=':')
    string = str(phase_cal) + " deg"
    plt.text(phase_cal+3,-48,string,rotation=90,va="bottom")
    if save_plots:
        string = "TEST_Rx_Phase_Cal_Vals.png"
        plt.savefig(string,dpi=300)
        plt.show()
    else:
        plt.show()

    
'Test by receiving signal and correcting hardware phase shifts----------------'

'Sample 1 buffer of Rx antenna baseband signal'
rx_data = sdr.rx()

X = FT_rows(rx_data,N,Nrx)    # conver to frequency domain

rx_data_shifted = np.zeros([N,Nrx],dtype='complex')     # initialise phase shifted data

'Create a plot comparing time domain signals before and after phase calibration'
fig, axs = plt.subplots(1,2,sharey=True)   # initialse subplot

for i in range(N):
    
    X_DELAY = X[i,:]*np.exp(1j*np.deg2rad(phase_cals[i][0]))
    rx_data_shifted[i][:] = np.fft.ifft(X_DELAY)
    string = "Rx"+str(n)
    axs[0].plot(t_rx*1e6,rx_data[i][:])
    axs[1].plot(t_rx*1e6,rx_data_shifted[i][:],label=string)

axs[0].set_xlim(0,1e6/Fb)
axs[1].set_xlim(0,1e6/Fb)
axs[0].set_ylabel("Signal")
axs[0].set_xlabel("Time [us]")
axs[1].set_xlabel("Time [us]")
axs[0].set_title("Before HW Phase Correction")
axs[1].set_title("After HW Phase Correction")
plt.legend(loc="upper right")
if save_plots:
    string = "TEST_Rx_Phase_Cal_time_domain.png"
    plt.savefig(string,dpi=300)
    plt.show()
else:
    plt.show()


'Work out DOA (should be 0 deg)'

# Now scan all angles and determine the DOA. Should be 0 degrees.
# Cancel out time delays for each possible DOA by applying equivalent opposite time delay in frequency domain.

X_phase_corrected = np.zeros([N,Nrx],dtype='complex')     # initialise phase shifted data

for i in range(N):
   
    X_phase_corrected[i,:] = X[i,:]*np.exp(1j*np.deg2rad(phase_cals[i][0]))   # perform phase cal correction

X_phase_corrected = np.asmatrix(X_phase_corrected)

# initialise signal power variable
signal_power_DAS = []

# scan through each angle in scan range variable
for angle in angles:
    
    w = array_factor(N,d,angle).T     # array weight vector
    Y = w.H @ X_phase_corrected       # apply array weights to find summed signal y(t)
    Y = np.asarray(Y,dtype="complex").squeeze()
    
    y = np.fft.ifft(Y)  # convert to time domain
    y_pwr = sig_pwr(y)  # calculate signal power
    
    signal_power_DAS.append(y_pwr)
    
signal_power_DAS = signal_power_DAS - np.max(signal_power_DAS)
peak_index = np.where(signal_power_DAS == np.max(signal_power_DAS))
DOA_angle_DAS = angles[peak_index]

print("\n DOA - DAS (deg): " + str(DOA_angle_DAS))
    
plt.plot(angles,signal_power_DAS)
plt.xlabel('Steering Angle [deg]')
plt.xlim(-90,90)
plt.xticks(np.arange(-90,90,30))
plt.grid()
plt.ylabel('Signal Strength [dBfs]')
plt.ylim(-50,10)
if save_plots:
    string = "SIM_DAS_rect.png"
    plt.savefig(string,dpi=300)
    plt.show()
    plt.show()
else:
    plt.show()
    
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(np.deg2rad(angles), signal_power_DAS) # MAKE SURE TO USE RADIAN FOR POLAR
ax.set_theta_zero_location('N') # make 0 degrees point up
ax.set_theta_direction(-1) # increase clockwise
ax.set_rlim(-20,5)
ax.set_rgrids(np.arange(-20,10,5))
ax.set_rlabel_position(22.5)  # Move grid labels away from other labels
if save_plots:
    string = "SIM_DAS_polar.png"
    plt.savefig(string,dpi=300)
    plt.show()
    plt.show()
else:
    plt.show()


input("\nCalibration complete. Press enter to continue.\n")
print("\nContinuously sampling Rx antennas. Hold q to end. Hold p to pause for saving plots\n")
print("\nHold t to change DOA estimation method\n")

DOA_print = "D"     # prints DAS method to plot by default. Change to C for Capon/MVDR and M for MUSIC

DOA_holder = []
DAS_results = []
MVDR_results = []
MUSIC_results =[]

z = False
g = False
h = False

while True:
    
    # receive rx data
    rx_data = sdr.rx()
    
    X = FT_rows(rx_data,N,Nrx)
    
    X_phase_corrected = np.zeros([N,Nrx],dtype='complex')     # initialise phase shifted data

    for i in range(N):
       
        X_phase_corrected[i,:] = X[i,:]*np.exp(1j*np.deg2rad(phase_cals[i][0]))   # perform phase cal correction

    X_phase_corrected = np.asmatrix(X_phase_corrected)
    
    # initialise variables for storing average power:
    signal_power_DAS = []
    signal_power_MVDR = []
    signal_power_MUSIC = []

    # DAS Method:
    
    for angle in angles:
        
        w = array_factor(N,d,angle).T     # array weight vector
        Y = w.H @ X_phase_corrected       # apply array weights to find summed signal y(t)
        Y = np.asarray(Y,dtype="complex").squeeze()
        
        y = np.fft.ifft(Y)  # convert to time domain
        y_pwr = sig_pwr(y)  # calculate signal power
        
        signal_power_DAS.append(y_pwr)
        
          
    signal_power_DAS = signal_power_DAS - np.max(signal_power_DAS)
    peak_index = np.where(signal_power_DAS == np.max(signal_power_DAS))
    DOA_angle_DAS = angles[peak_index]
        
    # MVDR Method:
    x = np.asmatrix(rx_data)

    Rxx = x @ x.H                    # covariance matrix of received data at every antenna
    Rxx_inv = np.linalg.inv(Rxx)     # inverse of covariance matrix

    signal_power_MVDR = []  # initialise variable to record MVDR spectrum

    angles = np.arange(-scan_range,scan_range,1)

    for angle in angles:
        
        steer_vec = array_factor(N,d,angle).T
        
        P = 1/(steer_vec.H @ Rxx_inv @ steer_vec)
        P = np.asarray(P,dtype="complex").squeeze()
        P = 10*np.log10(np.abs(P))
        signal_power_MVDR.append(P)
     
    peak_index = np.where(signal_power_MVDR == np.max(signal_power_MVDR))
    signal_power_MVDR = signal_power_MVDR - np.max(signal_power_MVDR)
    DOA_angle_MVDR = angles[peak_index]
    
    # MUSIC Method:
    Rxx = x @ x.H   # calculate covariance matrix

    eigenval, eigenvec = np.linalg.eig(Rxx) # find eigenvalues and eigenvectors

    order = np.argsort(np.abs(eigenval))    # find indices for sorting

    eigenvec = eigenvec[:,order]            # reorder eigenvectors accordingly.
                                            # ordered in ascending order, so the first 
                                            # eigenvectors are the noise subspace


    # separate noise subspace, assuming only 1 incident signal (max for 2 element array)
    noise_subspace = np.zeros((N,N-1),dtype="complex")
    noise_subspace = np.asmatrix(noise_subspace)

    for i in range(N-1):
        noise_subspace = eigenvec[:,i]  
    signal_power_MUSIC = []  # initialise variable to record MVDR spectrum

    angles = np.arange(-scan_range,scan_range,1)

    for angle in angles:
        
        steer_vec = array_factor(N,d,angle).T   # steering vector
        P = 1/(steer_vec.H @ noise_subspace @ noise_subspace.H @ steer_vec) # compute signal power
        P = 10*np.log10(np.abs(P))
        signal_power_MUSIC.append(P[0,0])
        
    signal_power_MUSIC = signal_power_MUSIC - np.max(signal_power_MUSIC)
    peak_index = np.where(signal_power_MUSIC == np.max(signal_power_MUSIC))
    DOA_angle_MUSIC = angles[peak_index]
    
    
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    if DOA_print == "D":
        ax.plot(np.deg2rad(angles), signal_power_DAS) # MAKE SURE TO USE RADIAN FOR POLAR
        plt.axvline(x=np.deg2rad(DOA_angle_DAS), color='r', linestyle=':')
    elif DOA_print == "C":
        ax.plot(np.deg2rad(angles), signal_power_MVDR) # MAKE SURE TO USE RADIAN FOR POLAR
    elif DOA_print == "M":
        ax.plot(np.deg2rad(angles), signal_power_MUSIC) # MAKE SURE TO USE RADIAN FOR POLAR
    else: 
        ax.plot(np.deg2rad(angles), signal_power_DAS) # MAKE SURE TO USE RADIAN FOR POLAR
    ax.set_theta_zero_location('N') # make 0 degrees point up
    ax.set_theta_direction(-1) # increase clockwise
    ax.set_rlim(-20,5)
    ax.set_rgrids(np.arange(-20,10,5))
    ax.set_rlabel_position(22.5)  # Move grid labels away from other labels
    plt.show()
    
    # if t is pressed, let user change what method:
    if keyboard.is_pressed("t"):
        print("t pressed, select DOA method to plot\n")
        print("D for DAS, C for Capon/MVDR, and M for MUSIC\n")
        
        DOA_print = input("\nSelect DOA method here: ")
        print("DOA Method Updated to" + DOA_print)
    
    # if q is preseed break while loop.
    if keyboard.is_pressed("q"):
        print("q pressed, ending loop\n")
        time.sleep(0.5)
        break
    
    # if p is pressed start saving
    if keyboard.is_pressed("p"):
        print("\nSampling paused.\n")
        time.sleep(0.5)
        phys_DOA = input("\nEnter physical DOA in degrees: ") 
        
        plt.plot(angles,signal_power_DAS)
        plt.xlabel('Steering Angle [deg]')
        plt.xlim(-90,90)
        plt.xticks(np.arange(-90,90,30))
        plt.grid()
        plt.ylabel('Relative Signal Strength [dB]')
        plt.ylim(-50,10)
        string = "!TEST_" + str(phys_DOA) +"_deg_DAS_rect.png"
        plt.savefig(string,dpi=300)
        plt.show()
        
        #input("\nPress enter to continue to next plot\n")
            
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.plot(np.deg2rad(angles), signal_power_DAS) # MAKE SURE TO USE RADIAN FOR POLAR
        ax.set_theta_zero_location('N') # make 0 degrees point up
        ax.set_theta_direction(-1) # increase clockwise
        ax.set_rlim(-20,5)
        ax.set_rgrids(np.arange(-20,10,5))
        ax.set_rlabel_position(22.5)  # Move grid labels away from other labels
        string = "!TEST_" + str(phys_DOA) +"_deg_DAS_polar.png"
        plt.savefig(string,dpi=300)
        plt.show()
        
        #input("\nPress enter to continue to next plot\n")
        
        plt.plot(angles,signal_power_MVDR)
        plt.xlabel('Steering Angle [deg]')
        plt.xlim(-90,90)
        plt.xticks(np.arange(-90,90,30))
        plt.grid()
        plt.ylabel('Relative Signal Strength [dB]')
        plt.ylim(-50,10)
        string = "!TEST_" + str(phys_DOA) +"_deg_MVDR_rect.png"
        plt.savefig(string,dpi=300)
        plt.show()
        
        #input("\nPress enter to continue to next plot\n")
            
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.plot(np.deg2rad(angles), signal_power_MVDR) # MAKE SURE TO USE RADIAN FOR POLAR
        ax.set_theta_zero_location('N') # make 0 degrees point up
        ax.set_theta_direction(-1) # increase clockwise
        ax.set_rlim(-20,5)
        ax.set_rgrids(np.arange(-20,10,5))
        ax.set_rlabel_position(22.5)  # Move grid labels away from other labels
        string = "!TEST_" + str(phys_DOA) +"_deg_mvdr_polar.png"
        plt.savefig(string,dpi=300)
        plt.show()
        
        #input("\nPress enter to continue to next plot\n")
        
        plt.plot(angles,signal_power_MUSIC)
        plt.xlabel('Steering Angle [deg]')
        plt.xlim(-90,90)
        plt.xticks(np.arange(-90,90,30))
        plt.grid()
        plt.ylabel('Relative Signal Strength [dB]')
        plt.ylim(-50,10)
        string = "!TEST_" + str(phys_DOA) +"_deg_MUSIC_rect.png"
        plt.savefig(string,dpi=300)
        plt.show()
        
        #input("\nPress enter to continue to next plot\n")
            
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.plot(np.deg2rad(angles), signal_power_MUSIC) # MAKE SURE TO USE RADIAN FOR POLAR
        ax.set_theta_zero_location('N') # make 0 degrees point up
        ax.set_theta_direction(-1) # increase clockwise
        ax.set_rlim(-20,5)
        ax.set_rgrids(np.arange(-20,10,5))
        ax.set_rlabel_position(22.5)  # Move grid labels away from other labels
        string = "!TEST_" + str(phys_DOA) +"_deg_MUSIC_polar.png"
        plt.savefig(string,dpi=300)
        plt.show()
        
        #input("\nPress enter to continue to next plot\n")
        
        plt.plot(angles,signal_power_DAS,label="DAS")
        plt.plot(angles,signal_power_MVDR,label="MVDR")    
        plt.plot(angles,signal_power_MUSIC,label="MUSIC")
        plt.legend(loc="upper right")
        plt.xlabel('Steering Angle [deg]')
        plt.xlim(-90,90)
        plt.xticks(np.arange(-90,90,30))
        plt.grid()
        plt.ylabel('Relative Signal Strength [dB]')
        plt.ylim(-50,10)
        string = "!TEST_" + str(phys_DOA) +"_deg_COMP_rect.png"
        plt.savefig(string,dpi=300)
        plt.show()
        
        #input("\nPress enter to continue to next plot\n")
        
        fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.plot(np.deg2rad(angles), signal_power_DAS,label="DAS")
        ax.plot(np.deg2rad(angles), signal_power_MVDR,label="MVDR")
        ax.plot(np.deg2rad(angles), signal_power_MUSIC,label="MUSIC") # MAKE SURE TO USE RADIAN FOR POLAR
        plt.legend(loc="upper right")
        ax.set_theta_zero_location('N') # make 0 degrees point up
        ax.set_theta_direction(-1) # increase clockwise
        ax.set_rlim(-20,5)
        ax.set_rgrids(np.arange(-20,10,5))
        ax.set_rlabel_position(22.5)  # Move grid labels away from other labels
        string = "!TEST_" + str(phys_DOA) +"_deg_COMP_polar.png"
        plt.savefig(string,dpi=300)
        plt.show()
        
        
        print("\n DOAs (Phys angle set to " + phys_DOA + " deg):\n")
        print("DAS: " + str(DOA_angle_DAS) + " deg\n")
        print("MVDR: " + str(DOA_angle_MVDR) + " deg\n")
        print("MUSIC: " + str(DOA_angle_MVDR) + " deg\n")
        print("MUSIC eigenvalues:\n")
        print(eigenval)
        print("\nMusic Eigenvecs:\n")
        print(eigenvec)
        
        #input("press enter to continue")
        print("\n plots saved. press p to save new plots or q to quit\n")
        
        DOA_holder.append(phys_DOA)
        DAS_results.append(DOA_angle_DAS)
        MVDR_results.append(DOA_angle_MVDR)
        MUSIC_results.append(DOA_angle_MUSIC)
        
    if keyboard.is_pressed("z"):
        
        z = True
        print("\nSampling paused.\n")
        time.sleep(0.5)
        input("\nPress Enter to Continue\n") 
        
        pos_Z_DAS = signal_power_DAS
        pos_Z_MVDR = signal_power_MVDR
        pos_Z_MUSIC = signal_power_MUSIC
        
    if keyboard.is_pressed("x"):
        g = True
        print("\nSampling paused.\n")
        time.sleep(0.5)
        input("\nPress Enter to Continue\n") 
        
        pos_X_DAS = signal_power_DAS
        pos_X_MVDR = signal_power_MVDR
        pos_X_MUSIC = signal_power_MUSIC
        
    if keyboard.is_pressed("y"):
        h = True
        print("\nSampling paused.\n")
        time.sleep(0.5)
        input("\nPress Enter to Continue\n") 
        
        pos_Y_DAS = signal_power_DAS
        pos_Y_MVDR = signal_power_MVDR
        pos_Y_MUSIC = signal_power_MUSIC
        
sdr.tx_destroy_buffer()
print("\nTransmission stopped\n")

if z and g and h:
    plt.plot(angles,pos_Z_DAS,label="Pos1")
    plt.plot(angles,pos_X_DAS,label="Pos2")    
    plt.plot(angles,pos_Y_DAS,label="Pos3")
    plt.legend(loc="upper right")
    plt.xlabel('Steering Angle [deg]')
    plt.xlim(-90,90)
    plt.xticks(np.arange(-90,90,30))
    plt.grid()
    plt.ylabel('Relative Signal Strength [dB]')
    plt.ylim(-50,10)
    string = "!TEST_DAS_POS_rect.png"
    plt.savefig(string,dpi=300)
    plt.show()
    
    plt.plot(angles,pos_Z_MVDR,label="Pos1")
    plt.plot(angles,pos_X_MVDR,label="Pos2")    
    plt.plot(angles,pos_Y_MVDR,label="Pos3")
    plt.legend(loc="upper right")
    plt.xlabel('Steering Angle [deg]')
    plt.xlim(-90,90)
    plt.xticks(np.arange(-90,90,30))
    plt.grid()
    plt.ylabel('Relative Signal Strength [dB]')
    plt.ylim(-50,10)
    string = "!TEST_MVDR_POS_rect.png"
    plt.savefig(string,dpi=300)
    plt.show()
    
    plt.plot(angles,pos_Z_MUSIC,label="Pos1")
    plt.plot(angles,pos_X_MUSIC,label="Pos2")    
    plt.plot(angles,pos_Y_MUSIC,label="Pos3")
    plt.legend(loc="upper right")
    plt.xlabel('Steering Angle [deg]')
    plt.xlim(-90,90)
    plt.xticks(np.arange(-90,90,30))
    plt.grid()
    plt.ylabel('Relative Signal Strength [dB]')
    plt.ylim(-50,10)
    string = "!TEST_MUSIC_POS_rect.png"
    plt.savefig(string,dpi=300)
    plt.show()


