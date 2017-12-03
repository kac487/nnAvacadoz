import os
import numpy as np
from scipy.io.wavfile import read

# Plotting Iports
import matplotlib.pyplot as plt
import seaborn as sns

# Load waveform stream data
wavData = os.listdir('./data')

# Specify index of wav files
fileName = wavData[2]
print(fileName)

xx = read('data/'+fileName)
# Read Sample Rate
fs = xx[0]

# Define points per word segment
sampleLen = int(fs*1.5)

# Cast to numpy float array and scale
data = np.array(xx[1],dtype=float)-2**7

# Specify Offset for each of the Split Sections ################################
offsets = [0,]

# Specify Stream Splitting Points ##############################################
splitPts = []#[fs*36,fs*108,fs*150,fs*126]

# ##############################################################################

# Split stream at specified splitting points
if len(splitPts) > 0:
    dataS = np.split(data,np.cumsum(splitPts))
else:
    dataS = [data]

dataX = np.empty((0,sampleLen))


for i in range(len(dataS)):

    # Shift data by ofset
    dataShift = np.roll(dataS[i],offsets[i])

    # Base the time vector off the split subset of the data
    timeVec = np.linspace(0,len(dataShift)/fs,len(dataShift))

    # Generate Plots
    sns.set()
    plt.plot(timeVec,dataShift)

    # Add Vertical lines to plot
    xcoords = timeVec[::sampleLen]

    for xc in xcoords:
        plt.axvline(x=xc,color='r')
    plt.show()

    # Pad the time vector up to a number thats divisible by the fs*1.5sec
    padLen = sampleLen - np.remainder(dataShift.shape[0],sampleLen)
    dataShift = np.append(dataShift,np.zeros((1,padLen)))

    # Reshape New Vector into sampLen x numSamps
    dataShift = np.reshape(dataShift,(-1,sampleLen))

    # Concatenate samples from between splits
    dataX = np.concatenate((dataX,dataShift),axis=0)
    print(dataX.shape)

    # Write out to .npy file
    np.save('trainingDat/'+fileName+'.npy',dataX)
