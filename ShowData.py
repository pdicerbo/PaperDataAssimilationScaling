import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

NameFiles = ["DoubleApril.dat", "SingleApril.dat", "DoubleJuly.dat", "SingleJuly.dat"]

NFiles = len(NameFiles)
NRep = np.zeros((NFiles), dtype = int)
NConf = np.zeros((NFiles), dtype = int)     # stores the number of configuration for each file
NTmp  = -1 * np.ones((NFiles))              # array used to store informations related to a new data series
TimeMes = []
NPConf  = []                                # list containing the number of processes used for a given data series
NIter   = []                                # number of iteration of a given series


# parsing source data files with timing measures:
# compute numbers of repetitions and
# the number of different configurations used
print("\n\tLoading Data\n")
for FileInd in range(NFiles):
    with open(NameFiles[FileInd], "r") as OpenFile:
        for line in OpenFile:
            # file prototype format:
            # using NP = 2, repetition number 1
            # Minimization done with          146 iterations
            # real 117.48
            # user 209.32
            # sys 22.88
            if 'using' in line:
                info = line.split()
                if NRep[FileInd] < int(info[-1]):
                    NRep[FileInd] = int(info[-1]) # store the last repetition number..

                NumProc = info[3]                           # for each line, store num process of this line..
                if NTmp[FileInd] < int(NumProc[0:-1]):      # the first time NTmp[FileInd] == -1, while NumProc can have more than one digit (-> [0:-1]..)
                    NTmp[FileInd] = int(NumProc[0:-1])      # retrieve NConf..
                    NPConf.append(NTmp[FileInd])            # ...and append it int NPConf list
                    NConf[FileInd] += 1                     # increase counter of this data series

            elif 'real' in line:
                TimeMes.append(float(line.split()[1]))      # append the measured execution time (supposing the same repetition number)
            elif 'Minimization' in line:
                NIter.append(float(line.split()[3]))        # retrieve the number of iteration required to minimize the equation for this configuration
            elif 'SERIAL' in line:
                break

NPE = np.array(NPConf, dtype=int)
print("\tNumber of Repetitions = ", NRep, "\n\tNumber of Configurations = ", NConf)

# Fill DataMatrix
MyIndex = 0
DataMatrix = np.zeros((NFiles, NConf[0], NRep[0]))

# for each data file and for each configuration, retrieve the correspondent time measure for each repetition
# (supposing the same repetition number for a given configuration)
for i in range(NFiles):
    # if we are executed 10 executions for each configuration, here range(NConf[i]) == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for j in range(NConf[i]):
        for k in range(NRep[i]):
            DataMatrix[i,j,k] = TimeMes[MyIndex]    # de-linearizing TimeMes array
            MyIndex += 1

# Plot Timing and Scaling
print("\n\tPrinting Data\n")
fig2 = plt.figure()
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax2 = fig2.add_subplot(111)
TheMask = np.array([i for i in range(-1,20,2)])
TheMask[0] = 0

# producing data files for the latex tabulars
# useful for the paper
TTSolFile = open("TimesToSolution_Latex.txt", "w")
ScalingFile = open("Scaling_Latex.txt", "w")
for i in range(NFiles):
    MeanTime = np.zeros((NConf[0]))
    StdDev   = np.zeros((NConf[0]))
    MyStr = NameFiles[i][:-4]
    ScalingStr = NameFiles[i][:-4]
    for j in range(NConf[i]):
        MeanTime[j] = np.mean(DataMatrix[i,j,:])
        StdDev[j]   = np.std(DataMatrix[i,j,:])

        if j==0 or j%2 == 1:
            MyStr += ' & ${:.2f} \pm {:.2f}$'.format(MeanTime[j], StdDev[j])
            ScalingStr += ' & ${:.2f}$'.format(MeanTime[0]/MeanTime[j])
    
    MyStr += " \\"
    MyStr += "\\"
    MyStr += '\n'
    TTSolFile.write(MyStr)

    ScalingStr += " \\"+"\\"+"\n"
    ScalingFile.write(ScalingStr)

ScalingFile.close()
TTSolFile.close()

# making plots
for i in range(NFiles):
    MeanTime = np.zeros((NConf[0]))
    StdDev   = np.zeros((NConf[0]))
    for j in range(NConf[i]):
        MeanTime[j] = np.mean(DataMatrix[i,j,:])
        StdDev[j]   = np.std(DataMatrix[i,j,:])

    # choose if you want to include all data or just the configurations with even NPE..
    ax1.errorbar(NPE[0:NConf[0]], MeanTime, StdDev, label = NameFiles[i][:-4])
    ax2.plot(NPE[0:NConf[0]], MeanTime[0]/MeanTime, ".-", label = NameFiles[i][:-4])
    
    print("\n------------------------------------------------------------------------------------------------\n")
    print("")
    print(NameFiles[i])
    print("\n\n  the maximum speedup is ")
    print(np.amax(MeanTime[0]/MeanTime))
    print("MeanTime: ", MeanTime, "pm ", StdDev, "\n\nSpeedup: ", MeanTime[0]/MeanTime, "\n\n")
    print("\n------------------------------------------------------------------------------------------------\n")

# ax2.plot(NPE[0:NConf[0]], NPE[0:NConf[0]], label = "Optimal Scaling")

MyAxis = np.array([i for i in range(0,21,2)])
MyAxis = np.append(MyAxis, [20])
MyYAxis = np.array([i for i in range(0,21,4)])

ax1.set_xticks(MyAxis)
ax2.set_xticks(MyAxis)
ax2.set_yticks(MyYAxis)
ax1.grid()
ax2.grid()
ax1.set_xlabel('Number of Processes')
ax1.set_ylabel("Time to Solution (s)")

ax2.set_xlabel('Number of Processes')
ax2.set_ylabel('Speedup')

print("\n\nTheMask = ", TheMask)
ax1.legend(numpoints=1)
ax2.legend(numpoints=1, bbox_to_anchor = [0.325, 1])

plt.show()
print("\n\tSaving results...\n")
fig1.savefig("ExecutionTimes.eps", format='eps', dpi=1000)
fig2.savefig("Scaling.eps", format='eps', dpi=1000)
