import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from LogS import H
from matplotlib.backends.backend_pdf import PdfPages
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['font.family'] = 'Calibri'

def getData(fileName, typeInfs, nbData, fold):
    outputFolder = "Output/"+fileName+"/"
    nom = outputFolder + fileName + "_" + typeInfs + "_" + str(nbData) + "_" + str(fold)

    training, test, usrToInt = {}, {}, {}

    iter = 0
    with open(nom+"_Fit_training.txt", "r") as f:
        for line in f:
            training[iter] = []
            tups = line.replace("\n", "").replace(")", "").replace(" ", "").replace("(", "").split("\t")[:-1]
            for t in tups:
                c, t, s = t.split(",")
                c, t, s = int(c), float(t), int(s)
                training[iter].append((c,t,s))

            if len(training[iter])<=1:
                print(line)
            iter += 1

    iter = 0
    with open(nom+"_Fit_test.txt", "r") as f:
        for line in f:
            test[iter] = []
            tups = line.replace("\n", "").replace(")", "").replace(" ", "").replace("(", "").split("\t")[:-1]
            for t in tups:
                c, t, s = t.split(",")
                c, t, s = int(c), float(t), int(s)
                test[iter].append((c,t,s))

            if len(test[iter])<=1:
                print(line)
            iter += 1

    with open(nom+"_Fit_usrToInt.txt", "r") as f:
        for line in f:
            id, usr = line.replace("\n", "").split("\t")
            id = int(id)

            usrToInt[usr]=id

    beta = np.load(nom+"_Fit_beta.npy")
    try:
        betaIC = np.load(nom+"_Fit_beta_IC.npy")
    except:
        betaIC = None

    with open("Data/" + fileName + "_" + typeInfs + "_" + str(nbData) + "_betaTrue.npy", "r") as f:
        try:
            if f.read()=="None":
                betaTrue = None
            else:
                betaTrue = np.load("Data/" + fileName + "_" + typeInfs + "_" + str(nbData) + "_betaTrue.npy")
        except:
            try:
                betaTrue = np.load("Data/" + fileName + "_" + typeInfs + "_" + str(nbData) + "_betaTrue.npy")
            except:
                betaTrue = None

    return training, test, usrToInt, beta, betaIC, betaTrue

def getMatInter(obs, lgStep, usri=None):
    dicTemp={}
    for u in obs:
        for (c,t,s) in obs[u]:
            if usri is not None:
                if usri!=c:
                    continue

            if c not in dicTemp: dicTemp[c]={}

            for (c2,t2,s2) in obs[u]:
                dt = t-t2+lgStep

                if dt<=1 or t<10 or dt>20:
                    continue

                if c2 not in dicTemp[c]: dicTemp[c][c2]={}
                if dt not in dicTemp[c][c2]: dicTemp[c][c2][dt]=[0,0]

                dicTemp[c][c2][dt][s] += 1

    if usri is not None:
        return dicTemp[usri]
    else:
        return dicTemp

def getCntFreq(obs, usri=None):
    cntFreq = {}
    cntFreqFin = {}
    for u in obs:
        for (c, t, s) in obs[u]:
            if usri is not None:
                if usri!=c:
                    continue

            if t>10:
                if c not in cntFreq: cntFreq[c]=[0,0]
                cntFreq[c][s] += 1

    for c in cntFreq:
        if c not in cntFreqFin: cntFreqFin[c] = 0
        cntFreqFin[c] = cntFreq[c][1] / (sum(cntFreq[c]) + 1e-20)

    if usri is not None:
        return cntFreqFin[usri]
    else:
        return cntFreqFin




def plotDistanceInteraction(fileName, typeInfs, nbData):
    tabHMGlob = []
    nameKeys = []
    nbDistSample=0
    for fold in range(5):
        print(fileName, typeInfs, nbData, fold)
        training, test, usrToInt, beta, betaIC, betaTrue = getData(fileName, typeInfs, nbData, fold)
        print(usrToInt)
        lgStep = 1
        if fileName == "PD":
            lgStep=1.01
        mat = getMatInter(training, lgStep=lgStep)
        P0 = getCntFreq(training)
        nbDistSample = len(beta[0,0])
        tabHM=[]
        nameKeys = []
        for c in range(len(beta)):
            for c2 in range(len(beta[c])):
                tabT = []
                rng = range(1, len(beta[c,c2]))
                if fileName=="PD":
                    rng = range(1, 11)
                for dt in rng:
                    tabT.append(H(dt, 0, beta[c,c2], nbDistSample=nbDistSample) - P0[c])
                tabHM.append(tabT)
                nameKeys.append(str(c)+"-"+str(c2))



        tabHMGlob.append(tabHM)

        xtickslabels = list(range(1, nbDistSample+1))
    tabHMGlob = np.array(tabHMGlob)
    tabHMGlob = tabHMGlob.mean(axis=0)
    maxAbsAmp = 0.2
    maxAbsAmp = np.min([np.max([np.max(tabHMGlob), -np.min(tabHMGlob)]), 0.2])
    ax = sns.heatmap(tabHMGlob, xticklabels=xtickslabels, yticklabels=nameKeys, cmap="RdBu_r", linewidths=.5, vmin=-maxAbsAmp, vmax=maxAbsAmp, cbar_kws={'label': r"$P_{ij}(\Delta t) - P_0$"}) #
    ax.vlines([0, nbDistSample], *ax.get_ylim())
    ax.hlines([0, len(beta)**2], *ax.get_xlim())
    ax.hlines([i*len(beta) for i in range(len(beta))], *ax.get_xlim())
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.xlabel(r"$\Delta$t", fontsize=18)
    plt.ylabel("Information pairs", fontsize=18)
    plt.tight_layout()
    plt.savefig("Misc/"+fileName+"/"+"DistanceInteraction_"+fileName+"_"+typeInfs+"_"+str(nbData)+".png")
    plt.savefig("Misc/"+fileName+"/"+"DistanceInteraction_"+fileName+"_"+typeInfs+"_"+str(nbData)+".pdf", dpi=600)
    #plt.show()
    plt.close()

def plotImportanceInteraction(fileName, typeInfs, nbData):
    tabGlob = []

    for fold in range(5):
        print(fileName, typeInfs, nbData, fold)
        training, test, usrToInt, beta, betaIC, betaTrue = getData(fileName, typeInfs, nbData, fold)
        mat = getMatInter(training, 1)
        tabInterFold = []
        for c in mat:
            for c2 in mat[c]:
                nbDistSample = len(beta[c][c2])

                for dt in mat[c][c2]:
                    qteSsInter = H(dt, 0, beta[c, c], nbDistSample=nbDistSample)
                    qte = H(dt, 0, beta[c,c2], nbDistSample=nbDistSample)

                    for r in range(sum(mat[c][c2][dt])):
                        tabInterFold.append((qte-qteSsInter)/qteSsInter)



        tabGlob += tabInterFold
    plt.hist(tabGlob)
    plt.semilogy()
    plt.xlabel(r"$\frac{P_{ij}(t) - P_{ii}(t)}{P_{ii}(t)}$", fontsize=18)
    plt.ylabel("Density", fontsize=18)
    plt.tight_layout()
    plt.savefig("Misc/"+fileName+"/"+"ImportanceInteraction_"+fileName+"_"+typeInfs+"_"+str(nbData)+".png")
    plt.savefig("Misc/"+fileName+"/"+"ImportanceInteraction_"+fileName+"_"+typeInfs+"_"+str(nbData)+".pdf", dpi=600)
    #plt.show()
    plt.close()

def plotImportanceTime(fileName, typeInfs, nbData):
    tabGlob = []

    for fold in range(5):
        print(fileName, typeInfs, nbData, fold)
        training, test, usrToInt, beta, betaIC, betaTrue = getData(fileName, typeInfs, nbData, fold)
        mat = getMatInter(training, 1)
        P0 = getCntFreq(training)
        tabInterFold = []
        for c in mat:
            for c2 in mat[c]:
                nbDistSample = len(beta[c][c2])

                for dt in mat[c][c2]:
                    qteSsInter = P0[c]
                    qte = H(dt, 0, beta[c,c2], nbDistSample=nbDistSample)

                    for r in range(sum(mat[c][c2][dt])):
                        tabInterFold.append((qte-qteSsInter)/qteSsInter)



        tabGlob += tabInterFold
    plt.hist(tabGlob)
    plt.semilogy()
    plt.xlabel(r"$\frac{P_{ij}(t) - P_{ij}}{P_{ij}}$", fontsize=18)
    plt.ylabel("Density", fontsize=18)
    plt.tight_layout()
    plt.savefig("Misc/"+fileName+"/"+"ImportanceTime_"+fileName+"_"+typeInfs+"_"+str(nbData)+".png")
    plt.savefig("Misc/"+fileName+"/"+"ImportanceTime_"+fileName+"_"+typeInfs+"_"+str(nbData)+".pdf", dpi=600)
    #plt.show()
    plt.close()

def printStats(fileName, typeInfs, nbData):
    for fold in range(1):
        print(fileName, typeInfs, nbData, fold)
        training, test, usrToInt, beta, betaIC, betaTrue = getData(fileName, typeInfs, nbData, fold)
        obs = training
        lg = 0
        sqrLg = 0
        for o in obs:
            lg += len(obs[o])
            sqrLg += len(obs[o])**2
        NCasc = len(usrToInt)
        N_Inter = len(obs)
        #lgTest, meanInter = getLenInfStep(obs)
        lgStep, meanInter = 1, 1  #  POUR COMPTER LES ECARTS DUS AUX INFOS PAS CONSIDEREES
        if fileName=="PD":
            lgStep=1.01

        # print(obs)
        print("Lg moyenne =", lg / (len(obs) + 1e-20))
        print("Lg moyenne squared =", sqrLg / (len(obs) + 1e-20))
        print("Tot interaciton =", sqrLg)
        print(usrToInt)
        print("Nb inter =", N_Inter)
        print("Nb casc =", NCasc)
        print("LgInf =", lgStep, "- LgMean =", meanInter)
        print()


listTypeInfsInteressants = [
                            ("Retail", "Products2", 50e6),
                            ("Twitter", "URL", 1e6),
                            ("Twitter", "SocialMedia", 1e6),
                            ("PD", "All", 300000),
                            ("Ads", "Ads2", 1e6),
                            ("Synth", "3", 20000),
                            ("Synth", "5", 20000),
                            ("Synth", "10", 20000),
                            ("Synth", "20", 20000),
                            ]

for (fileName, typeInfs, nbData) in listTypeInfsInteressants:
    nbData=int(nbData)
    #plotDistanceInteraction(fileName, typeInfs, nbData)
    #plotImportanceInteraction(fileName, typeInfs, nbData)
    #plotImportanceTime(fileName, typeInfs, nbData)
    printStats(fileName, typeInfs, nbData)











