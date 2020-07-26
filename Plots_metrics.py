import numpy as np
import matplotlib.pyplot as plt
import os
import GetResults
import random

def loadMetrics(nom):
    tabModeles, tabF1, tabROCAUC, tabPRAUC, tabAcc = [], [], [], [], []
    with open(nom + "_Metriques.txt", "r") as f:
        nomMetriques = f.readline().replace("\n", "").split("\t")
        for line in f:
            mod, F1, ROCAUC, PRAUC, acc = line.replace("\n", "").split("\t")
            F1, ROCAUC, PRAUC, acc = float(F1), float(ROCAUC), float(PRAUC), float(acc)
            tabModeles.append(mod)
            tabF1.append(F1)
            tabROCAUC.append(ROCAUC)
            tabPRAUC.append(PRAUC)
            tabAcc.append(acc)

    tabModeles = np.array(tabModeles)
    tabF1 = np.array(tabF1)
    tabROCAUC = np.array(tabROCAUC)
    tabPRAUC = np.array(tabPRAUC)
    tabAcc = np.array(tabAcc)
    return tabModeles, tabF1, tabROCAUC, tabPRAUC, tabAcc

def loadCalib(nom):
    tabModeles, tabL1, tabPearson = [], [], []
    with open(nom + "_Metriques_calib.txt", "r") as f:
        nomMetriques = f.readline().replace("\n", "").split("\t")
        for line in f:
            mod, L1, Pears = line.replace("\n", "").split("\t")
            L1, Pears = float(L1), float(Pears)
            tabModeles.append(mod)
            tabL1.append(L1)
            tabPearson.append(Pears)

    tabModeles = np.array(tabModeles)
    tabL1 = np.array(tabL1)
    tabPearson = np.array(tabPearson)
    return tabModeles, tabL1, tabPearson

def loadErrs(nom):
    tabModeles, tabMAE, tabMSE = [], [], []
    tabModeles.append("Modèle")
    with open(nom + "_Errors_.txt", "r") as f:
        nomMetriques = f.readline().replace("\n", "").split("\t")
        for line in f:
            MAE, MSE, NMAE = line.replace("\n", "").split("\t")
            MAE, MSE = float(MAE), float(MSE)
            tabMAE.append(MAE)
            tabMSE.append(MSE)

    tabModeles.append("IC")
    with open(nom + "_Errors_IC.txt", "r") as f:
        nomMetriques = f.readline().replace("\n", "").split("\t")
        for line in f:
            MAE, MSE, NMAE = line.replace("\n", "").split("\t")
            MAE, MSE = float(MAE), float(MSE)
            tabMAE.append(MAE)
            tabMSE.append(MSE)


    for i in range(6):  # Pour pouvoir passer au format array plus tard...
        tabModeles.append("")
        tabMAE.append(0)
        tabMSE.append(0)

    tabModeles = np.array(tabModeles)
    tabMAE = np.array(tabMAE)
    tabMSE = np.array(tabMSE)
    return tabModeles, tabMAE, tabMSE

def loadMetricsDist(nom):
    tabModeles, tabJS, tabBrierScore, tabCrossEntropy = [], [], [], []
    with open(nom + "_Metriques_dist.txt", "r") as f:
        nomMetriques = f.readline().replace("\n", "").split("\t")
        for line in f:
            mod, JS, Brier, CE = line.replace("\n", "").split("\t")
            JS, Brier, CE = float(JS), float(Brier), float(CE)
            tabModeles.append(mod)
            tabJS.append(JS)
            tabBrierScore.append(Brier)
            tabCrossEntropy.append(CE)

    tabModeles = np.array(tabModeles)
    tabJS = np.array(tabJS)
    tabBrierScore = np.array(tabBrierScore)
    tabCrossEntropy = np.array(tabCrossEntropy)

    return tabModeles, tabJS, tabBrierScore, tabCrossEntropy

def getFileTree(listTypeInfsInteressants=None):
    fileTree = {}
    if not listTypeInfsInteressants:
        folders = [name for name in os.listdir("Output")]
        for folder in folders:
            files = [name for name in os.listdir("Output/"+folder)]

            for file in files:
                if "Metriques" not in file:
                    pass
                    continue

                nomSec = file.split("_")
                fileName, typeInfs, nbData, fold = nomSec[0], nomSec[1], int(nomSec[2]), int(nomSec[3])

                try:
                    fileTree[fileName][typeInfs][fold].add(nbData)
                except:
                    try:
                        fileTree[fileName][typeInfs][fold]=set()
                        fileTree[fileName][typeInfs][fold].add(nbData)
                    except:
                        try:
                            fileTree[fileName][typeInfs] = {}
                            fileTree[fileName][typeInfs][fold] = set()
                            fileTree[fileName][typeInfs][fold].add(nbData)
                        except:
                            fileTree[fileName]={}
                            fileTree[fileName][typeInfs] = {}
                            fileTree[fileName][typeInfs][fold] = set()
                            fileTree[fileName][typeInfs][fold].add(nbData)

        for c in fileTree:
            for c2 in fileTree[c]:
                for fold in fileTree[c][c2]:
                    fileTree[c][c2][fold] = sorted(fileTree[c][c2][fold], reverse = True)

    else:
        for (fileName, typeInfs, nbData) in listTypeInfsInteressants:
            for fold in range(5):
                if fileName not in fileTree: fileTree[fileName]={}
                if typeInfs not in fileTree[fileName]: fileTree[fileName][typeInfs]={}
                if fold not in fileTree[fileName][typeInfs]: fileTree[fileName][typeInfs][fold]=set()
                fileTree[fileName][typeInfs][fold].add(int(nbData))

    return fileTree

def toNp(arr):
    return [np.array(a) for a in arr]



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

def getMatInter(obs, lgStep, usri=None, reduit=True):
    freq = getCntFreq(obs)
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

    if reduit:
        for c in dicTemp:
            for c2 in dicTemp[c]:
                for dt in dicTemp[c][c2]:
                    a, b = dicTemp[c][c2][dt][1], dicTemp[c][c2][dt][0]
                    P0 = freq[c]
                    asapb = a/(a+b+1e-20)
                    if b != 0:
                        dicTemp[c][c2][dt][1] = max([0, b*(asapb-P0)/(1-asapb+P0+1e-20)])

    if usri is not None:
        return dicTemp[usri]
    else:
        return dicTemp

def getBetaIMMSBM(nom):
    try:
        beta = np.load(nom+"_Fit_beta_IMMSBM.npy")
    except:
        beta=None

    return beta

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
        
    try:
        betaHawkes = np.load(nom+"_Fit_beta_Hawkes.npy")
    except:
        betaHawkes = None

    try:
        betaCoC = np.load(nom+"_Fit_beta_CoC.npy")
    except:
        betaCoC = None

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

    return training, test, usrToInt, beta, betaIC, betaHawkes, betaCoC, betaTrue




def plotRecap(tabGen, tabArrs, fileName, typeInfs, fold, tabStds=None):
    tabF1s, tabL1s, tabAUCROCs, tabAUCPRs, tabPearsons, tabAccs, tabMSEs, tabJSs, tabBriers, tabCrossEntropys, tabN = tabArrs
    tabNamesMet, tabNamesCal, tabNamesDist, tabNamesErr = tabGen
    tabN, tabF1s, tabL1s, tabAUCROCs, tabAUCPRs, tabPearsons, tabAccs, tabMSEs, tabJSs, tabBriers, tabCrossEntropys, tabNamesMet, tabNamesCal, tabNamesDist, tabNamesErr = toNp([tabN, tabF1s, tabL1s, tabAUCROCs, tabAUCPRs, tabPearsons, tabAccs, tabMSEs, tabJSs, tabBriers, tabCrossEntropys, tabNamesMet, tabNamesCal, tabNamesDist, tabNamesErr])

    if tabStds is not None:
        tabF1Stds, tabL1Stds, tabAUCROCStds, tabAUCPRStds, tabPearsonStds, tabAccStds, tabMSEStds, tabJSStds, tabBrierStds, tabCrossEntropyStds, tabStdN = tabStds
        tabNamesMet, tabNamesCal, tabNamesDist, tabNamesErr = tabGen
        tabN, tabF1Stds, tabL1Stds, tabAUCROCStds, tabAUCPRStds, tabPearsonStds, tabAccStds, tabMSEStds, tabJSStds, tabBrierStds, tabCrossEntropyStds, tabNamesMet, tabNamesCal, tabNamesDist, tabNamesErr = toNp([tabN, tabF1Stds, tabL1Stds, tabAUCROCStds, tabAUCPRStds, tabPearsonStds, tabAccStds, tabMSEStds, tabJSStds, tabBrierStds, tabCrossEntropyStds, tabNamesMet, tabNamesCal, tabNamesDist, tabNamesErr])

    plt.close()
    plt.figure(figsize=(20, 10))

    for modele in range(len(tabNamesMet)):
        if tabStds is not None:
            print("Tab N =", tabN[:, modele])
            plt.subplot(232)
            plt.errorbar(x=tabN[:, modele], y=tabF1s[:, modele], yerr=tabF1Stds[:, modele], label=tabNamesMet[modele] + " (%.7f +- %.7f)" %(tabF1s[0, modele], tabF1Stds[0, modele]))
            
            #plt.subplot(233)
            #plt.errorbar(x=tabN[:, modele], y=tabAUCROCs[:, modele], yerr=tabAUCROCStds[:, modele], label=tabNamesMet[modele])
            
            #plt.subplot(234)
            #plt.errorbar(x=tabN[:, modele], y=tabAUCPRs[:, modele], yerr=tabAUCPRStds[:, modele], label=tabNamesMet[modele])
            
            plt.subplot(236)
            plt.errorbar(x=tabN[:, modele], y=tabAccs[:, modele], yerr=tabAccStds[:, modele], label=tabNamesMet[modele] + " (%.7f +- %.7f)" %(tabAccs[0, modele], tabAccStds[0, modele]))

        else:
            plt.subplot(232)
            plt.plot(tabN[:, modele], tabF1s[:, modele], label=tabNamesMet[modele] + " (%.7f)" %(tabF1s[0, modele]))
    
            #plt.subplot(233)
            #plt.plot(tabN[:, modele], tabAUCROCs[:, modele], label=tabNamesMet[modele])
    
            #plt.subplot(234)
            #plt.plot(tabN[:, modele], tabAUCPRs[:, modele], label=tabNamesMet[modele])

            plt.subplot(236)
            plt.plot(tabN[:, modele], tabAccs[:, modele], label=tabNamesMet[modele] + " (%.7f)" %(tabAccs[0, modele]))


    for modele in range(len(tabNamesCal)):
        if tabStds is not None:
            pass
            #plt.subplot(231)
            #plt.errorbar(x=tabN[:, modele], y=tabL1s[:, modele], yerr=tabL1Stds[:, modele], label=tabNamesCal[modele])
        else:
            pass
            #plt.subplot(231)
            #plt.plot(tabN[:, modele], tabL1s[:, modele], label=tabNamesCal[modele])

    for modele in range(len(tabNamesDist)):
        if tabStds is not None:
            plt.subplot(231)
            plt.errorbar(x=tabN[:, modele], y=tabJSs[:, modele], yerr=tabJSStds[:, modele], label=tabNamesCal[modele] + " (%.7f +- %.7f)" %(tabJSs[0, modele], tabJSStds[0, modele]))

            plt.subplot(233)
            plt.errorbar(x=tabN[:, modele], y=tabBriers[:, modele], yerr=tabBrierStds[:, modele], label=tabNamesCal[modele] + " (%.7f +- %.7f)" %(tabBriers[0, modele], tabBrierStds[0, modele]))

            plt.subplot(234)
            plt.errorbar(x=tabN[:, modele], y=tabCrossEntropys[:, modele], yerr=tabCrossEntropyStds[:, modele], label=tabNamesCal[modele] + " (%.7f +- %.7f)" %(tabCrossEntropys[0, modele], tabCrossEntropyStds[0, modele]))
        else:
            plt.subplot(231)
            plt.plot(tabN[:, modele], tabJSs[:, modele], label=tabNamesCal[modele] + " (%.7f)" %(tabJSs[0, modele]))

            plt.subplot(233)
            plt.plot(tabN[:, modele], tabBriers[:, modele], label=tabNamesCal[modele] + " (%.7f)" %(tabBriers[0, modele]))

            plt.subplot(234)
            plt.plot(tabN[:, modele], tabCrossEntropys[:, modele], label=tabNamesCal[modele] + " (%.7f)" %(tabCrossEntropys[0, modele]))

    for modele in range(len(tabNamesErr)):
        if tabStds is not None:
            plt.subplot(235)
            plt.errorbar(x=tabN[:, modele], y=tabMSEs[:, modele], yerr=tabMSEStds[:, modele], label=tabNamesErr[modele] + " (%.7f +- %.7f)" %(tabMSEs[0, modele], tabMSEStds[0, modele]))
        else:
            plt.subplot(235)
            plt.plot(tabN[:, modele], tabMSEs[:, modele], label=tabNamesErr[modele] + " (%.7f)" %(tabMSEs[0, modele]))

    plt.subplot(231)
    plt.xlabel("# observations")
    plt.ylabel("JS divergence")
    plt.legend()

    plt.subplot(232)
    plt.xlabel("# observations")
    plt.ylabel("Max-F1 score")
    plt.legend()

    plt.subplot(233)
    plt.xlabel("# observations")
    plt.ylabel("RSS")
    plt.legend()

    plt.subplot(234)
    plt.xlabel("# observations")
    plt.ylabel("Cross entropy")
    plt.legend()

    plt.subplot(235)
    plt.xlabel("# observations")
    plt.ylabel(r"MSE $\beta_{infer}-\beta_{true}$")
    plt.legend()

    plt.subplot(236)
    plt.xlabel("# observations")
    plt.ylabel("Maximum accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig("Plots/" + fileName + "/" + fileName + "_" + typeInfs + "_" + str(fold) + "_Metrics.png", dpi=600)
    # plt.show()

def getGlobRes():
    runAll=True
    listTypeInfsInteressants = [
                                ("Synth", "20", 20000),
                                ("Synth", "5", 20000),
                                ("Ads", "Ads2", 1e6),
                                ("PD", "All", 300000),
                                ("Twitter", "URL", 1e6),]

    fileTree = getFileTree(listTypeInfsInteressants=listTypeInfsInteressants)




    for fileName in fileTree:
        if fileName!="Twitter":
            pass
            #continue
        for typeInfs in fileTree[fileName]:
            if typeInfs!="URL":
                pass
                #continue

            tabAllArr, tabAllGen = [], []
            for fold in fileTree[fileName][typeInfs]:
                tabNamesMet, tabNamesCal, tabNamesDist=[], [], []
                tabNs, tabF1s, tabAUCROCs, tabAUCPRs, tabL1s, tabPearsons, tabAccs, tabMSEs, tabJSs, tabBriers, tabCrossEntropys = [], [], [], [], [], [], [], [], [], [], []

                for i, nbData in enumerate(fileTree[fileName][typeInfs][fold]):
                    if nbData!=1e6:
                        pass
                        #continue

                    nbDistSample = len(np.load("Output/"+fileName+"/"+fileName+"_"+typeInfs+"_"+str(nbData)+"_"+str(fold) + "_Fit_beta.npy")[-1][-1])
                    if runAll:
                        GetResults.results(fileName, typeInfs, nbData, fold, nbDistSample)

                    training, test, usrToInt, beta, betaIC, betaHawkes, betaCoC, betaTrue = GetResults.getData(fileName, typeInfs, nbData, fold)
                    nom = "Output/"+fileName+"/" + fileName+"_"+typeInfs+"_"+str(nbData)+"_"+str(fold)
                    print(nom)
                    N_inter = len(training)


                    seeDistrib=False
                    if seeDistrib:
                        plt.close()
                        from LogS import H, HGen
                        lgStep = 1
                        if fileName=="PD":
                            lgStep=1.01
                        obs = training
                        dicTemp = getMatInter(obs, lgStep, reduit=False)
                        cntFreq = getCntFreq(training)
                        betaMMSBM = getBetaIMMSBM(nom)
                        training, test, usrToInt, beta, betaIC, betaHawkes, betaCoC, betaTrue = getData(fileName, typeInfs, nbData, fold)

                        print(usrToInt)
                        nbInfs = len(beta)
                        for c in dicTemp:
                            for c2 in dicTemp[c]:
                                if c!=0 or c2 !=3:
                                    continue
                                s=0
                                maxN = 0
                                for dt in dicTemp[c][c2]:
                                    if sum(dicTemp[c][c2][dt])>maxN:
                                        maxN=sum(dicTemp[c][c2][dt])
                                for dt in dicTemp[c][c2]:
                                    s+=sum(dicTemp[c][c2][dt])
                                    r = dicTemp[c][c2][dt][1] / (sum(dicTemp[c][c2][dt])+1e-20)
                                    plt.bar(dt, r, width=.5, color="orange")#, alpha=sum(dicTemp[c][c2][dt])/maxN)

                                #sm = plt.cm.ScalarMappable(cmap=plt.cm.Oranges, norm=plt.Normalize(0, maxN))
                                #sm.set_array([])
                                #cbar = plt.colorbar(sm)
                                #cbar.set_label('Number of observations', rotation=270, labelpad=15)

                                print(c, c2, betaHawkes[c][c2], s)

                                a=np.linspace(1, max(dicTemp[c][c2]), 10000)
                                arrH = np.array([H(a_val, 0, beta[c, c2], nbDistSample=nbDistSample) for a_val in a])
                                arrHIC = np.array([H(a_val, 0, betaIC[c, c2], nbDistSample=nbDistSample) for a_val in a])
                                arrHHawkes = np.array([np.exp(-betaHawkes[c,c2,0] - betaHawkes[c,c2,1]*a_val) for a_val in a])
                                plt.plot(a, arrH, "b", label="IR-RBF")
                                #plt.plot(a, arrHHawkes, "y", label="IR-EXP")
                                #plt.plot(a, [betaMMSBM[c][c2][nbInfs+1] for i in range(len(a))], "r", label="IMMSBM")
                                #plt.plot(a, [cntFreq[c] for i in range(len(a))], "g", label="Naive")

                                try:
                                    arrHTrue = np.array([HGen(a_val, 0, betaTrue[c, c2], nbDistSample=nbDistSample) for a_val in a])
                                    plt.plot(a, arrHTrue+cntFreq[c], "c", label="True")
                                except Exception as e: pass

                                plt.ylim([0,1])
                                plt.legend()
                                plt.xlabel("Time separation", fontsize=18)
                                plt.ylabel("Probability of contamination", fontsize=18)
                                plt.rcParams['pdf.fonttype'] = 42
                                plt.rcParams['font.family'] = 'Calibri'
                                plt.tight_layout()
                                #plt.savefig("Misc/ExDist_SocialMedia_N=350000_Flickr-Flickr.pdf", dpi=600)
                                plt.show()

                        pause()

                    tabModelesMet, tabF1, tabROCAUC, tabPRAUC, tabAcc = loadMetrics(nom)
                    tabModelesCal, tabL1, tabPearson = loadCalib(nom)
                    tabModelesDist, tabJS, tabBrierScore, tabCrossEntropy = loadMetricsDist(nom)
                    try: tabModelesErr, tabMAE, tabMSE = loadErrs(nom)
                    except: tabModelesErr, tabMAE, tabMSE = np.array(tabModelesMet), np.array([0 for i in range(len(tabModelesMet))]), np.array([0 for i in range(len(tabModelesMet))])

                    tabNamesMet=tabModelesMet
                    tabNamesCal=tabModelesCal
                    tabNamesDist=tabModelesDist
                    tabNamesErr=tabModelesErr

                    tabF1s.append(tabF1)
                    tabL1s.append(tabL1)
                    tabAUCROCs.append(tabROCAUC)
                    tabAUCPRs.append(tabPRAUC)
                    tabPearsons.append(tabPearson)
                    tabAccs.append(tabAcc)
                    tabMSEs.append(tabMSE)
                    tabJSs.append(tabJS)
                    tabBriers.append(tabBrierScore)
                    tabCrossEntropys.append(tabCrossEntropy)
                    tabNs.append([N_inter for _ in range(len(tabF1))])


                tabNs, tabF1s, tabL1s, tabAUCROCs, tabAUCPRs, tabPearsons, tabAccs, tabMSEs, tabNamesMet, tabNamesCal, tabNamesErr = toNp(([tabNs, tabF1s, tabL1s, tabAUCROCs, tabAUCPRs, tabPearsons, tabAccs, tabMSEs, tabNamesMet, tabNamesCal, tabNamesErr]))
                tabJSs, tabBriers, tabCrossEntropys = toNp([tabJSs, tabBriers, tabCrossEntropys])

                tabArrs = [tabF1s, tabL1s, tabAUCROCs, tabAUCPRs, tabPearsons, tabAccs, tabMSEs, tabJSs, tabBriers, tabCrossEntropys, tabNs]
                tabGen = [tabNamesMet, tabNamesCal, tabNamesDist, tabNamesErr]
                #plotRecap(tabGen, tabArrs, fileName, typeInfs, fold, tabStds=None)
                tabAllArr.append(tabArrs)
                tabAllGen.append(tabGen)

            tabAllGen = np.array(tabAllGen)
            tabAllArr = np.array(tabAllArr)
            tabGen = tabAllGen[-1]

            tabNamesMet, tabNamesCal, tabNamesErr = tabGen[0], tabGen[1], tabGen[2]

            tabAvgF1, tabStdF1 = np.average(tabAllArr[:, 0], axis=0), np.std(tabAllArr[:, 0], axis=0)
            tabAvgL1, tabStdL1 = np.average(tabAllArr[:, 1], axis=0), np.std(tabAllArr[:, 1], axis=0)
            tabAvgAUCROC, tabStdAUCROC = np.average(tabAllArr[:, 2], axis=0), np.std(tabAllArr[:, 2], axis=0)
            tabAvgAUCPR, tabStdAUCPR = np.average(tabAllArr[:, 3], axis=0), np.std(tabAllArr[:, 3], axis=0)
            tabAvgPearson, tabStdPearson = np.average(tabAllArr[:, 4], axis=0), np.std(tabAllArr[:, 4], axis=0)
            tabAvgAcc, tabStdAcc = np.average(tabAllArr[:, 5], axis=0), np.std(tabAllArr[:, 5], axis=0)
            tabAvgMSE, tabStdMSE = np.average(tabAllArr[:, 6], axis=0), np.std(tabAllArr[:, 6], axis=0)
            tabAvgJS, tabStdJS = np.average(tabAllArr[:, 7], axis=0), np.std(tabAllArr[:, 7], axis=0)
            tabAvgBriers, tabStdBriers = np.average(tabAllArr[:, 8], axis=0), np.std(tabAllArr[:, 8], axis=0)
            tabAvgCrossEntropy, tabStdMSECrossEntropy = np.average(tabAllArr[:, 9], axis=0), np.std(tabAllArr[:, 9], axis=0)
            tabAvgN, tabStdN = np.average(tabAllArr[:, 10], axis=0), np.std(tabAllArr[:, 10], axis=0)

            tabGen = [tabNamesMet, tabNamesCal, tabNamesDist, tabNamesErr]
            tabArrs = [tabAvgF1, tabAvgL1, tabAvgAUCROC, tabAvgAUCPR, tabAvgPearson, tabAvgAcc, tabAvgMSE, tabAvgJS, tabAvgBriers, tabAvgCrossEntropy, tabAvgN]
            tabStds = [tabStdF1, tabStdL1, tabStdAUCROC, tabStdAUCPR, tabStdPearson, tabStdAcc, tabStdMSE, tabStdJS, tabStdBriers, tabStdMSECrossEntropy, tabStdN]

            plotRecap(tabGen, tabArrs, fileName, typeInfs, fold="", tabStds=tabStds)


    print(fileTree)


getGlobRes()