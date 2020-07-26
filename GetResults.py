from LogS import H
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from copy import deepcopy
import os
import cvxpy as cp

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
        betaCoC = betaCoC * (betaCoC>0).astype(int) + (betaCoC<=0).astype(int)*1e-20
        betaCoC = betaCoC * (betaCoC<1).astype(int) + (betaCoC>=1).astype(int)*(1-1e-20)
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

def saveEval(nom, tabTrue, tabP, tabPIC, tabPBL, tabPBLC, tabPIMMSBM, tabPRand):
    np.savetxt(nom+"_Evl_TrueArr.txt", tabTrue)
    np.savetxt(nom+"_Evl_P.txt", tabP)
    np.savetxt(nom+"_Evl_PIC.txt", tabPIC)
    np.savetxt(nom+"_Evl_PBL.txt", tabPBL)
    np.savetxt(nom+"_Evl_PBLC.txt", tabPBLC)
    np.savetxt(nom+"_Evl_PIMMSBM.txt", tabPIMMSBM)
    np.savetxt(nom+"_Evl_PRand.txt", tabPRand)
    np.savetxt(nom+"_Evl_Keys.txt", tabKeys, fmt='%s')

def saveMetrics(nom, tabPRAUC, tabROCAUC, tabF1, tabAcc):
    modeles = ["IR-RBF", "ICIR-RBF", "IR-EXP", "Naive", "BLC", "IMMSBM", "CoC", "Random"]

    with open(nom+"_Metriques.txt", "w+") as f:
        f.write("Modèle\tmax-F1\tROCAUC\tPRAUC\tmax-accuracy\n")
        for i in range(len(modeles)):
            f.write(modeles[i]+"\t"+str(tabF1[i])+"\t"+str(tabROCAUC[i])+"\t"+str(tabPRAUC[i])+"\t"+str(tabAcc[i])+"\n")

def saveMetricsDist(nom, tabJS, tabBrierScore, tabCrossEntropy):
    modeles = ["IR-RBF", "ICIR-RBF", "IR-EXP", "Naive", "BLC", "IMMSBM", "CoC", "Random"]

    with open(nom+"_Metriques_dist.txt", "w+") as f:
        f.write("Modèle\tJensenShannonDiv\tBrierScore\tCrossEntropy\n")
        for i in range(len(modeles)):
            f.write(modeles[i]+"\t"+str(tabJS[i])+"\t"+str(tabBrierScore[i])+"\t"+str(tabCrossEntropy[i])+"\n")

def saveCalib(nom, tabL1, tabPearson):
    modeles = ["IR-RBF", "ICIR-RBF", "IR-EXP", "Naive", "BLC", "IMMSBM", "CoC", "Random"]

    with open(nom + "_Metriques_calib.txt", "w+") as f:
        f.write("Modèle\tL1\tPearson\n")
        for i in range(len(modeles)):
            f.write(modeles[i]+"\t"+str(tabL1[i])+"\t"+str(tabPearson[i])+"\n")

def toNp(arr):
    return [np.array(a) for a in arr]



def getProbsBL(training):
    probs = {}
    cntSeq = {}
    for u in training:
        training[u] = sorted(training[u], key=lambda x: x[1])
        for (c, t, s) in training[u]:
            if s == 0:
                try:
                    cntSeq[c][0] += 1
                except:
                    cntSeq[c] = np.zeros((2))
                    cntSeq[c][0] += 1
            else:
                try:
                    cntSeq[c][1] += 1
                except:
                    cntSeq[c] = np.zeros((2))
                    cntSeq[c][1] += 1

    for u in cntSeq:
        probs[u] = cntSeq[u][1] / (cntSeq[u][0] + cntSeq[u][1])

    return probs

def getProbsBLC(data):
    cntSeq = {}
    for u in data:
        data[u] = sorted(data[u], key=lambda x: x[1])
        for (c, t, s) in data[u]:
            for (c2, t2, s2) in data[u]:
                if t2 <= t and s!=1:
                    try:
                        cntSeq[c][c2][0]+=1
                    except:
                        try:
                            cntSeq[c][c2] = np.zeros((2))
                            cntSeq[c][c2][0] += 1
                        except:
                            cntSeq[c]={}
                            cntSeq[c][c2] = np.zeros((2))
                            cntSeq[c][c2][0] += 1

                elif t2 <= t and s==1:
                    try:
                        cntSeq[c][c2][1]+=1
                    except:
                        try:
                            cntSeq[c][c2] = np.zeros((2))
                            cntSeq[c][c2][1] += 1
                        except:
                            cntSeq[c]={}
                            cntSeq[c][c2] = np.zeros((2))
                            cntSeq[c][c2][1] += 1

    for c in cntSeq:
        for c2 in cntSeq[c]:
            cntSeq[c][c2] = cntSeq[c][c2][1] / sum(cntSeq[c][c2])

    return cntSeq

def rescaleArrays(tabArr):
    tabArrMod=[]
    for arr in tabArr:
        a = (arr-min(arr))/(max(arr)-min(arr)+1e-20)
        tabArrMod.append(a)
    return tabArrMod



def getTabs(training, test, beta, betaIC, betaHawkes, betaIMMSBM, betaCoC, nbDistSample, lgStep, cntFreq):
    probsBL = getCntFreq(training)
    probsBLC = getProbsBLC(training)
    nbInfs = len(beta)

    mat = getMatInter(test, reduit=False)

    tabP, tabPIC, tabPBL, tabPRand, tabPBLC, tabPIMMSBM, tabPCoC, tabPHawkes, tabTrue, tabInfs = [], [], [], [], [], [], [], [], [], []
    for c in mat:
        if c not in probsBL:
            probsBL[c]=0
        for c2 in mat[c]:
            for dt in mat[c][c2]:
                if dt <= 0 or dt>20:
                    continue

                p = H(dt, 0, beta[c, c2], nbDistSample)
                if betaIC is not None:
                    if c==c2:
                        pIC = H(dt, 0, betaIC[c, c2], nbDistSample)
                    else:
                        pIC = probsBL[c]
                else:
                    pIC = 1

                if betaHawkes is not None:
                    pHawkes = np.exp(-betaHawkes[c,c2,0] - betaHawkes[c,c2,1]*dt)
                else:
                    pHawkes = 1

                if betaIMMSBM is not None:
                    pIMMSBM = betaIMMSBM[c,c2,nbInfs+1]  # Prob de survie
                else:
                    pIMMSBM = 1
                    
                if betaCoC is not None:
                    pCoC = betaCoC[c,c2,int(dt)]  # Prob de survie
                else:
                    pCoC = 1

                try:
                    pBLC = probsBLC[c][c2]
                except:
                    pBLC = 1

                pBL = probsBL[c]
                pRand = random.random()
                ptest = mat[c][c2][dt][1]/(sum(mat[c][c2][dt])+1e-20)

                tabTrue.append(ptest)
                tabInfs.append(sum(mat[c][c2][dt]))
                tabP.append(p)
                tabPIC.append(pIC)
                tabPHawkes.append(pHawkes)
                tabPBL.append(pBL)
                tabPBLC.append(pBLC)
                tabPIMMSBM.append(pIMMSBM)
                tabPCoC.append(pCoC)
                tabPRand.append(pRand)


    tabP, tabPIC, tabPHawkes, tabPBL, tabPBLC, tabPIMMSBM, tabPCoC, tabPRand, tabTrue, tabInfs = np.array(tabP), np.array(tabPIC), np.array(tabPHawkes), np.array(tabPBL), np.array(tabPBLC), np.array(tabPIMMSBM), np.array(tabPCoC), np.array(tabPRand), np.array(tabTrue), np.array(tabInfs)

    return tabTrue, tabInfs, tabP, tabPIC, tabPBL, tabPBLC, tabPIMMSBM, tabPCoC, tabPHawkes, tabPRand

def getTabsClassic(training, test, beta, betaIC, betaIMMSBM, nbDistSample, lgStep, cntFreq):
    probsBL = getCntFreq(training)
    probsBLC = getProbsBLC(training)
    nbInfs = len(beta)

    iter=0
    tabP, tabPIC, tabPBL, tabPRand, tabPBLC, tabPIMMSBM, tabTrue, tabInfs = [], [], [], [], [], [], [], []
    dicTabsP, dicTabsPIC, dicTabsPBL, dicTabsPBLC, dicTabsPIMMSBM, dicTabsPRand, dicTabsInfs, dicTabsTrue = {}, {}, {}, {}, {}, {}, {}, {}
    for u in test:
        for (c, t, s) in test[u]:
            if t<10:
                continue

            p, pIC, pIMMSBM, pBLC = 1., 1., 1., 1.
            tabPtemp, tabPICtemp=[], []

            for (c2, t2, s2) in test[u]:
                tdiff = t - t2 + lgStep
                if tdiff <= 0 or tdiff>20:
                    continue

                p *= 1-max([H(tdiff, 0, beta[c, c2], nbDistSample)-probsBL[c]*0, 0])
                tabPtemp.append(max([H(tdiff, 0, beta[c, c2], nbDistSample)-probsBL[c], 0]))
                #print(c, c2, s, p)
                if betaIC is not None:
                    #pIC *= 1-H(tdiff, 0, betaIC[c2, c3], nbDistSample)
                    pIC *= 1 - max([H(tdiff, 0, betaIC[c, c2], nbDistSample)-probsBL[c]*0, 0])
                    tabPICtemp.append(max([H(tdiff, 0, betaIC[c, c2], nbDistSample)-probsBL[c], 0]))
                else:
                    pIC *= 1

                if betaIMMSBM is not None:
                    pIMMSBM *= 1. - betaIMMSBM[c,c2,nbInfs]/(betaIMMSBM[c,c2,nbInfs]+betaIMMSBM[c,c2,nbInfs+1])  # Prob de survie
                else:
                    pIMMSBM = 1

                try:
                    pBLC *= 1-probsBLC[c][c2]
                except:
                    pass


            p = max([1. - p, 0])
            p = min([np.sum(tabPtemp)+probsBL[c], 1])
            #p = H(tdiff, 0, beta[c, c2], nbDistSample)+cntFreq[c]
            pIC = max([1. - pIC, 0])
            pIC = min([np.sum(tabPICtemp)+probsBL[c], 1])
            #pIC = H(tdiff, 0, betaIC[c, c2], nbDistSample)+cntFreq[c]
            pBLC = 1. - pBLC
            #pBLC = probsBLC[c][c2]
            pIMMSBM = 1 - pIMMSBM
            pBL = probsBL[c]
            pRand = random.random()

            tabInfs.append(c)
            tabTrue.append(s)
            tabP.append(p)
            tabPIC.append(pIC)
            tabPBL.append(pBL)
            tabPBLC.append(pBLC)
            tabPIMMSBM.append(pIMMSBM)
            tabPRand.append(pRand)


            iter+=1

    tabP, tabPIC, tabPBL, tabPBLC, tabPIMMSBM, tabPRand, tabTrue, tabInfs = np.array(tabP), np.array(tabPIC), np.array(tabPBL), np.array(tabPBLC), np.array(tabPIMMSBM), np.array(tabPRand), np.array(tabTrue), np.array(tabInfs)


    return tabTrue, tabInfs, tabP, tabPIC, tabPBL, tabPBLC, tabPIMMSBM, tabPRand

def getTabsCalib(training, beta, betaIC, betaHawkes, betaIMMSBM, betaCoC, nbDistSample, cntSeq, cntFreq):
    probsBL = getProbsBL(training)
    probsBLC = getProbsBLC(training)
    tabP, tabPIC, tabPHawkes, tabPBL, tabPBLC, tabPIMMSBM, tabPCoC, tabPRand, tabF, tabW = [], [], [], [], [], [], [], [], [], []
    tabKeys = []

    mat = getMatInter(training, reduit=False)

    for c in mat:
        for c2 in mat[c]:
            for tdiff in mat[c][c2]:
                s = sum(mat[c][c2][tdiff])
                f = mat[c][c2][tdiff][1] / s

                p = H(tdiff, 0, beta[c,c2], nbDistSample)
                if betaIC is not None:
                    if c==c2:
                        pIC = H(tdiff, 0, betaIC[c, c2], nbDistSample)

                    else:
                        pIC = probsBLC[c][c2]
                else:
                    pIC = -1
                if betaHawkes is not None:
                    pHawkes = np.exp(-betaHawkes[c,c2,0] - betaHawkes[c,c2,1]*tdiff)
                else:
                    pHawkes = -1
                pBL = probsBL[c]
                try:
                    pBLC = probsBLC[c][c2]
                except:
                    pBLC = 0
                if betaIMMSBM is not None:
                    pIMMSBM = betaIMMSBM[c,c2,c]
                else:
                    pIMMSBM = 0
                if betaCoC is not None:
                    pCoC = betaCoC[c,c2,int(tdiff)]
                else:
                    pCoC = 0
                pRand = random.random()

                tabP.append(p)
                tabPIC.append(pIC)
                tabPHawkes.append(pHawkes)
                tabPBL.append(pBL)
                tabPBLC.append(pBLC)
                tabPIMMSBM.append(pIMMSBM)
                tabPCoC.append(pCoC)
                tabPRand.append(pRand)
                tabF.append(f)
                tabW.append(s)
                tabKeys.append((c,c2,tdiff))

    tabP, tabPIC, tabPHawkes, tabPBL, tabPBLC, tabPIMMSBM, tabPCoC, tabPRand, tabF, tabW, tabKeys = np.array(tabP), np.array(tabPIC), np.array(tabPHawkes), np.array(tabPBL), np.array(tabPBLC), np.array(tabPIMMSBM), np.array(tabPCoC), np.array(tabPRand), np.array(tabF), np.array(tabW), np.array(tabKeys)

    return tabP, tabPIC, tabPHawkes, tabPBL, tabPBLC, tabPIMMSBM, tabPCoC, tabPRand, tabF, tabW, tabKeys

def getCntSeq(data, lgStep, res=1):
    res = int(res)
    cntSeq = {}

    for u in data:
        data[u] = sorted(data[u], key=lambda x: x[1])
        for (c, t, s) in data[u]:
            for (c2, t2, s) in data[u]:
                diff = t + lgStep - t2

                if res!=0:
                    diff = diff // res
                    diff *= res
                else:
                    diff=diff

                if t2 <= t and s==0:
                    try:
                        cntSeq[c][c2][diff][0]+=1
                    except:
                        try:
                            cntSeq[c][c2][diff] = np.zeros((2))
                            cntSeq[c][c2][diff][0] += 1
                        except:
                            try:
                                cntSeq[c][c2] = {}
                                cntSeq[c][c2][diff] = np.zeros((2))
                                cntSeq[c][c2][diff][0] += 1
                            except:
                                cntSeq[c]={}
                                cntSeq[c][c2] = {}
                                cntSeq[c][c2][diff] = np.zeros((2))
                                cntSeq[c][c2][diff][0] += 1

                elif t2 <= t and s==1:
                    try:
                        cntSeq[c][c2][diff][1]+=1
                    except:
                        try:
                            cntSeq[c][c2][diff] = np.zeros((2))
                            cntSeq[c][c2][diff][1] += 1
                        except:
                            try:
                                cntSeq[c][c2] = {}
                                cntSeq[c][c2][diff] = np.zeros((2))
                                cntSeq[c][c2][diff][1] += 1
                            except:
                                cntSeq[c]={}
                                cntSeq[c][c2] = {}
                                cntSeq[c][c2][diff] = np.zeros((2))
                                cntSeq[c][c2][diff][1] += 1

    return cntSeq

def getConfMat(tab, tabTrue, tabInfs, seuil):
    mat = np.zeros((2,2))

    '''
    mat[1,1] = np.sum(tab * (tab > seuil).astype(int) * tabTrue)
    mat[0,0] = np.sum(tab * (tab <= seuil).astype(int) * (1-tabTrue))

    mat[1,0] = np.sum(tab * (tab <= seuil).astype(int) * tabTrue)
    mat[0,1] = np.sum(tab * (tab > seuil).astype(int) * (1-tabTrue))
    '''

    propTruePos = tabTrue * (tab>=tabTrue).astype(int) + tab * (tab<tabTrue).astype(int)
    propTrueNeg = (1-tabTrue) * ((1-tab) >= (1-tabTrue)).astype(int) + (1-tab) * ((1-tab) < (1-tabTrue)).astype(int)
    propFalsePos = (tab-tabTrue) * (tab >= tabTrue).astype(int)
    propFalseNeg = (tabTrue-tab) * (tab < tabTrue).astype(int)

    mat[1,1] = np.sum(propTruePos * tabInfs)
    mat[0,0] = np.sum(propTrueNeg * tabInfs)

    mat[1,0] = np.sum(propFalseNeg * tabInfs)
    mat[0,1] = np.sum(propFalsePos * tabInfs)


    prec, rec, TPR, FPR, acc = 0., 0., 0., 0., 0.
    if mat[1,1]+mat[0,1] != 0:
        prec = mat[1,1]/(mat[1,1]+mat[0,1])
    if mat[1,1]+mat[1,0] != 0:
        rec = mat[1,1]/(mat[1,1]+mat[1,0])
        TPR = rec
    if mat[0,1]+mat[0,0] != 0:
        FPR = mat[0,1]/(mat[0,1]+mat[0,0])
    if np.sum(mat) != 0:
        acc = (mat[1,1]+mat[0,0])/np.sum(mat)

    return prec, rec, TPR, FPR, acc

def getLenInfStep(obs):
    l, div = 0., 0
    ltot, divtot = 0., 0.
    for u in obs:
        l+=obs[u][-1][1] - obs[u][-2][1]
        div+=1
        for i in range(len(obs[u][:-1])):
            ltot += obs[u][i+1][1] - obs[u][i][1]
            divtot += 1

    return l/div, ltot/divtot

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

def getMatInter(obs, usri=None, reduit=True):
    freq = getCntFreq(obs)
    dicTemp={}
    for u in obs:
        for (c,t,s) in obs[u]:
            if usri is not None:
                if usri!=c:
                    continue

            if c not in dicTemp: dicTemp[c]={}

            for (c2,t2,s2) in obs[u]:
                dt = t-t2+1

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
                    if b!=0:
                        dicTemp[c][c2][dt][1] = max([0, b*(asapb-P0)/(1-asapb+P0+1e-20)])

    if usri is not None:
        return dicTemp[usri]
    else:
        return dicTemp



def PRROC(training, test, beta, betaIC, betaHawkes, betaIMMSBM, betaCoC, nbDistSample, lgStep, cntFreq, nom, save=False):
    tabTrue, tabInfs, tabP, tabPIC, tabPBL, tabPBLC, tabPIMMSBM, tabPCoC, tabPHawkes, tabPRand = getTabs(training, test, beta, betaIC, betaHawkes, betaIMMSBM, betaCoC, nbDistSample, lgStep, cntFreq)

    pas=0.05
    thres = -pas
    tabPrec, tabRec, tabTPR, tabFPR, tabAcc = [], [], [], [], []
    tabPrecIC, tabRecIC, tabTPRIC, tabFPRIC, tabAccIC = [], [], [], [], []
    tabPrecHawkes, tabRecHawkes, tabTPRHawkes, tabFPRHawkes, tabAccHawkes = [], [], [], [], []
    tabPrecBL, tabRecBL, tabTPRBL, tabFPRBL, tabAccBL = [], [], [], [], []
    tabPrecBLC, tabRecBLC, tabTPRBLC, tabFPRBLC, tabAccBLC = [], [], [], [], []
    tabPrecIMMSBM, tabRecIMMSBM, tabTPRIMMSBM, tabFPRIMMSBM, tabAccIMMSBM = [], [], [], [], []
    tabPrecCoC, tabRecCoC, tabTPRCoC, tabFPRCoC, tabAccCoC = [], [], [], [], []
    tabPrecRand, tabRecRand, tabTPRRand, tabFPRRand, tabAccRand = [], [], [], [], []
    tabSeuil = []
    while thres <= 1+pas:
        #print(thres)
        prec, rec, TPR, FPR, acc = getConfMat(tabP, tabTrue, tabInfs, thres)
        precIC, recIC, TPRIC, FPRIC, accIC = getConfMat(tabPIC, tabTrue, tabInfs, thres)
        precHawkes, recHawkes, TPRHawkes, FPRHawkes, accHawkes = getConfMat(tabPHawkes, tabTrue, tabInfs, thres)
        precBL, recBL, TPRBL, FPRBL, accBL = getConfMat(tabPBL, tabTrue, tabInfs, thres)
        precBLC, recBLC, TPRBLC, FPRBLC, accBLC = getConfMat(tabPBLC, tabTrue, tabInfs, thres)
        precIMMSBM, recIMMSBM, TPRIMMSBM, FPRIMMSBM, accIMMSBM = getConfMat(tabPIMMSBM, tabTrue, tabInfs, thres)
        precCoC, recCoC, TPRCoC, FPRCoC, accCoC = getConfMat(tabPCoC, tabTrue, tabInfs, thres)
        precRand, recRand, TPRRand, FPRRand, accRand = getConfMat(tabPRand, tabTrue, tabInfs, thres)

        if thres > 1 and False:  # Definition premier point PR
            prec, rec = tabPrec[-1], 0
            precIC, recIC = tabPrecIC[-1], 0
            precHawkes, recHawkes = tabPrecHawkes[-1], 0
            precBL, recBL = tabPrecBL[-1], 0
            precBLC, recBLC = tabPrecBLC[-1], 0
            precIMMSBM, recIMMSBM = tabPrecIMMSBM[-1], 0
            precCoC, recCoC = tabPrecCoC[-1], 0
            precRand, recRand = tabPrecRand[-1], 0

        tabPrec.append(prec)
        tabRec.append(rec)
        tabTPR.append(TPR)
        tabFPR.append(FPR)
        tabAcc.append(acc)

        tabPrecIC.append(precIC)
        tabRecIC.append(recIC)
        tabTPRIC.append(TPRIC)
        tabFPRIC.append(FPRIC)
        tabAccIC.append(accIC)

        tabPrecHawkes.append(precHawkes)
        tabRecHawkes.append(recHawkes)
        tabTPRHawkes.append(TPRHawkes)
        tabFPRHawkes.append(FPRHawkes)
        tabAccHawkes.append(accHawkes)

        tabPrecBL.append(precBL)
        tabRecBL.append(recBL)
        tabTPRBL.append(TPRBL)
        tabFPRBL.append(FPRBL)
        tabAccBL.append(accBL)

        tabPrecBLC.append(precBLC)
        tabRecBLC.append(recBLC)
        tabTPRBLC.append(TPRBLC)
        tabFPRBLC.append(FPRBLC)
        tabAccBLC.append(accBLC)
        
        tabPrecIMMSBM.append(precIMMSBM)
        tabRecIMMSBM.append(recIMMSBM)
        tabTPRIMMSBM.append(TPRIMMSBM)
        tabFPRIMMSBM.append(FPRIMMSBM)
        tabAccIMMSBM.append(accIMMSBM)
        
        tabPrecCoC.append(precCoC)
        tabRecCoC.append(recCoC)
        tabTPRCoC.append(TPRCoC)
        tabFPRCoC.append(FPRCoC)
        tabAccCoC.append(accCoC)

        tabPrecRand.append(precRand)
        tabRecRand.append(recRand)
        tabTPRRand.append(TPRRand)
        tabFPRRand.append(FPRRand)
        tabAccRand.append(accRand)

        tabSeuil.append(thres)

        thres += pas

    [tabPrec, tabRec, tabTPR, tabFPR, tabAcc] = toNp([tabPrec, tabRec, tabTPR, tabFPR, tabAcc])
    [tabPrecIC, tabRecIC, tabTPRIC, tabFPRIC, tabAccIC] = toNp([tabPrecIC, tabRecIC, tabTPRIC, tabFPRIC, tabAccIC])
    [tabPrecHawkes, tabRecHawkes, tabTPRHawkes, tabFPRHawkes, tabAccHawkes] = toNp([tabPrecHawkes, tabRecHawkes, tabTPRHawkes, tabFPRHawkes, tabAccHawkes])
    [tabPrecBL, tabRecBL, tabTPRBL, tabFPRBL, tabAccBL] = toNp([tabPrecBL, tabRecBL, tabTPRBL, tabFPRBL, tabAccBL])
    [tabPrecBLC, tabRecBLC, tabTPRBLC, tabFPRBLC, tabAccBLC] = toNp([tabPrecBLC, tabRecBLC, tabTPRBLC, tabFPRBLC, tabAccBLC])
    [tabPrecIMMSBM, tabRecIMMSBM, tabTPRIMMSBM, tabFPRIMMSBM, tabAccIMMSBM] = toNp([tabPrecIMMSBM, tabRecIMMSBM, tabTPRIMMSBM, tabFPRIMMSBM, tabAccIMMSBM])
    [tabPrecCoC, tabRecCoC, tabTPRCoC, tabFPRCoC, tabAccCoC] = toNp([tabPrecCoC, tabRecCoC, tabTPRCoC, tabFPRCoC, tabAccCoC])
    [tabPrecRand, tabRecRand, tabTPRRand, tabFPRRand, tabAccRand] = toNp([tabPrecRand, tabRecRand, tabTPRRand, tabFPRRand, tabAccRand])

    PRAUC, ROCAUC, maxF1, acc = -np.trapz(tabPrec, tabRec), -np.trapz(tabTPR, tabFPR), max(2. * tabPrec * tabRec / (tabPrec + tabRec+1e-20)), max(tabAcc)
    PRAUCIC, ROCAUCIC, maxF1IC, accIC = -np.trapz(tabPrecIC, tabRecIC), -np.trapz(tabTPRIC, tabFPRIC), max(2. * tabPrecIC * tabRecIC / (tabPrecIC + tabRecIC+1e-20)), max(tabAccIC)
    PRAUCHawkes, ROCAUCHawkes, maxF1Hawkes, accHawkes = -np.trapz(tabPrecHawkes, tabRecHawkes), -np.trapz(tabTPRHawkes, tabFPRHawkes), max(2. * tabPrecHawkes * tabRecHawkes / (tabPrecHawkes + tabRecHawkes+1e-20)), max(tabAccHawkes)
    PRAUCBL, ROCAUCBL, maxF1BL, accBL = -np.trapz(tabPrecBL, tabRecBL), -np.trapz(tabTPRBL, tabFPRBL), max(2. * tabPrecBL * tabRecBL / (tabPrecBL + tabRecBL+1e-20)), max(tabAccBL)
    PRAUCBLC, ROCAUCBLC, maxF1BLC, accBLC = -np.trapz(tabPrecBLC, tabRecBLC), -np.trapz(tabTPRBLC, tabFPRBLC), max(2. * tabPrecBLC * tabRecBLC / (tabPrecBLC + tabRecBLC+1e-20)), max(tabAccBLC)
    PRAUCIMMSBM, ROCAUCIMMSBM, maxF1IMMSBM, accIMMSBM = -np.trapz(tabPrecIMMSBM, tabRecIMMSBM), -np.trapz(tabTPRIMMSBM, tabFPRIMMSBM), max(2. * tabPrecIMMSBM * tabRecIMMSBM / (tabPrecIMMSBM + tabRecIMMSBM+1e-20)), max(tabAccIMMSBM)
    PRAUCCoC, ROCAUCCoC, maxF1CoC, accCoC = -np.trapz(tabPrecCoC, tabRecCoC), -np.trapz(tabTPRCoC, tabFPRCoC), max(2. * tabPrecCoC * tabRecCoC / (tabPrecCoC + tabRecCoC+1e-20)), max(tabAccCoC)
    PRAUCRand, ROCAUCRand, maxF1Rand, accRand = -np.trapz(tabPrecRand, tabRecRand), -np.trapz(tabTPRRand, tabFPRRand), max(2. * tabPrecRand * tabRecRand / (tabPrecRand + tabRecRand+1e-20)), max(tabAccRand)

    tabPRAUC = [PRAUC, PRAUCIC, PRAUCHawkes, PRAUCBL, PRAUCBLC, PRAUCIMMSBM, PRAUCCoC, PRAUCRand]
    tabROCAUC = [ROCAUC, ROCAUCIC, ROCAUCHawkes, ROCAUCBL, ROCAUCBLC, ROCAUCIMMSBM, ROCAUCCoC, ROCAUCRand]
    tabF1 = [maxF1, maxF1IC, maxF1Hawkes, maxF1BL, maxF1BLC, maxF1IMMSBM, maxF1CoC, maxF1Rand]
    tabAccuracy = [acc, accIC, accHawkes, accBL, accBLC, accIMMSBM, accCoC, accRand]

    if save:
        #saveEval(nom, tabTrue, tabP, tabPIC, tabPBL, tabPBLC, tabPIMMSBM, tabPRand)
        saveMetrics(nom, tabPRAUC, tabROCAUC, tabF1, tabAccuracy)

    '''
    inds = []
    g, d = [], []
    for (p, r) in [(tabPrec, tabRec), (tabPrecIC, tabRecIC), (tabPrecBL, tabRecBL), (tabPrecBLC, tabRecBLC), (tabPrecIMMSBM, tabRecIMMSBM), (tabPrecRand, tabRecRand)]:
        ind = np.where(2. * p * r / (p + r + 1e-20) == max(2. * p * r / (p + r + 1e-20)))[0][0]
        g.append(r[ind])
        d.append(p[ind])
    '''

    plt.close()
    plt.figure(figsize=(18, 6))

    plt.subplot(131)
    plt.plot(tabRec, tabPrec, label="Modèle, maxF1=%.3f" %maxF1)
    plt.plot(tabRecIC, tabPrecIC, label="IC, maxF1=%.3f" %maxF1IC)
    plt.plot(tabRecHawkes, tabPrecHawkes, label="Hawkes, maxF1=%.3f" %maxF1Hawkes)
    plt.plot(tabRecBL, tabPrecBL, label="BL, maxF1=%.3f" %maxF1BL)
    plt.plot(tabRecBLC, tabPrecBLC, label="BLC, maxF1=%.3f" %maxF1BLC)
    plt.plot(tabRecIMMSBM, tabPrecIMMSBM, label="IMMSBM, maxF1=%.3f" %maxF1IMMSBM)
    plt.plot(tabRecCoC, tabPrecCoC, label="CoC, maxF1=%.3f" %maxF1CoC)
    plt.plot(tabRecRand, tabPrecRand, label="Rand, maxF1=%.3f" %maxF1Rand)
    #plt.plot(g, d, "or", markersize=5)


    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend()
    plt.subplot(132)
    plt.plot(tabFPR, tabTPR, label="Modèle, AUC=%.3f" %ROCAUC)
    plt.plot(tabFPRIC, tabTPRIC, label="IC, AUC=%.3f" %ROCAUCIC)
    plt.plot(tabFPRHawkes, tabTPRHawkes, label="Hawkes, AUC=%.3f" %ROCAUCHawkes)
    plt.plot(tabFPRBL, tabTPRBL, label="BL, AUC=%.3f" %ROCAUCBL)
    plt.plot(tabFPRBLC, tabTPRBLC, label="BLC, AUC=%.3f" %ROCAUCBLC)
    plt.plot(tabFPRIMMSBM, tabTPRIMMSBM, label="IMMSBM, AUC=%.3f" %ROCAUCIMMSBM)
    plt.plot(tabFPRCoC, tabTPRCoC, label="CoC, AUC=%.3f" %ROCAUCCoC)
    plt.plot(tabFPRRand, tabTPRRand, label="Rand, AUC=%.3f" %ROCAUCRand)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend()
    plt.subplot(133)
    plt.plot(tabSeuil, tabAcc, label="Modèle, maxAcc=%.3f" %max(tabAcc))
    plt.plot(tabSeuil, tabAccIC, label="IC, maxAcc=%.3f" %max(tabAccIC))
    plt.plot(tabSeuil, tabAccHawkes, label="Hawkes, maxAcc=%.3f" %max(tabAccHawkes))
    plt.plot(tabSeuil, tabAccBL, label="BL, maxAcc=%.3f" %max(tabAccBL))
    plt.plot(tabSeuil, tabAccBLC, label="BLC, maxAcc=%.3f" %max(tabAccBLC))
    plt.plot(tabSeuil, tabAccIMMSBM, label="IMMSBM, maxAcc=%.3f" %max(tabAccIMMSBM))
    plt.plot(tabSeuil, tabAccCoC, label="CoC, maxAcc=%.3f" %max(tabAccCoC))
    plt.plot(tabSeuil, tabAccRand, label="Rand, maxAcc=%.3f" %max(tabAccRand))
    plt.xlabel("Threshold")
    plt.ylabel("Accuracy")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend()
    if save:
        plt.savefig(nom+"_Metriques.png")
    #plt.show()

    return tabTrue, tabP, tabPIC, tabPHawkes, tabPBL, tabPBLC, tabPIMMSBM, tabPCoC, tabPRand, None

def evaluation(alphaInfer, alphaTrue=None, nom="", modele="", save=False):
    if alphaTrue is None:
        #print("Give alphaTrue for evaluation")
        return -1
    if alphaInfer is None:
        #print("Give alphaInfer for evaluation")
        return -1

    absErr = np.mean(abs(alphaInfer[alphaTrue.nonzero()] - alphaTrue[alphaTrue.nonzero()]))
    sqrAbsErr = np.mean(abs(alphaInfer[alphaTrue.nonzero()] - alphaTrue[alphaTrue.nonzero()])**2)
    normAbsErr = np.mean(abs(alphaInfer[alphaTrue.nonzero()] - alphaTrue[alphaTrue.nonzero()])/(alphaTrue[alphaTrue.nonzero()]))

    absErr = np.mean(abs(alphaInfer - alphaTrue))
    sqrAbsErr = np.mean(abs(alphaInfer - alphaTrue)**2)


    if save:
        with open(nom+"_Errors_"+modele+".txt", "w+") as f:
            f.write("AbsErr\tSqrAbsErr\tNormAbsErr\n")
            f.write(str(absErr)+"\t"+str(sqrAbsErr)+"\t"+str(normAbsErr)+"\n")

    printRes = False
    if printRes:
        print("Mean abs error alpha:", absErr)
        print("Mean squared abs error alpha:", sqrAbsErr)
        print("Normalized mean abs error alpha:", normAbsErr)

    return sqrAbsErr

def getMetricsDist(training, test, beta, betaIC, betaHawkes, betaIMMSBM, betaCoC, nbDistSample, lgStep, cntFreq, nom, save=False):
    nbInfs = len(beta)
    distTest = getMatInter(test, reduit=False)
    probsBL = getProbsBL(training)
    probsBLC = getProbsBLC(training)
    tabFreqs, tabProbs, tabProbsIC, tabProbsHawkes, tabProbsBL, tabProbsBLC, tabProbsIMMSBM, tabProbsCoC, tabProbsRand = [], [], [], [], [], [], [], [], []
    dist, distIC, distHawkes, distBL, distBLC, distIMMSBM, distCoC, distRand = {}, {}, {}, {}, {}, {}, {}, {}
    crossEntropy, crossEntropyIC, crossEntropyHawkes, crossEntropyBL, crossEntropyBLC, crossEntropyIMMSBM, crossEntropyCoC, crossEntropyRand, div = 0, 0, 0, 0, 0, 0, 0, 0, 0
    for c in distTest:
        if c not in dist: dist[c] = {}
        if c not in distIC: distIC[c] = {}
        if c not in distHawkes: distHawkes[c] = {}
        if c not in distBL: distBL[c] = {}
        if c not in distBLC: distBLC[c] = {}
        if c not in distRand: distRand[c] = {}
        if c not in distIMMSBM: distIMMSBM[c] = {}
        if c not in distCoC: distCoC[c] = {}
        for c2 in distTest[c]:
            if c2 not in dist[c]: dist[c][c2]={}
            if c2 not in distIC[c]: distIC[c][c2]={}
            if c2 not in distHawkes[c]: distHawkes[c][c2]={}
            if c2 not in distBL[c]: distBL[c][c2]={}
            if c2 not in distBLC[c]: distBLC[c][c2]={}
            if c2 not in distIMMSBM[c]: distIMMSBM[c][c2]={}
            if c2 not in distCoC[c]: distCoC[c][c2]={}
            if c2 not in distRand[c]: distRand[c][c2]={}
            for dt in distTest[c][c2]:
                if dt==1:
                    continue

                dist[c][c2][dt] = H(dt, 0, beta[c][c2], nbDistSample=nbDistSample)
                try:
                    if c==c2:
                        distIC[c][c2][dt] = H(dt, 0, betaIC[c][c2], nbDistSample=nbDistSample)
                    else:
                        distIC[c][c2][dt] = probsBL[c]
                except: distIC[c][c2][dt] = 0
                try: distHawkes[c][c2][dt] = np.exp(-betaHawkes[c,c2,0] - betaHawkes[c,c2,1]*dt)
                except: distHawkes[c][c2][dt] = 0
                distBL[c][c2][dt] = probsBL[c]
                try: distBLC[c][c2][dt] = probsBLC[c][c2]
                except: distBLC[c][c2][dt] = 0
                try: distIMMSBM[c][c2][dt] = betaIMMSBM[c,c2,nbInfs+1]#/(betaIMMSBM[c,c2,c+nbInfs]+betaIMMSBM[c,c2,c])
                except: distIMMSBM[c][c2][dt] = 0
                try: distCoC[c][c2][dt] = betaCoC[c,c2,int(dt)]#/(betaCoC[c,c2,c+nbInfs]+betaCoC[c,c2,c])
                except: distCoC[c][c2][dt] = 0
                distRand[c][c2][dt] = random.random()
                ptest = distTest[c][c2][dt][1] / (sum(distTest[c][c2][dt]))
                
                tabFreqs.append(ptest)
                tabProbs.append(dist[c][c2][dt])
                tabProbsIC.append(distIC[c][c2][dt])
                tabProbsHawkes.append(distHawkes[c][c2][dt])
                tabProbsBL.append(distBL[c][c2][dt])
                tabProbsBLC.append(distBLC[c][c2][dt])
                tabProbsIMMSBM.append(distIMMSBM[c][c2][dt])
                tabProbsCoC.append(distCoC[c][c2][dt])
                tabProbsRand.append(distRand[c][c2][dt])

                if False:
                    crossEntropy += distTest[c][c2][dt][1] * dist[c][c2][dt]
                    crossEntropy += distTest[c][c2][dt][0] * (1. - dist[c][c2][dt])

                    crossEntropyIC += distTest[c][c2][dt][1] * distIC[c][c2][dt]
                    crossEntropyIC += distTest[c][c2][dt][0] * (1. - distIC[c][c2][dt])

                    crossEntropyBL += distTest[c][c2][dt][1] * distBL[c][c2][dt]
                    crossEntropyBL += distTest[c][c2][dt][0] * (1. - distBL[c][c2][dt])

                    crossEntropyBLC += distTest[c][c2][dt][1] * distBLC[c][c2][dt]
                    crossEntropyBLC += distTest[c][c2][dt][0] * (1. - distBLC[c][c2][dt])

                    crossEntropyIMMSBM += distTest[c][c2][dt][1] * distIMMSBM[c][c2][dt]
                    crossEntropyIMMSBM += distTest[c][c2][dt][0] * (1. - distIMMSBM[c][c2][dt])

                    crossEntropyCoC += distTest[c][c2][dt][1] * distCoC[c][c2][dt]
                    crossEntropyCoC += distTest[c][c2][dt][0] * (1. - distCoC[c][c2][dt])

                    crossEntropyRand += distTest[c][c2][dt][1] * distRand[c][c2][dt]
                    crossEntropyRand += distTest[c][c2][dt][0] * (1. - distRand[c][c2][dt])

                if True:
                    div += 1

                    if dist[c][c2][dt] >=1: dist[c][c2][dt]=1
                    elif dist[c][c2][dt]<=0: dist[c][c2][dt]=0
                    crossEntropy += distTest[c][c2][dt][1] * np.log2(1e-10+ dist[c][c2][dt])
                    crossEntropy += distTest[c][c2][dt][0] * np.log2(1e-10+ 1 - dist[c][c2][dt])

                    if distIC[c][c2][dt] >1: distIC[c][c2][dt]=1
                    elif distIC[c][c2][dt]<0: distIC[c][c2][dt]=0
                    crossEntropyIC += distTest[c][c2][dt][1] * np.log2(1e-10+ distIC[c][c2][dt])
                    crossEntropyIC += distTest[c][c2][dt][0] * np.log2(1e-10+ 1. - distIC[c][c2][dt])
                    
                    if distHawkes[c][c2][dt] >1: distHawkes[c][c2][dt]=1
                    elif distHawkes[c][c2][dt]<0: distHawkes[c][c2][dt]=0
                    crossEntropyHawkes += distTest[c][c2][dt][1] * np.log2(1e-10+ distHawkes[c][c2][dt])
                    crossEntropyHawkes += distTest[c][c2][dt][0] * np.log2(1e-10+ 1. - distHawkes[c][c2][dt])

                    crossEntropyBL += distTest[c][c2][dt][1] * np.log2(1e-10+ distBL[c][c2][dt])
                    crossEntropyBL += distTest[c][c2][dt][0] * np.log2(1e-10+ 1. - distBL[c][c2][dt])

                    crossEntropyBLC += distTest[c][c2][dt][1] * np.log2(1e-10+ distBLC[c][c2][dt])
                    crossEntropyBLC += distTest[c][c2][dt][0] * np.log2(1e-10+ 1. - distBLC[c][c2][dt])

                    crossEntropyIMMSBM += distTest[c][c2][dt][1] * np.log2(1e-10+ distIMMSBM[c][c2][dt])
                    crossEntropyIMMSBM += distTest[c][c2][dt][0] * np.log2(1e-10+ 1. - distIMMSBM[c][c2][dt])

                    crossEntropyCoC += distTest[c][c2][dt][1] * np.log2(1e-10+ distCoC[c][c2][dt])
                    crossEntropyCoC += distTest[c][c2][dt][0] * np.log2(1e-10+ 1. - distCoC[c][c2][dt])

                    crossEntropyRand += distTest[c][c2][dt][1] * np.log2(1e-10+ distRand[c][c2][dt])
                    crossEntropyRand += distTest[c][c2][dt][0] * np.log2(1e-10+ 1. - distRand[c][c2][dt])

    if div==0: div=1e-10
    crossEntropy, crossEntropyIC, crossEntropyHawkes, crossEntropyBL, crossEntropyBLC, crossEntropyIMMSBM, crossEntropyCoC, crossEntropyRand = crossEntropy/div, crossEntropyIC/div, crossEntropyHawkes/div, crossEntropyBL/div, crossEntropyBLC/div, crossEntropyIMMSBM/div, crossEntropyCoC/div, crossEntropyRand/div
    tabFreqs, tabProbs, tabProbsIC, tabProbsHawkes, tabProbsBL, tabProbsBLC, tabProbsIMMSBM, tabProbsCoC, tabProbsRand = toNp([tabFreqs, tabProbs, tabProbsIC, tabProbsHawkes, tabProbsBL, tabProbsBLC, tabProbsIMMSBM, tabProbsCoC, tabProbsRand])


    if True:
        M = (tabFreqs + tabProbs) / 2 +1e-20
        JS = np.mean(0.5 * tabFreqs*np.log2(tabFreqs/M + 1e-20) + 0.5 * tabProbs*np.log2(tabProbs/M + 1e-20))

        M = (tabFreqs + tabProbsIC) / 2 +1e-20
        JSIC = np.mean(0.5 * tabFreqs * np.log2(tabFreqs / M + 1e-20) + 0.5 * tabProbsIC * np.log2(tabProbsIC / M + 1e-20))

        M = (tabFreqs + tabProbsHawkes) / 2 +1e-20
        JSHawkes = np.mean(0.5 * tabFreqs * np.log2(tabFreqs / M + 1e-20) + 0.5 * tabProbsHawkes * np.log2(tabProbsHawkes / M + 1e-20))

        M = (tabFreqs + tabProbsBL) / 2 +1e-20
        JSBL = np.mean(0.5 * tabFreqs * np.log2(tabFreqs / M + 1e-20) + 0.5 * tabProbsBL * np.log2(tabProbsBL / M + 1e-20))

        M = (tabFreqs + tabProbsBLC) / 2 +1e-20
        JSBLC = np.mean(0.5 * tabFreqs * np.log2(tabFreqs / M + 1e-20) + 0.5 * tabProbsBLC * np.log2(tabProbsBLC / M + 1e-20))

        M = (tabFreqs + tabProbsIMMSBM) / 2 +1e-20
        JSIMMSBM = np.mean(0.5 * tabFreqs * np.log2(tabFreqs / M + 1e-20) + 0.5 * tabProbsIMMSBM * np.log2(tabProbsIMMSBM / M + 1e-20))

        M = (tabFreqs + tabProbsCoC) / 2 +1e-20
        JSCoC = np.mean(0.5 * tabFreqs * np.log2(tabFreqs / M + 1e-20) + 0.5 * tabProbsCoC * np.log2(tabProbsCoC / M + 1e-20))

        M = (tabFreqs + tabProbsRand) / 2 +1e-20
        JSRand = np.mean(0.5 * tabFreqs * np.log2(tabFreqs / M + 1e-20) + 0.5 * tabProbsRand * np.log2(tabProbsRand / M + 1e-20))


    if False:
        brierScore = np.mean((tabFreqs-tabProbs)**2)
        brierScoreIC = np.mean((tabFreqs-tabProbsIC)**2)
        brierScoreBL = np.mean((tabFreqs-tabProbsBL)**2)
        brierScoreBLC = np.mean((tabFreqs-tabProbsBLC)**2)
        brierScoreIMMSBM = np.mean((tabFreqs-tabProbsIMMSBM)**2)
        brierScoreCoC = np.mean((tabFreqs-tabProbsCoC)**2)
        brierScoreRand = np.mean((tabFreqs-tabProbsRand)**2)

    if True:  # RSS
        brierScore = np.sum((tabFreqs-tabProbs)**2)
        brierScoreIC = np.sum((tabFreqs-tabProbsIC)**2)
        brierScoreHawkes = np.sum((tabFreqs-tabProbsHawkes)**2)
        brierScoreBL = np.sum((tabFreqs-tabProbsBL)**2)
        brierScoreBLC = np.sum((tabFreqs-tabProbsBLC)**2)
        brierScoreIMMSBM = np.sum((tabFreqs-tabProbsIMMSBM)**2)
        brierScoreCoC = np.sum((tabFreqs-tabProbsCoC)**2)
        brierScoreRand = np.sum((tabFreqs-tabProbsRand)**2)

    #print(JS, JSIC, JSBL, JSBLC, JSIMMSBM, JSRand)
    #print(brierScore, brierScoreIC, brierScoreBL, brierScoreBLC, brierScoreIMMSBM, brierScoreRand)
    #print(crossEntropy, crossEntropyIC, crossEntropyBL, crossEntropyBLC, crossEntropyIMMSBM, crossEntropyRand)

    tabJS = [JS, JSIC, JSHawkes, JSBL, JSBLC, JSIMMSBM, JSCoC, JSRand]
    tabBrierScore = [brierScore, brierScoreIC, brierScoreHawkes, brierScoreBL, brierScoreBLC, brierScoreIMMSBM, brierScoreCoC, brierScoreRand]
    tabCrossEntropy = [crossEntropy, crossEntropyIC, crossEntropyHawkes, crossEntropyBL, crossEntropyBLC, crossEntropyIMMSBM, crossEntropyCoC, crossEntropyRand]

    saveMetricsDist(nom, tabJS, tabBrierScore, tabCrossEntropy)

def getMetricsCalib(tabP, tabPIC, tabPHawkes, tabPBL, tabPBLC, tabPIMMSBM, tabPCoC, tabPRand, tabF, tabW, nom="", save=True):
    if sum(tabW)==0:
        print("========================== SUM NULL")
        return -1

    pearsonr(tabF, tabPBL)[0]
    pearsonr(tabF, tabPBLC)[0]
    pearsonr(tabF, tabPIMMSBM)[0]
    pearsonr(tabF, tabPCoC)[0]
    pearsonr(tabF, tabPRand)[0]
    pearsonr(tabF, tabP)[0]
    pearsonr(tabF, tabPIC)[0]
    pearsonr(tabF, tabPHawkes)[0]

    L1Dist, L1DistIC, L1DistHawkes, L1DistBL, L1DistBLC, L1DistIMMSBM, L1DistCoC, L1DistRand = np.average(abs(tabF - tabP), weights=tabW), np.average(abs(tabF - tabPIC), weights=tabW), np.average(abs(tabF - tabPHawkes), weights=tabW), np.average(abs(tabF - tabPBL), weights=tabW), np.average(abs(tabF - tabPBLC), weights=tabW), np.average(abs(tabF - tabPIMMSBM), weights=tabW), np.average(abs(tabF - tabPCoC), weights=tabW), np.average(abs(tabF - tabPRand), weights=tabW)
    Pears, PearsIC, PearsHawkes, PearsBL, PearsBLC, PearsIMMSBM, PearsCoC, PearsRand = pearsonr(tabF, tabP)[0], pearsonr(tabF, tabPIC)[0], pearsonr(tabF, tabPHawkes)[0], pearsonr(tabF, tabPBL)[0], pearsonr(tabF, tabPBLC)[0], pearsonr(tabF, tabPIMMSBM)[0], pearsonr(tabF, tabPCoC)[0], pearsonr(tabF, tabPRand)[0]
    tabL1 = [L1Dist, L1DistIC, L1DistHawkes, L1DistBL, L1DistBLC, L1DistIMMSBM, L1DistCoC, L1DistRand]
    tabPearson = [Pears, PearsIC, PearsHawkes, PearsBL, PearsBLC, PearsIMMSBM, PearsCoC, PearsRand]

    if save:
        saveCalib(nom, tabL1, tabPearson)

    printRes=False
    if printRes:
        print("L1 DISTANCES")
        print("Modèle", L1Dist)
        print("IC", L1DistIC)
        print("Hawkes", L1DistHawkes)
        print("BL", L1DistBL)
        print("BLC", L1DistBLC)
        print("IMMSBM", L1DistIMMSBM)
        print("CoC", L1DistCoC)
        print("Rand", L1DistRand)
        print()
        print("PEARSON CORRELATION")
        print("Modèle", Pears)
        print("IC", PearsIC)
        print("Hawkes", PearsHawkes)
        print("BL", PearsBL)
        print("BLC", PearsBLC)
        print("IMMSBM", PearsIMMSBM)
        print("CoC", PearsCoC)
        print("Rand", PearsRand)

    return L1Dist, L1DistIC, L1DistHawkes, L1DistBL, L1DistBLC, L1DistIMMSBM, L1DistCoC, L1DistRand, Pears, PearsIC, PearsHawkes, PearsBL, PearsBLC, PearsIMMSBM, PearsCoC, PearsRand

def getBetaIMMSBM(nom):
    try:
        beta = np.load(nom+"_Fit_beta_IMMSBM.npy")
    except:
        beta=None

    return beta

def results(fileName=None, typeInfs=None, nbData=None, fold=0, nbDistSample=None):
    if fileName is None:
        nbData = 10000
        fileName = "Twitter"
        typeInfs = "SocialMedia"
        nbDistSample = 3

    save = True

    outputFolder = "Output/"+fileName+"/"
    nom = outputFolder + fileName + "_" + typeInfs + "_" + str(nbData) + "_" + str(fold)

    training, test, usrToInt, beta, betaIC, betaHawkes, betaCoC, betaTrue = getData(fileName, typeInfs, nbData, fold)
    betaIMMSBM = getBetaIMMSBM(nom)

    nbCasc = len(beta)

    cntFreq = getCntFreq(training)

    #obs = {}
    #for d in (training, test): obs.update(d)
    #lgStep, meanInter = getLenInfStep(obs)
    lgStep, meanInter = 1, 1

    PRROC(training, test, beta, betaIC, betaHawkes, betaIMMSBM, betaCoC, nbDistSample, lgStep, cntFreq, nom, save=save)
    evaluation(alphaInfer = beta, alphaTrue=betaTrue, nom=nom, save=save, modele="")
    evaluation(alphaInfer = betaIC, alphaTrue=betaTrue, nom=nom, save=save, modele="IC")

    getMetricsDist(training, test, beta, betaIC, betaHawkes, betaIMMSBM, betaCoC, nbDistSample, lgStep, cntFreq, nom, save=save)

    cntSeq = getCntSeq(test, lgStep=lgStep, res=meanInter)
    tabPCalib, tabPICCalib, tabPHawkesCalib, tabPBLCalib, tabPBLCCalib, tabPIMMSBMCalib, tabPCoCCalib, tabPRandCalib, tabFCalib, tabWCalib, tabKeysCalib = getTabsCalib(training, beta, betaIC, betaHawkes, betaIMMSBM, betaCoC, nbDistSample, cntSeq, cntFreq)
    metriques = getMetricsCalib(tabPCalib, tabPICCalib, tabPHawkesCalib, tabPBLCalib, tabPBLCCalib, tabPIMMSBMCalib, tabPCoCCalib, tabPRandCalib, tabFCalib, tabWCalib, nom, save=save)

    if False:
        v = [0 for i in range(nbDistSample)]
        col=["r", "b", "y", "g", "k", "c", "m"]
        iter=0
        plt.close()
        for i in range(len(beta)):
            for j in range(len(beta)):
                for k in range(nbDistSample):
                    plt.scatter(x=[iter], y=[beta[i,j,k]], c=col[k], s=4)
                    if beta[i,j,k]>1:
                        plt.scatter(x=[iter], y=[1], marker="x", c="k", s=4)
                iter+=1
                v[np.where(beta[i,j]==max(beta[i,j]))[0][0]]+=1
        plt.ylim([-0.1,1])
        plt.savefig(nom+"_DistBeta.png", dpi=300)
        #plt.show()

    plt.plot([0, 1],[0, 1],"r-")
    plt.plot(tabFCalib, tabPCalib, "o", markersize=3, label="model")
    plt.plot(tabFCalib, tabPICCalib, "o", markersize=3, label="IC")
    plt.plot(tabFCalib, tabPHawkesCalib, "o", markersize=3, label="Hawkes")
    plt.plot(tabFCalib, tabPBLCalib, "o", markersize=3, label="BL")
    plt.plot(tabFCalib, tabPBLCCalib, "o", markersize=3, label="BLT")
    plt.plot(tabFCalib, tabPIMMSBMCalib, "o", markersize=3, label="IMMSBM")
    plt.plot(tabFCalib, tabPCoCCalib, "o", markersize=3, label="CoC")
    plt.legend()
    #plt.show()

#results(fileName="Twitter", typeInfs="News", nbData="25000", fold=0, nbDistSample=7)

