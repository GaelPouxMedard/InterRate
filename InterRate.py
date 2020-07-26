import GetData
import GetResults
import BL_IMMSBM
from LogS import H

import numpy as np
import cvxpy as cp
import random
import multiprocessing
import tqdm


'''
import pprofile
profiler = pprofile.Profile()
with profiler:
    RhovsT(g, 0.1, 0.28999, 1, 1000)
profiler.print_stats()
profiler.dump_stats("Benchmark.txt")
pause()

'''

np.random.seed(1111)
random.seed(1111)


def saveParams(nom, IC, beta, usrToInt, test, training, fold, Hawkes=False):
    particle=""
    if IC:
        particle = "_IC"
    elif Hawkes:
        particle = "_Hawkes"


    nom += "_"+str(fold)
    np.save(nom+"_Fit_beta"+particle, beta)

    with open(nom+"_Fit_usrToInt"+particle+".txt", "w+") as f:
        for u in usrToInt:
            f.write(str(usrToInt[u])+"\t"+str(u)+"\n")

    with open(nom+"_Fit_test"+particle+".txt", "w+") as f:
        for u in test:
            for tup in test[u]:
                f.write(str(tup))
                f.write("\t")
            f.write("\n")

    with open(nom+"_Fit_training"+particle+".txt", "w+") as f:
        for u in training:
            for tup in training[u]:
                f.write(str(tup))
                f.write("\t")
            f.write("\n")



def getObsFromFile(file):
    iter = 0
    obs = {}
    with open("Data/"+file+"_obs.txt", "r") as f:
        for line in f:
            obs[iter] = []
            tups = line.replace("\n", "").replace(")", "").replace(" ", "").replace("(", "").split("\t")[:-1]
            for t in tups:
                c, t, s = t.split(",")
                c, t, s = int(c), float(t), int(s)
                obs[iter].append((c,t,s))

            if len(obs[iter])<=1:
                print(line)
            iter += 1

    usrToInt = {}
    with open("Data/"+file+"_usrToInt.txt", "r") as f:
        for line in f:
            id, usr = line.replace("\n", "").split("\t")
            id = int(id)

            usrToInt[usr]=id

    with open("Data/" + file + "_betaTrue.npy", "r") as f:
        try:
            if f.read()=="None":
                betaTrue = None
            else:
                betaTrue = np.load("Data/" + file + "_betaTrue.npy")
        except:
            try:
                betaTrue = np.load("Data/" + file + "_betaTrue.npy")
            except:
                betaTrue = None

    return obs, usrToInt, betaTrue

def getTrTe(obs, perc=0.8, nbFolds=5):
    keys = list(obs.keys())
    random.shuffle(keys)
    lgObs = len(obs)
    allTrainings, allTests = [], []
    allTrainingsUsr, allTestUsr = [], []
    for i in range(nbFolds):
        training = {}
        test = {}
        keysTest = keys[int(i*lgObs*(1-perc)):int((i+1)*lgObs*(1-perc))]
        keysTraining = [u for u in keys if u not in keysTest]
        for u in keysTraining:
            training[u] = obs[u]
        for u in keysTest:
            test[u] = obs[u]


        setInfs = set()
        for u in training:
            for (c,t,s) in training[u]:
                setInfs.add(c)
        keysTest = list(test.keys())
        for u in keysTest:
            for (c,t,s) in test[u]:
                if c not in setInfs:
                    del test[u]
                    break

        obsUsrTr = getObsUsr(training)
        obsUsrTe = getObsUsr(test)

        for u in training:
            training[u] = sorted(training[u], key=lambda x: x[1])
        for u in test:
            test[u] = sorted(test[u], key=lambda x: x[1])

        allTrainings.append(training)
        allTests.append(test)
        allTrainingsUsr.append(obsUsrTr)
        allTestUsr.append(obsUsrTe)



    return allTrainings, allTests, allTrainingsUsr, allTestUsr

def getICData(training, test):
    for u in training:
        training[u] = sorted(training[u], key=lambda x: x[1])
        toRem = []
        for i in range(len(training[u])):
            if training[u][i][0] != training[u][-1][0]:
                toRem.append(training[u][i])
        for tup in toRem:
            training[u].remove(tup)

    setInfs = set()
    keysTraining = list(training.keys())
    for u in keysTraining:
        if len(training[u])<=1:
            del training[u]
            continue
        for (c,t,s) in training[u]:
            setInfs.add(c)

    obsUsrTr = getObsUsr(training)
    obsUsrTe = getObsUsr(test)

    return training, test, obsUsrTr, obsUsrTe

def getObsUsr(obs):
    obsUsr={}
    for c in obs:
        for (u, t, s) in obs[c]:
            try:
                obsUsr[u].append((c, t, s))
            except:
                obsUsr[u]=[(c, t, s)]

    return obsUsr

def getCascUsr(obs):
    cascUsr = {}
    for u in obs:
        for (usri, ti, s) in obs[u]:
            try:
                cascUsr[usri].add(u)
            except:
                cascUsr[usri] = set()
                cascUsr[usri].add(u)

    return cascUsr

def getNeverInteractingNodes(obsUsr):
    nonVus = {}

    for u in obsUsr:
        for v in obsUsr:
            inter = False
            for (c, t, s) in obsUsr[u]:
                for (c2, t2, s2) in obsUsr[v]:
                    if c==c2 and t>t2:
                        inter=True
                        break
                if inter:
                    break

            if not inter:
                try:
                    nonVus[u].add(v)
                except:
                    nonVus[u] = set()
                    nonVus[u].add(v)

    for u in obsUsr:
        if u not in nonVus:
            nonVus[u]=set()
        nonVus[u]=list(nonVus[u])

    return nonVus

def getLenInfStep(obs):
    l, div = 0., 0
    ltot, divtot = 0., 0.
    for u in obs:
        l+=obs[u][-1][1] - obs[u][-2][1]
        div+=1
        for i in range(len(obs[u][:-1])):
            ltot += obs[u][i+1][1] - obs[u][i][1]
            divtot += 1

        print(obs[u])

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
                    if b!=0:
                        dicTemp[c][c2][dt][1] = max([0, b*(asapb-P0)/(1-asapb+P0+1e-20)])

    if usri is not None:
        return dicTemp[usri]
    else:
        return dicTemp



def likelihood(alpha, dicUrls, T, nbDistSample, lgStep=1):
    L=0

    for c in dicUrls:
        vecCascTime=[]
        vecCascUsr=[]
        ti = -1
        for (usr, timei) in dicUrls[c]:
            vecCascTime.append(timei)
            vecCascUsr.append(usr)
            if timei==max(vecCascTime):
                ti = timei
                usri = usr


        vecCascTime = np.array(vecCascTime)
        vecCascUsr = np.array(vecCascUsr)

        usrInfTi = vecCascUsr[vecCascTime < ti]
        timeInfTi = vecCascTime[vecCascTime < ti]

        sumHinfTi, logSinfTi, logSsupTi = 0., 0., 0.
        sum2 = 0.
        for usrInfTi_i in range(len(usrInfTi)):
            sumHinfTi += np.log(1. - np.exp(logS(ti, timeInfTi[usrInfTi_i], alpha[usri, usrInfTi[usrInfTi_i]], nbDistSample)) + 1e-20)

            for usrInfTi_ii in range(len(usrInfTi)):
                if timeInfTi[usrInfTi_ii] <= timeInfTi[usrInfTi_i]:  # Parce qu'il a survécu à son propre retweet
                    sum2 += logS(timeInfTi[usrInfTi_i] + lgStep, timeInfTi[usrInfTi_ii], alpha[usrInfTi[usrInfTi_i], usrInfTi[usrInfTi_ii]], nbDistSample)

        if len(timeInfTi) != 0:
            indMax = np.where(timeInfTi == max(timeInfTi))[0][0]
            if usrInfTi[indMax] == usri:  # Parce que i n'a pas survécu à son dernier retweet
                for usrInfTi_i in range(len(usrInfTi)):
                    if timeInfTi[usrInfTi_i] <= timeInfTi[indMax]:
                        sum2 -= logS(timeInfTi[indMax] + lgStep, timeInfTi[usrInfTi_i], alpha[usrInfTi[indMax], usrInfTi[usrInfTi_i]], nbDistSample)

        L += logSinfTi
        L += (sumHinfTi)
        L += logSsupTi
        L += sum2

    return L

def likelihoodCVX(alpha, dicUrls, T, nbDistSample, lgStep=1):
    L=0

    for c in dicUrls:
        vecCascTime=[]
        vecCascUsr=[]
        ti = -1
        for (usr, timei) in dicUrls[c]:
            vecCascTime.append(timei)
            vecCascUsr.append(usr)
            if timei==max(vecCascTime):
                ti = timei
                usri = usr


        vecCascTime = np.array(vecCascTime)
        vecCascUsr = np.array(vecCascUsr)

        usrInfTi = vecCascUsr[vecCascTime < ti]
        timeInfTi = vecCascTime[vecCascTime < ti]

        sumHinfTi, logSinfTi, logSsupTi = 0., 0., 0.
        sum2 = 0.
        for usrInfTi_i in range(len(usrInfTi)):
            sumHinfTi += cp.log(1. - cp.exp(logS(ti, timeInfTi[usrInfTi_i], alpha[usri, usrInfTi[usrInfTi_i]], nbDistSample)) + 1e-20)

            for usrInfTi_ii in range(len(usrInfTi)):
                if timeInfTi[usrInfTi_ii] <= timeInfTi[usrInfTi_i]:  # Parce qu'il a survécu à son propre retweet
                    sum2 += logS(timeInfTi[usrInfTi_i] + lgStep, timeInfTi[usrInfTi_ii], alpha[usrInfTi[usrInfTi_i], usrInfTi[usrInfTi_ii]], nbDistSample)

        if len(timeInfTi) != 0:
            indMax = np.where(timeInfTi == max(timeInfTi))[0][0]
            if usrInfTi[indMax] == usri:  # Parce que i n'a pas survécu à son dernier retweet
                for usrInfTi_i in range(len(usrInfTi)):
                    if timeInfTi[usrInfTi_i] <= timeInfTi[indMax]:
                        sum2 -= logS(timeInfTi[indMax] + lgStep, timeInfTi[usrInfTi_i], alpha[usrInfTi[indMax], usrInfTi[usrInfTi_i]], nbDistSample)

        L += logSinfTi
        L += (sumHinfTi)
        L += logSsupTi
        L += sum2

    return L

def likelihoodCVXUsrIndiv(alphai, dicUrls, cascUsr, T, usri, nbDistSample, lgStep=1):
    L=0

    for u in cascUsr[usri]:
        vecCascTime=[]
        vecCascUsr=[]
        vecCascStatus=[]
        for (usr, timei, si) in dicUrls[u]:
            vecCascTime.append(timei)
            vecCascUsr.append(usr)
            vecCascStatus.append(si)

        vecCascTime = np.array(vecCascTime)
        vecCascUsr = np.array(vecCascUsr)

        for (usr, ti, si) in dicUrls[u]:
            timeInfTi = vecCascTime[vecCascTime <= ti]
            usrInfTi = vecCascUsr[vecCascTime <= ti]

            for usrInfTi_i in range(len(usrInfTi)):
                if ti+lgStep-timeInfTi[usrInfTi_i]>10:  # On considère que les infos avant dt=x n'ont plus aucune influence
                    continue

                if si == 1 and usr==usri:
                    #L += cp.log(1. - cp.exp(logS(ti+lgStep, timeInfTi[usrInfTi_i], alphai[usrInfTi[usrInfTi_i]], nbDistSample)) + 1e-20)
                    L += logF(ti + lgStep, timeInfTi[usrInfTi_i], alphai[usrInfTi[usrInfTi_i]], nbDistSample)

                if si == 0 and usr==usri:
                    if random.random() < 0.01 or True:  # Subsampling
                        L += logS(ti+lgStep, timeInfTi[usrInfTi_i], alphai[usrInfTi[usrInfTi_i]], nbDistSample)


    return L

def likelihoodFromMatrix(obs, usri, alphai, lgStep, N, nbDistSample=1):
    mat = getMatInter(obs, lgStep, usri, reduit=False)
    L=0
    for c2 in mat:
        for dt in mat[c2]:
            #L += logF(dt, 0, alphai[c2], nbDistSample) * mat[c2][dt][1]
            #L += logS(dt, 0, alphai[c2], nbDistSample) * mat[c2][dt][0]

            L += cp.log(H(dt, 0, alphai[c2], nbDistSample)) * mat[c2][dt][1]
            L += cp.log(1.-H(dt, 0, alphai[c2], nbDistSample)) * mat[c2][dt][0]

    return L

def likelihoodFromMatrix_Hawkes(obs, usri, alphai, lgStep, N, nbDistSample=2):
    mat = getMatInter(obs, lgStep, usri, reduit=False)
    L=0
    for c2 in mat:
        for dt in mat[c2]:
            #L += logF(dt, 0, alphai[c2], nbDistSample) * mat[c2][dt][1]
            #L += logS(dt, 0, alphai[c2], nbDistSample) * mat[c2][dt][0]

            L += (-alphai[c2, 0] - alphai[c2, 1] * dt) * mat[c2][dt][1]
            L += cp.log(1.-cp.exp(-alphai[c2, 0] - alphai[c2, 1] * dt)) * mat[c2][dt][0]

    return L

def fitNode(args):
    node, func, obs, cascUsr, T, N, firstGuess, nonVusi, nbDistSample, lgStep, IC, Hawkes = args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7], args[8], args[9], args[10], args[11]
    alphai = cp.Variable((N, nbDistSample))
    if Hawkes:
        func = likelihoodFromMatrix_Hawkes

    objective = cp.Maximize(func(obs, node, alphai, lgStep, N, nbDistSample))
    nonUsri = list(range(N))
    nonUsri.remove(node)

    if not IC:
        constraints = [alphai >= 0]  # , alphai[nonVusi]==0  ?
    else:
        constraints = [alphai >= 0] + [alphai[j][:] == 0 for j in range(N) if j != node]

    cont, iter=True, 0
    while cont and iter<3:
        try:
            prob = cp.Problem(objective, constraints)
            result = prob.solve(solver = cp.SCS, verbose=False, max_iters=10000)
            if alphai.value is None:
                cont=True
            else:
                cont=False
        except Exception as e:
            print("ERROR FIT", node, e, " ========================================================================")
        iter+=1

    if cont==True:
        print("Infeasible line, ", prob.status)
        alphai.value = np.zeros((N, nbDistSample))

    return alphai, node

def getAlpha(obs=None, obsUsr=None, alphaTrue=None, firstGuessAlpha=None, N=-1, nbDistSample=1, lgStep=1, IC=False, Hawkes=False):
    if __name__ == '__main__':
        if obsUsr is None:
            obsUsr = getObsUsr(obs)

        #print("Treat data - Cascade usr")
        cascUsr = getCascUsr(obs)
        #print("End treat data")

        #nonVus = getNeverInteractingNodes(obsUsr)
        nonVus = {a:{} for a in range(len(cascUsr))}
        #lgStep, meanStep = getLenInfStep(obs)
        #print("Lg last step, Lg mean step", lgStep, meanStep)

        T = 1e20

        # Likelihood test
        alpha = np.random.random((N, N, nbDistSample))
        #L = likelihood(alpha, obs, T, nbDistSample, lgStep=lgStep)
        #print("L random", L)
        if firstGuessAlpha is None:
            firstGuessAlpha = np.zeros((N, N, nbDistSample))

        # Fit
        fromMat=True
        if not fromMat:
            alphaInfer = np.zeros((N, N, nbDistSample))
            with multiprocessing.Pool(processes=6) as p:
                with tqdm.tqdm(total=N) as progress:
                    args = [(usri, likelihoodCVXUsrIndiv, obs, cascUsr, T, N, firstGuessAlpha[usri], nonVus[usri], nbDistSample, lgStep) for usri in cascUsr]
                    for i, res in enumerate(p.imap(fitNode, args)):
                        progress.update()
                        usri = res[1]
                        alphai = res[0].value
                        alphaInfer[usri] = alphai
        else:
            alphaInfer = np.zeros((N, N, nbDistSample))
            with multiprocessing.Pool(processes=6) as p:
                with tqdm.tqdm(total=N) as progress:
                    args = [(usri, likelihoodFromMatrix, obs, cascUsr, T, N, firstGuessAlpha[usri], nonVus[usri], nbDistSample, lgStep, IC, Hawkes) for usri in cascUsr]

                    for i, res in enumerate(p.imap(fitNode, args)):
                        progress.update()
                        usri = res[1]
                        alphai = res[0].value
                        alphaInfer[usri] = alphai


        alphaInfer = alphaInfer * (alphaInfer > 0).astype(int)


        #optValue = likelihood(alphaInfer, obs, T, nbDistSample, lgStep=lgStep)
        #print("Optimal value:", optValue)

        return alphaInfer, alphaTrue



def run(fileName="Retail", typeInfs="Products2", nbData=1000000):
    if __name__ == "__main__":
        nbDistSample = 20

        IR = False
        IMMSBM = False
        IC = False
        Hawkes = True
        save = True
        rewrite = False
        outputFolder = "Output/"+fileName+"/"
        nbFolds = 5

        nom = outputFolder + fileName + "_" + typeInfs + "_" + str(nbData)

        # Get observations
        try:
            if rewrite:
                obs, usrToInt, betaTrue = GetData.getObs(type=fileName, infs=typeInfs, nbData=nbData, nbDistSample=nbDistSample)
            else:
                obs, usrToInt, betaTrue = getObsFromFile(fileName + "_" + typeInfs + "_" + str(nbData))

        except Exception as e:
            print(e)
            print("Data file not found ; creating new one ("+fileName+", "+typeInfs+", N="+str(nbData)+")")
            obs, usrToInt, betaTrue = GetData.getObs(type=fileName, infs=typeInfs, nbData=nbData, nbDistSample=nbDistSample)


        lg = 0
        for o in obs:
            lg += len(obs[o])
        NCasc = len(usrToInt)
        N_Inter = len(obs)
        #lgTest, meanInter = getLenInfStep(obs)
        lgStep, meanInter = 1, 1  #  POUR COMPTER LES ECARTS DUS AUX INFOS PAS CONSIDEREES
        if fileName=="PD":
            lgStep=1.01

        printMeta = True
        print(nom)
        if printMeta:
            #print(obs)
            print("Lg moyenne =", lg / (len(obs) + 1e-20))
            print(usrToInt)
            print("Nb inter =", N_Inter)
            print("Nb casc =", NCasc)
            print("LgInf =", lgStep, "- LgMean =", meanInter)

        allTrainings, allTests, allTrainingsUsr, allTestUsr = getTrTe(obs, perc=0.8, nbFolds = nbFolds)

        for fold, (training, test, trainingUsr, testUsr) in enumerate(zip(allTrainings, allTests, allTrainingsUsr, allTestUsr)):
            print("FOLD",fold)
            if IR:
                beta, _ = getAlpha(obs=training, obsUsr=trainingUsr, nbDistSample=nbDistSample, N=NCasc, lgStep=lgStep, IC=False)
                if save:
                    saveParams(nom, False, beta, usrToInt, test, training, fold)

            if IC:
                beta, _ = getAlpha(obs=training, obsUsr=trainingUsr, nbDistSample=nbDistSample, N=NCasc, lgStep=lgStep, IC=IC)
                # Remove the random unfit terms if IC
                if IC:
                    betaDiag = np.zeros((NCasc, NCasc, nbDistSample))

                    for i in range(NCasc):
                        betaDiag[i, i, :] = beta[i, i, :]
                    beta = betaDiag

                if save:
                    saveParams(nom, IC, beta, usrToInt, test, training, fold)

            if Hawkes:
                beta, _ = getAlpha(obs=training, obsUsr=trainingUsr, nbDistSample=2, N=NCasc, lgStep=lgStep, IC=False, Hawkes=True)
                if save:
                    saveParams(nom, False, beta, usrToInt, test, training, fold, Hawkes=True)

            if IMMSBM:
                BL_IMMSBM.fitIMMSBM(fileName, typeInfs, nbData, fold, T=5)
                print("IMMSBM complete")

            GetResults.results(fileName=fileName, typeInfs=typeInfs, nbData=nbData, fold=fold, nbDistSample=nbDistSample)


def runEverything():
    listFileNames = ["Ads", "Retail"]#, "Twitter","Synth", ]
    listTypeInfsTwitter = ["News", "SocialMedia", "URL", "All"]
    listTypeInfsRetail = ["Products4", "Products5", "Products6", "Products2", "Products3", "Products1"]
    listTypeInfsSynth = ["3", "5", "10", "20"]
    listTypeAds = ["Ads2Brand", "Ads2", "Ads3Brand", "Ads4Brand", "Ads1Brand", "Ads3", "Ads1", ]
    listTypeInfsPD = ["All"]
    listNbDataTwitter = [10000, 50000, 100000, 150000, 200000, 250000]
    listNbDataRetail = [1e6, 2e6, 3e6, 4e6, 5e6, 6e6, 7e6, 8e6, 9e6, 10e6, 20e6]
    listNbDataSynth = [100, 500, 1000, 5000, 10000, 15000, 20000, 25000]
    listNbDataPD = [1000, 10000, 20000, 30000, 100000]
    listNbDataAds = [100000, 200000, 300000, 1000000]

    listTypeInfsInteressants = [#("Retail", "Products2", 50e6),
                                ("Ads", "Ads2", 1e6),
                                ("PD", "All", 300000),
                                ("Twitter", "URL", 1e6),
                                ("Synth", "3", 20000),
                                ("Synth", "5", 20000),
                                ("Synth", "10", 20000),
                                ("Synth", "20", 20000),]

    #listTypeInfsInteressants = [("Synth", "3", 20000)]

    infoInter = True
    if infoInter:
        for (fileName, typeInfs, nbData) in listTypeInfsInteressants:
            #run(fileName, typeInfs, int(nbData))
            try:
                run(fileName, typeInfs, int(nbData))
            except Exception as e:
                print("ERREUUUUUUUUUUUUUUUUUUUUUUUUUUR", e)

    else:
        for fileName in listFileNames:
            if fileName == "Twitter":
                listTypeInfs = listTypeInfsTwitter
                listNbData = listNbDataTwitter
            elif fileName == "Retail":
                listTypeInfs = listTypeInfsRetail
                listNbData = listNbDataRetail
            elif fileName == "Synth":
                listTypeInfs = listTypeInfsSynth
                listNbData = listNbDataSynth
            elif fileName == "PD":
                listTypeInfs = listTypeInfsPD
                listNbData = listNbDataPD
            elif fileName == "Ads":
                listTypeInfs = listTypeAds
                listNbData = listNbDataAds
            else:
                continue

            for typeInfs in listTypeInfs:
                for nbData in listNbData:
                    run(fileName, typeInfs, int(nbData))
                    try:
                        pass
                        #run(fileName, typeInfs, int(nbData))
                    except Exception as e:
                        print("ERREUUUUUUUUUUUUUUUUUUUUUUUUUUR", e)

#run("PD", "All", int(100000))
runEverything()