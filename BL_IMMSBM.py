import numpy as np
import random
import sparse
import datetime


np.random.seed(1111)
random.seed(1111)

# // region Manipulates the data files

def writeMatrix(arr, filename):
    try:
        sparse.save_npz(filename.replace(".txt", ""), arr)
    except:

        with open(filename, 'a') as outfile:
            outfile.truncate(0)
            outfile.write('# Array shape: {0}\n'.format(arr.shape))
            for slice_2d in arr:
                np.savetxt(outfile, slice_2d)
                outfile.write("# New slice\n")

    # np.savetxt(filename, arr)


def readMatrix(filename):
    try:
        return sparse.load_npz(filename.replace(".txt", ".npz"))
    except:
        with open(filename, 'r') as outfile:
            dims = outfile.readline().replace("# Array shape: (", "").replace(")", "").replace("\n", "").split(", ")
            for i in range(len(dims)):
                dims[i] = int(dims[i])

        new_data = np.loadtxt(filename).reshape(dims)
        return new_data

    # return sparse.csr_matrix(new_data)


def writeToFile_params(folder, theta, p, maxL, HOL, T, selfInter=False):
    while True:
        try:
            I=len(theta)
            if selfInter:
                s="Self"
            else:
                s=""

            writeMatrix(theta, folder + "_Fit_theta_IMMSBM.txt")
            writeMatrix(p, folder + "_Fit_p_IMMSBM.txt")
            beta = theta.dot(theta.dot(p))
            np.save(folder+"_Fit_beta_IMMSBM", beta)

            break

        except Exception as e:
            print("Retrying to write file -", e)


def recoverData(folder):
    folderData = "Data/"+folder+"/"
    alpha_Tr, alpha_Te = readMatrix(folderData + "/Inter_alpha_Tr.txt"), readMatrix(folderData + "/Inter_alpha_Te.txt")


    return alpha_Tr, alpha_Te


def getAlphaSelfInter(alpha):
    I = len(alpha)
    data, coords = [], [[], [], []]
    for (i,k) in zip(alpha.nonzero()[0], alpha.nonzero()[2]):
        data.append(alpha[i,i,k])
        coords[0].append(i)
        coords[1].append(i)
        coords[2].append(k)
    return sparse.COO(coords, data, shape=(I, I, I), )


def stopAtTime(string="1970-01-01 00:00:00"):
    if datetime.datetime.now() > datetime.datetime.strptime(string, "%Y-%m-%d %H:%M:%S"):
        print("TIME OUT")
        pause()


def startAtTime(string="1970-01-01 00:00:00"):
    while True:
        if datetime.datetime.now() > datetime.datetime.strptime(string, "%Y-%m-%d %H:%M:%S"):
            print("STARTED")
            break
        else:
            import time
            time.sleep(60)


# // endregion


# // region Fit tools

def likelihood(theta, p, alpha):
    I=len(alpha)

    '''
    I = len(alpha)
    for i in range(I):
        for j in range(I):
            for inf in range(I):
                temp = 0
                for k in range(T):
                    for l in range(T):
                        temp+=theta[j,l]*theta[i,k]*p[k,l,inf]

                L+=alpha[i,j,inf]*np.log(temp+1e-10)


    I = len(alpha)
    probs = theta.dot((theta.dot(p)))
    for i in range(I):
        for j in range(I):
            for inf in range(I):
                tmp=0.
                for t in range(T):
                    for s in range(T):
                        tmp+=theta[i,t]*theta[j,s]*p[t,s,inf]
                print(probs[i, j, inf] - tmp)

    '''


    coords = alpha.nonzero()
    vals=[]
    for (i,j,k) in zip(coords[0], coords[1], coords[2]):
        vals.append(theta[j].dot(theta[i].dot(p[:, :, k])))
    probs = sparse.COO(coords, vals, shape=(I,I,I))

    L = (alpha * (np.log(1e-10 + probs))).sum()

    # L = np.sum(alpha * (np.log(1e-10 + theta.dot((theta.dot(p))))))

    return L


def maximization_Theta(alpha, I, T, thetaPrev, p):
    '''
    theta2 = np.zeros((I, T))
    for m in range(I):
        nonZG, nonZD = alpha[m, :, :].nonzero(), alpha[:, m, :].nonzero()
        for n in range(T):
            tmp=0.
            for s in range(T):
                for (i, inf) in zip(nonZG[0], nonZG[1]):
                    tmp+=alpha[m, i, inf] * omega[m, i, s, n, inf]
                for (i, inf) in zip(nonZD[0], nonZD[1]):
                    tmp+=alpha[i, m, inf] * omega[i, m, n, s, inf]

            theta2[m,n] = tmp / Cm[m]
    '''

    '''  Memory consuming
    divix = (thetaPrev.dot(thetaPrev.dot(p))) + 1e-10  # mrx
    divix = np.swapaxes(divix, 0, 1)  # rmx  # Parce que alpha c'est dans l'ordre rmx

    terme1 = np.swapaxes(alpha/divix, 0, 1)  # mrx
    terme2 = np.swapaxes(thetaPrev.dot(p), 1, 2)  # rxk
    theta = np.tensordot(terme1, terme2, axes=2)  # mk


    terme1 = np.swapaxes(terme1, 0, 1)  # rmx
    terme2 = np.swapaxes(thetaPrev.dot(np.swapaxes(p, 0, 1)), 1, 2)  # mxl
    theta += np.tensordot(terme1, terme2, axes=2)  # rl

    theta = theta / Cm[:, None]
    theta *= thetaPrev
    '''

    # Combinaisons : rl, mk, klx  ;  alpha(rmx)!=alpha(mrx) car on considere ici alpha_Tr

    coords = alpha.nonzero()
    vals=[]
    for (r,m,k) in zip(coords[0], coords[1], coords[2]):
        vals.append(thetaPrev[r].dot(thetaPrev[m].dot(p[:, :, k])))  # rmx
    divix = sparse.COO(coords, np.array(vals), shape=(I,I,I))+1e-10

    Cm = (alpha.sum(axis=0).sum(axis=1) + alpha.sum(axis=1).sum(axis=1)).todense()+1e-10

    terme1 = alpha / divix  # rmx
    terme2 = np.swapaxes(thetaPrev.dot(np.swapaxes(p, 0, 1)), 1, 2)  # mxl
    theta = sparse.tensordot(terme1, terme2, axes=2)  # rl

    terme1 = terme1.transpose(axes=(1, 0, 2))  # mrx
    terme2 = np.swapaxes(thetaPrev.dot(p), 1, 2)  # rxk
    theta += sparse.tensordot(terme1, terme2, axes=2)  # mk

    theta = theta / Cm[:, None]
    theta *= thetaPrev

    return theta


def maximization_p(alpha, I, T, theta, pPrev):
    '''
    nonZ = alpha.nonzero()

    p2 = np.zeros((T, T, I))
    for m in range(T):
        for n in range(T):
            div = 0.
            for (i, j, inf) in zip(nonZ[0], nonZ[1], nonZ[2]):
                div+=alpha[i,j,inf]*omega[i,j,m,n,inf]


            for (i, j, inf) in zip(nonZ[0], nonZ[1], nonZ[2]):
                p2[m,n,inf] += alpha[i,j,inf]*omega[i,j,m,n,inf]


            p2[m,n] = p2[m,n, :]/div
    '''

    ''' Memory consuming
    divrm = (theta.dot(theta.dot(np.swapaxes(pPrev, 0, 1)))) + 1e-10  # rmx

    terme1 = np.swapaxes(alpha/divrm, 0, 2)  # xmr
    p = np.tensordot(terme1, theta, axes=1)  # xml
    p = np.swapaxes(p, 1, 2)  # xlm
    p = np.tensordot(p, theta, axes=1)  # xlk
    p = np.swapaxes(p, 0, 2)  # klx

    grandDiv = np.sum(p * pPrev, axis=2)[:, :, None] + 1e-10
    p = p * pPrev / grandDiv
    '''


    coords = alpha.nonzero()
    vals=[]
    for (r,m,k) in zip(coords[0], coords[1], coords[2]):
        vals.append(theta[r].dot(theta[m].dot(pPrev[:, :, k])))  # rmx
    divrm = sparse.COO(coords, np.array(vals), shape=(I,I,I))+1e-10

    terme1 = (alpha/divrm).transpose((2, 1, 0))  # xmr
    p = terme1.dot(theta)  # xml
    p = p.transpose((0, 2, 1))  # xlm
    p = p.dot(theta)  # xlk
    p = p.transpose((2, 1, 0))  # klx

    grandDiv = np.sum(p * pPrev, axis=2)[:, :, None] + 1e-10
    p = p * pPrev / grandDiv

    return p


def initVars(I, T):
    theta, p = np.random.rand(I, T), np.random.random((T, T, I))

    for k in range(I):
        p[:,:,k]=(p[:,:,k]+p[:,:,k].T)/2

    p = p / np.sum(p, axis=2)[:, :, None]
    theta = theta / np.sum(theta, axis=1)[:, None]

    return theta, p


def EMLoop(alpha, T, I, maxCnt, prec, alpha_Te, folder, selfInter):
    theta, p = initVars(I, T)
    maxTheta, maxP = initVars(I, T)

    prevL, L, maxL = -1e10, 0., 0.
    cnt = 0

    i = 0
    iPrev=0
    while i < 10000:
        #print(i)

        if i%10==0:
            L = likelihood(theta, p, alpha)
            #print("L =", L)

            if abs((L - prevL) / L) < prec:
                cnt += i-iPrev
                if cnt > maxCnt:
                    break
            else:
                cnt = 0

            iPrev=i

            if L > prevL:
                maxTheta, maxP = theta, p
                maxL = L
                HOL = likelihood(theta, p, alpha_Te)
                #writeToFile_params(folder, maxTheta, maxP, maxL, HOL, T, selfInter)
                #print("Saved")

            prevL = L
        thetaNew = maximization_Theta(alpha, I, T, theta, p)
        pNew = maximization_p(alpha, I, T, theta, p)
        p = pNew
        theta = thetaNew

        i += 1

    return maxTheta, maxP, maxL


# // endregion

def getObs(fileName):
    nbInfs = len(np.load(fileName+"_Fit_beta.npy"))

    alphaTr, alphaTe = np.zeros((nbInfs+2, nbInfs+2, nbInfs+2)), np.zeros((nbInfs+2, nbInfs+2, nbInfs+2))
    with open(fileName+"_Fit_training.txt", "r") as f:
        for line in f:
            tups = line.replace("\n", "").replace(")", "").replace(" ", "").replace("(", "").split("\t")[:-1]
            seqTups = []
            for t in tups:
                c, t, s = t.split(",")
                c, t, s = int(c), float(t), int(s)
                seqTups.append((c,t,s))

            seqTups = sorted(seqTups, key = lambda k:k[1])

            for (c,t,s) in seqTups:
                for (c2,t2,s2) in seqTups:
                    dt = t-t2+1
                    if dt<=1 or t<10 or dt>20:
                        continue

                    alphaTr[c, c2, nbInfs+s] += 1

    with open(fileName+"_Fit_test.txt", "r") as f:
        for line in f:
            tups = line.replace("\n", "").replace(")", "").replace(" ", "").replace("(", "").split("\t")[:-1]
            seqTups = []
            for t in tups:
                c, time, s = t.split(",")
                c, time, s = int(c), float(time), int(s)
                seqTups.append((c,time,s))

            seqTups = sorted(seqTups, key = lambda k:k[1])

            for (c, t, s) in seqTups:
                for (c2, t2, s2) in seqTups:
                    dt = t - t2 + 1
                    if dt <= 1 or t<10 or dt > 20:
                        continue

                    alphaTe[c, c2, nbInfs + s] += 1


    alphaTr, alphaTe = sparse.COO.from_numpy(alphaTr), sparse.COO.from_numpy(alphaTe)

    return alphaTr, alphaTe

def runFit(name, T, prec, maxCnt, saveToFile, nbRuns, seuil, selfInter):

    alpha_Tr, alpha_Te = getObs(name)


    I = len(alpha_Tr)

    maxL = -1e100
    for i in range(nbRuns):
        #print("RUN", i)
        theta, p, L = EMLoop(alpha_Tr, T, I, maxCnt, prec, alpha_Te, name, selfInter)
        HOL = likelihood(theta, p, alpha_Te)
        if L > maxL:
            maxL = L
            writeToFile_params(name, theta, p, L, HOL, T, selfInter)
            #print("######saved####### MAX L =", L)
        #print("=============================== END EM ==========================")


# // endregion

def fitIMMSBM(folder, typeInfs, nbData, fold, T=5):
    name = "Output/"+folder+"/"+folder+"_"+typeInfs+"_"+str(nbData)+"_"+str(fold)
    selfInter=False


    prec = 1e-4
    maxCount = 30
    saveToFile = True
    nbRuns = 10

    #print(folder)
    #print("Self inter =", selfInter)


    listT = [T]
    seuil = 0



    for T in listT:
        #print("================ %.0f ================" %T)
        runFit(name, T, prec, maxCount, saveToFile, nbRuns, seuil, selfInter)
        treatData=False

#fitIMMSBM("Synth", "3", 100, 0, T=5)