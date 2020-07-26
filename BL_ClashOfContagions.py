import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import random
matplotlib.use('TkAgg')


'''
import pprofile
profiler = pprofile.Profile()
with profiler:
    RhovsT(g, 0.1, 0.28999, 1, 1000)
profiler.print_stats()
profiler.dump_stats("Benchmark.txt")

'''

def to_integer(dt_time):
    dt_time = datetime.datetime.strptime(dt_time, "%Y-%m-%d %H:%M:%S")
    return (dt_time - datetime.datetime(1970, 1, 1)).total_seconds()

def getTimeTweet(tupleTweet):
    return tupleTweet[1]
def getContentTweet(tupleTweet):
    return tupleTweet[0]


def stopAtTime(string="1970-01-01 00:00:00"):
    if datetime.datetime.now() > datetime.datetime.strptime(string, "%Y-%m-%d %H:%M:%S"):
        print("TIME OUT")
        pause()


def getSeqUsr(u, histUsr, tweetsUsr, infToInt):
    seq = []
    indTweet = 0

    for (inf, time) in histUsr[u]:
        if time < tweetsUsr[u][indTweet][1]:
            while time < tweetsUsr[u][indTweet][1] and indTweet < len(tweetsUsr[u]) - 1:
                seq.append(("rtw", infToInt[tweetsUsr[u][indTweet][0]]))
                indTweet += 1
        seq.append(("inf", infToInt[inf]))

    return seq


def getAlphaBeta(histUsr, tweetsUsr, listInfs, propTrainingSet, K=3):
    infToInt = {}
    ind = 0
    for i in listInfs:
        infToInt[i] = ind
        ind += 1

    alpha_training = np.array([[[0 for i in range(K)] for i in range(len(listInfs))] for i in range(len(listInfs))])
    beta_training = np.array([[[0 for i in range(K)] for i in range(len(listInfs))] for i in range(len(listInfs))])
    alpha_test = np.array([[[0 for i in range(K)] for i in range(len(listInfs))] for i in range(len(listInfs))])
    beta_test = np.array([[[0 for i in range(K)] for i in range(len(listInfs))] for i in range(len(listInfs))])

    tmp=0
    with open(folder + "/Seq_Tr.txt", "a") as f:
        f.truncate(0)
    with open(folder + "/Seq_Te.txt", "a") as f:
        f.truncate(0)

    for u in histUsr:
        if u not in tweetsUsr:
            continue

        # Construction de la séquence
        seq = getSeqUsr(u, histUsr, tweetsUsr, infToInt)

        training = True
        saveTrain=False
        indEoT=len(seq)

        # Construction des observations
        ## Considère toutes les paires (en respectant causalité) dans l'intervalle entre deux tweets, et observe pour
        ## chaque paire si un tweet a été émis après l'exposition au plus vieux member, comme dans le schéma de l'article CoC
        ## Ne pas oublier : alpha_ijk est le nombre de fois où l'info i a été tweetée après que l'usr ait été exposé à i au temps t, et à j au temps t-k
        for i in range(len(seq)):  # Pour chaque entrée de la séquence entière
            if seq[i][0]=="rtw":
                if not training and not saveTrain:
                    indEoT=i
                    with open(folder+"/Seq_Tr.txt", "a") as file:
                        file.write(str(seq[:indEoT])+"\n")
                    saveTrain=True
                k=0
                packtweets=True
                listPackTweets=np.array([seq[i][1]])
                for j in range(i+1, len(seq)):  # On parcourt toutes les entrées plus vieilles (à droite)
                    if seq[j][0]=="inf":
                        if packtweets: # Si l'entrée suivante est une info, on sort du packet de tweets
                            packtweets=False
                            continue

                        if k < K:
                            k2=0
                            nbPres = len(np.where(listPackTweets == seq[j][1]))
                            for j2 in range(j+1, len(seq)):  # On fait des paires chronologiques dans l'intervalle <K, et on compare avec le set de tweets suivant
                                if k + k2 < K:
                                    if training:
                                        if seq[j][1] in listPackTweets:
                                            alpha_training[seq[j][1]][seq[j2][1]][k] += nbPres
                                        else:
                                            beta_training[seq[j][1]][seq[j2][1]][k] += 1.
                                    else:
                                        if seq[j][1] in listPackTweets:
                                            alpha_test[seq[j][1]][seq[j2][1]][k] += nbPres
                                        else:
                                            beta_test[seq[j][1]][seq[j2][1]][k] += 1.
                                else:
                                    break
                                k2+=1

                        else:
                            break
                        k+=1

                    elif seq[j][0]=="rtw" and not packtweets:
                        break
                    elif seq[j][0]=="rtw" and packtweets:
                        #listPackTweets.append(seq[j][1])
                        np.append(listPackTweets, seq[j][1])

            elif training:
                if float(i)/len(seq)>propTrainingSet:
                    training=False

        with open(folder + "/Seq_Te.txt", "a") as file:
            file.write(str(seq[indEoT:])+"\n")

    return np.array(alpha_training), np.array(beta_training), np.array(alpha_test), np.array(beta_test)


def getP(histUsr, tweetsUsr, listInfs):
    P=[0 for i in range(len(listInfs))]
    num=[0 for i in range(len(listInfs))]
    denum=[0 for i in range(len(listInfs))]
    infToInt = {}
    ind = 0
    for i in listInfs:
        infToInt[i] = ind
        ind += 1

    for u in tweetsUsr:
        for (i,t) in tweetsUsr[u]:
            num[infToInt[i]] += 1
            denum[infToInt[i]] += 1

    for u in histUsr:
        for (i,t) in histUsr[u]:
            denum[infToInt[i]]+=1

    for i in range(len(num)):
        if denum[i]==0: denum[i]=1
        P[i] = num[i]/denum[i]

    return P


def likelihood(P, M, Delta, alpha, beta, listInfs, K, T, pairs, maskPairs, withPenal=True, returnBoth=False):
    allPijk = M.dot((M.dot(Delta)))
    diagsDelta=[]
    for k in range(K):
        diagsDelta.append(np.diag(Delta[:, :, k]))
    diagsDelta=np.array(diagsDelta)
    for i in range(len(allPijk)):
        allPijk[i][i] = M[i].dot(diagsDelta.T)
        allPijk[i] = allPijk[i] + P[i]
    maskInf= (allPijk < 0).astype(int)
    allPijk = allPijk * (1 - maskInf) + maskInf * 1e-15
    maskSup = (allPijk > 1).astype(int)
    allPijk = allPijk * (1 - maskSup) + maskSup * (1-1e-15)

    LssPenal = np.sum(maskPairs * np.sum(alpha*np.log(allPijk) + beta*np.log(1 - allPijk), axis=2))
    penal = np.sum(np.sum(penaltyProb(allPijk), axis=2)*maskPairs)

    L = LssPenal-penal

    if returnBoth:
        return L, LssPenal
    elif withPenal:
        return L
    else:
        return LssPenal


def derivM(P, M, Delta, alpha, beta, listInfs, K, T, i, j):  # Note : dérivée de l'opposé de la likelihood
    nbInf = len(listInfs)
    lamb_Norm=1000
    dM = []
    rngT, rngK = range(T), range(K)

    Pi = P[i]
    Mj = M[j]
    Mi = M[i]

    for m in range(nbInf):
        dM.append([])
        Mm = M[m]
        Pm = P[m]
        alphaim, betaim = alpha[i][m], beta[i][m]
        alphamj, betamj = alpha[m][j], beta[m][j]
        sumMm = sum(abs(Mm))

        for n in rngT:
            if i!=j:
                Pmjk = Pm + Mm.dot(Mj.T.dot(Delta))
                Pimk = Pi + Mi.dot(Mm.T.dot(Delta))

                terme1 = (alphamj).dot(Mj.dot(Delta[n, :]) / Pmjk)
                terme2 = (betamj).dot(Mj.dot(Delta[n, :]) / (1. - Pmjk))
                terme3 = (alphaim).dot(Mi.dot(Delta[:, n]) / Pimk)
                terme4 = (betaim).dot(Mi.dot(Delta[:, n]) / (1. - Pimk))


                '''
                ProbInf1 = Pmjk
                penalInf1 = -Mj.dot(Delta[n, :]).dot(np.maximum(-ProbInf1, np.zeros(K))/(-ProbInf1))

                ProbInf2 = Pimk
                penalInf2 = -Mi.dot(Delta[:, n]).dot(np.maximum(-ProbInf2, np.zeros(K))/(-ProbInf2))

                ProbSup1 = ProbInf1
                penalSup1 = -Mj.dot(Delta[n, :]).dot(np.minimum(1-ProbSup1, np.zeros(K))/(1-ProbSup1))

                ProbSup2 = ProbInf2
                penalSup2 = -Mi.dot(Delta[:, n]).dot(np.minimum(1-ProbSup2, np.zeros(K))/(1-ProbSup2))

                lamb_Prob = (-terme1 + terme2 - terme3 + terme4)/(penalInf1 + penalInf2 - penalSup1 - penalSup2+1e-10) + 10  # Suppression du gradient de base + retour dans les bornes
                lamb_Prob=0
                '''


                somme = -terme1 + terme2 - terme3 + terme4 # + lamb_Prob*(penalInf1 + penalInf2 - penalSup1 - penalSup2)


            else:
                somme=0.
                for k in rngK:
                    num3 = Delta[n][n][k]
                    denum3 = 0.
                    for t in rngT:
                        denum3+=M[m][t]*Delta[t][t][k]
                    num3A = (alpha[n][n][k]) * num3
                    denum3A = P[m]+denum3
                    num3B = (beta[n][n][k]) * num3
                    denum3B = 1-P[m]-denum3

                    somme += -num3A/denum3A + num3B/denum3B

            absMmn = abs(Mm[n])
            if absMmn==0:
                absMmn=1

            dM[m].append(somme)


    return np.array(dM)


def transfoM(phi):
    mat=[]
    for i in range(len(phi)):
        lineSqr=phi[i]**2
        s=sum(lineSqr)
        mat.append(lineSqr / s)

    return np.array(mat)


def derivee_transfoM(phi):
    tM = transfoM(phi)
    mat=(1.-tM)*tM*2./phi

    return np.array(mat)


def penaltyProb(x):
    param = 75
    p = np.exp(-param*(x))+np.exp(param*(x-1))
    return p


def derivPenalProb(x):
    param=75
    d = param*(-np.exp(-param*(x))+np.exp(param*(x-1)))
    return d


def derivPhi(P, Delta, phi, alpha, beta, listInfs, K, T, i, j):
    dphi = np.zeros((len(listInfs), T))
    terme1, terme2, terme3, terme4 = 0., 0., 0., 0.
    nbInf = len(listInfs)
    tM, dtM = transfoM(phi), derivee_transfoM(phi)
    alphai, betai = alpha[i], beta[i]
    Pi =P[i]
    tMi, tMj = tM[i], tM[j]
    if i != j:
        for m in range(nbInf):
            alphamj, betamj = alpha[m][j], beta[m][j]
            alphaim, betaim = alphai[m], betai[m]
            Pm = P[m]
            tMm = tM[m]
            Pmjk = Pm + tMm.dot(tMj.T.dot(Delta)) + 1e-10
            Pimk = Pi + tMi.dot(tMm.T.dot(Delta)) + 1e-10
            derivPenalProbmj = derivPenalProb(Pmjk)
            derivPenalProbim = derivPenalProb(Pimk)
            for n in range(T):
                dTmn = dtM[m][n]

                sumDnstMj = tMj.dot(Delta[n, :])
                sumDsntMi = tMi.dot(Delta[:, n])

                terme1 = dTmn * alphamj.dot((sumDnstMj) / (Pmjk))
                terme2 = dTmn * betamj.dot((sumDnstMj) / (1 - Pmjk))
                terme3 = dTmn * alphaim.dot((sumDsntMi) / (Pimk))
                terme4 = dTmn * betaim.dot((sumDsntMi) / (1 - Pimk))

                somme = -terme1 + terme2 - terme3 + terme4 + dTmn*(sumDnstMj.dot(derivPenalProbmj) + sumDsntMi.dot(derivPenalProbim))

                dphi[m][n]=somme
    else:
        for m in range(nbInf):
            alphamm, betamm = alpha[m][m], beta[m][m]
            Pm = P[m]
            tMm = tM[m]
            for n in range(T):
                Deltann = Delta[n][n]
                dTmn = dtM[m][n]
                terme1, terme2, terme3, terme4 = 0., 0., 0., 0.
                penalTerm = 0.
                for k in range(K):
                    num  = Deltann[k]

                    Pmmk = Pm + tMm.dot(np.diag(Delta[:, :, k])) + 1e-10

                    terme1 += dTmn * num * alphamm[k] / (Pmmk)
                    terme2 += dTmn * num * betamm[k] / (1 - Pmmk)
                    penalTerm += dTmn * num * derivPenalProb(Pmmk)

                somme = -terme1 + terme2 - terme3 + terme4 + penalTerm

                dphi[m][n]=somme

    return np.array(dphi)


def derivDelta(P, M, Delta, alpha, beta, listInfs, K, T, i, j):
    dDelta = np.zeros((T, T, K))
    alphaij, betaij = alpha[i][j], beta[i][j]
    Mi, Mj = M[i], M[j]
    Pi = P[i]
    if i!=j:
        for m in range(T):
            Mim = Mi[m]
            for n in range(T):
                MimMjn = Mim * Mj[n]
                for l in range(K):
                    Pijl = Pi + Mi.dot(Delta[:, :, l].dot(Mj.T))
                    terme1 = alphaij[l] * MimMjn / (Pijl+1e-10)
                    terme2 = betaij[l] * MimMjn / (1 - Pijl+1e-10)

                    somme = -terme1 + terme2 + MimMjn*derivPenalProb(Pijl)

                    dDelta[m][n][l] = somme

    elif i==j:
        for m in range(T):
            Mim = Mi[m]
            for l in range(K):
                Piil = Pi + Mi.dot(np.diag(Delta[:, :, l]))
                terme1 = alphaij[l] * Mim / (Piil+1e-10)
                terme2 = betaij[l] * Mim / (1 - Piil+1e-10)

                somme = -terme1 + terme2 + Mim*derivPenalProb(Piil)

                dDelta[m][m][l]=somme

    return np.array(dDelta)


def optLearningRate(Mprev, Deltaprev, M, Delta, dM, dDelta, dMprev, dDeltaprev):
    sM = M - Mprev
    yM = dM - dMprev
    eta_M = np.linalg.norm(sM**2) / np.linalg.norm(sM.T.dot(yM))

    sDelta = Delta - Deltaprev
    yDelta = dDelta - dDeltaprev
    eta_Delta = np.linalg.norm(sDelta**2) / np.linalg.norm(sDelta.T.dot(yDelta))

    return eta_M, eta_Delta


def optAdaDelta(grad, s, prevUpt, t):  # Inutile pour CoC/trop lent et diverge aussi
    fac=0.95

    s=s*fac+np.multiply(grad, grad)*(1.-fac)

    t=t*fac+np.multiply(prevUpt, prevUpt)*(1.-fac)

    return t, s


def optAdagrad(grad, s):
    fac=0.95

    s = s*fac + np.multiply(grad, grad)*(1-fac)

    return 1., s


def optLineSearch(var, grad, P, M, Delta, phi, alpha, beta, listInfs, K, T, sample, maskPairsSample):
    minEta, maxEta = -6, 3
    tabEta = []
    tabL = []

    for eta in np.logspace(minEta, maxEta, 5*(maxEta-minEta)):
        if var=="M":
            newPhi = phi - eta * grad
            M_new = transfoM(newPhi)
            L = likelihood(P, M_new, Delta, alpha, beta, listInfs, K, T, sample, maskPairsSample, returnBoth=False)

        else:
            newDelta = Delta - eta * grad
            L = likelihood(P, M, newDelta, alpha, beta, listInfs, K, T, sample, maskPairsSample, returnBoth=False)

        tabL.append(L)
        tabEta.append(eta)

    if False:
        plt.close()
        plt.plot(tabEta, tabL)
        plt.semilogx()
        plt.savefig("OutputCoC/LvsEta"+var+".png")

    return tabEta[tabL.index(max(tabL))], max(tabL)


def getCoeff(var, t, s, x, xprec, gradx, eps, eta, P, M, Delta, phi, alpha, beta, listInfs, K, T, sample, maskPairsSample):
    useAdaDelta=False
    useAdagrad=False
    mixedLSAdad=False
    lineSearchAdaDelta=False

    allAtOnce=True

    if var == "M":
        if useAdaDelta:
            t, s = optAdaDelta(gradx, s, x - xprec, t)
            coeff_phi = np.sqrt((t + eps) / (s + eps))
        elif useAdagrad:
            t, s = optAdagrad(gradx, s)
            coeff_phi = eta / np.sqrt(s + eps)
        elif mixedLSAdad:
            coeff_phi_LS, LLS = optLineSearch("M", gradx, P, M, Delta, phi, alpha, beta, listInfs, K, T, sample, maskPairsSample)
            t, s = optAdaDelta(gradx, s, x - xprec, t)
            coeff_phi_Adad = np.sqrt((t + eps) / (s + eps))
            LAdad = likelihood(P, transfoM(x-gradx*coeff_phi_Adad), Delta, alpha, beta, listInfs, K, T, sample, maskPairsSample, returnBoth=False)
            if LLS > LAdad:
                coeff_phi = coeff_phi_LS
            else:
                coeff_phi = coeff_phi_Adad
        elif lineSearchAdaDelta:
            t, s = optAdaDelta(gradx, s, x - xprec, t)
            coeff_phi_Adad = np.sqrt((t + eps) / (s + eps))
            coeff_phi_LS, LLS = optLineSearch("M", gradx*coeff_phi_Adad, P, M, Delta, phi, alpha, beta, listInfs, K, T, sample, maskPairsSample)
            coeff_phi = coeff_phi_LS*coeff_phi_Adad
        elif allAtOnce:
            t, s = optAdaDelta(gradx, s, x - xprec, t)
            coeff_phi_Adad = np.sqrt((t + eps) / (s + eps))
            coeff_phi_LS_Adad, LLS_Adad = optLineSearch("M", gradx*coeff_phi_Adad, P, M, Delta, phi, alpha, beta, listInfs, K, T, sample, maskPairsSample)
            coeff_phi_LS_Simple, LLS_Simple = optLineSearch("M", gradx, P, M, Delta, phi, alpha, beta, listInfs, K, T, sample, maskPairsSample)
            LAdad = likelihood(P, transfoM(x-gradx*coeff_phi_Adad), Delta, alpha, beta, listInfs, K, T, sample, maskPairsSample, returnBoth=False)
            arrL = [LLS_Adad, LLS_Simple, LAdad]
            if LLS_Adad >= max(arrL):
                coeff_phi = coeff_phi_Adad*coeff_phi_LS_Adad
            elif LLS_Simple >= max(arrL):
                coeff_phi = coeff_phi_LS_Simple
            elif LAdad >= max(arrL):
                coeff_phi = coeff_phi_Adad
        else:
            coeff_phi = optLineSearch("M", gradx, P, M, Delta, phi, alpha, beta, listInfs, K, T, sample, maskPairsSample)

        return coeff_phi, t, s

    else:
        if useAdaDelta:
            t, s = optAdaDelta(gradx, s, x - xprec, t)
            coeff_Delta = np.sqrt((t + eps) / (s + eps))
        elif useAdagrad:
            t, s = optAdagrad(gradx, s)
            coeff_Delta = eta / np.sqrt(s + eps)
        elif mixedLSAdad:
            coeff_Delta_LS, LLS = optLineSearch("Delta", gradx, P, M, Delta, phi, alpha, beta, listInfs, K, T, sample, maskPairsSample)
            t, s = optAdaDelta(gradx, s, x - xprec, t)
            coeff_Delta_Adad = np.sqrt((t + eps) / (s + eps))
            LAdad = likelihood(P, M, x-gradx*coeff_Delta_Adad, alpha, beta, listInfs, K, T, sample, maskPairsSample, returnBoth=False)
            if LLS > LAdad:
                coeff_Delta = coeff_Delta_LS
            else:
                coeff_Delta = coeff_Delta_Adad
        elif lineSearchAdaDelta:
            t, s = optAdaDelta(gradx, s, x - xprec, t)
            coeff_Delta_Adad = np.sqrt((t + eps) / (s + eps))
            coeff_Delta_LS, LLS = optLineSearch("Delta", gradx*coeff_Delta_Adad, P, M, Delta, phi, alpha, beta, listInfs, K, T, sample, maskPairsSample)
            coeff_Delta = coeff_Delta_LS * coeff_Delta_Adad
        elif allAtOnce:
            t, s = optAdaDelta(gradx, s, x - xprec, t)
            coeff_Delta_Adad = np.sqrt((t + eps) / (s + eps))
            coeff_Delta_LS_Adad, LLS_Adad = optLineSearch("Delta", gradx*coeff_Delta_Adad, P, M, Delta, phi, alpha, beta, listInfs, K, T, sample, maskPairsSample)
            coeff_Delta_LS_Simple, LLS_Simple = optLineSearch("Delta", gradx, P, M, Delta, phi, alpha, beta, listInfs, K, T, sample, maskPairsSample)
            LAdad = likelihood(P, M, x-gradx*coeff_Delta_Adad, alpha, beta, listInfs, K, T, sample, maskPairsSample, returnBoth=False)
            arrL = [LLS_Adad, LLS_Simple, LAdad]
            if LLS_Adad >= max(arrL):
                coeff_Delta = coeff_Delta_Adad*coeff_Delta_LS_Adad
            elif LLS_Simple >= max(arrL):
                coeff_Delta = coeff_Delta_LS_Simple
            elif LAdad >= max(arrL):
                coeff_Delta = coeff_Delta_Adad
        else:
            coeff_Delta = optLineSearch("Delta", gradx, P, M, Delta, phi, alpha, beta, listInfs, K, T, sample)

        return coeff_Delta, t, s


def gradientDescent(P, M, Delta, phi, alpha, beta, listInfs, K, T, prec, folder, sample, maskPairsSample, maxCount, saveToFile):
    eta_Delta=0.001
    eta_phi=0.01

    infToInt = {}
    ind = 0
    for i in listInfs:
        infToInt[i] = ind
        ind += 1

    eps = 1e-6

    maxIter=200

    tailleMiniBatch = min([1, len(listInfs)*30/(18*len(sample))])  # Le deuxième terme est la proportion paires/info de CoC
    tailleMiniBatch = 1
    print("Taille mini-batch =", tailleMiniBatch)

    tabL, tabLssPenal, tabHOL, tabHOLssPenal, tabIter = [], [], [], [], []
    likelihood_beginning, likelihood_beginningssPenal=likelihood(P, M, Delta, alpha, beta, listInfs, K, T, sample, maskPairsSample, returnBoth=True)
    print(likelihood_beginning, likelihood_beginningssPenal)
    if saveToFile:
        f=open(folder+"/L.txt", "w+")
        f.write(str(likelihood_beginningssPenal)+"\n")
        f.close()

    prevL, L, maxL = -1e10, -1e9, -1e10
    count=0
    iter=0
    iterPrecSave=0
    iterRenewSample=0


    s_phi, t_phi = np.array([[0. for i in range(len(M[0]))] for j in range(len(M))]), np.array([[0. for i in range(len(M[0]))] for j in range(len(M))])
    s_Delta, t_Delta = np.array([[[0. for k in range(len(Delta[0][0]))] for i in range(len(Delta[0]))] for j in range(len(Delta))]), np.array([[[0. for k in range(len(Delta[0][0]))] for i in range(len(Delta[0]))] for j in range(len(Delta))])

    phi_prec, Delta_prec = phi.copy(), Delta.copy()

    while count<maxCount and iter<maxIter:
        if iterRenewSample>=10:
            np.random.shuffle(sample)
            iterRenewSample=0
            print("Shuffled")

        dDelta = np.array([[[0. for k in range(len(Delta[0][0]))] for i in range(len(Delta[0]))] for j in range(len(Delta))])
        dphi = np.array([[0. for i in range(len(M[0]))] for j in range(len(M))])

        miniBatch = sample[:int(len(sample)*tailleMiniBatch)] # Select the mini batch

        for (i,j) in miniBatch:
            dphi += derivPhi(P, Delta, phi, alpha, beta, listInfs, K, T, i, j)

        coeff_phi, t_phi, s_phi = getCoeff("M", t_phi, s_phi, phi, phi_prec, dphi, eps, eta_phi, P, M, Delta, phi, alpha, beta, listInfs, K, T, sample, maskPairsSample)

        phi -= dphi * coeff_phi
        phi_prec = phi.copy()
        M = transfoM(phi)

        for (i, j) in miniBatch:
            dDelta += derivDelta(P, M, Delta, alpha, beta, listInfs, K, T, i, j)

        coeff_Delta, t_Delta, s_Delta = getCoeff("Delta", t_Delta, s_Delta, Delta, Delta_prec, dDelta, eps, eta_Delta, P, M, Delta, phi, alpha, beta, listInfs, K, T, sample, maskPairsSample)

        Delta_prec = Delta.copy()
        Delta -= dDelta * coeff_Delta

        L, LssPenal = likelihood(P, M, Delta, alpha, beta, listInfs, K, T, sample, maskPairsSample, returnBoth=True)

        tabL.append(L)
        tabLssPenal.append(LssPenal)
        tabIter.append(iter)
        if abs((L - prevL)/L) < prec:
            count+=1
        else:
            count=0

        if maxL < L:
            maxL = L
            maxM = M.copy()
            maxDelta = Delta.copy()
            if iter - iterPrecSave >= 10 and saveToFile:
                writeToFile_SGD(folder, M, Delta, maxL)
                print("Saved")
                iterPrecSave = iter
        prevL = L

        printOffProbs=True
        if printOffProbs:
            nbIncorr=0
            Pijk=[]
            allPijk=[]
            for k in range(K):
                for y in sample:
                    (i, j)=y
                    Pij = P[i]
                    if i!=j:
                        for t in range(T):
                            for s in range(T):
                                Pij+=M[i][t]*Delta[t][s][k]*M[j][s]
                    else:
                        for t in range(T):
                            Pij+=M[i][t]*Delta[t][t][k]
                    if Pij<0 or Pij>1:
                        nbIncorr+=1
                        Pijk.append((i, j, k, Pij))
                    allPijk.append(Pij)

            print(nbIncorr, "/", len(allPijk))

        plotStats = False
        if iter % 1 == 0 and plotStats:
            delts = []
            for t in range(T):
                for s in range(T):
                    for k in range(K):
                        delts.append(Delta[t][s][k])
            Ms = []
            infModif = set()
            for (i, j) in sample:
                infModif.add(i)
                infModif.add(j)
            for i in list(infModif):
                for s in range(T):
                    Ms.append(M[i][s])

            if iter == 0:
                plt.title(str(datetime.datetime.now()))
            plt.ion()
            plt.clf()
            plt.subplot(131)
            plt.hist(delts, bins=100)
            # plt.ylim([0, 3.5])

            plt.subplot(132)
            plt.hist(allPijk, bins=100)
            # plt.ylim([0, 5.5])
            # plt.xlim([-0.2, 1.2])

            plt.subplot(133)
            plt.hist(Ms, bins=100)
            # plt.ylim([0, 5.5])
            plt.xlim([-0.05, 1.05])

            plt.pause(0.05)

        #stopAtTime("2019-11-29 10:00:00")


        print(iter, iterPrecSave, L, LssPenal) #, nbIncorr, len(pairs)*K, Pijk)
        iter+=1
        iterRenewSample += 1

    return maxM, maxDelta, maxL


def getPairs(alpha, beta):
    pairs=[]

    for i in range(len(alpha)):
        for j in range(len(alpha[i])):
            for k in range(len(alpha[i][j])):
                if alpha[i][j][k]!=0 or beta[i][j][k]!=0:
                    pairs.append((i, j))
                    break

    pairs = list(sorted(set(pairs)))

    pairs = pairs[:int(len(pairs)*1)]

    return pairs


def getMaskPairs(pairs, I):
    mask = np.zeros((I, I))
    for (i,j) in pairs:
        mask[i,j]=1
    return mask


def readMatrix(filename):
    with open(filename, 'r') as outfile:
        dims = outfile.readline().replace("# Array shape: (", "").replace(")", "").replace("\n", "").split(", ")
        for i in range(len(dims)):
            dims[i]=int(dims[i])

    new_data = np.loadtxt(filename).reshape(dims)
    return new_data


def writeToFile_SGD(folder, M, Delta, maxL):
    while True:
        try:
            writeMatrix(M, folder + "/M.txt")

            writeMatrix(Delta, folder + "/Delta.txt")

            f = open(folder + "/L.txt", "a")
            f.write(str(maxL) + "\n")
            f.close()

            break
        except:
            print("Retrying to write file")
            pass


def writeToFile_data(folder, histUsr, tweetsUsr, listInfs, P, alpha_Tr, beta_Tr, alpha_Te, beta_Te, propTrainingSet):
    while True:
        try:
            with open(folder + "/histUsr.txt", "a", encoding="utf-8") as f:
                f.truncate(0)
                for u in histUsr:
                    f.write(u + "\t")
                    premPass = True
                    for v in histUsr[u]:
                        if not premPass:
                            f.write(" ")
                        f.write(str(v[0]) + ";" + str(v[1]))
                        premPass = False
                    f.write("\n")

            with open(folder + "/tweetsUsr.txt", "a", encoding="utf-8") as f:
                f.truncate(0)
                for u in tweetsUsr:
                    f.write(u + "\t")
                    premPass = True
                    for v in tweetsUsr[u]:
                        if not premPass:
                            f.write(" ")
                        f.write(str(v[0]) + ";" + str(v[1]))
                        premPass = False
                    f.write("\n")

            txtInfs = ""
            i = 0
            for u in listInfs:
                txtInfs += str(i) + "\t" + u + "\n"
                i += 1
            f = open(folder + "/listInfs.txt", "w+")
            f.write(txtInfs)
            f.close()

            f = open(folder + "/propTrainingSet.txt", "w+")
            f.write(str(propTrainingSet))
            f.close()

            np.savetxt(folder + "/P.txt", P)

            writeMatrix(alpha_Tr, folder + "/alpha_Tr.txt")
            writeMatrix(beta_Tr, folder + "/beta_Tr.txt")

            writeMatrix(alpha_Te, folder + "/alpha_Te.txt")
            writeMatrix(beta_Te, folder + "/beta_Te.txt")

            break

        except:
            print("Retrying to write file")
            pass


def initVars(nbInf, T, K, pairs, P):
    #Delta = np.random.rand(T, T, K)
    #Delta = (Delta - 0.5) / 100

    Delta = np.zeros((T, T, K))
    Delta = (np.random.random((T,T,K))-0.5)/100

    phi = np.random.rand(nbInf, T)

    return Delta, phi


def getObs(fileName):
    nbInfs = len(np.load(fileName+"_Fit_beta.npy"))

    alphaTr, alphaTe = np.zeros((nbInfs, nbInfs, 21, 2)), np.zeros((nbInfs, nbInfs, 21, 2))
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
                    dt = int(t-t2+1)
                    if dt<=1 or t<10 or dt>20:
                        continue


                    alphaTr[c, c2, dt, s] += 1

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
                    dt = int(t - t2 + 1)
                    if dt <= 1 or t<10 or dt > 20:
                        continue

                    alphaTe[c, c2, dt, s] += 1


    return alphaTr, alphaTe, nbInfs


def writeMatrix(arr, filename):
    try:
        sparse.save_npz(filename.replace(".txt", ""), arr)
    except:
        try:
            np.save(filename, arr)
        except:
            with open(filename, 'a') as outfile:
                outfile.truncate(0)
                outfile.write('# Array shape: {0}\n'.format(arr.shape))
                for slice_2d in arr:
                    np.savetxt(outfile, slice_2d)
                    outfile.write("# New slice\n")

    # np.savetxt(filename, arr)


def writeToFile_params(folder, M, Delta, P0):
    while True:
        try:
            writeMatrix(M, folder + "_Fit_M_CoC")
            writeMatrix(Delta, folder + "_Fit_Delta_CoC")
            writeMatrix(P0, folder + "_Fit_Delta_P0")
            beta = P0[:, None, None] + M.dot(M.dot(Delta))
            np.save(folder+"_Fit_beta_CoC", beta)

            break

        except Exception as e:
            print("Retrying to write file -", e)


def runFit(folder, name, K, T, prec, maxCount, saveToFile, propTrainingSet, alpha, I, nbRuns):
    np.random.seed(111)
    random.seed(111)

    alpha_Tr, beta_Tr = alpha[:,:,:,1], alpha[:,:,:,0]
    P0_temp = alpha.sum(axis=1).sum(axis=1)
    P = P0_temp[:, 1]/P0_temp.sum(axis=1)
    listInfs = list(range(I))

    I=len(listInfs)

    pairs_sample = getPairs(alpha_Tr, beta_Tr)

    maskPairsSample = getMaskPairs(pairs_sample, I)

    nbInf = len(listInfs)

    maxM, maxDelta = initVars(nbInf, T, K, pairs_sample, P)
    maxL = -1e20
    for i in range(nbRuns):
        print("RUN", i)
        Delta, phi = initVars(nbInf, T, K, pairs_sample, P)
        M = transfoM(phi)

        L_baseline = likelihood(P, M, Delta * 0, alpha_Tr, beta_Tr, listInfs, K, T, pairs_sample, maskPairsSample, returnBoth=True)

        print("Random likelihood :", likelihood(np.random.random(len(P)), M, Delta * 0, alpha_Tr, beta_Tr, listInfs, K, T, pairs_sample, maskPairsSample, returnBoth=True))
        print("Baseline likelihood :", L_baseline)

        M, Delta, maxL_2 = gradientDescent(P, M, Delta, phi, alpha_Tr, beta_Tr, listInfs, K, T, prec, folder, pairs_sample, maskPairsSample, maxCount, saveToFile)

        if maxL_2>maxL:
            writeToFile_params(name, M, Delta, P)
            maxM, maxDelta = M, Delta
            maxL = maxL_2

    return maxM, maxDelta, P


filename="Allege_TweetsMacron2410.txt"
folder="OutputCoC/"
K=21
T=5
prec = 1e-6
maxCount=30  # Nb de fois où la likelihood varie peu de suite
saveToFile=False
propTrainingSet = 0.9
nbRuns = 100

listTypeInfsInteressants = [
                            ("Synth", "5", 20000),
                            ("Synth", "20", 20000),
                            #("Ads", "Ads2", 1e6),
                            #("PD", "All", 300000),
                            #("Twitter", "URL", 1e6),
                            ]


for folder, typeInfs, nbData in listTypeInfsInteressants:
    nbData = int(nbData)
    for fold in range(5):
        name = "Output/"+folder+"/"+folder+"_"+typeInfs+"_"+str(nbData)+"_"+str(fold)
        alphaTr, alphaTe, I = getObs(name)

        M, Delta, P = runFit(folder, name, K, T, prec, maxCount, saveToFile, propTrainingSet, alphaTr, I, nbRuns)

        writeToFile_params(name, M, Delta, P)
