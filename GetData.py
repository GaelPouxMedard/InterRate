import numpy as np
import random
from LogS import H, HGen
import datetime
from copy import deepcopy
from scipy import sparse


def saveObs(file, obs, usrToInt, betaTrue = None):
    with open("Data/"+file+"_usrToInt.txt", "w+") as f:
        for u in usrToInt:
            f.write(str(usrToInt[u])+"\t"+str(u)+"\n")

    with open("Data/"+file+"_obs.txt", "w+") as f:
        for u in obs:
            for i, tup in enumerate(obs[u]):
                f.write(str(tup)+"\t")
            f.write("\n")

    if betaTrue is not None:
        np.save("Data/" + file + "_betaTrue", betaTrue)
    else:
        with open("Data/" + file + "_betaTrue.npy", "w+") as f:
            f.write("None")

def dateToSec(date="2019-11-06 03:51:03"):
    d = datetime.datetime.strptime(date, "%Y-%m-%dT%H:%M:%S.%fZ")
    return d.timestamp()

def shortenURL(txt):
    txt = txt.replace("http://", "").replace("https://", "")
    txt = txt.replace("www.", "").replace("https://", "")

    if "/" in txt:
        ind = txt.find("/")
        return txt[:ind+1]

    else:
        return txt

def showGraphInter(obs, num, itmToInt, intToItm):

    coocDic = {}
    for u in obs:
        obs[u] = sorted(obs[u], key = lambda x:x[1], reverse=True)
        for i1 in range(len(obs[u])):
            (obj1, t1, s1) = obs[u][i1]
            obj1_int = itmToInt[obj1]
            if obj1_int not in coocDic: coocDic[obj1_int]={}
            for i2 in range(i1, len(obs[u])):
                (obj2, t2, s2) = obs[u][i2]
                obj2_int = itmToInt[obj2]
                if obj2_int not in coocDic[obj1_int]: coocDic[obj1_int][obj2_int]=0
                if t1>t2 and obj1_int != obj2_int and t1-t2<20 and s1==1:  # osef des self inter
                    coocDic[obj1_int][obj2_int]+=1

    print("Fin build lists init sparse")
    list1, list2, data = [], [], []
    for c in coocDic:
        for c2 in coocDic[c]:
            list1.append(c)
            list2.append(c2)
            data.append(coocDic[c][c2])

    list1, list2, data = np.array(list1), np.array(list2), np.array(data)

    print(np.max(data))
    print(list1[data!=0])
    print(list2[data!=0])
    print(data[data!=0])
    data = data * (data > 2).astype(int)

    print("Indices", list1[data!=0])
    print("Indices", list2[data!=0])
    print("Indices", data[data!=0])

    print("Items", [intToItm[i] for i in list1[data!=0]])
    print("Items", [intToItm[i] for i in list2[data!=0]])
    print("Items", data[data!=0])

    listNodes = set(list1[data!=0]) | set(list2[data!=0])
    listEdges = list(zip(list1[data!=0], list2[data!=0]))
    labels = {i: intToItm[i] for i in listNodes}
    print([intToItm[i] for i in listNodes])
    print(len(listNodes))

    import networkx as nx
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans

    coords = (list1, list2)
    cooc = sparse.csr_matrix((data, coords))

    nclus = 5
    labeler = KMeans(n_clusters=nclus)
    labeler.fit(cooc)
    print(len(labeler.labels_), list(labeler.labels_))

    for i in range(nclus):
        print([intToItm[i] for i in np.where(labeler.labels_ == i)[0]])

    pause()
    G = nx.from_scipy_sparse_matrix(cooc)
    nx.draw_networkx(G, pos=nx.spring_layout(G, k=1./np.sqrt(len(listNodes))), labels=labels, nodelist=listNodes, edgelist=listEdges, node_size=20, with_labels=True)
    plt.show()



def treatData(data, alphaTrue=None):
    obs = {}
    obsUsr = {}
    usrToInt = {}
    cascToInt = {}
    intToUsr = {}
    intToCasc = {}
    numUsr = 0
    numCasc = 0
    indTranspose = []

    for casc in data:
        for (u,t,s) in data[casc]:
            if u not in usrToInt:
                usrToInt[u] = numUsr
                intToUsr[numUsr]=u
                indTranspose.append(u)
                numUsr+=1

            if casc not in cascToInt:
                cascToInt[casc] = numCasc
                intToCasc[numCasc] = casc
                numCasc+=1

            try:
                obs[cascToInt[casc]].append((usrToInt[u],t,s))
            except:
                obs[cascToInt[casc]] = [(usrToInt[u],t,s)]

            try:
                obsUsr[usrToInt[u]].append((cascToInt[casc],t,s))
            except:
                obsUsr[usrToInt[u]] = [(cascToInt[casc],t,s)]

    if alphaTrue is not None:
        alphaTrue = alphaTrue[indTranspose, :, :]
        alphaTrue = alphaTrue[:, indTranspose, :]


    #showGraphInter(obs, len(usrToInt), usrToInt, intToUsr)
    #pause()

    return obs, obsUsr, alphaTrue, usrToInt

def removeUsers(obs, obsUsr, seuil=0):
    usrToRem = set()
    for u in obsUsr:
        if len(obsUsr[u])<=seuil:
            usrToRem.add(u)
    for u in obs:
        indToRem=[]
        for i in range(len(obs[u])):
            if obs[u][i][0] in usrToRem:
                indToRem.append(i)
        for t in sorted(indToRem, reverse=True):
            del obs[u][t]
    for u in list(obs.keys()):
        if len(obs[u])==0:
            del obs[u]

    return obs

def getObsUsr(obs):
    obsUsr={}
    for c in obs:
        for (u, t, s) in obs[c]:
            try:
                obsUsr[u].append((c, t, s))
            except:
                obsUsr[u]=[(c, t, s)]

    return obsUsr

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

                if dt<=1 or dt>20 or t<10:
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



def getObsTwitter(nbData=10000, listInfos=None):
    twUsr = {}
    histUsr = {}
    obs = {}
    cntUrls = {}
    lim = nbData
    with open("Data/Twitter/histUsrSsAds.txt") as f:
        for i, l in enumerate(f):
            if i%100==0:
                print(i*100./nbData)
            idUsr, urls = l.replace("\n", "").split("\t")
            idUsr = int(idUsr)
            urls = urls[:-1].split(";")  # Pour virer le dernier ";"
            tabTuples = []
            if urls == ['']:
                continue
            for u in urls:
                try:
                    t, url = u.split(" ")
                    url = shortenURL(url)
                    t=int(t)
                except:
                    print("ERREUR")
                    continue

                txtCnt = url.replace("http://", "").replace("https://", "")
                try:
                    ind = txtCnt.index("/")
                except:
                    ind=-1
                txtCnt = txtCnt[:ind]
                try:
                    cntUrls[txtCnt] += 1
                except:
                    cntUrls[txtCnt] = 0

                tabTuples.append((url.lower(), t))

            histUsr[idUsr] = sorted(tabTuples, key = lambda kv:(kv[1]))

            if i>lim:
                break

    iterObs = 0
    with open("Data/Twitter/twUsrSsAds.txt") as f:
        iterTweet=0
        for l in f:
            if iterTweet%100==0:
                print(iterTweet*100./len(histUsr), "%")
            idUsr, urls = l.replace("\n", "").split("\t")
            idUsr = int(idUsr)
            if idUsr not in histUsr.keys():
                continue
            iterTweet+=1
            urls = urls[:-1].split(";")
            tabTuples = []
            for u in urls:
                t, url = u.split(" ")
                url = shortenURL(url)
                t = int(t)
                tabTuples.append((url.lower(), t))

            if len(tabTuples)==0:
                continue
            twUsr[idUsr]=sorted(tabTuples, key = lambda kv:(kv[1]), reverse=True)
            histUsr[idUsr]=sorted(histUsr[idUsr], key = lambda kv:(kv[1]), reverse=True)
            obsTemp = []
            for (c, t) in histUsr[idUsr]:
                obsTemp.append((c, t, 0))
            for (c, t) in twUsr[idUsr]:
                time=-1
                for (c2, t2) in histUsr[idUsr]:
                    if t2 <= t and c2 == c:
                        time = t2
                        break
                if time != -1:
                    try:
                        obsTemp.remove((c, time, 0))
                        obsTemp.append((c, time, 1))
                    except:
                        pass


            obsTemp = sorted(obsTemp, key = lambda kv:(kv[1]), reverse=False)

            order = True
            obsTemp2, toRem, toKeep = [], [], []
            if order:
                for i, (c,t,s) in enumerate(obsTemp):
                    obsTemp2.append((c,i,s))
                obsTemp = obsTemp2
            presRT = False
            for i, (c, t, s) in enumerate(obsTemp):
                pres = False
                for inf in listInfos:
                    if inf in c:
                        pres = True
                        toKeep.append((inf, t, s))
                        if s == 1:
                            presRT = True
                        break
                if not pres:
                    toRem.append((c,t,s))
            if not presRT:
                continue
            for r in toRem:
                obsTemp.remove(r)
            obsTemp = toKeep


            consOrdreAcEcarts = True
            if order:
                if not consOrdreAcEcarts:
                    for i, (c,t,s) in enumerate(obsTemp):
                        obsTemp2.append((c,i,s))
                    obsTemp = obsTemp2


            obs[iterObs] = obsTemp
            if len(obs[iterObs]) <= 1:
                del obs[iterObs]
            iterObs += 1



            if iterObs > lim:
                break

    return obs

def getObsSynth(betaTrue=None, nbInter=500, nbCasc=4, nbDistSample=1):
    window=50

    nbDistSample = 20

    if betaTrue is None:
        np.random.seed(1111)
        random.seed(1111)
        beta = np.random.random((nbCasc, nbCasc, nbDistSample))
        for i in range(nbCasc):
            for j in range(nbCasc):
                a = np.zeros((nbDistSample))
                a[np.random.randint(0, nbDistSample, 1)[0]]+=1
                beta[i,j] *= a
                #beta[i,j] = (beta[i,j] + 1)/3

        beta /= 1
        beta[1,0] *= 0
        beta[1,0,3] = 0.7
        beta[1,0,12] = 0.1

    else:
        beta = betaTrue

    obs = {}
    step = 1

    for k in range(nbInter):
        obs[k]=[]

        clock = 0
        hist = []
        for j in range(window):
            inf = np.random.randint(0, nbCasc, 1)[0]
            hist.append((inf, round(clock, 10), 0))

            tc = clock+1
            clock += random.randint(1,10)

            S = 1.
            for (c2, tc2, s2) in hist:
                if clock>10:
                    S *= max([(1. - HGen(tc, tc2, beta[inf, c2], nbDistSample)), 0])

            arrProb = np.array([(1 - S), S])
            c = np.random.choice([inf, -1], 1, p=arrProb)[0]

            if c==-1:
                continue
            else:
                (c,t,s)=hist[-1]
                hist[-1]=(c,t,1)

        obs[k]=deepcopy(hist)

    if False:
        import matplotlib.pyplot as plt
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['font.family'] = 'Calibri'
        mat = getMatInter(obs, reduit=False)
        cntFreq = getCntFreq(obs)
        for c in mat:
            for c2 in mat[c]:
                if c!=1:
                    continue
                s = 0
                for dt in mat[c][c2]:
                    s += sum(mat[c][c2][dt])
                    r = mat[c][c2][dt][1] / (sum(mat[c][c2][dt]) + 1e-20)
                    if dt==4:
                        plt.bar(dt, r, width=.5, alpha=1, color="orange", label = "Observed data")
                    else:
                        plt.bar(dt, r, width=.5, alpha=1, color="orange")

                a = np.linspace(0,20, 1000)
                plt.plot(a, HGen(a, 0, beta[c,c2], nbDistSample=nbDistSample), "r", label="Underlying generation process")
                plt.plot(a, [cntFreq[c] for a_i in a], "--y", label=r"$P_0$")
                print(c,c2,s)
                print(beta[c])
                print(c, c2, beta[c,c2])
                plt.ylim([0, 1])
                plt.xlabel(r"$\Delta t$", fontsize=18)
                plt.ylabel("Density", fontsize=18)
                plt.legend()
                #plt.show()
                plt.tight_layout()
                plt.savefig("Misc/N=%.0f_CompDistTrueAndObs_%.0f-%.0f.pdf" %(nbInter, c, c2), dpi=600)
                plt.close()

        pause()


    return obs, beta

def getObsRetail(nbData=10000, listInfos=None):
    seqs = {}
    itmToInt = {}
    intToItm = {}
    with open("Data/Yoochoose/yoochoose-clicks.dat", "r") as f:
        for i, l in enumerate(f):
            if i%10000==0:
                print(i*100./nbData)
            sessionID, time, itemID, cat = l.replace("\n", "").split(",")
            itemID = int(itemID)
            time = dateToSec(time)
            sessionID = int(sessionID)
            try:
                seqs[sessionID].append((itemID, time, 0))
            except:
                seqs[sessionID]=[(itemID, time, 0)]

            if i > nbData:
                break

    with open("Data/Yoochoose/yoochoose-buys.dat", "r") as f:
        for l in f:
            sessionID, time, itemID, price, qty = l.replace("\n", "").split(",")
            itemID = int(itemID)
            time = dateToSec(time)
            sessionID = int(sessionID)

            if sessionID not in seqs:
                continue

            seqs[sessionID].append((itemID, time, 1))

    k = list(seqs.keys())
    for i, u in enumerate(k):
        bought=False
        for (_,_,type) in seqs[u]:
            if type == 1:
                bought=True
                break
        if not bought:
            del seqs[u]

    print("Data collected")

    obs = {}
    num = 0
    order = True
    for iter, u in enumerate(seqs):
        obs[u] = sorted(seqs[u], key=lambda kv: (kv[1]), reverse=True)

        '''
        toRem = set()
        for i1 in range(len(obs[u])):
            (c, t, s) = obs[u][i1]
            if s==0:
                continue


            for i2 in range(i1, len(obs[u])):
                (c2, t2, s2) = obs[u][i2]
                if s2==1:
                    continue

                if c==c2 and t2<t:
                    toRem.add((c,t,s))
                    toRem.add((c2,t2,s2))

                    obs[u].append((c,t2,s))
                    break
        for r in toRem:
            try:
                obs[u].remove(r)
            except Exception as e:
                print(e)
        '''

        obs[u] = sorted(obs[u], key=lambda kv: (kv[1]), reverse=False)
        if order:
            iter, tprec = 0, 0
            for i in range(len(obs[u])):
                (c,t,type) = obs[u][i]
                if not t-tprec < 0.5:
                    iter += 1

                obs[u][i]=(c,iter,type)  # HERE CHOOSE WHETHER TO CONSIDER ABSOLUTE TIMES


                tprec = t

        toRem = []
        for i in range(len(obs[u])):
            (c, t, type) = obs[u][i]
            if listInfos is None or c in listInfos:
                obs[u][i]=(c,t,type)

                if c not in itmToInt:
                    itmToInt[c] = num
                    intToItm[num] = c
                    num += 1
            else:
                toRem.append((c,t,type))

        for r in toRem:
            obs[u].remove(r)

        if len(obs[u])<=1:
            del obs[u]
            continue


        consOrdreAcEcarts = True
        if not consOrdreAcEcarts and order:
            obs[u] = sorted(obs[u], key=lambda kv: (kv[1]), reverse=False)
            obsTemp = []

            time = 0
            for (c2, t2, s2) in reversed(obs[u]):
                obsTemp.append((c2, time, s2))
                time += 1

            obs[u] = obsTemp

        obs[u] = sorted(obs[u], key=lambda kv: (kv[1]), reverse=False)


    if False:
        import matplotlib.pyplot as plt
        mat = getMatInter(obs, reduit=False)
        cntFreq = getCntFreq(obs)
        for c in mat:
            for c2 in mat[c]:
                s = 0
                for dt in mat[c][c2]:
                    s += sum(mat[c][c2][dt])
                    r = mat[c][c2][dt][1] / (sum(mat[c][c2][dt]) + 1e-20)
                    plt.bar(dt, r, width=.5, alpha=1, color="b")

                print(c,c2,s)
                plt.ylim([0, 1])
                plt.legend()
                plt.show()

    return obs

def getObsPD(nbData):
    obs = {}
    precRnd = -1
    with open("Data/Prisonner/all_data.csv", "r") as f:
        f.readline()
        iterInter=0
        for i, line in enumerate(f):
            if i>nbData:
                break
            tabLine = line.replace("\n", "").split(",")
            round=int(tabLine[1])
            decision=tabLine[2]
            r1, r2 = float(tabLine[5]), float(tabLine[6])
            risk = int(tabLine[3])
            precDec, precDecAdv = tabLine[9], tabLine[18]
            if precDec=="NA" or precDecAdv=="NA" or (r1, r2) != (0.18, 0.59) or risk==1:
                continue

            if decision.replace("\"", "")=="coop":
                decision=1
            else:
                decision=0

            if round<=precRnd:
                iterInter+=1
                precRnd = -1
            else:
                precRnd=round

            situation = str(precDec) + "-" + str(precDecAdv)
            elemSeq = (situation, (round+5), decision)
            # Le +5 c'est pcq le fit évite les rounds avant 5 (manque d'historique : fausse les données)
            # Le 1.01 c'est pour considérer dt=1 dans ce cas

            if iterInter not in obs: obs[iterInter]=[]
            obs[iterInter].append(elemSeq)

    print(obs)
    return obs

def getObsAds(nbData, typeInfs, listInfos=None):
    sizeFile = 26600000
    obs = {}
    adToInt, intToAd, num = {}, {}, 0

    '''
    adToCat = {}
    with open("Data/Ads/ad_feature.csv", "r") as f:
        f.readline()
        for i, l in enumerate(f):
            adgroup_id, cate_id, campaign_id, customer, brand, price = l.replace("\n", "").split(",")
            if adgroup_id not in adToCat:
                adToCat[adgroup_id] = brand #cate_id

    with open("Data/Ads/raw_sample.csv", "r") as f:
        f.readline()
        for i, line in enumerate(f):
            if i%100000==0:
                pass
                print(100 * i / 26600000, "%")
            usr, time_stamp, adgroup_id, pid, nonclk, clk = line.replace("\n", "").split(",")

            time_stamp = int(time_stamp)
            clk = int(clk)
            if usr not in obs: obs[usr] = []

            adCat = adToCat[adgroup_id]
            obs[usr].append((adCat, time_stamp, clk))

            if adCat not in adToInt:
                adToInt[adCat] = num
                intToAd[num] = adCat
                num+=1

    with open("Data/Ads/treated_sample_brands.csv", "w+") as f:
        for u in obs:
            f.write(u+"\t")
            for (c,t,s) in obs[u]:
                f.write(str((c,t,s))+" - ")
            f.write("\n")
    pause()
    '''


    if "NoGpe" in typeInfs:
        fname = "Data/Ads/treated_sample_noGroups.csv"
    elif "Brand" in typeInfs:
        fname = "Data/Ads/treated_sample_brands.csv"
    else:
        fname = "Data/Ads/treated_sample.csv"
    with open(fname, "r") as f:
        for i, line in enumerate(f):
            u, tabTupsStr = line.replace("\n", "").split("\t")
            u = int(u)
            tabTups = tabTupsStr.replace("(", "").replace(")", "").split(" - ")[:-1]
            if u not in obs: obs[u] = []

            for tupStr in tabTups:
                tup = tupStr.replace("'", "").split(", ")
                if tup[0] == "NULL":
                    tup[0] = -1
                (c,t,s) = (int(tup[0]), int(tup[1]), int(tup[2]))

                if c not in adToInt:
                    adToInt[c] = num
                    intToAd[num] = c
                    num += 1

                obs[u].append((c,t,s))


            if i>nbData:
                pass
                break



    k = list(obs.keys())
    for numU, u in enumerate(k):
        obs[u] = sorted(obs[u], key=lambda x:x[1], reverse = False)


        for i in range(len(obs[u])):
            (c,t,s) = obs[u][i]
            obs[u][i] = (c,i+5,s)  # Pour éviter de n'avoir aucuen donnée car les séquences font plutôt moins de 10 steps

        toRem=[]
        for (c,t,s) in obs[u]:
            if listInfos is not None:
                if c not in listInfos:
                    toRem.append((c, t, s))
            elif c == -1:  # Symbole pour non-défini/null
                toRem.append((c, t, s))

        for r in toRem:
            obs[u].remove(r)

        bought = False
        for (c,t,s) in obs[u]:
            if s==1:
                bought=True
                break

        if not bought or len(obs[u])<=1:
            del obs[u]
            continue

    #showGraphInter(obs, num, adToInt, intToAd)

    return obs


def getObs(type="Twitter", infs="SocialMedia", nbData=10000, nbDistSample=3):
    obs = None
    betaTrue = None

    listInfosNews = ["newzfor.me", "on.cnn", "nyti", "bbc", "huff", "mtv.co.uk"]
    listInfosURL = ["migre.me", "bit.ly", "tinyurl", "t.co/"]
    listInfosSocialMedia = ["t.co/", "tumblr", "mysp", "facebook", "instagram", "quora", "flickr", "linkedin", "deviantArt", "youtube.com/"]


    listProducts1 = [214716928,214717007,214717003,214717005,214716926,214716982]
    listProducts2 = [214827005,214827000,214827007,214826925,214834865]
    listProducts3 = [214826702,214821292,214821277,214821285,214821290,214826803]
    listProducts4 = [214826606,214826835,214826955,214826715,214826627]
    listProducts5 = [214753505,214753513,214748291,214748304,214748295,214753507,214748297,214748300,214748293,214748338]
    listProducts6 = [214839997,214829313,214839999,214839995,214840001]

    listAds1 = [6261, 4282, 4284, 6426, 4281, 1665, 4280, 6423, 4520, 4521, 4283, 4292]
    listAds2 = [1665, 4280, 4520, 4282]
    listAds3 = [6736, 562, 6421, 6423, 6300, 4521, 4283, 4292, 4284, 6426]
    listAds1Brand = [184190, 188059, 412482, 364403]
    listAds2Brand = [353787, 220468, 146115, 82527, 234846]
    listAds3Brand = [247789, 234846, 82527, 146115]
    listAds4Brand = [425589, 220468, 264740]

    listAll = ["a", "e", "i", "o", "u", "y", "h"]

    if type=="Twitter":
        if infs == "SocialMedia":
            listInfos = listInfosSocialMedia
        elif infs == "News":
            listInfos = listInfosNews
        elif infs == "URL":
            listInfos = listInfosURL
        elif infs == "All":
            listInfos = listAll
        else:
            print("CHOISIR LIST INFS")
            return -1

        obs = getObsTwitter(nbData=nbData, listInfos=listInfos)

    elif type == "Retail":
        if infs == "Products1":
            listInfos = listProducts1
        elif infs == "Products2":
            listInfos = listProducts2
        elif infs == "Products3":
            listInfos = listProducts3
        elif infs == "Products4":
            listInfos = listProducts4
        elif infs == "Products5":
            listInfos = listProducts5
        elif infs == "Products6":
            listInfos = listProducts6
        elif infs == "All":
            listInfos = None #Code pour dire de tout prendre
        else:
            print("CHOISIR LIST INFS")
            return -1

        obs = getObsRetail(nbData=nbData, listInfos=listInfos)

    elif type == "Synth":
        pass
        obs, betaTrue = getObsSynth(nbInter=nbData, nbCasc=int(infs), nbDistSample=nbDistSample)

    elif type == "PD":
        pass
        obs = getObsPD(nbData=nbData)

    elif type == "Ads":
        if infs == "Ads1":
            listInfos = listAds1
        elif infs == "Ads2":
            listInfos = listAds2
        elif infs == "Ads3":
            listInfos = listAds3
        elif infs == "Ads1Brand":
            listInfos = listAds1Brand
        elif infs == "Ads2Brand":
            listInfos = listAds2Brand
        elif infs == "Ads3Brand":
            listInfos = listAds3Brand
        elif infs == "Ads4Brand":
            listInfos = listAds4Brand
        elif "All" in infs:
            listInfos = None  # Code pour dire de tout prendre
        else:
            print("CHOISIR LIST INFS")
            return -1


        obs = getObsAds(nbData=nbData, typeInfs=infs, listInfos = listInfos)

    obsUsr = getObsUsr(obs)
    obs = removeUsers(obs, obsUsr, seuil=0)
    obs, obsUsr, betaTrue, usrToInt = treatData(obs, alphaTrue=betaTrue)

    saveObs(type+"_"+infs+"_"+str(nbData), obs, usrToInt, betaTrue)

    return obs, usrToInt, betaTrue


#getObs(type="Retail", infs="Products3", nbData=int(100e6), nbDistSample=1)

