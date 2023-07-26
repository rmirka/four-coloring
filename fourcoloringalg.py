import mosek
import sys
import scipy.linalg
import numpy as np


def streamprinter(text):
    sys.stdout.write(text)
    sys.stdout.flush()


def mosek_check(n, m, k, edges, ci, cj ,cval):
    with mosek.Env() as env:
        with env.Task(0, 0) as task:
            task.set_Stream(mosek.streamtype.log, None)

            task.putobjsense(mosek.objsense.minimize)

            # Angle variable
            task.appendvars(1)
            inf = 0.0
            task.putvarbound(0, mosek.boundkey.fr, -inf, +inf)  # -inf/+inf = 0????
            # task.putcj(0, 1.0)

            # SDP variable
            mdim = [n]
            task.appendbarvars(mdim)

            task.appendcons(n + m)

            # Edge and diagonal constraints
            for i in range(n):
                syma = task.appendsparsesymmat(n, [i], [i], [1.0])
                task.putbaraij(i, 0, [syma], [1.0])
                task.putconbound(i, mosek.boundkey.fx, 1.0, 1.0)
            for i in range(m):  # where do we set vi.vj = alpha
                # task.putarow(i + n, [0], [-2.0])
                syma = task.appendsparsesymmat(n, [edges[i][1]], [edges[i][0]], [1.0])
                task.putbaraij(i + n, 0, [syma], [1.0])
                #                task.putconbound(i+n, mosek.boundkey.fx, 0.0, 0.0)
                task.putconbound(i + n, mosek.boundkey.fx, -2.0 / 3, -2.0 / 3)

            symc = task.appendsparsesymmat(n, ci, cj, cval)
            task.putbarcj(0, [symc], [1.0])

            task.optimize()
            task.solutionsummary(mosek.streamtype.msg)

            solsta = task.getsolsta(mosek.soltype.itr)

            if (solsta == mosek.solsta.optimal):
                alpha = [0.0]
                task.getxx(mosek.soltype.itr, alpha)

                xmat = [0.0 for i in range(n * (n + 1) // 2)]
                task.getbarxj(mosek.soltype.itr, 0, xmat)
                X = np.zeros((n, n))
                l = 0
                for i in range(n):
                    for j in range(n - i):
                        X[i][j + i] = xmat[l]
                        l = l + 1
                        X[j + i][i] = X[i][j + i]
                r = np.linalg.matrix_rank(X, tol=.001)

                xsat = [0.0 for i in range(n * (n + 1) // 2)]
                task.getbarsj(mosek.soltype.itr, 0, xsat)
                S = np.zeros((n, n))
                q = 0
                for i in range(n):
                    for j in range(n - i):
                        if abs(xsat[q] < .0001):
                            S[i][j + i] = xsat[q]
                            # S[i][j + i] = 0
                        else:
                            S[i][j + i] = xsat[q]
                        q = q + 1
                        S[j + i][i] = S[i][j + i]
                t, u = np.linalg.eig(S)
                #print(t)
                p = np.linalg.matrix_rank(S, tol=.001)


                return r,p,X,S


def translate_input(line):
    line = line[line.index(':'):]
    line = line.replace('[', ' ')
    line = line.replace(']', ',')
    line = line.replace('32', 'F')
    line = line.replace('31', 'E')
    line = line.replace('30', 'D')
    line = line.replace('29', 'C')
    line = line.replace('28', 'B')
    line = line.replace('27', 'A')
    line = line.replace('26', 'z')
    line = line.replace('25', 'y')
    line = line.replace('24', 'x')
    line = line.replace('23', 'w')
    line = line.replace('22', 'v')
    line = line.replace('21', 'u')
    line = line.replace('20', 't')
    line = line.replace('19', 's')
    line = line.replace('18', 'r')
    line = line.replace('17', 'q')
    line = line.replace('16', 'p')
    line = line.replace('15', 'o')
    line = line.replace('14', 'n')
    line = line.replace('13', 'm')
    line = line.replace('12', 'l')
    line = line.replace('11', 'k')
    line = line.replace('10', 'j')
    line = line.replace('9', 'i')
    line = line.replace('8', 'h')
    line = line.replace('7', 'g')
    line = line.replace('6', 'f')
    line = line.replace('5', 'e')
    line = line.replace('4', 'd')
    line = line.replace('3', 'c')
    line = line.replace('2', 'b')
    line = line.replace('1', 'a')
    line = line.replace(' ', '')
    line = line[:-2]
    return line


def findK4(edges, n):
    for i in range(n):
        for j in range(i+1, n):
            if [i,j] in edges:
                for k in range(j+1, n):
                    if [i,k] in edges and [j,k] in edges:
                        for l in range(k+1, n):
                            if [i,l] in edges and [j,l] in edges and [k,l] in edges:
                                return [i,j,k,l]
    return 0


def color_switch(goodset, colored):
    for i in range(len(goodset)):
        if goodset[i] not in colored:
            return True
    return False

def color_change(prevColors, colors):
    for i in range(4):
        for j in prevColors[i]:
            if j not in colors[i]:
                return True
    return False


def read_graph():
    noK4 = 0
    numswitch = 0
    leaving = []
    count=0
    dual_count = 0
    indices = []
    failure_indices = []
    k = 0
    index = 1
    fail = 0
    iter_avg = 0
    format_change = False
    s = 'abcdefghijklmnopqrstuvwxyzABCDEFGHI'
    with open("graph files/samplefailures.txt") as f:
        for line in f:
            line.strip()
            if format_change:
                n = 13
                elts = translate_input(line)
                adj = elts.split(",")
            else:
                elts = line.split()
                n = int(elts[0])
                adj = elts[1].split(",")

            m = 0
            edges = []
            for i in range(n): #creates the edges
                for j in range(len(adj[i])):
                    endpt = s.find(adj[i][j])
                    if endpt > i:
                        edges.append([0, 0])
                        edges[m][0] = i
                        edges[m][1] = endpt
                        m += 1
            print(index)

            K = findK4(edges, n) #identifies a K4 in the input graph
            # output.write(K)
            print(K)


            r = n
            if K == 0: #checks to see if the graph has a K4, if not we skip it
                r = 0
                p = 0
                goodset = []
                noK4 += 1
                print("No K4, skipping")
            else: goodset = [K[0], K[1], K[2], K[3]] #initiates the colored vertices with the K4

            i = -1
            j = 0
            ci = [] #creates the array of x values of nonzero cost matrix entries
            cj = [] #creates the array of y values of nonzero cost matrix entries
            cval = [] #creates the array of nonzero cost matrix entries
            iter = 0
            wasbad = False
            badcolors = []
            everSwitch = False
            failToIncrease = False
            numFail = 0
            prevColors = [[],[],[],[]]
            colorChange = False
            numColorChange = 0
            while r > 3 and len(goodset) != n and iter<100: #loops through SDP until desired rank is achieved
                good = True
                if iter > 0:
                     prevColors = [[], [], [], []]
                     for g in range(4):
                         for h in range(len(colors[g])):
                             prevColors[g].append(colors[g][h])
                r,p,X,S = mosek_check(n, m, k, edges, ci, cj, cval) #solves the SDP with the given cost matrix assignments

                colored = [K[0], K[1], K[2], K[3]]
                colors = [[K[0]],[K[1]],[K[2]],[K[3]]]
                #print(K)
                for k in range(4): #checks to see which vertices were successfully colored and stores them with the matching K4 vertex
                    for g in range(n):
                        # print("k: ", K[k])
                        # print("g: ", g)
                        # print(abs(X[g][K[k]] - 1.0))
                        # print(g not in K)
                        if g not in K and abs(X[g][K[k]] - 1.0) <= .025:
                            colored.append(g)
                            colors[k].append(g)

                # print("goodset: ", goodset)
                # print("new set: ",colored)

                dualrank = "dual rank is " + str(p) + " \n"
                primalrank = "primal rank is " + str(r) + " \n"
                output.write(dualrank)

                for ii in range(n):
                    ss = ""
                    for jj in range(n):
                        ss = ss + str(round(S[ii][jj],3)) + " "
                    ss = ss + "\n"
                    output.write(ss)

                # output.write(S)
                output.write(primalrank)

                for ii in range(n):
                    ss = ""
                    for jj in range(n):
                        ss = ss + str(round(X[ii][jj],3)) + " "
                    ss = ss + "\n"
                    output.write(ss)
                # output.write(X)
                # print(p,S)
                # print(r, X)
                # print("index: ", index)
                #print(goodset)
                iter += 1

                if len(cval) > 0 and (cval[len(cval)-1] == -1.0 and abs(X[i][j] - 1.0) > .015): #checks to see if the Xij value corresponding to most recent Cij value was set to 1
                    print("here")
                    good = False #reruns SDP after removing 'bad' cost matrix choice
                    wasbad = True
                    badcolors.append(j)
                    cval[len(cval)-1] = 0.0 #if entry cannot be 1, removes nonzero cost matrix entry

                if good and color_switch(goodset, colored): everSwitch = True #checks if a vertex that was previously assigned a color is now not

                # print("previous colors: ", prevColors)
                # print("current colors: ", colors)
                if color_change(prevColors, colors): colorChange = True
                if len(goodset) >= len(colored): failToIncrease = True
                goodset = colored

                if good and i < n-1:
                    if not wasbad:
                        i += 1
                        badcolors = []
                    else: wasbad = False


                if j == n - 1 and good:
                    i += 1
                    if(i == n): i = 0
                    j = i + 1
                elif j < n-1 and good:
                    j += 1
                if good: #this option builds the chain of colors
                    ci = []
                    cj = []
                    cval = []

                    for k in range(4):
                        for l in range(len(colors[k])-1):
                            ci.append(max(colors[k][l], colors[k][l+1]))
                            cj.append(min(colors[k][l], colors[k][l+1]))
                            cval.append(-1.0)


                #use this code to set entries of all possible colors equal to -1
                # if len(goodset) != n and r > 3 and good:
                #     stopper = True
                #     while stopper:
                #
                #         if i not in colored and stopper:
                #             possible_colors = []
                #             for k in range(4):
                #                 if abs(1.0-X[i][K[k]])>=.01 and abs(-1.0/3 -X[i][K[k]])>=.01:
                #                     possible_colors.append(K[k])
                #             for d in range(len(possible_colors)):
                #                 ci += [max(i, possible_colors[d])]
                #                 cj += [min(i, possible_colors[d])]
                #                 cval += [1.0]
                #                 for e in range(d, len(possible_colors)):
                #                     ci += [max(possible_colors[e], possible_colors[d])]
                #                     cj += [min(possible_colors[e], possible_colors[d])]
                #                     cval += [-1.0]
                #             stopper = False
                #         if stopper:
                #             if i < n-1:
                #                 i += 1
                #             else: i = 0

                # output.write("colored: ", colored)
                # output.write("ci: ", ci)
                # output.write("cj: ", cj)
                print("colored: ", colored)
                print("ci: ", ci)
                print("cj: ", cj)


                if len(goodset) != n and r > 3 and good:
                    stopper = True
                    while stopper:
                        for k in range(4):
                            if i not in colored and K[k] not in badcolors and stopper and abs(1.0-X[i][K[k]])>=.015 and abs(-1.0/3 -X[i][K[k]])>=.015:
                                # print(X[i][K[k]], [i], [K[k]])
                                ci.append(max(colors[k][len(colors[k])-1], i)) #this option builds the chain of colors
                                cj.append(min(colors[k][len(colors[k]) - 1], i))
                                cval.append(-1.0)
                                j = K[k]


                                # ci += [max(i, K[k])]  #this only changes entries involving the K4
                                # cj += [min(i, K[k])]
                                # cval += [-1.0]
                                # j = K[k]
                                stopper = False



                        if stopper:
                            if i < n-1:
                                i += 1
                            else: i = 0
                            badcolors = []

                # output.write(colors)
                # output.write(ci)
                # output.write(cj)
                # output.write(cval)
                print(colors)
                print(ci)
                print(cj)
                print(cval)
                print(X)

            if r>=4:
                fail+=1

            if r == 3 and p < n-3: #if we have low primal but not high dual, try running one more time
                colored = [K[0], K[1], K[2], K[3]]
                colors = [[K[0]], [K[1]], [K[2]], [K[3]]]
                ci = []
                cj = []
                cval = []

                for k in range(4):
                    for g in range(n):
                        if g not in K and abs(X[g][K[k]] - 1.0) <= .01:
                            ci.append(max(g, K[k]))
                            cj.append(min(g, K[k]))
                            cval.append(-1.0)

                r, p, X, S = mosek_check(n, m, k, edges, ci, cj, cval)


                    # while stopper and i < n - 1:
                    #     if abs(1.0 - X[i][j]) >= .001 and abs(-1.0 / 3 - X[i][j]) >= .001:
                    #         print(X[i][j], [i], [j])
                    #         ci += [max(i, j)]
                    #         cj += [min(i, j)]
                    #         cval += [-1.0]
                    #         stopper = False
                    #     elif j == n - 1:
                    #         i += 1
                    #         j = i + 1
                    #     else:
                    #         j += 1
            #print(r)
            print(p)
            # print(S)
            iter_avg += iter
            if r ==3 or len(goodset) == n:
                count +=1
            else:
                failure_indices.append(index)
            if p >= n-3:
                dual_count += 1
            elif p != 0:
                indices.append(index)
            k += 1
            index +=1

            if everSwitch:
                numswitch += 1
                leaving.append(index)
            if failToIncrease:
                numFail += 1
            if colorChange:
                numColorChange +=1
    print(count)
    print(dual_count)
    print(failure_indices)
    print("noK4s: ", noK4)
    print("average number of iterations: ", iter_avg/(1.0*(index-noK4)))
    print("num graphs with vertex leaving good set: ", numswitch)
    print(leaving)
    # print("num graphs with size good set decreasing: ", numFail)
    # print("num graphs where a vertex changes colors after being colored: ", numColorChange)

output = open("test.txt", "w")
read_graph()
output.close()