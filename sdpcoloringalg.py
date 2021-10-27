import mosek
import sys
import numpy as np


def streamprinter(text):
    sys.stdout.write(text)
    sys.stdout.flush()


def mosek_check(n, m, edges, ci, cj ,cval):
    """
    mosek_check: constructs and solves a semidefinite program with costs using the mosek optimizer
    :param n: the number of vertices
    :param m: the number of edges
    :param edges: the list of edges of the graph
    :param ci: ci[r] is the column corresponding to the rth nonzero entry in the cost matrix
    :param cj: cj[r] is the row corresponding to the rth nonzero entry in the cost matrix
    :param cval: cval[r] is the value of the rth nonzero entry in the cost matrix
    :return:
        r: the rank of the primal solution
        X: the primal solution
        p: the rank of the dual solution
        S: the dual solution
    """
    with mosek.Env() as env:
        with env.Task(0, 0) as task:
            task.set_Stream(mosek.streamtype.log, None)

            task.putobjsense(mosek.objsense.minimize)

            # Angle variable
            task.appendvars(1)
            inf = 0.0
            task.putvarbound(0, mosek.boundkey.fr, -inf, +inf)
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
                syma = task.appendsparsesymmat(n, [edges[i][1]], [edges[i][0]], [1.0])
                task.putbaraij(i + n, 0, [syma], [1.0])
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
                        S[i][j + i] = xsat[q]
                        q = q + 1
                        S[j + i][i] = S[i][j + i]
                t, u = np.linalg.eig(S)
                p = np.linalg.matrix_rank(S, tol=.001)


                return r,p,X,S


def translate_input(line):
    """
    translate_input translates the format of a graph from adjacency list format to ascii
    :param line: a graph encoded via an adjacency list
    :return: the graph represented by line in ascii format
    """
    line = line[line.index(':'):]
    line = line.replace('[', ' ')
    line = line.replace(']', ',')
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
    """
    findK4 searches for a clique (K4) in the graph
    :param edges: the edges in the graph
    :param n: the number of vertices
    :return: 4 vertices composing a K4 and a True boolean if one exists, otherwise 0 and False
    """
    for i in range(n):
        for j in range(i+1, n):
            if [i,j] in edges:
                for k in range(j+1, n):
                    if [i,k] in edges and [j,k] in edges:
                        for l in range(k+1, n):
                            if [i,l] in edges and [j,l] in edges and [k,l] in edges:
                                return [i,j,k,l], True
    return 0, False


def updateColors(X,K,n):
    """
    updateColors records which vertices were successfully colored and the vertices in each color class
    :param X: the primal solution being considered
    :param K: the clique used as a baseline for colors
    :param n: the number of vertices
    :return: the set of colored vertices and an array of the 4 color classes
    """
    colored = [K[0], K[1], K[2], K[3]]
    colors = [[K[0]], [K[1]], [K[2]], [K[3]]]
    for k in range(4):  # checks to see which vertices were successfully colored and stores them with the matching K4 vertex
        for g in range(n):
            if g not in K and abs(X[g][K[k]] - 1.0) <= .001:
                colored.append(g)
                colors[k].append(g)

    return colored, colors

def checkColorAttempt(cval, X, i, j):
    """
    checkColorAttempt checks if the primal entry corresponding to the most recent cost matrix entry change is equal to 1
    :param cval: the nonzero entries in the cost matrix
    :param X: the primal solution being considered
    :param i: the vertex most recently assigned a potential color
    :param j: the vertex corresponding to i's assigned color
    :return: a boolean indicating whether the assignment was successful or not
    """
    if len(cval) > 0 and (cval[len(cval) - 1] == -1.0 and abs(X[i][j] - 1.0) > .001):
        return False
    return True

def buildCosts(heuristicToggle, colors,colored,badcolors,ci,cj,cval,l,n,r,X,K,i):
    """
    buildCosts constructs the cost matrix for the next iteration
    :param heuristicToggle: a boolean indicating which cost matrix heuristic is being used
    :param colors: an array of the 4 color classes
    :param colored: a set of all colored vertices
    :param badcolors: the set of colors that have previously been assigned to vertex i and failed
    :param ci: ci[r] is the column corresponding to the rth nonzero entry in the cost matrix
    :param cj: cj[r] is the row corresponding to the rth nonzero entry in the cost matrix
    :param cval: cval[r] is the value of the rth nonzero entry in the cost matrix
    :param l: the number of vertices that have been colored
    :param n: the total number of vertices
    :param r: the rank of the primal solution being considered
    :param X: the primal solution being considered
    :param K: the clique used as a baseline for colors
    :param i: the vertex currently considering possible colors
    :return:
        ci: ci[r] is the column corresponding to the rth nonzero entry in the updated cost matrix
        cj: cj[r] is the row corresponding to the rth nonzero entry in the updated cost matrix
        cval: cval[r] is the value of the rth nonzero entry in the updated cost matrix
        badcolors: the set of colors that have previously been assigned to vertex i and failed
        i: the vertex receiving a new color assignment
        j: the clique member corresponding to the color i is being assigned
    """
    if l != n and r > 3:
        while True:
            for k in range(4):
                if i not in colored and K[k] not in badcolors and abs(1.0 - X[i][K[k]]) >= .001 and abs(-1.0 / 3 - X[i][K[k]]) >= .001:
                    if heuristicToggle:
                        newci = []
                        newcj = []
                        newcval = []

                        for m in range(4):
                            for q in range(len(colors[m]) - 1):
                                newci.append(max(colors[m][q], colors[m][q + 1]))
                                newcj.append(min(colors[m][q], colors[m][q + 1]))
                                newcval.append(-1.0)

                        newci.append(max(colors[k][len(colors[k]) - 1], i))  # this option builds the chain of colors
                        newcj.append(min(colors[k][len(colors[k]) - 1], i))
                        newcval.append(-1.0)
                        j = K[k]
                        return newci,newcj,newcval,badcolors,i,j
                    else:
                        ci += [max(i, K[k])]  # this only changes entries involving the K4
                        cj += [min(i, K[k])]
                        cval += [-1.0]
                        j = K[k]
                        return ci,cj,cval,badcolors,i,j

            i = (i+1)%n
            badcolors = []
    return ci,cj,cval,badcolors,i,i

def read_graph(line, format_change):
    """
    read_graph reads a line representing a graph and translates it to a list of edges
    :param line: an edge list or ascii representation of a graph
    :param format_change: a boolean indicating whether line is an edge list (True) or ascii representation (False)
    :return:
        n: the number of vertices in the graph
        m: the number of edges in the graph
        edges: the list of edges in the graph
    """
    s = 'abcdefghijklmnopqrstuvwxyz'
    line.strip()
    if format_change:
        elts = translate_input(line)
        adj = elts.split(",")
        n = len(adj)
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
    return n,m,edges




def color():
    """
    color opens a file with each line corresponding to a graph and attempts the 4-coloring algorithm on each
    :return: prints out (1) the total number of graphs in the file, (2) the number of graphs that were successfully colored, and (3) the number that were skipped due to lack of a K4
    """
    noK4 = 0
    num_successes=0
    k = 0
    index = 1
    fail = 0
    format_change = True #True for files in edge code (5/6/7/8/9/10/11/12/13maxplanar.txt,problem12.txt) False for files in ascii format (10apollonian.txt,14/15maxplanar.txt,kempefails.txt)
    heuristicToggle = True #True if using heuristic 1 (-1 entries corresponding to creating a chain for each color) and False if using heuristic 2 (-1 entries only between a vertex in the K4 and one not in the K4)
    with open("11maxplanar.txt") as f:
        for line in f:
            n,m,edges = read_graph(line, format_change)

            K, hasK4 = findK4(edges, n) #identifies a K4 in the input graph

            if not hasK4:
                noK4 +=1
                index+=1
            else:
                r = n
                goodset = [K[0], K[1], K[2], K[3]] #initiates the colored vertices with the K4

                i = -1
                j = 0
                ci = [] #creates the array of x values of nonzero cost matrix entries
                cj = [] #creates the array of y values of nonzero cost matrix entries
                cval = [] #creates the array of nonzero cost matrix entries
                iter = 0
                wasbad = False
                badcolors = []
                while r > 3 and len(goodset) != n and iter<100: #loops through SDP until desired rank is achieved or 100 iterations have been attempted (enough to try all remaining colors on all remaining vertices)

                    r,p,X,S = mosek_check(n, m, edges, ci, cj, cval) #solves the SDP with the given cost matrix assignments

                    colored, colors = updateColors(X,K,n)

                    colorSuccess = checkColorAttempt(cval, X, i, j)

                    if not colorSuccess: #checks to see if the Xij value corresponding to most recent Cij value was set to 1
                        wasbad = True
                        badcolors.append(j)
                        ci = ci[:-1]
                        cj = cj[:-1]
                        cval = cval[:-1] #if entry cannot be 1, removes nonzero cost matrix entry
                    else:
                        goodset = colored

                        if i < n-1:
                            if not wasbad:
                                i += 1
                                badcolors = []
                            else: wasbad = False

                        ci,cj,cval,badcolors,i,j = buildCosts(heuristicToggle, colors,colored,badcolors, ci, cj, cval,len(goodset),n,r,X,K,i)

                    iter+=1
                if r>=4:
                    fail+=1
                if r <=3 or len(goodset) == n: #accounts for lack of precision
                    num_successes +=1
                k += 1
                index +=1

    print("The total number of graphs was %d" %(index-1))
    print("The number of graphs successfully colored was %d" %num_successes)
    print("The number of graphs not containing a K4 was %d" %noK4)


color()

