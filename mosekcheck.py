import mosek
import sys
import scipy.linalg
import numpy as np

def streamprinter(text):
    sys.stdout.write(text)
    sys.stdout.flush()


def mosek_check(n, m, k, edges):
    with mosek.Env() as env:
        with env.Task(0, 0) as task:
            task.set_Stream(mosek.streamtype.log, streamprinter)

            task.putobjsense(mosek.objsense.minimize)

            # Angle variable
            task.appendvars(1)
            inf = 0.0
            task.putvarbound(0, mosek.boundkey.fr, -inf, +inf)  #-inf/+inf = 0????
            #task.putcj(0, 1.0)

            # SDP variable
            mdim = [n]
            task.appendbarvars(mdim)

            task.appendcons(n + m)

            # Edge and diagonal constraints
            for i in range(n):
                syma = task.appendsparsesymmat(n, [i], [i], [1.0])
                task.putbaraij(i, 0, [syma], [1.0])
                task.putconbound(i, mosek.boundkey.fx, 1.0, 1.0)
            for i in range(m):   #where do we set vi.vj = alpha
                #task.putarow(i + n, [0], [-2.0])
                syma = task.appendsparsesymmat(n, [edges[i][1]], [edges[i][0]], [1.0])
                task.putbaraij(i + n, 0, [syma], [1.0])
                #task.putconbound(i+n, mosek.boundkey.fx, 0.0, 0.0)
                task.putconbound(i + n, mosek.boundkey.fx, -2.0/3, -2.0/3)

            # task.appendcons(1)
            # syma = task.appendsparsesymmat(n,[9],[2],[1.0])
            # task.putbaraij(n+m, 0, [syma], [1.0])
            # task.putconbound(n+m, mosek.boundkey.fx, 2.0,2.0)


            # ci = [9]
            # cj = [2]
            # cval = []
            # for j in range(n):
            #     for i in range(i, n):
            #         if [j,i] not in edges:
            #             ci.append(i)
            #             cj.append(j)
            #             cval.append(1)


            # symc = task.appendsparsesymmat(n, ci, cj, cval)
            # task.putbarcj(0, [symc], [1.0])


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
                #print(t)
                s = np.linalg.matrix_rank(S, tol=.001)
                print(S)
                print(X)
                print(r)
                print(s)
                # B = scipy.linalg.sqrtm(S).real
                # print B
            else:
                print("Optimal solution not found for graph %d" % k)




def read_graph():
    k = 0
    index = 1
    s = 'abcdefghijklmnopqrstuvwxyz'
    with open("c7.txt") as f:
        for line in f:
            line.strip()
            elts = line.split()
            n = int(elts[0])
            m = 0
            adj = elts[1].split(",")
            edges = []
            for i in range(n):
                for j in range(len(adj[i])):
                    endpt = s.find(adj[i][j])
                    if endpt > i:
                        edges.append([0, 0])
                        edges[m][0] = i
                        edges[m][1] = endpt
                        m += 1
            print(index)
            mosek_check(n, m, k, edges)
            k += 1
            index +=1

read_graph()






#testing
            # task.appendcons(1)
            # syma = task.appendsparsesymmat(n, [4], [1], [1.0])
            # task.putbaraij(n+m, 0, [syma], [1.0])
            # task.putconbound(n+m, mosek.boundkey.fx, 2.0, 2.0)
            #
            #
            # task.appendcons(1)
            # syma = task.appendsparsesymmat(n, [8], [3], [1.0])
            # task.putbaraij(n + m+1, 0, [syma], [1.0])
            # task.putconbound(n + m + 1, mosek.boundkey.fx, 2.0, 2.0)
            #
            # task.appendcons(1)
            # syma = task.appendsparsesymmat(n, [5], [1], [1.0])
            # task.putbaraij(n + m + 2, 0, [syma], [1.0])
            # task.putconbound(n + m + 2, mosek.boundkey.fx, -2.0/3, -2.0/3)
            #
            # task.appendcons(1)
            # syma = task.appendsparsesymmat(n, [7], [1], [1.0])
            # task.putbaraij(n + m + 3, 0, [syma], [1.0])
            # task.putconbound(n + m + 3, mosek.boundkey.fx, 2.0, 2.0)

            # c=n+m
            # task.appendcons(10)
            # for i in range(10):
            #     syma = task.appendsparsesymmat(n, [i], [0], [1.0])
            #     task.putbaraij(c, 0, [syma], [1.0])
            #     if i == 9:
            #         task.putconbound(c, mosek.boundkey.fx, 2.0, 2.0)
            #     elif i == 0:
            #         task.putconbound(c, mosek.boundkey.fx, 1.0, 1.0)
            #     else:
            #         task.putconbound(c, mosek.boundkey.fx, -2.0/3, -2.0/3)
            #     c +=1
            #
            # task.appendcons(10)
            # for i in range(10):
            #     syma = task.appendsparsesymmat(n, [max(i,1)], [min(i,1)], [1.0])
            #     task.putbaraij(c, 0, [syma], [1.0])
            #     if i == 6:
            #         task.putconbound(c, mosek.boundkey.fx, 2.0, 2.0)
            #     elif i == 1:
            #         task.putconbound(c, mosek.boundkey.fx, 1.0, 1.0)
            #     else:
            #         task.putconbound(c, mosek.boundkey.fx, -2.0 / 3, -2.0 / 3)
            #     c += 1




            # c=n+m
            # #Triangle constraints <= 1/3
            # for i in range(n):
            #     for j in range(i+1,n):
            #         if [i,j] in edges:
            #             for k in range(j+1,n):
            #                 if [i,k] in edges and [j,k] in edges: #identified a triangle composed of i,j,k
            #                     for l in range(n):
            #                         task.appendcons(1)
            #                         syma = task.appendsparsesymmat(n, [max(i,l), max(j,l), max(k,l)], [min(i,l), min(j,l), min(k,l)], [1.0, 1.0, 1.0])
            #                         task.putbaraij(c, 0, [syma], [1.0])
            #                         task.putconbound(c, mosek.boundkey.up, -inf, 2.0/3)
            #                         c += 1
                                # for l in range(i,j):
                                #     task.appendcons(1)
                                #     syma = task.appendsparsesymmat(n, [l, j, k], [i, l, l], [1.0, 1.0, 1.0])
                                #     task.putbaraij(c, 0, [syma], [1.0])
                                #     task.putconbound(c, mosek.boundkey.up, -inf, 2.0 / 3)
                                #     c += 1
                                # for l in range(j,k)
                                #     task.appendcons(1)
                                #     syma = task.appendsparsesymmat(n, [l, l, k], [i, j, l], [1.0, 1.0, 1.0])
                                #     task.putbaraij(c, 0, [syma], [1.0])
                                #     task.putconbound(c, mosek.boundkey.up, -inf, 2.0 / 3)
                                #     c += 1
                                # for l in range(k,n)
                                #     task.appendcons(1)
                                #     syma = task.appendsparsesymmat(n, [l, l, l], [i, j, k], [1.0, 1.0, 1.0])
                                #     task.putbaraij(c, 0, [syma], [1.0])
                                #     task.putconbound(c, mosek.boundkey.up, -inf, 2.0 / 3)
                                #     c += 1




            # task.appendcons(8*(n-4))
            # c=n+m
            # #sum over clique constraint
            # for i in range(4,n):
            #     syma = task.appendsparsesymmat(n, [i,i,i], [0,1,2], [1.0,1.0,1.0])
            #     task.putbaraij(c, 0, [syma], [1.0])
            #     task.putconbound(c, mosek.boundkey.ra, -2.0, 2.0/3)
            #     c += 1
            #     syma = task.appendsparsesymmat(n, [i, i, i], [0, 1, 3], [1.0, 1.0, 1.0])
            #     task.putbaraij(c, 0, [syma], [1.0])
            #     task.putconbound(c, mosek.boundkey.ra, -2.0, 2.0 / 3)
            #     c += 1
            #     syma = task.appendsparsesymmat(n, [i, i, i], [0, 2, 3], [1.0, 1.0, 1.0])
            #     task.putbaraij(c, 0, [syma], [1.0])
            #     task.putconbound(c, mosek.boundkey.ra, -2.0, 2.0 / 3)
            #     c += 1
            #     syma = task.appendsparsesymmat(n, [i, i, i], [1, 2, 3], [1.0, 1.0, 1.0])
            #     task.putbaraij(c, 0, [syma], [1.0])
            #     task.putconbound(c, mosek.boundkey.ra, -2.0, 2.0 / 3)
            #     c += 1
            #
            #     if(i >= 7):
            #         syma = task.appendsparsesymmat(n, [i, i, i], [0, 1, 6], [1.0, 1.0, 1.0])
            #         task.putbaraij(c, 0, [syma], [1.0])
            #         task.putconbound(c, mosek.boundkey.ra, -2.0, 2.0 / 3)
            #         c += 1
            #         syma = task.appendsparsesymmat(n, [i, i, i], [0, 1, 7], [1.0, 1.0, 1.0])
            #         task.putbaraij(c, 0, [syma], [1.0])
            #         task.putconbound(c, mosek.boundkey.ra, -2.0, 2.0 / 3)
            #         c += 1
            #         syma = task.appendsparsesymmat(n, [i, i, i], [0, 6, 7], [1.0, 1.0, 1.0])
            #         task.putbaraij(c, 0, [syma], [1.0])
            #         task.putconbound(c, mosek.boundkey.ra, -2.0, 2.0 / 3)
            #         c += 1
            #         syma = task.appendsparsesymmat(n, [i, i, i], [1, 6, 7], [1.0, 1.0, 1.0])
            #         task.putbaraij(c, 0, [syma], [1.0])
            #         task.putconbound(c, mosek.boundkey.ra, -2.0, 2.0 / 3)
            #         c += 1
            #
            #     elif (i == 6):
            #         syma = task.appendsparsesymmat(n, [i, i, i], [0, 1, 6], [1.0, 1.0, 1.0])
            #         task.putbaraij(c, 0, [syma], [1.0])
            #         task.putconbound(c, mosek.boundkey.ra, -2.0, 2.0 / 3)
            #         c += 1
            #         syma = task.appendsparsesymmat(n, [i, i, 7], [0, 1, i], [1.0, 1.0, 1.0])
            #         task.putbaraij(c, 0, [syma], [1.0])
            #         task.putconbound(c, mosek.boundkey.ra, -2.0, 2.0 / 3)
            #         c += 1
            #         syma = task.appendsparsesymmat(n, [i, i, 7], [0, 6, i], [1.0, 1.0, 1.0])
            #         task.putbaraij(c, 0, [syma], [1.0])
            #         task.putconbound(c, mosek.boundkey.ra, -2.0, 2.0 / 3)
            #         c += 1
            #         syma = task.appendsparsesymmat(n, [i, i, 7], [1, 6, i], [1.0, 1.0, 1.0])
            #         task.putbaraij(c, 0, [syma], [1.0])
            #         task.putconbound(c, mosek.boundkey.ra, -2.0, 2.0 / 3)
            #         c += 1
            #     else:
            #         syma = task.appendsparsesymmat(n, [i, i, 6], [0, 1, i], [1.0, 1.0, 1.0])
            #         task.putbaraij(c, 0, [syma], [1.0])
            #         task.putconbound(c, mosek.boundkey.ra, -2.0, 2.0 / 3)
            #         c += 1
            #         syma = task.appendsparsesymmat(n, [i, i, 7], [0, 1, i], [1.0, 1.0, 1.0])
            #         task.putbaraij(c, 0, [syma], [1.0])
            #         task.putconbound(c, mosek.boundkey.ra, -2.0, 2.0 / 3)
            #         c += 1
            #         syma = task.appendsparsesymmat(n, [i, 6, 7], [0, i, i], [1.0, 1.0, 1.0])
            #         task.putbaraij(c, 0, [syma], [1.0])
            #         task.putconbound(c, mosek.boundkey.ra, -2.0, 2.0 / 3)
            #         c += 1
            #         syma = task.appendsparsesymmat(n, [i, 6, 7], [1, i, i], [1.0, 1.0, 1.0])
            #         task.putbaraij(c, 0, [syma], [1.0])
            #         task.putconbound(c, mosek.boundkey.ra, -2.0, 2.0 / 3)
            #         c += 1