# This notebook solve the Heat Equation using a first-order FE method
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags

def get_mass_matrix(n,h):
    Msuper_diag = Msub_diag = (1./6.)*h
    Mdiag = (2./3.)*h
    return Mdiag*np.eye(n) + Msuper_diag*np.eye(n,n,1) + Msub_diag*np.eye(n,n,-1)
    
def get_A_matrix(n,h,dt):
    Asuper_diag = Asub_diag = (1./6.)*h - (dt/h)
    Adiag = (2./3.)*h + 2*(dt/h)
    return Adiag*np.eye(n) + Asuper_diag*np.eye(n,n,1) + Asub_diag*np.eye(n,n,-1)
    
def update_discrete_solution_heat(n,x,h,dt,u_old,A,M,db0,dbn,ul_old,ur_old,reduced_flag):
    b = np.matmul(M,u_old)
    if reduced_flag == 1:
        Msub_diag = M[0,1]
        Asub_diag = A[0,1]
        b[0] = b[0] + Msub_diag * ul_old - Asub_diag * db0
        b[n-1] = b[n-1] + Msub_diag * ur_old - Asub_diag * dbn
    else:
        b[0] = db0
        b[n-1] = dbn
    #print("b: ",b)
    u = np.linalg.solve(A, b)
    return  u

def create_grid(n,xl,xr):
    h = (xr-xl)/(n-1)
    x = np.linspace(xl,xr,n)
    return h, x

def heat_solve(xl,xr,nx,tl,tr,nt,reduced,u_analytic):
    dt,t = create_grid(nt,tl,tr)
    h,xn = create_grid(nx,xl,xr)
    un = u_analytic(xn,t[0])

    if reduced == 1:
        ni = nx - 2
        M = get_mass_matrix(ni,h)
        A = get_A_matrix(ni,h,dt)
        x = xn[1:nx-1]
        u = un[1:nx-1]
    else:
        ni = nx
        M = get_mass_matrix(nx,h)
        A = get_A_matrix(nx,h,dt)
        A[0,:] = 0   # zeroes out row 0
        A[nx-1,:] = 0   # zeroes out row n
        A[0,0] = 1
        A[nx-1,nx-1] = 1
        x = xn
        u = un
    solution = []
    solution.append(un.tolist())
    #print("u(x,",round(tl,2),"): ",un)
    for n1 in range(nt-1):
        time = t[n1]+dt
        db0 = u_analytic(xl,time) # updated bc at next time
        dbn = u_analytic(xr,time) # updated bc at next time
        u = update_discrete_solution_heat(ni,x,h,dt,u,A,M,db0,dbn,un[0],un[nx-1],reduced)
        if reduced == 1:
            un[0] = db0
            un[nx-1] = dbn
            un[1:nx-1] = u[0:ni]
        else:
            un = u
        solution.append(un.tolist())
        #print("u(x,",round(time,2),"): ",u)
    return solution, xn, t, M, A, dt, h

def wave_solve(c,xl,xr,nx,tl,tr,nt,reduced,u_analytic):
    # note :: Un is full grid, u is internal grid
    # note :: C MUST BE 1
    assert(c==1)
    
    dt,t = create_grid(nt,tl,tr)
    h,xn = create_grid(nx,xl,xr)
    
    n1 = 0
    older_time = t[n1]
    old_time   = t[n1]+dt
    #print("older time: ",older_time);
    #print("old time: ",old_time);
    
    un_older = u_analytic(xn,older_time)
    un_old = u_analytic(xn,old_time)

    # multiply through by h and dt^2
    if reduced == 1:
        ni = nx - 2
        K = diags([-1, 2, -1], [-1, 0, 1], shape=(ni, ni)).toarray()
        M = h * h * diags([1/6, 2/3, 1/6], [-1, 0, 1], shape=(ni, ni)).toarray()
        A = M + dt*dt*K
        x = xn[1:nx-1]
        u_older = un_older[1:nx-1]
        u_old = un_old[1:nx-1]
#    else:
#         ni = nx
#         K = (1/h) * diags([-1, 2, -1], [-1, 0, 1], shape=(nx, nx)).toarray()
#         M = h *h * diags([1/6, 2/3, 1/6], [-1, 0, 1], shape=(nx, nx)).toarray()
#         A = M + dt*dt*K
#         A[0,:] = 0   # zeroes out row 0
#         A[nx-1,:] = 0   # zeroes out row n
#         A[0,0] = 1
#         A[nx-1,nx-1] = 1
#         x = xn
#         u_older = un_older
#         u_old = un_old
#         u_new = un_new
        
    solution = []
    solution.append(un_older.tolist())
    solution.append(un_old.tolist())

    un = np.zeros(nx)
    #print("u(x,",round(tl,2),"): ",un)
    for n1 in range(2,nt):
        
        older_time = t[n1]-2*dt
        old_time   = t[n1]-dt
        new_time   = t[n1]
        
        #print("older time: ",older_time);
        #print("old time: ",old_time);
        #print("new time: ",new_time);
        
        ul_older = u_analytic(xl,older_time) # updated bc at next time
        ur_older = u_analytic(xr,older_time) # updated bc at next time
        ul_old = u_analytic(xl,old_time) # updated bc at next time
        ur_old = u_analytic(xr,old_time) # updated bc at next time
        ul_new = u_analytic(xl,new_time) # updated bc at next time
        ur_new = u_analytic(xr,new_time) # updated bc at next time
        
        b = 2*np.matmul(M,u_old) - np.matmul(M,u_older)
        if reduced == 1:
            b[0] = b[0] - A[0,1] * ul_new + (2 * M[0,1]*ul_old - M[0,1]*ul_older)
            b[ni-1] = b[ni-1]  - A[0,1] * ur_new + (2 * M[0,1]*ur_old - M[0,1]*ur_older)
        else:
            b[0] = db0
            b[nx-1] = dbn
        #print("b: ",b)
        u_new = np.linalg.solve(A, b)
        
        if reduced == 1:
            un[0] = ul_new
            un[nx-1] = ur_new
            un[1:nx-1] = u_new[0:ni]
        else:
            un = u_new
        solution.append(un.tolist())
        
        u_older = u_old
        u_old = u_new
        
        #print("u(x,",round(time,2),"): ",u)
    return solution, xn, t, M, A, dt, h



