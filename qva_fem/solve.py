from fem_1d import *
import matplotlib.pyplot as plt

xl = 0; xr = 1; nx = 6;
tl = 1; tr = 3; nt = 11
reduced=1;
u, x, t, M, A, dt, h = heat_solve(xl,xr,nx,tl,tr,nt,reduced) 


hh,xx = create_grid(101,xl,xr) # for plotting analytic
for n1 in range(nt):
    print("u(x,",round(t[n1],2),") =  ",u[n1]);
    fig = plt.figure(figsize=(12, 12),facecolor=(1, 1, 1))
    plt.plot(x,u[n1],color="black",label='Discrete Solution')
    plt.plot(xx,u_heat(xx,t[n1]),color="blue",linestyle='--',marker='o',label='Analytic Solution')
    plt.title('time: %.1fs'%(n1*dt), fontsize=16)
    plt.legend(fontsize=16,edgecolor="black",loc="upper right")
    plt.show()
