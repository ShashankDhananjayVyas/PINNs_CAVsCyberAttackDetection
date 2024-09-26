"""
@author: Maziar Raissi
Modified 09/26/2024: Shashank Dhananjay Vyas
"""

import sys
sys.path.insert(0, '../../Utilities/')

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from plotting import newfig, savefig
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import time
import xlsxwriter

np.random.seed(1234)
tf.set_random_seed(1234)

class PhysicsInformedNN:
    # Initialize the class
##    def __init__(self, X, u, h, kp, kd, layers, lb, ub):
    def __init__(self, X, u, h, kp, kd, delta, layers):
        
##        self.lb = lb
##        self.ub = ub
        
        self.t = X
        self.vc = u[:,0:1]
        self.ac = u[:,1:2]
        self.di = u[:,2:3]
        self.vi = u[:,3:4]
        self.ai = u[:,4:5]
        
        self.h = h
        self.kp = kp
        self.kd = kd
        self.delta = delta
        
        self.layers = layers
        
        # Initialize NNs
        self.weights, self.biases = self.initialize_NN(layers)
        
        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

        # Initialize parameters
        self.lambda_1 = tf.Variable([-9], dtype=tf.float32)
##        self.lambda_2 = tf.Variable([0], dtype=tf.float32)
##        self.lambda_3 = tf.Variable([0], dtype=tf.float32)
        
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])
        self.vc_tf = tf.placeholder(tf.float32, shape=[None, self.vc.shape[1]])
        self.ac_tf = tf.placeholder(tf.float32, shape=[None, self.ac.shape[1]])
        self.di_tf = tf.placeholder(tf.float32, shape=[None, self.di.shape[1]])
        self.vi_tf = tf.placeholder(tf.float32, shape=[None, self.vi.shape[1]])
        self.ai_tf = tf.placeholder(tf.float32, shape=[None, self.ai.shape[1]])

        self.u_pred = self.net_u(self.t_tf)
        self.f_pred = self.net_f(self.t_tf, self.h, self.kp, self.kd, self.delta)
        
        self.loss = tf.reduce_mean(tf.square(np.hstack((self.vc, self.ac, self.di, self.vi, self.ai)) - self.u_pred)) + \
                    tf.reduce_mean(tf.square(self.f_pred))

        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 50000,
                                                                           'maxfun': 50000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})
    
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
        
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        
        H = X
##        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_u(self, t):
        u = self.neural_net(t, self.weights, self.biases)
        return u
    
    def net_f(self, t, h, kp, kd, delta):
        lambda_1 = self.lambda_1        
##        lambda_2 = self.lambda_2
##        lambda_3 = self.lambda_3
        u = self.net_u(t)
        a_dot = tf.gradients(u[:,4], t)
##        f = (kp/h)*u[:,2] - (kp+kd/h)*u[:,3] -(1/h+kd)*u[:,4] +(kd/h)*u[:,0] +(1/h)*u[:,1] + lambda_1*(kp*delta/h)
        f = (kp/h)*u[:,2] - (kp+kd/h)*u[:,3] -(1/h+kd)*u[:,4] +(kd/h)*u[:,0] +(1/h)*u[:,1] + lambda_1
        f = f - a_dot
        return f
    
    def callback(self, loss, lambda_1):
        print('Loss: %e, l1: %.5f' % (loss, lambda_1))
##    def callback(self, loss, lambda_1, lambda_2):
##        print('Loss: %e, l1: %.5f, l2: %.5f' % (loss, lambda_1, lambda_2))
##    def callback(self, loss, lambda_1, lambda_2, lambda_3):
##        print('Loss: %e, l1: %.5f, l2: %.5f, l3: %.5f' % (loss, lambda_1, lambda_2, lambda_3))
        
        
    def train(self, nIter):
        tf_dict = {self.t_tf: self.t, self.vc_tf: self.vc, self.ac_tf: self.ac, self.di_tf: self.di, self.vi_tf: self.vi, self.ai_tf: self.ai}
        
        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)
            
            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                lambda_1_value = self.sess.run(self.lambda_1)
##                lambda_2_value = self.sess.run(self.lambda_2)
##                lambda_3_value = self.sess.run(self.lambda_3)
                print('It: %d, Loss: %.3e, Lambda_1: %.3f, Time: %.2f' %
                      (it, loss_value, lambda_1_value, elapsed))
##                print('It: %d, Loss: %.3e, Lambda_1: %.3f, Lambda_2: %.3f, Time: %.2f' %
##                      (it, loss_value, lambda_1_value, lambda_2_value, elapsed))
##                print('It: %d, Loss: %.3e, Lambda_1: %.3f, Lambda_2: %.3f, Lambda_3: %.3f, Time: %.2f' % 
##                      (it, loss_value, lambda_1_value, lambda_2_value, lambda_3_value, elapsed))
                start_time = time.time()
        
        self.optimizer.minimize(self.sess,
                                feed_dict = tf_dict,
                                fetches = [self.loss, self.lambda_1],
##                                fetches = [self.loss, self.lambda_1, self.lambda_2],
##                                fetches = [self.loss, self.lambda_1, self.lambda_2, self.lambda_3],
                                loss_callback = self.callback)
        
        
    def predict(self, X_star):
        tf_dict = {self.t_tf: X_star}
        
        u_star = self.sess.run(self.u_pred, tf_dict)
        f_star = self.sess.run(self.f_pred, tf_dict)
        
        return u_star, f_star

    
if __name__ == "__main__": 

    file_lambda = xlsxwriter.Workbook('param_vals_cycles.xlsx')
    sheet = file_lambda.add_worksheet()
    for i in range(1,4):
        nu = 0.01/np.pi

        N_u = 1500#5000
        layers = [1, 20, 20, 20, 20, 10, 10, 10, 10, 5]
        
        d_path = '../Data/PINN_data_'
        d_idx = str(i)
        d_ext = '.mat'
        d_file = d_path + d_idx + d_ext
        data = scipy.io.loadmat(d_file)

        h=data['h']
        kp=data['kp']
        kd=data['kd']
        delta=data['delta']
        
        t = data['timet'].flatten()[:,None]
##        dc = data['dc'].flatten()[:,None]
        vc = data['vc'].flatten()[:,None]
        ac = data['ac'].flatten()[:,None]
        di = data['di'].flatten()[:,None]
        vi = data['vi'].flatten()[:,None]
        ai = data['ai'].flatten()[:,None]
        
        X_star = t
        u_star = np.hstack((vc, ac, di, vi, ai))

    # Doman bounds
##    lb = X_star.min(0)[0]
##    ub = X_star.max(0)[0]    
    
    ######################################################################
    ######################## Noiseles Data ###############################
    ######################################################################
        noise = 0.0            
                 
        idx = np.random.choice(X_star.shape[0], N_u, replace=False)
        X_u_train = X_star[idx,:]
        u_train = u_star[idx,:]
    
##    model = PhysicsInformedNN(X_u_train, u_train, h, kp, kd, layers, lb, ub)
        model = PhysicsInformedNN(X_u_train, u_train, h, kp, kd, delta, layers)

        model.train(0)
    
##    u_pred, f_pred = model.predict(X_star)
            
##    error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
    
##    U_pred = griddata(X_star, u_pred.flatten(), (X, T), method='cubic')
        
        lambda_1_value = model.sess.run(model.lambda_1)
##        file_lambda = xlsxwriter.Workbook('param_vals_cycles.xlsx')
##        sheet = file_lambda.add_worksheet()
        cell_no = 'A' + str(i)
        sheet.write(cell_no,lambda_1_value)

##    lambda_2_value = model.sess.run(model.lambda_2)
##    lambda_3_value = model.sess.run(model.lambda_3)
    
        error_lambda_1 = np.abs(lambda_1_value - (-kp*delta/h))/(kp*delta/h) * 100
    
##    print('Error u: %e' % (error_u))    
        print('Error l1: %.5f%%' % (error_lambda_1))
        print('File ',str(i),' done.')
    file_lambda.close()
    ######################################################################
    ########################### Noisy Data ###############################
    ######################################################################
##    noise = 0.01        
##    u_train = u_train + noise*np.std(u_train)*np.random.randn(u_train.shape[0], u_train.shape[1])
##        
##    model = PhysicsInformedNN(X_u_train, u_train, layers, lb, ub)
##    model.train(10000)
##    
##    u_pred, f_pred = model.predict(X_star)
##        
##    lambda_1_value_noisy = model.sess.run(model.lambda_1)
##    lambda_2_value_noisy = model.sess.run(model.lambda_2)
##    lambda_2_value_noisy = np.exp(lambda_2_value_noisy)
##            
##    error_lambda_1_noisy = np.abs(lambda_1_value_noisy - 1.0)*100
##    error_lambda_2_noisy = np.abs(lambda_2_value_noisy - nu)/nu * 100
##    
##    print('Error lambda_1: %f%%' % (error_lambda_1_noisy))
##    print('Error lambda_2: %f%%' % (error_lambda_2_noisy))                           

 
##    ######################################################################
##    ############################# Plotting ###############################
##    ######################################################################    
##    
##    fig, ax = newfig(1.0, 1.4)
##    ax.axis('off')
##    
##    ####### Row 0: u(t,x) ##################    
##    gs0 = gridspec.GridSpec(1, 2)
##    gs0.update(top=1-0.06, bottom=1-1.0/3.0+0.06, left=0.15, right=0.85, wspace=0)
##    ax = plt.subplot(gs0[:, :])
##    
##    h = ax.imshow(U_pred.T, interpolation='nearest', cmap='rainbow', 
##                  extent=[t.min(), t.max(), x.min(), x.max()], 
##                  origin='lower', aspect='auto')
##    divider = make_axes_locatable(ax)
##    cax = divider.append_axes("right", size="5%", pad=0.05)
##    fig.colorbar(h, cax=cax)
##    
##    ax.plot(X_u_train[:,1], X_u_train[:,0], 'kx', label = 'Data (%d points)' % (u_train.shape[0]), markersize = 2, clip_on = False)
##    
##    line = np.linspace(x.min(), x.max(), 2)[:,None]
##    ax.plot(t[25]*np.ones((2,1)), line, 'w-', linewidth = 1)
##    ax.plot(t[50]*np.ones((2,1)), line, 'w-', linewidth = 1)
##    ax.plot(t[75]*np.ones((2,1)), line, 'w-', linewidth = 1)
##    
##    ax.set_xlabel('$t$')
##    ax.set_ylabel('$x$')
##    ax.legend(loc='upper center', bbox_to_anchor=(1.0, -0.125), ncol=5, frameon=False)
##    ax.set_title('$u(t,x)$', fontsize = 10)
##    
##    ####### Row 1: u(t,x) slices ##################    
##    gs1 = gridspec.GridSpec(1, 3)
##    gs1.update(top=1-1.0/3.0-0.1, bottom=1.0-2.0/3.0, left=0.1, right=0.9, wspace=0.5)
##    
##    ax = plt.subplot(gs1[0, 0])
##    ax.plot(x,Exact[25,:], 'b-', linewidth = 2, label = 'Exact')       
##    ax.plot(x,U_pred[25,:], 'r--', linewidth = 2, label = 'Prediction')
##    ax.set_xlabel('$x$')
##    ax.set_ylabel('$u(t,x)$')    
##    ax.set_title('$t = 0.25$', fontsize = 10)
##    ax.axis('square')
##    ax.set_xlim([-1.1,1.1])
##    ax.set_ylim([-1.1,1.1])
##    
##    ax = plt.subplot(gs1[0, 1])
##    ax.plot(x,Exact[50,:], 'b-', linewidth = 2, label = 'Exact')       
##    ax.plot(x,U_pred[50,:], 'r--', linewidth = 2, label = 'Prediction')
##    ax.set_xlabel('$x$')
##    ax.set_ylabel('$u(t,x)$')
##    ax.axis('square')
##    ax.set_xlim([-1.1,1.1])
##    ax.set_ylim([-1.1,1.1])
##    ax.set_title('$t = 0.50$', fontsize = 10)
##    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)
##    
##    ax = plt.subplot(gs1[0, 2])
##    ax.plot(x,Exact[75,:], 'b-', linewidth = 2, label = 'Exact')       
##    ax.plot(x,U_pred[75,:], 'r--', linewidth = 2, label = 'Prediction')
##    ax.set_xlabel('$x$')
##    ax.set_ylabel('$u(t,x)$')
##    ax.axis('square')
##    ax.set_xlim([-1.1,1.1])
##    ax.set_ylim([-1.1,1.1])    
##    ax.set_title('$t = 0.75$', fontsize = 10)
##    
##    ####### Row 3: Identified PDE ##################    
##    gs2 = gridspec.GridSpec(1, 3)
##    gs2.update(top=1.0-2.0/3.0, bottom=0, left=0.0, right=1.0, wspace=0.0)
##    
##    ax = plt.subplot(gs2[:, :])
##    ax.axis('off')
##    s1 = r'$\begin{tabular}{ |c|c| }  \hline Correct PDE & $u_t + u u_x - 0.0031831 u_{xx} = 0$ \\  \hline Identified PDE (clean data) & '
##    s2 = r'$u_t + %.5f u u_x - %.7f u_{xx} = 0$ \\  \hline ' % (lambda_1_value, lambda_2_value)
##    s3 = r'Identified PDE (1\% noise) & '
##    s4 = r'$u_t + %.5f u u_x - %.7f u_{xx} = 0$  \\  \hline ' % (lambda_1_value_noisy, lambda_2_value_noisy)
##    s5 = r'\end{tabular}$'
##    s = s1+s2+s3+s4+s5
##    ax.text(0.1,0.1,s)
##        
##    # savefig('./figures/Burgers_identification')  
    



