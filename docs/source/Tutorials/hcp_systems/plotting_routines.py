import matplotlib
import numpy as np
import matplotlib.pyplot as plt

def plot_2d_pts_box(twoD_pts, twoD_mat, orig):
    ################################################################################
    l1 = np.copy(twoD_mat);
    avec = l1[:,0]; bvec=l1[:,1];
    pts = np.zeros((4,2));
    pts[0,:] = orig;
    pts[1,:] = orig+avec; pts[2,:] = orig+avec+bvec; pts[3,:] = orig+bvec;
    tinds = [0,1,2,3,0];
    x1=pts[tinds,0];y1=pts[tinds,1];
    plt.plot(x1,y1);

    pts = np.copy(twoD_pts);
    x1=pts[:,0];y1=pts[:,1];
    plt.scatter(x1,y1);
    ################################################################################

def plot_3d_pts_box(fig, pts, tmat, sim_orig):
    ################################################################################
    ax = fig.add_subplot(111, projection='3d')

    box_pts = np.zeros((8,3));
    box_pts[1,:] = tmat[:,0]; box_pts[2,:] = tmat[:,0]+tmat[:,1];
    box_pts[3,:] = tmat[:,1];
    box_pts[4,:] = box_pts[0,:]+tmat[:,2];
    box_pts[5,:] = box_pts[1,:]+tmat[:,2];
    box_pts[6,:] = box_pts[2,:]+tmat[:,2];
    box_pts[7,:] = box_pts[3,:]+tmat[:,2];
    box_pts = box_pts + sim_orig;

    tinds = [0,1,2,3,0,4,5,6,7,4,0,1,5,6,2,3,7];
    x1 = box_pts[tinds,0]; y1 = box_pts[tinds,1]; z1 = box_pts[tinds,2];
    ax.plot(x1,y1,z1);

    x1=pts[:,0];y1=pts[:,1];z1=pts[:,2];
    ax.scatter(x1,y1,z1);

    ################################################################################
