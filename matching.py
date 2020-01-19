import os
import numpy as np
import matplotlib.pyplot as plt

def match_viz_shapes(t4viz, t8viz):
    cols = t4viz.shape[1]
    zerosvec = np.zeros((1,cols))

    while t4viz.shape[0] - t8viz.shape[0] > 1:
        t8viz = np.r_[zerosvec, t8viz, zerosvec]

    if t4viz.shape[0] == t8viz.shape[0]:
        return t8viz

    assert t4viz.shape[0] - t8viz.shape[0] == 1, 'sth wrong in difference: ' + str(t4viz.shape[0]) + ' ' + str(t8viz.shape[0])

    t8viz_up = np.r_[zerosvec, t8viz]
    t8viz_down = np.r_[zerosvec, t8viz]

    if np.sum((t4viz>0)*(t8viz_up>0)) > np.sum((t4viz>0)*(t8viz_down>0)):
        return t8viz_up
    else:
        return t8viz_down

cnt = 0
for ff in os.listdir('data/t4'):
    try:
        t4data = np.load(os.path.join('data/t4', ff))
        t8data = np.load(os.path.join('data/t8', ff))
    except FileNotFoundError as e:
        print(e)
        print('Not such file: ' + ff)
        continue
    vizname = ff.split('.')[0]
    t4viz = t4data['visualization'] 
    t8viz = t8data['visualization'] 
    if t4viz.shape == t8viz.shape:
        plt.figure()
        plt.imshow(t4viz>0)
        plt.imshow(t8viz>0,alpha = 0.5) 
        plt.savefig('matching_' + vizname+'.png')
        plt.close()
    else:
        print('Different sizes: ' + ff + ' t4: ' + str(t4viz.shape) + ' t8: ' + str(t8viz.shape))
        cnt += 1
        plt.figure(figsize=(9,4))
        plt.subplot(121)
        plt.imshow(t4viz>0)
        plt.subplot(122)
        plt.imshow(t8viz>0)
        plt.savefig('nonmatching_' + vizname+'.png')
        plt.close()
        if t4viz.shape[0] < t8viz.shape[0]:
            t4viz = match_viz_shapes(t8viz, t4viz)
        else:
            t8viz = match_viz_shapes(t4viz, t8viz)
        plt.figure()
        plt.imshow(t4viz>0)
        plt.imshow(t8viz>0,alpha = 0.5) 
        plt.savefig('newmatching_' + vizname+'.png')
        plt.close()



print('Non matching sizes:' + str(cnt))

