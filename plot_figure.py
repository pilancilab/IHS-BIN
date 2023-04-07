import matplotlib as mpl
mpl.rcParams['text.usetex']=True
# mpl.rcParams['text.latex.unicode']=True
import matplotlib.pyplot as plt
import pickle
import argparse
plt.rcParams.update({'font.size': 24})
import numpy as np

lw=3
alpha=0.5

def get_parser():
    parser = argparse.ArgumentParser(description='ridge regression')
    parser.add_argument("--path", type=str, default='.')
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--ymin",type=int,default=0)
    parser.add_argument("--ymax",type=int,default=99)
    return parser

def main():
    fontsize=28
    parser = get_parser()
    args = parser.parse_args()
    repeat = args.repeat
    state_dict, eigATA_sort,eigAAT_sort, lbd_list, output_dict = pickle.load(open('{}/results.p'.format(args.path),'rb'))
    test_name = args.path
    ymin = args.ymin
    ymax = args.ymax
    if state_dict['eigATA']:
        fig = plt.figure(figsize=(10,10))
        ax = fig.gca()
        ax.plot(eigATA_sort,linewidth=lw)
        ax.set_yscale('log')
        ax.set_title('Eigenvalues of '+r'$A^TA$',fontsize=fontsize)
        fig.savefig('{}/ATA_eig.eps'.format(test_name), format='eps')
        fig.savefig('{}/ATA_eig.png'.format(test_name))
    if state_dict['eigAAT']:
        fig = plt.figure(figsize=(10,10))
        ax = fig.gca()
        ax.plot(eigAAT_sort,linewidth=lw)
        ax.set_yscale('log')
        ax.set_title('eigenvalues of '+r'$AA^T$',fontsize=fontsize)
        fig.savefig('{}/AAT_eig.eps'.format(test_name), format='eps')
        fig.savefig('{}/AAT_eig.png'.format(test_name))
    # time
    fig = plt.figure(figsize=(10,10))
    ax = fig.gca()
    for key in output_dict.keys():
        time = output_dict[key][0]
        if key=='cg' or key=='cg-over':
            label_plt = 'CG'
            linestyle = 'dotted'
            colorer_plt = 'r'
        elif key=='svd' or key=='svd-over':
            label_plt='SVD'
            linestyle = 'dashed'
            colorer_plt = 'g'
        elif key=='IHS-BIN-over' or key=='IHS-BIN':
            label_plt='IHS-BIN'
            linestyle = 'solid'
            colorer_plt = 'b'
        elif key=='native-over' or key=='native':
            label_plt='native'
            linestyle = 'dashdot'
            colorer_plt = 'c'
        elif key=='ridge-sketch':
            label_plt='ridge-sketch'
            linestyle = 'solid'
            colorer_plt = 'gold'
        
        if key=='IHS-BIN-over' or key=='IHS-BIN' or key == 'ridge-sketch':
            if repeat==1:
                ax.plot(time,label=label_plt,linewidth=lw,linestyle=linestyle,color=colorer_plt,alpha=alpha)
            else:
                time_list =  [x[0].reshape([-1,1]) for x in output_dict[key]]
                time_list = np.concatenate(time_list,axis=1)
                time_mean = np.mean(time_list,axis=1)
                time_std = np.sqrt(np.var(time_list,axis=1))
                ax.plot(time_mean, label=label_plt,linewidth=lw,linestyle=linestyle,color=colorer_plt,alpha=alpha)
                ax.fill_between(np.arange(len(time_mean)), time_mean-time_std, time_mean+time_std, color=colorer_plt, alpha=.2)
        else:
            ax.plot(time,label=label_plt,linewidth=lw,linestyle=linestyle,color=colorer_plt,alpha=alpha)

    ax.legend(fontsize=fontsize)
    ax.set_xlabel(r'$T$',fontsize=fontsize)
    ax.set_title('Time',fontsize=fontsize)
    # ax.set_ylabel('Time',fontsize=fontsize)
    fig.savefig('{}/time.eps'.format(test_name), format='eps')
    fig.savefig('{}/time.png'.format(test_name))

    # sub iter
    # if state_dict['IHS-BIN']:
    #     fig = plt.figure(figsize=(10,10))
    #     ax = fig.gca()
    #     info_ihs_bin = output_dict['IHS-BIN'][1]
    #     # if data_type=='random':
    #     #   ax.plot(info_ihs['sub_iters'],label='IHS')
    #     #   ax.plot(info_ihs_mom['sub_iters'],label='IHS-MOM')
    #     ax.plot(info_ihs_bin['sub_iters'],label='IHS-BIN',linewidth=lw)
    #     ax.legend(fontsize=fontsize)
    #     ax.set_title('Sub-iteration number',fontsize=fontsize)
    #     fig.savefig('{}/sub_iters.eps'.format(test_name), format='eps')
    #     fig.savefig('{}/sub_iters.png'.format(test_name))


    # train loss
    fig = plt.figure(figsize=(10,10))
    ax = fig.gca()
    for key in output_dict.keys():
        if key=='cg' or key=='cg-over':
            label_plt = 'CG'
            linestyle = 'dotted'
            colorer_plt = 'r'
        elif key=='svd' or key=='svd-over':
            label_plt='SVD'
            linestyle = 'dashed'
            colorer_plt = 'g'
        elif key=='IHS-BIN-over' or key=='IHS-BIN':
            label_plt='IHS-BIN'
            linestyle = 'solid'
            colorer_plt = 'b'
        elif key=='native-over' or key=='native':
            label_plt='native'
            linestyle = 'dashdot'
            colorer_plt = 'c'
        elif key=='ridge-sketch':
            label_plt='ridge-sketch'
            linestyle = 'solid'
            colorer_plt = 'gold'
        
            
        if key=='IHS-BIN-over' or key=='IHS-BIN' or key == 'ridge-sketch':
            if repeat==1:
                info = output_dict[key][1]
                ax.plot(lbd_list,info['train_loss'],label=label_plt,linestyle=linestyle,color=colorer_plt, linewidth=lw,alpha=alpha)

            else:
                time_list =  [x[1]['train_loss'].reshape([-1,1]) for x in output_dict[key]]
                time_list = np.concatenate(time_list,axis=1)
                time_mean = np.mean(time_list,axis=1)
                time_std = np.sqrt(np.var(time_list,axis=1))
                ax.plot(lbd_list, time_mean, label=label_plt,linewidth=lw,linestyle=linestyle,color=colorer_plt,alpha=alpha)
                ax.fill_between(lbd_list, time_mean-time_std, time_mean+time_std, color=colorer_plt, alpha=.2)

        else:
            info = output_dict[key][1]
            ax.plot(lbd_list,info['train_loss'],label=label_plt,linestyle=linestyle,color=colorer_plt, linewidth=lw,alpha=alpha)

    # ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend(fontsize=fontsize)
    ax.set_title('Train loss',fontsize=fontsize)
    # ax.set_ylabel('Train loss',fontsize=fontsize)
    ax.set_xlabel(r'$\lambda$',fontsize=fontsize)
    fig.savefig('{}/train_loss.eps'.format(test_name), format='eps')
    fig.savefig('{}/train_loss.png'.format(test_name))

    # test loss
    fig = plt.figure(figsize=(10,10))
    ax = fig.gca()
    for key in output_dict.keys():
        if key=='cg' or key=='cg-over':
            label_plt = 'CG'
            linestyle = 'dotted'
            colorer_plt = 'r'
        elif key=='svd' or key=='svd-over':
            label_plt='SVD'
            linestyle = 'dashed'
            colorer_plt = 'g'
        elif key=='IHS-BIN-over' or key=='IHS-BIN':
            label_plt='IHS-BIN'
            linestyle = 'solid'
            colorer_plt = 'b'
        elif key=='native-over' or key=='native':
            label_plt='native'
            linestyle = 'dashdot'
            colorer_plt = 'c'
        elif key=='ridge-sketch':
            label_plt='ridge-sketch'
            linestyle = 'solid'
            colorer_plt = 'gold'
        if key=='IHS-BIN-over' or key=='IHS-BIN' or key == 'ridge-sketch':
            if repeat==1:
                info = output_dict[key][1]
                ax.plot(lbd_list,info['test_loss'],label=label_plt,linestyle=linestyle,color=colorer_plt, linewidth=lw,alpha=alpha)
            else:
                time_list =  [x[1]['test_loss'].reshape([-1,1]) for x in output_dict[key]]
                time_list = np.concatenate(time_list,axis=1)
                time_mean = np.mean(time_list,axis=1)
                time_std = np.sqrt(np.var(time_list,axis=1))
                ax.plot(lbd_list, time_mean, label=label_plt,linewidth=lw,linestyle=linestyle,color=colorer_plt,alpha=alpha)
                ax.fill_between(lbd_list, time_mean-time_std, time_mean+time_std, color=colorer_plt, alpha=.2)
        else:
            info = output_dict[key][1]
            ax.plot(lbd_list,info['test_loss'],label=label_plt,linestyle=linestyle,color=colorer_plt, linewidth=lw,alpha=alpha)
    # ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend(fontsize=fontsize)
    ax.set_xlabel(r'$\lambda$',fontsize=fontsize)
    ax.set_title('Test loss',fontsize=fontsize)
    fig.savefig('{}/test_loss.eps'.format(test_name), format='eps')
    fig.savefig('{}/test_loss.png'.format(test_name))


if __name__ == '__main__':
    main()