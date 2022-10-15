import matplotlib as mpl
mpl.rcParams['text.usetex']=True
# mpl.rcParams['text.latex.unicode']=True
import matplotlib.pyplot as plt
import pickle
import argparse
plt.rcParams.update({'font.size': 20})

def get_parser():
    parser = argparse.ArgumentParser(description='ridge regression')
    parser.add_argument("--path", type=str, default='.')

    return parser

def main():
    fontsize=28
    parser = get_parser()
    args = parser.parse_args()
    state_dict, eigATA_sort,eigAAT_sort, lbd_list, output_dict = pickle.load(open('{}/results.p'.format(args.path),'rb'))
    test_name = args.path
    if state_dict['eigATA']:
        fig = plt.figure(figsize=(10,10))
        ax = fig.gca()
        ax.plot(eigATA_sort)
        ax.set_yscale('log')
        ax.set_title('eigenvalues of ATA',fontsize=fontsize)
        fig.savefig('{}/ATA_eig.eps'.format(test_name), format='eps')
        fig.savefig('{}/ATA_eig.png'.format(test_name))
    if state_dict['eigAAT']:
        fig = plt.figure(figsize=(10,10))
        ax = fig.gca()
        ax.plot(eigAAT_sort)
        ax.set_yscale('log')
        ax.set_title('eigenvalues of AAT',fontsize=fontsize)
        fig.savefig('{}/AAT_eig.eps'.format(test_name), format='eps')
        fig.savefig('{}/AAT_eig.png'.format(test_name))
    # time
    fig = plt.figure(figsize=(10,10))
    ax = fig.gca()
    for key in output_dict.keys():
        time = output_dict[key][0]
        if key=='cg' or key=='cg-over':
            label_plt = 'CG'
        elif key=='svd' or key=='svd-over':
            label_plt='SVD'
        elif key=='IHS-BIN-over':
            label_plt='IHS-BIN'
        elif key=='native-over':
            label_plt='native'
        else:
            label_plt = key
        ax.plot(time,label=label_plt)
    ax.legend(fontsize=fontsize)
    ax.set_xlabel(r'$T$',fontsize=fontsize)
    ax.set_title('Time',fontsize=fontsize)
    fig.savefig('{}/time.eps'.format(test_name), format='eps')
    fig.savefig('{}/time.png'.format(test_name))

    # sub iter
    if state_dict['IHS-BIN']:
        fig = plt.figure(figsize=(10,10))
        ax = fig.gca()
        info_ihs_bin = output_dict['IHS-BIN'][1]
        # if data_type=='random':
        #   ax.plot(info_ihs['sub_iters'],label='IHS')
        #   ax.plot(info_ihs_mom['sub_iters'],label='IHS-MOM')
        ax.plot(info_ihs_bin['sub_iters'],label='IHS-BIN')
        ax.legend(fontsize=fontsize)
        ax.set_title('Sub-iteration number',fontsize=fontsize)
        fig.savefig('{}/sub_iters.eps'.format(test_name), format='eps')
        fig.savefig('{}/sub_iters.png'.format(test_name))


    # train loss
    fig = plt.figure(figsize=(10,10))
    ax = fig.gca()
    for key in output_dict.keys():
        if key=='cg' or key=='cg-over':
            label_plt = 'CG'
        elif key=='svd' or key=='svd-over':
            label_plt='SVD'
        elif key=='IHS-BIN-over':
            label_plt='IHS-BIN'
        elif key=='native-over':
            label_plt='native'
        else:
            label_plt = key
        info = output_dict[key][1]
        ax.plot(lbd_list,info['train_loss'],label=label_plt)
    # ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend(fontsize=fontsize)
    ax.set_title('Train loss',fontsize=fontsize)
    ax.set_xlabel(r'$\lambda$',fontsize=fontsize)
    fig.savefig('{}/train_loss.eps'.format(test_name), format='eps')
    fig.savefig('{}/train_loss.png'.format(test_name))

    # test loss
    fig = plt.figure(figsize=(10,10))
    ax = fig.gca()
    for key in output_dict.keys():
        info = output_dict[key][1]
        if key=='cg' or key=='cg-over':
            label_plt = 'CG'
        elif key=='svd' or key=='svd-over':
            label_plt='SVD'
        elif key=='IHS-BIN-over':
            label_plt='IHS-BIN'
        elif key=='native-over':
            label_plt='native'
        else:
            label_plt = key
        ax.plot(lbd_list,info['test_loss'],label=label_plt)
    # ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend(fontsize=fontsize)
    ax.set_xlabel(r'$\lambda$',fontsize=fontsize)
    ax.set_title('Test loss',fontsize=fontsize)
    fig.savefig('{}/test_loss.eps'.format(test_name), format='eps')
    fig.savefig('{}/test_loss.png'.format(test_name))


if __name__ == '__main__':
    main()