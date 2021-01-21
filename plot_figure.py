import matplotlib.pyplot as plt
import pickle
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description='ridge regression')
    parser.add_argument("--path", type=str, default='.')

    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    state_dict, eigATA_sort,eigAAT_sort, lbd_list, output_dict = pickle.load(open('{}/results.p'.format(args.path),'rb'))
    test_name = args.path
    if state_dict['eigATA']:
        fig = plt.figure(figsize=(10,10))
        ax = fig.gca()
        ax.plot(eigATA_sort)
        ax.set_yscale('log')
        ax.set_title('eigenvalues of ATA')
        fig.savefig('{}/ATA_eig.png'.format(test_name))
    if state_dict['eigAAT']:
        fig = plt.figure(figsize=(10,10))
        ax = fig.gca()
        ax.plot(eigAAT_sort)
        ax.set_yscale('log')
        ax.set_title('eigenvalues of AAT')
        fig.savefig('{}/AAT_eig.png'.format(test_name))
    # time
    fig = plt.figure(figsize=(10,10))
    ax = fig.gca()
    for key in output_dict.keys():
        time = output_dict[key][0]
        ax.plot(time,label=key)
    ax.legend()
    ax.set_title('Time')
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
        ax.legend()
        ax.set_title('Sub-iteration number')
        fig.savefig('{}/sub_iters.png'.format(test_name))


    # train loss
    fig = plt.figure(figsize=(10,10))
    ax = fig.gca()
    for key in output_dict.keys():
        info = output_dict[key][1]
        ax.plot(lbd_list,info['train_loss'],label=key)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend()
    ax.set_title('Train loss')
    fig.savefig('{}/train_loss.png'.format(test_name))

    # test loss
    fig = plt.figure(figsize=(10,10))
    ax = fig.gca()
    for key in output_dict.keys():
        info = output_dict[key][1]
        ax.plot(lbd_list,info['test_loss'],label=key)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.legend()
    ax.set_title('Test loss')
    fig.savefig('{}/test_loss.png'.format(test_name))


if __name__ == '__main__':
    main()