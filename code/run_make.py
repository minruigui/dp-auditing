import multiprocessing as mp
import os

def run_exp(cmd):
    cmd = "CUDA_VISIBLE_DEVICES= "+cmd
    print(cmd)
    os.system(cmd)

if __name__ == "__main__":

    from itertools import product
    import auditing_args
    from collections import defaultdict

    BATCH_SIZE = 50

    data_dir = auditing_args.args["save_dir"]
    h5s = [fname for fname in os.listdir(data_dir) if fname.endswith('.h5')]


    def get_cfg(h5):
        splt = h5.split('-')
        if 'no' in h5:
                            # clip_norm noise, init_mult, bkd_trial
            return ('no', '.', splt[2], splt[3], splt[4])
        else:
                            # clip_norm, noise, init_mult, bkd_trial
            return ('new', splt[1], splt[2], splt[3], splt[4])

    cfg_map = defaultdict(list)

    for h5 in h5s:
        cfg_map[get_cfg(h5)].append(h5)

    args = {d: len(cfg_map[d]) for d in cfg_map if len(cfg_map[d]) > 0}
    print(args)

    all_exp = []


    fmt_cmd = "python3  code/make_nps.py {} {} {} {} {} {} {} >/dev/null"
    for arg in args:
        for start in range(0, args[arg], BATCH_SIZE):
            cmd = fmt_cmd.format(arg[0], start, start +
                                BATCH_SIZE, arg[1], arg[2], arg[3], arg[4])
            all_exp.append(cmd)
    print(len(args), len(all_exp))

    p = mp.Pool(16)
    p.map(run_exp, all_exp)
