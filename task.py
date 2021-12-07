#!/usr/bin/env python
import sys, os, logging
import time
import statistics
import argparse
import pandas as pd
import collections
import pickle
import numpy as np
sys.path.insert(0, '.')

import sacrebleu
from sacrebleu.metrics import BLEU
from seq2seq import utils

# define the search space
beam_sizes = [i for i in range(1,15)] + [i for i in range(15, 40, 5)]

alphas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

N_REPEATS = 5

ref_files = ['data/en-fr/raw/test.en']

metrics = [
    # BLEU
    (BLEU, {}),
    (BLEU, {'tokenize': 'intl'}),
    (BLEU, {'tokenize': 'none', 'force': True}),
]


def create_metric(klass, kwargs, refs=None):
    if refs:
        # caching mode
        kwargs['references'] = refs
    return klass(**kwargs)


def read_files(*args):
    lines = []
    for fname in args:
        cur_lines = []
        with open(fname) as f:
            for line in f:
                cur_lines.append(line.strip())
        lines.append(cur_lines)
    return lines


def calculate_stat(scores):
    scores = sorted(scores)
    median = scores[len(scores) // 2]
    std = statistics.pstdev(scores)
    mean = sum(scores) / len(scores)
    logging.info(f' || mean: {mean:.3f} -- median: {median:.3f} -- stdev: {std:.3f}')
    return mean, median, std


def measure(thisdict, i, metric_klass, metric_kwargs,
            beam_sizes,
            systems,
            refs, cache=False,
            decoding_time_dict=None,
            alpha=0):
    scores = []
    durations = []
    bps = []
    if cache:
        # caching mode
        metric_kwargs['references'] = refs
    st = time.time()
    metric = metric_klass(**metric_kwargs)

    for beam_size, system in zip(beam_sizes, systems):
        Score = metric.corpus_score(system, None if cache else refs)
        sc = Score.score
        bp = Score.bp
        dur = time.time() - st
        thisdict['BLUE Version'].append(i)
        thisdict['Beam Size'].append(beam_size)
        thisdict['alpha'].append(alpha)
        if decoding_time_dict and len(decoding_time_dict) != 0:
            if f'{beam_size}-{alpha}' in decoding_time_dict.keys():
                thisdict['Duration'].append(decoding_time_dict[f'{beam_size}-{alpha}'])
            else:
                thisdict['Duration'].append(np.nan)
        thisdict['BLUE Score'].append(sc)
        thisdict['Brevity Penalty'].append(bp)
        st = time.time()
        logging.info(f' || Beam Size: {beam_size} alpha: {alpha} '
              f'-- BLUE: {sc:.3f} '
              f'-- BP: {bp:.3f} -- Duration: {dur:.3f}')
    # scores_stat = calculate_stat(scores)
    # bp_stat = calculate_stat(bps)
    # duration_stat = calculate_stat(durations)

def get_translation_output(beam_sizes, cache=False, alphas=[0],
                           output_dict=None):
    decoding_time_dict = {}
    for beam_size in beam_sizes:
        for alpha in alphas:
            if cache and os.path.isfile(f'{output_dict}/model_translations_beam-{beam_size}_alpha-{alpha}.p.txt'):
                continue
            st = time.time()
            os.system(f'python translate_beam.py '
                      f'--data data/en-fr/prepared -'
                      f'-dicts data/en-fr/prepared  '
                      f'--checkpoint-path assignments/03/baseline/checkpoints/checkpoint_best.pt '
                      f'--batch-size 500 '
                      f'--alpha {alpha} '
                      f'--beam-size {beam_size} '
                      f'--output {output_dict}/model_translations_beam-{beam_size}_alpha-{alpha}.txt '
                      f'--cuda True')
            decoding_time = time.time() - st
            logging.info(f"the decoding time of beam size {beam_size} alpha {alpha}: {decoding_time}")
            decoding_time_dict[f"{beam_size}-{alpha}"] = decoding_time
            os.system(f'./scripts/postprocess.sh '
                      f'{output_dict}/model_translations_beam-{beam_size}_alpha-{alpha}.txt '
                      f'{output_dict}/model_translations_beam-{beam_size}_alpha-{alpha}.p.txt en ')
            os.remove(f'{output_dict}/model_translations_beam-{beam_size}_alpha-{alpha}.txt')
    if not cache:
        pickle.dump(decoding_time_dict, open(f"{output_dict}/decoding_time_dict.pkl", "wb"))
    return decoding_time_dict


def search(beam_sizes=(11), alphas=(0), output_dict=None, cache=True):
    get_translation_output(beam_sizes, cache=cache, alphas=alphas,
                           output_dict=output_dict)
    # decoding_time_dict = pickle.load(open(f"{output_dict}/decoding_time_dict.pkl", "rb"))
    decoding_time_dict = None
    thisdict = collections.defaultdict(list)
    for alpha in alphas:
        sys_files = [f'{output_dict}/model_translations_beam-{beam_size}_alpha-{alpha}.p.txt' for beam_size in beam_sizes]
        systems = read_files(*sys_files)
        refs = read_files(*ref_files)
        msg = f'SacreBLEU {sacrebleu.__version__} performance tests'
        logging.info('-' * len(msg) + '\n' + msg + '\n' + '-' * len(msg))
        for i, (klass, kwargs) in enumerate(metrics):
            logging.info(klass.__name__, f"VERSION {i}")

            # logging.info(' > [no-cache] ', end='')
            # measure(klass, kwargs, systems, refs, cache=False)

            # logging.info(' >   [cached] ', end='')
            measure(thisdict, i, klass, kwargs, beam_sizes, systems, refs,
                    cache=True, decoding_time_dict=decoding_time_dict,
                    alpha=alpha)
        # logging.info(thisdict)
        # save results to csv
        df = pd.DataFrame.from_dict(thisdict)
    df.to_csv(f'{output_dict}/df.csv', index=False)
    return df


def task3_1(best_beam_size=13, cache=True):
    # TASK3.1: search for the best alpha given the best beam size = 13
    output_dict = f'output_alpha_beam{best_beam_size}'
    if not os.path.isdir(output_dict):
        os.mkdir(output_dict)
    df = search(beam_sizes=[best_beam_size], alphas=alphas, output_dict=output_dict, cache=cache)
    data = df[df['BLUE Version'] == 0]
    best_alpha = data[data['BLUE Score'] == data['BLUE Score'].max()]['alpha'].item()
    logging.info(f"The best alpha when beam_size = {best_beam_size}: {best_alpha}")
    return best_alpha


def task1(best_alpha=0, cache=True):
    # TASK1: search for the beam size when alpha == 0
    # TASK3.2: earch for the beam size when alpha == best_alpha
    output_dict = f'output_beamsize_alpha{best_alpha}'
    if not os.path.isdir(output_dict):
        os.mkdir(output_dict)
    df = search(beam_sizes=beam_sizes, alphas=[best_alpha], output_dict=output_dict, cache=cache)
    data = df[df['BLUE Version'] == 0]
    best_beam_size = data[data['BLUE Score'] == data['BLUE Score'].max()]['Beam Size'].item()
    logging.info(f"The best beam_size when alpha = {best_alpha}: {best_beam_size}")
    return best_beam_size


if __name__ == '__main__':
    parser = argparse.ArgumentParser('task')
    parser.add_argument('--log_file', default='./task.log', help='specify the log file')
    args = parser.parse_args()
    utils.init_logging(args)
    # task1(best_alpha=0, cache=True)
    # best_alpha = task3_1(best_beam_size=13, cache=True)
    task1(best_alpha=0.2, cache=True)

