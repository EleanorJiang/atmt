#!/usr/bin/env python
import logging
import time, os, pickle, argparse
import pandas as pd
from seq2seq import utils
import collections
from distinct_n import distinct_n_sentence_level

gammas = [0, 0.2, 0.4, 0.6, 0.8, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35]

def get_translation_output(beam_sizes, cache=True, alphas=[0], gammas=[0],
                           output_dict=None):
    decoding_time_dict = {}
    for gamma in gammas:
        for beam_size in beam_sizes:
            for alpha in alphas:
                if cache and os.path.isfile(f'{output_dict}/model_translations_beam-{beam_size}_alpha-{alpha}.p.txt'):
                    continue
                st = time.time()
                os.system(f'python translate_beam_N_best.py '
                          f'--data data/en-fr/prepared -'
                          f'-dicts data/en-fr/prepared  '
                          f'--checkpoint-path assignments/03/baseline/checkpoints/checkpoint_best.pt '
                          f'--batch-size 500 '
                          f'--alpha {alpha} '
                          f'--beam-size {beam_size} '
                          f'--gamma {gamma} --N 3 '
                          f'--output {output_dict}/model_translations_beam-{beam_size}_alpha-{alpha}_gamma-{gamma}.txt '
                          f'--cuda True')
                decoding_time = time.time() - st
                logging.info(f"the decoding time of beam size {beam_size} alpha {alpha} gamma{gamma}: {decoding_time}")
                decoding_time_dict[f"{beam_size}-{alpha}"] = decoding_time
                os.system(f'./scripts/postprocess.sh '
                          f'{output_dict}/model_translations_beam-{beam_size}_alpha-{alpha}_gamma-{gamma}.txt '
                          f'{output_dict}/model_translations_beam-{beam_size}_alpha-{alpha}_gamma-{gamma}.p.txt en ')
                os.remove(f'{output_dict}/model_translations_beam-{beam_size}_alpha-{alpha}_gamma-{gamma}.txt')
    if not cache:
        pickle.dump(decoding_time_dict, open(f"{output_dict}/decoding_time_dict.pkl", "wb"))
    return decoding_time_dict

def distinct_N(output_dict, beam_size=13, alpha=0.2, N=3):
    # read to file
    thisdict = collections.defaultdict(list)
    for gamma in gammas:
        all_scores = collections.defaultdict(list)
        out_file = f'{output_dict}/model_translations_beam-{beam_size}_alpha-{alpha}_gamma-{gamma}.p.txt'
        with open(out_file, 'r') as f:
            hypothesis = [sentence.split() for sentence in f.readlines()]

        for i in range(int(len(hypothesis)/N)):
            tmp = []
            for ii in range(N):
                tmp += hypothesis[i+ii]
            all_scores[1].append(distinct_n_sentence_level(tmp, 1))
            all_scores[2].append(distinct_n_sentence_level(tmp, 2))
        thisdict['gamma'].append(gamma)
        thisdict['distinct1'].append(sum(all_scores[1]) / len(all_scores[1]))
        thisdict['distinct2'].append(sum(all_scores[2]) / len(all_scores[2]))
    df = pd.DataFrame.from_dict(thisdict)
    df.to_csv(f'{output_dict}/gamma_df.csv', index=False, float_format='%.3f')
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser('task')
    parser.add_argument('--log_file', default='./task4.log', help='specify the log file')
    parser.add_argument('--output_dict', default='output_diverse', help='output_dict')
    args = parser.parse_args()
    utils.init_logging(args)
    output_dict = args.output_dict
    if not os.path.isdir(output_dict):
        os.mkdir(output_dict)
    # get_translation_output([13], cache=True, alphas=[0.2], gammas=gammas,
    #                       output_dict=output_dict)
    distinct_N(output_dict)
