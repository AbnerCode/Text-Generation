#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : create_dictionary.py
# @Author: harry
# @Date  : 18-8-6 下午5:37
# @Desc  : Create dictionary.txt

from DataHelpers import *
from HyperParameter import HyperParameter

if __name__ == '__main__':
    # Load data
    hp = HyperParameter()
    sources = load_and_cut_data(hp.sources_txt)
    targets = load_and_cut_data(hp.targets_txt)

    # Create dictionary in mem
    word_dic_new = list(set([character for line in sources for character in line]))
    word_dic_new_t = list(set([character for line in targets for character in line]))

    # Save to file
    with open(hp.dictionary_sources, 'w') as f:
        f.write('\n'.join(word_dic_new))

    with open(hp.dictionary_targets, 'w') as f:
        f.write('\n'.join(word_dic_new_t))

    print("Dictionary saved in {}".format(hp.dictionary_sources))
    print("Dictionary saved in {}".format(hp.dictionary_targets))

