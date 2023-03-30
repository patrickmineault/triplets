import numpy as np
import pandas as pd
from scipy.stats import multinomial

in_files = ['data/raw/testset1.txt', 
            'data/raw/testset2.txt',
            'data/raw/testset2_repeat.txt',
            'data/raw/testset3.txt']

out_files = ['data/interim/testset1.csv',
             'data/interim/testset2.csv',
             'data/interim/testset2_repeat.csv',
             'data/interim/testset3.csv']

def upper_bound(df):
    # Evaluate best possible answer combo.
    nboot = 100
    P = df[['im0_chosen', 'im1_chosen', 'im2_chosen']].values
    N_total = P.sum(axis=1)
    P = P / P.sum(axis=1, keepdims=True)

    samples = []
    for i in range(1000):
        samples.append(np.argmax(multinomial.rvs(N_total[i], P[i, :], size=100), axis=1))
    the_ceil = (df.best_answer.values.reshape((-1, 1)) == np.array(samples)).mean(axis=0)
    return(the_ceil.mean(), the_ceil.std())

if __name__ == '__main__':
    things = pd.read_csv('data/raw/things_concepts.csv', sep='\t')
    things.set_index(things.index + 1, inplace=True)
    num_to_word = {i: row['Word'] for i, row in things.iterrows()}

    for in_file, out_file in zip(in_files, out_files):
        df = pd.read_csv(in_file, sep=' ', header=None)
        assert df.min().min() == 0
        df = df + 1
        df.columns = ['c0', 'c1', 'c2']
        df['triplet'] = df.apply(lambda x: tuple(sorted(x)), axis=1)
        df['im0_chosen'] = df.triplet.map(lambda x: x[0]) == df.c2
        df['im1_chosen'] = df.triplet.map(lambda x: x[1]) == df.c2
        df['im2_chosen'] = df.triplet.map(lambda x: x[2]) == df.c2

        df = df.groupby('triplet').sum()
        df['word0'] = df.index.map(lambda x: num_to_word[x[0]])
        df['word1'] = df.index.map(lambda x: num_to_word[x[1]])
        df['word2'] = df.index.map(lambda x: num_to_word[x[2]])

        df['best_answer'] = df[['im0_chosen', 'im1_chosen', 'im2_chosen']].idxmax(axis=1).map(lambda x: int(x[2]))
        df.to_csv(out_file, index=False)

        mean, sd = upper_bound(df)
        print(f"{in_file}: {mean:.3f} +/- {sd:.3f}")
