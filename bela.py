import sys
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import sklearn.svm
import numpy as np
from xgboost import plot_tree
import sklearn.manifold

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt


assert False, "obsoleted by bela_kazarokkal.py"


def logg(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


colorblind_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']

from cycler import cycler
mpl.rcParams['axes.prop_cycle'] = cycler(color=colorblind_color_cycle) # cycler(color='bgrcmyk')


# source = "szerb.tsv"
# source = "bela.tsv"
# source = "bela_tigris.tsv"
# source = "bela_merged.tsv"
# source = "karosiak.tsv"
source = sys.argv[1]
if len(sys.argv)>2:
    np.random.seed(int(sys.argv[2]))

data = pd.read_csv(source, sep='\t')

standardize = False
if standardize:
    markers = data.iloc[:, 2:]
    markers_std = markers.std()
    data.iloc[:, 2:] = (markers - markers.mean()) / markers_std
    logg("standardized data, marker stds:")
    logg(markers_std)

label_column = "Haplogroup"
index_column = "Kit Number"

bela = data.iloc[:1]
bela = bela.drop([label_column, index_column], axis=1)
data = data.iloc[1:]

le = LabelEncoder()
data[label_column] = le.fit_transform(data[label_column])
logg(np.unique(data[label_column], return_counts=True))

n = len(data)

data = data.sample(frac=1)

train_size = n-1 # 9*n//10
train = data.iloc[:train_size]
test = data.iloc[train_size:]

logg("train on", len(train), "test on", len(test))

balance = False
if balance:
    grouped = train.groupby(label_column)
    train = grouped.apply(lambda x: x.sample(grouped.size().max(), replace=True)).reset_index(drop=True)
    print("rebalanced, now train on", len(train), "test still on", len(test))

train_X = train.drop([label_column, index_column], axis=1)
train_y = train[label_column]
test_X = test.drop([label_column, index_column], axis=1)
test_y = test[label_column]

labels = np.array(le.inverse_transform(range(np.max(train_y)+1)))


do_tsne = False
if do_tsne:
    embedder = sklearn.manifold.TSNE(n_components=2, perplexity=30, metric='l2') # l1 also makes some sense.
    # embedder = sklearn.decomposition.PCA(n_components=2, whiten=True)
    train_and_bela = np.concatenate((train_X, bela))
    reduced = embedder.fit_transform(train_and_bela)
    for i, l in enumerate(labels):
        plt.scatter(reduced[:-1][train_y==i, 0], reduced[:-1][train_y==i, 1], label=l)
    plt.scatter(reduced[-1, 0], reduced[-1, 1], marker='s', c='r', s=100, label="III. Béla")
    plt.legend()

    plt.savefig("tsne.png")
    plt.show()


# that was good for bela.tsv:
#classifier = xgb.XGBClassifier(max_depth=7, n_estimators=1000, learning_rate=0.05)
# that's tuned for bela_tigris.tsv:
classifier = xgb.XGBClassifier(max_depth=7, n_estimators=1000)
# classifier = sklearn.svm.SVC(kernel='linear', C=1000)

classifier = classifier.fit(train_X, train_y)

predictions = classifier.predict(test_X)

logg("test gold histogram", np.unique(test_y, return_counts=True))
logg("majority baseline", np.max(np.bincount(test_y)) / len(test_y))
logg("prediction histogram", np.unique(predictions, return_counts=True))

acc = float((predictions == test_y).sum()) / len(test_y)

print("acc", acc)

print("III Bela's haplogroup:", le.inverse_transform(classifier.predict(bela)[0]))

bela_y = classifier.predict_proba(bela)

logg(le.inverse_transform(range(np.max(train_y)+1)))
np.set_printoptions(suppress=True)
print("III Bela's haplogroup probs:", bela_y)

# xgb.plot_importance(classifier)
# plt.show()

vis = False
if vis:
    plot_tree(classifier, num_trees=2) # 2 is Z280
    fig = plt.gcf()
    fig.set_size_inches(75, 50)
    plt.savefig("bela.Z280.tree0.png")

