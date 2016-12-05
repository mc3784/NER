import numpy as np
from sklearn import cluster
import pickle
from sklearn.decomposition import PCA
from sklearn import metrics

#Starting pca:
#pca = PCA(n_components=0.7)
#ent_emb = pca.fit_transform(entity_embedding)
#print ent_emb.shape
print("starting do spectral")
entity_embedding = pickle.load(open("embedding.p", "rb"))
labels_true = pickle.load(open("true_labels.p", "rb"))
print("starting spectral clustering")

#Spectral CLustering
print("before statrting specral")
spectral = cluster.SpectralClustering(n_clusters=4, eigen_solver='arpack', n_init=1, n_jobs=-1)#, affinity="nearest_neighbors"
print("after spectral declaration")
spectral.fit(entity_embedding.astype(np.float))
print("after spectral fit")
labels_pred = spectral.labels_
print labels_pred

accuracy = metrics.adjusted_rand_score(labels_true, labels_pred)
print accuracy

pickle.dump(spectral.labels_, open("predicted_labels.p", "wb"))
#labels = pickle.load(open("labels.p", "rb"))