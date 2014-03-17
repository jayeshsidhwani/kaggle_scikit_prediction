import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC

class Estimator:

    def __init__(self):
        self.train_data = np.asarray(pd.read_csv('data/train.csv', header=None))
        self.train_target = np.asarray(pd.read_csv('data/trainLabels.csv', header=None)).ravel()
        self.test_data = np.asarray(pd.read_csv('data/test.csv', header=None))

    def decompose(self):
        pca_for_components = PCA()
        pca_for_components.fit(self.train_data)
        # A initial pca on the dataset calculates the variance ratio for different number of components.
        # Cumsum is used to find the number of components that has 80 percentile variance
        # 80 is a choice. More would being in noise, less would bring invalid results. Optimum -- you try and figure out!
        components = np.where(np.cumsum(pca_for_components.explained_variance_ratio_) >= 0.80)[0][0]
        print 'Components used: %s' %components

        pca_for_decomposition = PCA(n_components=components)
        self.transformed_train_data = pca_for_decomposition.fit(self.train_data).transform(self.train_data)
        self.transformed_test_data = pca_for_decomposition.fit(self.test_data).transform(self.test_data)

        return self

    def scale(self):
        self.transformed_train_data = StandardScaler().fit_transform(self.transformed_train_data)
        return self

    def svm_classifier(self):
        return SVC(probability=True, gamma=0.03)

    def classify(self, classifier, data=None, target=None):
        classifier = getattr(self, classifier)()
        if data is not None: data = self.transformed_train_data
        if target is not None: target = self.train_target

        classifier.fit(data, target)
        self.classifier = classifier
        return self

    def predict(self, input=None):
        if not input: input = self.transformed_test_data
        return self.classifier.predict(input)

    def test(self):
        train_data, test_data, train_target, test_target = train_test_split(self.transformed_train_data, self.train_target, test_size=0.4, random_state=0)
        self.classify("svm_classifier", train_data, train_target)
        # import code; code.interact(local=locals())
        print self.classifier.score(test_data, test_target)



if __name__ == '__main__':
    Estimator().decompose().scale().test()
    # svm_estimator = Estimator().decompose().scale().classify('svm_classifier')
    # estimator_result = svm_estimator.predict()
    # pd.DataFrame(dict(Id = np.arange(1, estimator_result.shape[0]+1), Solution=estimator_result)).\
    #     to_csv('data/submission.csv', header=True, index=None)