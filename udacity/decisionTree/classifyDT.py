def classify(features_train, labels_train):
    from sklearn import tree
    clf = tree.DecisionTreeClassifier(min_samples_split=50)
    clf = clf.fit(features_train, labels_train)
    return clf
