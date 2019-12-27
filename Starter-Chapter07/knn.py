import sys
sys.path.append("../")
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pyimagesearch.preprocessing.simplepreprocessor import SimplePreprocessor
from pyimagesearch.datasets.simpledatasetloader import SimpleDatasetLoader
from imutils import paths
import argparse

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="path to input dataset")
ap.add_argument("-k", "--neighbors", type=int, default=1, help="# nearest neighbors for classification")
ap.add_argument("-j", "--jobs", type=int, default=-1, help="# of jobs for k-NN distance (-1 uses all available cores)")

args = vars(ap.parse_args())

#List of images
imagePaths = list(paths.list_images(args["dataset"]))

# initialize the image preprocessor, load the dataset from disk, and reshape the data matrix
sp = SimplePreprocessor(32, 32)
sdl = SimpleDatasetLoader(preprocessors=[sp])

print("[INFO] loading data......")
(data, labels) = sdl.load(imagePaths, verbose=250)
print("[INFO] data shape {}".format(data.shape))
data = data.reshape((data.shape[0], 3072))

#memory taken by images
print("[INFO] features matrix: {:.1f}MB".format(data.nbytes / (1024 * 1000.0)))

#Encoding the labes as integers
le = LabelEncoder()
labels = le.fit_transform(labels)

#partition the data into train and test set 75% as train and rest as test
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

#training and evaluatin knn classifier
print("[INFO] evaluating knn classifier....")

model = KNeighborsClassifier(n_neighbors=args['neighbors'], n_jobs=args["jobs"])
model.fit(trainX, trainY)
print(classification_report(testY, model.predict(testX), target_names=le.classes_))