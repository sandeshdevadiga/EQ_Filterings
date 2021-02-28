#This code is from Music-Classification repo
#I dont take any credits for this. I am here just exploring my ideas.
#If you want to get this removed, please contact me at sandesh7@outlook.com


import numpy as np
import load_data
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import librosa as lb 

SR = 22050
N_FFT = 512
HOP_LENGTH = N_FFT // 2
N_MELS = 64 



data = load_data.loadall('melspects.npz')
x_tr = data['x_tr']
y_tr = data['y_tr']
x_te = data['x_te']
y_te = data['y_te']
x_cv = data['x_cv']
y_cv = data['y_cv']

print('here1', x_tr.shape)

x_tr = x_tr.reshape(x_tr.shape[0], x_tr.shape[1]*x_tr.shape[2])
x_cv = x_cv.reshape(x_cv.shape[0], x_cv.shape[1]*x_cv.shape[2])
x_te = x_te.reshape(x_te.shape[0], x_te.shape[1]*x_te.shape[2])
print("what is the shape here",x_tr.shape)
scaler = StandardScaler()
# Fit on training set only.
scaler.fit(x_tr)
# Apply transform to both the training set and the test set.
train_sc = scaler.transform(x_tr)
cv_sc = scaler.transform(x_cv)
test_sc = scaler.transform(x_te)

print('here2',test_sc.shape)

pca = PCA(n_components = 15)
pca.fit(train_sc)

train_pca = pca.transform(train_sc)
cv_pca = pca.transform(cv_sc)
test_pca = pca.transform(test_sc)
print("here3");
print ("shape after pca",test_pca.shape);
print("here3");
print(pca.n_components_)

classifier = svm.SVC(gamma='scale', verbose=True)
classifier.fit(train_pca, y_tr)

# preds = classifier.predict(cv_pca)
# acc = np.sum(preds == y_cv)
# acc = acc / len(y_cv)
# print('Accuracy is {}'.format(acc))
# print(preds)

train_preds = classifier.predict(train_pca)
train_acc = np.sum(train_preds == y_tr)
train_acc = train_acc / len(y_tr)

cv_preds = classifier.predict(cv_pca)
cv_acc = np.sum(cv_preds == y_cv)
cv_acc = cv_acc / len(y_cv)

test_preds = classifier.predict(test_pca)
test_acc = np.sum(test_preds == y_te)
test_acc = test_acc / len(y_te)
scale_file = "debussy2ms.wav"
scale, sr = lb.load(scale_file)

#S_scale = librosa.stft(scale, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)

mel_spectrogram_TestSong = lb.feature.melspectrogram(scale, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
#trained = scaler.transform(mel_spectrogram_TestSong)

melspectrogram_TestSong=lb.power_to_db(mel_spectrogram_TestSong**2)
melspectrogram_TestSong=melspectrogram_TestSong.reshape(1, melspectrogram_TestSong.shape[0]*melspectrogram_TestSong.shape [1]);

print("Shape that i pass to predicter",melspectrogram_TestSong.shape);                                
scaler1 = StandardScaler()
scaler1.fit(melspectrogram_TestSong)
melspectrogram_TestSong = scaler1.transform(melspectrogram_TestSong)

#pca2 = PCA(n_components = 1)
#pca2.fit(melspectrogram_TestSong)
#mel_spectrogram_TestSong = pca2.transform(melspectrogram_TestSong)

print("after pca",melspectrogram_TestSong.shape);

melspectrogram_TestSong.shape;
melspectrogram_TestSong=melspectrogram_TestSong[:,0:15]
print("new share",melspectrogram_TestSong.shape);
test_preds = classifier.predict(melspectrogram_TestSong)
print (test_preds);

##################################################################3
##################################################################3
##################################################################

#
#import numpy as np
#import load_data
#from sklearn import svm
#from sklearn.preprocessing import StandardScaler
#from sklearn.decomposition import PCA
#import librosa as lb 
#
#SR = 22050
#N_FFT = 512
#HOP_LENGTH = N_FFT // 2
#N_MELS = 64 
#
#
#
#data = load_data.loadall('../melspects.npz')
#x_tr = data['x_tr']
#y_tr = data['y_tr']
#x_te = data['x_te']
#y_te = data['y_te']
#x_cv = data['x_cv']
#y_cv = data['y_cv']
#
#print('here1', x_tr.shape)
#
#x_tr = x_tr.reshape(x_tr.shape[0], x_tr.shape[1]*x_tr.shape[2])
#x_cv = x_cv.reshape(x_cv.shape[0], x_cv.shape[1]*x_cv.shape[2])
#x_te = x_te.reshape(x_te.shape[0], x_te.shape[1]*x_te.shape[2])
#print("what is the shape here",x_tr.shape)
#scaler = StandardScaler()
## Fit on training set only.
#scaler.fit(x_tr)
## Apply transform to both the training set and the test set.
#train_sc = scaler.transform(x_tr)
#cv_sc = scaler.transform(x_cv)
#test_sc = scaler.transform(x_te)
#
#print('here2',test_sc.shape)
#
#pca = PCA(n_components = 15)
#pca.fit(train_sc)
#
#train_pca = train_sc
#cv_pca = cv_sc
#test_pca = test_sc
#print("here3");
#print ("shape after pca",test_pca.shape);
#print("here3");
#print(pca.n_components_)
#
#classifier = svm.SVC(gamma='scale', verbose=True)
#classifier.fit(train_pca, y_tr)
#
## preds = classifier.predict(cv_pca)
## acc = np.sum(preds == y_cv)
## acc = acc / len(y_cv)
## print('Accuracy is {}'.format(acc))
## print(preds)
#
#train_preds = classifier.predict(train_pca)
#train_acc = np.sum(train_preds == y_tr)
#train_acc = train_acc / len(y_tr)
#
#cv_preds = classifier.predict(cv_pca)
#cv_acc = np.sum(cv_preds == y_cv)
#cv_acc = cv_acc / len(y_cv)
#
#test_preds = classifier.predict(test_pca)
#test_acc = np.sum(test_preds == y_te)
#test_acc = test_acc / len(y_te)
#scale_file = "debussy2ms.wav"
#scale, sr = lb.load(scale_file)
#
##S_scale = librosa.stft(scale, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
#
#mel_spectrogram_TestSong = lb.feature.melspectrogram(scale, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
##trained = scaler.transform(mel_spectrogram_TestSong)
#
#melspectrogram_TestSong=lb.power_to_db(mel_spectrogram_TestSong**2)
#melspectrogram_TestSong=melspectrogram_TestSong.reshape(1, melspectrogram_TestSong.shape[0]*melspectrogram_TestSong.shape [1]);
#
#print("Shape that i pass to predicter",melspectrogram_TestSong.shape);                                
#scaler1 = StandardScaler()
#scaler1.fit(melspectrogram_TestSong)
#melspectrogram_TestSong = scaler1.transform(melspectrogram_TestSong)
#
##pca2 = PCA(n_components = 1)
##pca2.fit(melspectrogram_TestSong)
##mel_spectrogram_TestSong = pca2.transform(melspectrogram_TestSong)
#
#print("after pca",melspectrogram_TestSong.shape);
#
#melspectrogram_TestSong.shape;
##melspectrogram_TestSong=melspectrogram_TestSong[:,1:15]
#test_preds = classifier.predict(melspectrogram_TestSong)
#print (test_preds);
