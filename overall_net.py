import tensorflow as tf # tensorflow module
import numpy as np # numpy module
import pandas as pd
import math
import os # path join
from heapq import merge

FEATURES_COUNT = 25
TARGET_COUNT = 1

train_data = pd.read_csv('./data/Train_reduced.csv')
val_data = pd.read_csv('./data/Validation.csv')

train_X = train_data.iloc[:,2:-3]
train_X = train_X.drop(['Team', 'DefaultPosition', 'OriginalPosition', 'Greed', 'Happiness', 'Loyalty',
	'Winner', 'DisplayHeight', 'Potential', 'KnownPotential', 'CurrentRating', 'FutureRating', 'Scoring'
	, 'OutsideScoring', 'Stamina'], axis=1)
print(train_X.columns)
train_y = train_data.iloc[:,-1:]
print(train_y.columns)
train_playername = train_data.iloc[:,1]

train_X = train_X.as_matrix()
train_y = train_y.as_matrix()
train_playername = train_playername.as_matrix()

print(train_X.shape)
print(train_y.shape)

val_X = val_data.iloc[:,2:-3]
val_X = val_X.drop(['Team', 'DefaultPosition', 'OriginalPosition', 'Greed', 'Happiness', 'Loyalty',
	'Winner', 'DisplayHeight', 'Potential', 'KnownPotential', 'CurrentRating', 'FutureRating', 'Scoring'
	, 'OutsideScoring', 'Stamina'], axis=1)
val_y = val_data.iloc[:,-1:]
val_playername = val_data.iloc[:,1]

val_X = val_X.as_matrix()
val_y = val_y.as_matrix()
val_playername = val_playername.as_matrix()

print(val_X.shape)
print(val_y.shape)

class myOnet(object):
	def __init__(self):
		self.batch_size = 10
		self.old_best = 500
		self.val_loss = 500
		self.fail_counter = 0

	def get_batch(self, trainX, trainY):
		random_int = np.random.randint(0, len(trainX))
		return trainX[random_int:int(random_int+self.batch_size)], trainY[random_int:int(random_int+self.batch_size)]

	def network(self):

		model = tf.keras.Sequential()
		model.add(tf.keras.layers.Dense(20480, input_dim=FEATURES_COUNT, activation=None))
		model.add(tf.keras.layers.Dense(10240, activation=None))
		model.add(tf.keras.layers.Dense(5120, activation=None))
		model.add(tf.keras.layers.Dense(2560, activation=None))
		model.add(tf.keras.layers.Dense(1280, activation=None))
		model.add(tf.keras.layers.Dense(640, activation=None))
		model.add(tf.keras.layers.Dense(320, activation=None))
		model.add(tf.keras.layers.Dense(160, activation=None))
		model.add(tf.keras.layers.Dense(80, activation=None))
		model.add(tf.keras.layers.Dense(1, activation=None))
		model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(3e-8), metrics=['mean_absolute_error'])
		model.summary()
		return model

	def train(self):
		try:
			my_model = tf.keras.models.load_model('./results/iter.h5')
			print("loaded chk")
		except: 
			print("couldnt load chk")
			my_model = self.network()

		loss = []
		valloss = []
		#train model for an epoch
		epoch_length = int(len(train_X)/self.batch_size)

		for epoch in range(50000):
			print(epoch)
			epoch_mean_loss = []
			if epoch != 0:
				if self.val_loss < self.old_best:
					print(str(self.old_best) + ' was the old best val_loss. ' +str(self.val_loss) + ' is the new best val loss!')
					self.old_best = self.val_loss
					my_model.save('./results/val_loss.h5', overwrite=True)
					self.fail_counter = 0
				else:
					self.fail_counter+=1
					print("consecutive val fails: " + str(self.fail_counter))
					if self.fail_counter % 10 == 0:
						print("val loss failed to improve 10 epochs in a row")
						#print("Current LR: " + str(my_model.optimizer.lr) + "reducing learning rate by 1%")
						tf.keras.backend.set_value(my_model.optimizer.lr, my_model.optimizer.lr*.90)
						print("New LR: " + str(tf.keras.backend.get_value(my_model.optimizer.lr)))

			for i in range(epoch_length):
				train_batch, target_batch = self.get_batch(train_X, train_y)
				train_history = my_model.train_on_batch(train_batch, target_batch)
				epoch_mean_loss = np.append(epoch_mean_loss, train_history[0])
			print("Epoch Loss: " + str(np.mean(epoch_mean_loss)))
			loss = np.append(loss, train_history[0])
			my_model.save('./results/iter.h5', overwrite=True)
			val_history = my_model.evaluate(val_X, val_y, batch_size=5, verbose=1)
			valloss = np.append(valloss, val_history[0])
			self.val_loss = float(val_history[0])

	def eval(self):
		my_model = tf.keras.models.load_model('./results/val_loss.h5')
		print("loaded chk")
		infer_out = my_model.predict(val_X)
		df = pd.DataFrame(val_playername)
		tX = pd.DataFrame(val_X)
		ty = pd.DataFrame(val_y)
		io = pd.DataFrame(infer_out)
		df3 = [df, ty, io]
		result = pd.concat(df3, axis=1)
		print(result)
		result.to_csv("foo.csv")

if __name__ == '__main__':

   Onet = myOnet()
   Onet.train()
   Onet.eval()