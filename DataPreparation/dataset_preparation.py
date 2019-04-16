# -*- coding: utf-8 -*-
'''
Copyright(c) 2019, Soroush Saryazdi
All rights reserved.
2019/02/02
'''
import numpy as np
import pandas as pd
import dask.dataframe as dd
import os
from random import randint, choice

def get_LANL_dataset(data_dir, validation_split=0.2, numpy=False, normalize=False):
	'''
	Reads LANL dataset from data_dir, then splits the training data based on
	the validation_split ratio and also the optional seed. Then it proceeds to normalize the
	data using the mean and standard deviation of the training set (X_train). If we're using
	data augmentation on X_train, we might want to normalize X_train after the augmentation
	and not before: normalize_train flag allows us to do that.

	Inputs:
	- data_dir: should contain trainset and testset folders
	OR train_dataset.npy, train_labels.npy and X_test.npy.
	- validation_split: Determines how to split the train_dataset into train and validation
	sets. validation_split is the ratio of X_val size to the entire train_dataset.
	- seed: (Optional) Performs the random split of training and validation set using this
	seed.
	- normalize_tarin: We might want to normalize X_train after the augmentation
	and not before: normalize_train flag allows us to do that.

	Returns:
	- X_train: Training data.
	- y_train: Training data labels.
	- X_val: Validation data.
	- y_val: Validation data labels.
	- X_test: Test data.
	- X_train_moments: Tuple of mean and standard deviation of X_train. (Used for
	visualizing original samples)
	'''
	assert validation_split < 1
	data = {}
	X_train_moments = None

	try: # Try to load numpy dataset
		train_dir = data_dir + 'trainset/'
		test_dir = data_dir + 'testset/'
		train_file_names = os.listdir(f'{train_dir}')
		if (f'train_split_{validation_split}.csv' not in train_file_names) or (f'validation_split_{validation_split}.csv' not in train_file_names):
			print('train_split.csv & validation_split.csv were not found')
			print('Trying to build them from train.csv...')
			print('Warning! This could take a while...')
			traindata_df = pd.read_csv(f'{train_dir}train.csv')
			len_total = traindata_df.acoustic_data.count()
			len_train = int((1-validation_split) * len_total)
			traindata_df[:len_train].to_csv(f'{train_dir}train_split_{validation_split}.csv', index=False)
			traindata_df[len_train:].to_csv(f'{train_dir}validation_split_{validation_split}.csv', index=False)

		train_df = dd.read_csv(f'{train_dir}train_split_{validation_split}.csv')
		val_df = dd.read_csv(f'{train_dir}validation_split_{validation_split}.csv')

		test_file_names = os.listdir(f'{test_dir}')
		test_dict = {}
		for test_file_name in test_file_names:
			test_df = dd.read_csv(f'{test_dir}{test_file_name}')		    	
			test_dict.update({test_file_name[:-4]:test_df})

	except EnvironmentError: # If numpy dataset doesn't exist, convert jpg data to numpy
		raise ValueError('Could not find LANL dataset in: "' + data_dir + '"')
	
	print('Dask dataframes loaded.')
	if normalize:
		print('Preprocessing...')
		X_train_mean = train_df['acoustic_data'].mean().compute()
		X_train_std = train_df['acoustic_data'].std().compute()

		train_df['acoustic_data'] -= X_train_mean
		train_df['acoustic_data'] /= X_train_std

		val_df['acoustic_data'] -= X_train_mean
		val_df['acoustic_data'] /= X_train_std

		for seg_id, test_df in test_dict.items():
			test_df['acoustic_data'] -= X_train_mean
			test_df['acoustic_data'] /= X_train_std
			test_dict.update({seg_id: test_df})

		X_train_moments = (X_train_mean, X_train_std)

	if numpy:
		print('Converting to numpy...')
		X_train = train_df.acoustic_data.values.compute()
		y_train = train_df.time_to_failure.values.compute()

		X_val = val_df.acoustic_data.values.compute()
		y_val = val_df.time_to_failure.values.compute()

		for seg_id, test_df in test_dict.items():
			test_dict.update({seg_id: test_df.acoustic_data.values.compute()})

		data = {'X_train':X_train, 'y_train':y_train,
		'X_val':X_val, 'y_val':y_val, 'test_dict':test_dict}
	else:
		data = {'train_df':train_df, 'val_df':val_df, 'test_dict':test_dict}
	
	print('Done.')
	return data, X_train_moments




def get_LANL_data(data_dir, method='sample', n=5000, normalize=True):
	'''
	Reads LANL dataset from data_dir, then splits the training data based on
	the validation_split ratio and also the optional seed. Then it proceeds to normalize the
	data using the mean and standard deviation of the training set (X_train). If we're using
	data augmentation on X_train, we might want to normalize X_train after the augmentation
	and not before: normalize_train flag allows us to do that.

	Inputs:
	- data_dir: should contain trainset and testset folders
	OR train_dataset.npy, train_labels.npy and X_test.npy.
	- validation_split: Determines how to split the train_dataset into train and validation
	sets. validation_split is the ratio of X_val size to the entire train_dataset.
	- seed: (Optional) Performs the random split of training and validation set using this
	seed.
	- normalize_tarin: We might want to normalize X_train after the augmentation
	and not before: normalize_train flag allows us to do that.

	Returns:
	- X_train: Training data.
	- y_train: Training data labels.
	- X_val: Validation data.
	- y_val: Validation data labels.
	- X_test: Test data.
	- X_train_moments: Tuple of mean and standard deviation of X_train. (Used for
	visualizing original samples)
	'''
	X_train_moments = None

	print('Reading data...')
	try: # Try to load numpy dataset
		train_dir = data_dir + 'trainset/'
		test_dir = data_dir + 'testset/'
		train_file_names = os.listdir(f'{train_dir}')

		train_df = pd.read_csv(f'{train_dir}train.csv')
		X_merged = train_df['acoustic_data'].values
		y_merged = train_df['time_to_failure'].values

		test_file_names = os.listdir(f'{test_dir}')
		X_test_dict = {}
		for test_file_name in test_file_names:
			X_test_i = pd.read_csv(f'{test_dir}{test_file_name}').values
			X_test_dict.update({test_file_name[:-4]:X_test_i})

	except EnvironmentError: # If numpy dataset doesn't exist, convert jpg data to numpy
		raise ValueError('Could not find LANL dataset in: "' + data_dir + '"')

	seq_len = 150000
	earthquake_ind = [0, 5656574, 50085878, 104677356,
					  138772453, 187641820, 218652630, 245829585,
					  307838917, 338276287, 375377848, 419368880,
					  461811623, 495800225, 528777115, 585568144,
					  621985673]
	earthquake_ind = np.asarray(earthquake_ind)
	len_train_total = 629145480

	X_train = X_merged[:earthquake_ind[-4]]
	X_val = X_merged[earthquake_ind[-4]:]
	y_train = y_merged[:earthquake_ind[-4]]
	y_val = y_merged[earthquake_ind[-4]:]

	X_train_mean = np.mean(X_train)
	X_train_std = np.std(X_train)
	print('Splitting & Sampling...')

	if method == 'random':
		val_eq_ind = np.hstack([earthquake_ind[-4:], len_train_total])
		val_eq_ind -= np.min(val_eq_ind)
		val_slicing_ranges = [(val_eq_ind[i], (val_eq_ind[i+1]-seq_len)) for i in range(len(val_eq_ind)-1)]
		val_sample_size = 1000 # 838
		val_sample_ind = uniform_from_slices(val_slicing_ranges, val_sample_size)
		X_val, y_val = sample_sequences(X_val, y_val, val_sample_ind, seq_len)
		X_val = X_val[:,:,None]

		train_eq_ind = earthquake_ind[:-4]
		train_slicing_ranges = [(train_eq_ind[i], (train_eq_ind[i+1]-seq_len)) for i in range(len(train_eq_ind)-1)]
		train_sample_size = n # 3355
		train_sample_ind = uniform_from_slices(train_slicing_ranges, train_sample_size)
		X_train, y_train = sample_sequences(X_train, y_train, train_sample_ind, seq_len)
		X_train = X_train[:,:,None]

	elif method == 'slicing':
		val_eq_ind = np.hstack([earthquake_ind[-4:], len_train_total])
		val_eq_ind -= np.min(val_eq_ind)
		val_slicing_ranges = [(val_eq_ind[i], (val_eq_ind[i+1]-seq_len)) for i in range(len(val_eq_ind)-1)]
		val_sample_size = 1000 # 838
		val_sample_ind = uniform_from_slices(val_slicing_ranges, val_sample_size)
		X_val, y_val = sample_sequences(X_val, y_val, val_sample_ind, seq_len)
		X_val = X_val[:,:,None]
	
		X_train = X_train[:int(np.floor(X_train.shape[0] / seq_len))*seq_len]
		y_train = y_train[:int(np.floor(y_train.shape[0] / seq_len))*seq_len]
		X_train= X_train.reshape((-1, seq_len, 1))
		y_train = y_train[seq_len-1::seq_len]
	
	elif method == 'raw':
		pass

	else:
		raise ValueError('Uknown method: %s' % method)

	if normalize:
		print('Standardizing...')
		X_train = X_train.astype(float)
		X_val = X_val.astype(float)

		X_train -= X_train_mean
		X_train /= X_train_std

		X_val -= X_train_mean
		X_val /= X_train_std

		for seg_id, X_test_i in X_test_dict.items():
			X_test_i = X_test_i.astype(float)
			X_test_i -= X_train_mean
			X_test_i /= X_train_std
			X_test_dict.update({seg_id: X_test_i})

		X_train_moments = (X_train_mean, X_train_std)

	print('[DONE]')
	return X_train, y_train, X_val, y_val, X_test_dict, X_train_moments

diff = lambda l: np.array([(l[i][1]-l[i][0]) for i in range(len(l))], dtype=float)

def uniform_from_slices(slices, n):
	out = []
	for _ in range(n):
		prob = diff(slices)
		prob /= np.sum(prob)
		ind = np.random.choice(len(slices), p=prob)
		r = slices[ind]
		out.append(randint(*r))
	return out

def sample_sequences(x, y, indices, seq_len):
	num_samples = len(indices)
	x_ = np.zeros((num_samples, seq_len))
	y_ = np.zeros(num_samples)
	for i, ind in enumerate(indices):
		x_[i, :] = x[ind:ind+seq_len]
		y_[i] = y[ind+seq_len-1]
	return x_, y_