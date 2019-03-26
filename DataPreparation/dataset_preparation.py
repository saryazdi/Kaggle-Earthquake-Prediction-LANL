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