import numpy as np
import random
import tensorflow as tf
from tensorflow.python.platform import gfile
from lib import models as models

def read_train_set(train_data_path):
	raw_data = open(train_data_path, 'r', encoding='utf-8').readlines()
	feature_set = []
	label_set = []
	for line in raw_data:
		ids, label = line.split('\u241E')
		feature_set.append(ids.strip())
		label_set.append(label.replace('\n',''))
	idx = random.sample(range(len(feature_set)), len(label_set))
	return np.array(feature_set)[idx], np.array(label_set)[idx]

def get_batch(features, labels, batch_size, num_epochs, max_document_length):
	data_size = len(labels)
	num_batches_per_epoch = int((len(labels) - 1) / batch_size) + 1
	for epoch in range(num_epochs):
		for batch_num in range(num_batches_per_epoch):
			start_index = batch_num * batch_size
			end_index = min((batch_num + 1) * batch_size, data_size)
			batch_features_raw = features[start_index:end_index]
			batch_features = np.zeros((end_index-start_index, max_document_length), dtype=int)
			for i, batch_feature_raw in enumerate(batch_features_raw):
				splited_batch_feature_raw = batch_feature_raw.split()
				current_max_length = min(len(splited_batch_feature_raw), max_document_length)
				for j in range(current_max_length):
					batch_features[i,j] = int(splited_batch_feature_raw[j])
			batch_labels_raw = labels[start_index:end_index]
			batch_labels = []
			for batch_label_raw in batch_labels_raw:
				if float(batch_label_raw) > 0.5:
					batch_labels.append([1, 0])
				else:
					batch_labels.append([0, 1])
			yield batch_features, np.array(batch_labels)

def create_or_load_model(session, config):
	model = models.TextCNN(config=config)
	checkpoint = tf.train.get_checkpoint_state(config.checkpoint_path)
	finW = 0
	if checkpoint:
		checkpoint_path = checkpoint.model_checkpoint_path
		if gfile.Exists("%s.index" % checkpoint.model_checkpoint_path):
			print("reading model parameters from %s" % checkpoint_path)
			model.saver.restore(session, checkpoint_path)
			finW = session.run(model.finW)
		else:
			print("checkpoint file does not exists. create new model")
			session.run(tf.global_variables_initializer())
	else:
		print("created model with new parameters.")
		session.run(tf.global_variables_initializer())
	return model, finW