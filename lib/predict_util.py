#-*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import re
from lib import train_utils
from konlpy.tag import Mecab

class CAM2(object):

	def __init__(self, config):

		tf.reset_default_graph()
		self.sess = tf.Session()
		self.config = config
		self.is_training = False
		self.tokenizer = Mecab()
		self.model, finW = train_utils.create_or_load_model(self.sess, self.config)
		self.pos_w = finW[:, 0]
		self.neg_w = finW[:, 1]
		vocab = open(self.config.vocab_path, mode="rt", encoding="utf-8").readlines()
		self.vocab_dic = {}
		for voca in vocab:
			word, index = voca.replace("\n", "").split("\u241E")
			self.vocab_dic[word] = index

	def get_test_batch(self, features, max_document_length):
		batch_features = np.zeros((1, max_document_length), dtype=int)
		for idx, id in enumerate(features):
			if idx < max_document_length:
				batch_features[0, idx] = int(id)
		return batch_features

	def CAM2(self, tokens, actmaps, predictions, size=1):
		results = []

		for batch_idx in range(size):
			combined_actmap_trigram = \
				actmaps[0][batch_idx].reshape(
					((self.config.sequence_length + 2), self.config.num_filters))
			combined_actmap_quadgram = \
				actmaps[1][batch_idx].reshape(
					((self.config.sequence_length + 3), self.config.num_filters))
			combined_actmap_5gram = \
				actmaps[2][batch_idx].reshape(
					((self.config.sequence_length + 4), self.config.num_filters))

			if predictions[batch_idx] == 0:
				trigram_scores = np.dot(combined_actmap_trigram, self.pos_w[:self.config.num_filters])
				quadgram_scores = np.dot(combined_actmap_quadgram, self.pos_w[self.config.num_filters:2 * (self.config.num_filters)])
				fivegram_scores = np.dot(combined_actmap_5gram, self.pos_w[2 * (self.config.num_filters):])
			else:
				trigram_scores = np.dot(combined_actmap_trigram, self.neg_w[:self.config.num_filters])
				quadgram_scores = np.dot(combined_actmap_quadgram, self.neg_w[self.config.num_filters:2 * (self.config.num_filters)])
				fivegram_scores = np.dot(combined_actmap_5gram, self.neg_w[2 * (self.config.num_filters):])

			mean_trigram_scores = []
			mean_quadgram_scores = []
			mean_fivegram_scores = []

			for tri_idx in range(len(trigram_scores)):
				if tri_idx + 3 < len(trigram_scores) + 1:
					mean_trigram_scores.append(np.mean(trigram_scores[tri_idx:tri_idx + 3]))

			for quad_idx in range(len(quadgram_scores)):
				if quad_idx + 4 < len(quadgram_scores) + 1:
					mean_quadgram_scores.append(np.mean(quadgram_scores[quad_idx:quad_idx + 4]))

			for five_idx in range(len(fivegram_scores)):
				if five_idx + 5 < len(fivegram_scores) + 1:
					mean_fivegram_scores.append(np.mean(fivegram_scores[five_idx:five_idx + 5]))

			mean_scores = np.array(mean_trigram_scores) + np.array(mean_quadgram_scores) + np.array(mean_fivegram_scores)

			fin_result = []
			for word_idx, score in enumerate(mean_scores):
				if word_idx < len(tokens):
					fin_result.append([tokens[word_idx], score])
				else:
					fin_result.append([word_idx, score])
			fin_result = fin_result[:min(len(tokens), self.config.sequence_length)]

			preinfo = predictions[batch_idx]
			results.append([preinfo, fin_result])
			return results

	def visualize(self, x):
		re = '<p>'
		predict, scores = x
		words, values = [], []
		for el in scores:
			word, value = el
			words.append(word)
			values.append(value)
		values_idx = np.argsort(values)
		toSelect = round(len(words) * self.config.ratio)
		if toSelect == 0: toSelect = 1
		toSelect_idx = values_idx[-toSelect:]
		for i in range(len(words)):
			if (i in toSelect_idx and predict == 1):
				re += '<span class="label label-danger">  ' + words[i] + '  </span>'
			elif (i in toSelect_idx and predict == 0):
				re += '<span class="label label-success">  ' + words[i] + '  </span>'
			else:
				re += '<span>  ' + words[i] + '  </span>'

		if predict == 0:
			re += '<span class="label label-info"> Positive </span>'
		else:
			re += '<span class="label label-info"> Negative </span>'

		re += '</p>'
		return re

	def get_scores(self, string):
		sentence = re.sub(self.config.pattern, " ", string)
		tokens = self.tokenizer.morphs(sentence)
		token_ids = []
		fake_output = np.zeros((1, 2))
		for token in tokens:
			token_ids.append(self.vocab_dic.get(token, self.config.UNK_ID))
		input = self.get_test_batch(token_ids, self.config.sequence_length)
		actmaps, predictions = self.model.step(self.sess, self.config, input, fake_output, self.is_training)
		cam_results = self.CAM2(tokens, actmaps, predictions)
		return cam_results

	def get_visualized_scores(self, string):
		cam_results = self.get_scores(string)
		return self.visualize(cam_results[0])
