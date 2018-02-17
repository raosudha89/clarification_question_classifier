import sys
import theano, lasagne
import theano.tensor as T

def build_list_lstm(content_list, content_masks_list, N, max_len, word_embeddings, word_emb_dim, hidden_dim, len_voc, batch_size):
	out = [None]*N
	l_in = lasagne.layers.InputLayer(shape=(batch_size, max_len), input_var=content_list[0])
	l_mask = lasagne.layers.InputLayer(shape=(batch_size, max_len), input_var=content_masks_list[0])
	l_emb = lasagne.layers.EmbeddingLayer(l_in, len_voc, word_emb_dim, W=word_embeddings)
	l_lstm = lasagne.layers.LSTMLayer(l_emb, hidden_dim, mask_input=l_mask, )
	out[0] = lasagne.layers.get_output(l_lstm)
	out[0] = T.mean(out[0] * content_masks_list[0][:,:,None], axis=1)
	for i in range(1, N):
		l_in_ = lasagne.layers.InputLayer(shape=(batch_size, max_len), input_var=content_list[i])
		l_mask_ = lasagne.layers.InputLayer(shape=(batch_size, max_len), input_var=content_masks_list[i])
		l_emb_ = lasagne.layers.EmbeddingLayer(l_in_, len_voc, word_emb_dim, W=l_emb.W)
		l_lstm_ = lasagne.layers.LSTMLayer(l_emb_, hidden_dim, mask_input=l_mask_,\
											ingate=lasagne.layers.Gate(W_in=l_lstm.W_in_to_ingate,\
																		W_hid=l_lstm.W_hid_to_ingate,\
																		b=l_lstm.b_ingate,\
																		nonlinearity=l_lstm.nonlinearity_ingate),\
											outgate=lasagne.layers.Gate(W_in=l_lstm.W_in_to_outgate,\
																		W_hid=l_lstm.W_hid_to_outgate,\
																		b=l_lstm.b_outgate,\
																		nonlinearity=l_lstm.nonlinearity_outgate),\
											forgetgate=lasagne.layers.Gate(W_in=l_lstm.W_in_to_forgetgate,\
																		W_hid=l_lstm.W_hid_to_forgetgate,\
																		b=l_lstm.b_forgetgate,\
																		nonlinearity=l_lstm.nonlinearity_forgetgate),\
											cell=lasagne.layers.Gate(W_in=l_lstm.W_in_to_cell,\
																		W_hid=l_lstm.W_hid_to_cell,\
																		b=l_lstm.b_cell,\
																		nonlinearity=l_lstm.nonlinearity_cell),\
											peepholes=False,\
											)
		out[i] = lasagne.layers.get_output(l_lstm_)
		out[i] = T.mean(out[i] * content_masks_list[i][:,:,None], axis=1)
	l_emb.params[l_emb.W].remove('trainable')
	params = lasagne.layers.get_all_params(l_lstm, trainable=True)
	print 'Params in lstm: ', lasagne.layers.count_params(l_lstm)
	return out, params

def build_list_lstm_bi(content_list, content_masks_list, N, max_len, word_embeddings, word_emb_dim, hidden_dim, len_voc, batch_size):
	out = [None]*N
	l_in = lasagne.layers.InputLayer(shape=(batch_size, max_len), input_var=content_list[0])
	l_mask = lasagne.layers.InputLayer(shape=(batch_size, max_len), input_var=content_masks_list[0])
	l_emb = lasagne.layers.EmbeddingLayer(l_in, len_voc, word_emb_dim, W=word_embeddings)

	l_lstm_forward = lasagne.layers.LSTMLayer(l_emb, hidden_dim, mask_input=l_mask, )
	l_lstm_backward = lasagne.layers.LSTMLayer(l_emb, hidden_dim, mask_input=l_mask, backwards=True)
	
	l_lstm = lasagne.layers.ConcatLayer([l_lstm_forward, l_lstm_backward])

	out[0] = lasagne.layers.get_output(l_lstm)
	out[0] = T.mean(out[0] * content_masks_list[0][:,:,None], axis=1)

	for i in range(1, N):
		l_in_ = lasagne.layers.InputLayer(shape=(batch_size, max_len), input_var=content_list[i])
		l_mask_ = lasagne.layers.InputLayer(shape=(batch_size, max_len), input_var=content_masks_list[i])
		l_emb_ = lasagne.layers.EmbeddingLayer(l_in_, len_voc, word_emb_dim, W=l_emb.W)
		l_lstm_forward_ = lasagne.layers.LSTMLayer(l_emb_, hidden_dim, mask_input=l_mask_,\
											ingate=lasagne.layers.Gate(W_in=l_lstm_forward.W_in_to_ingate,\
																		W_hid=l_lstm_forward.W_hid_to_ingate,\
																		b=l_lstm_forward.b_ingate,\
																		nonlinearity=l_lstm_forward.nonlinearity_ingate),\
											outgate=lasagne.layers.Gate(W_in=l_lstm_forward.W_in_to_outgate,\
																		W_hid=l_lstm_forward.W_hid_to_outgate,\
																		b=l_lstm_forward.b_outgate,\
																		nonlinearity=l_lstm_forward.nonlinearity_outgate),\
											forgetgate=lasagne.layers.Gate(W_in=l_lstm_forward.W_in_to_forgetgate,\
																		W_hid=l_lstm_forward.W_hid_to_forgetgate,\
																		b=l_lstm_forward.b_forgetgate,\
																		nonlinearity=l_lstm_forward.nonlinearity_forgetgate),\
											cell=lasagne.layers.Gate(W_in=l_lstm_forward.W_in_to_cell,\
																		W_hid=l_lstm_forward.W_hid_to_cell,\
																		b=l_lstm_forward.b_cell,\
																		nonlinearity=l_lstm_forward.nonlinearity_cell),\
											peepholes=False,\
											)
		l_lstm_backward_ = lasagne.layers.LSTMLayer(l_emb_, hidden_dim, mask_input=l_mask_,\
											ingate=lasagne.layers.Gate(W_in=l_lstm_backward.W_in_to_ingate,\
																		W_hid=l_lstm_backward.W_hid_to_ingate,\
																		b=l_lstm_backward.b_ingate,\
																		nonlinearity=l_lstm_backward.nonlinearity_ingate),\
											outgate=lasagne.layers.Gate(W_in=l_lstm_backward.W_in_to_outgate,\
																		W_hid=l_lstm_backward.W_hid_to_outgate,\
																		b=l_lstm_backward.b_outgate,\
																		nonlinearity=l_lstm_backward.nonlinearity_outgate),\
											forgetgate=lasagne.layers.Gate(W_in=l_lstm_backward.W_in_to_forgetgate,\
																		W_hid=l_lstm_backward.W_hid_to_forgetgate,\
																		b=l_lstm_backward.b_forgetgate,\
																		nonlinearity=l_lstm_backward.nonlinearity_forgetgate),\
											cell=lasagne.layers.Gate(W_in=l_lstm_backward.W_in_to_cell,\
																		W_hid=l_lstm_backward.W_hid_to_cell,\
																		b=l_lstm_backward.b_cell,\
																		nonlinearity=l_lstm_backward.nonlinearity_cell),\
											peepholes=False,\
											backwards=True,\
											)
	
		l_lstm_ = lasagne.layers.ConcatLayer([l_lstm_forward_, l_lstm_backward_])
		out[i] = lasagne.layers.get_output(l_lstm_)
		out[i] = T.mean(out[i] * content_masks_list[i][:,:,None], axis=1)
	l_emb.params[l_emb.W].remove('trainable')
	params = lasagne.layers.get_all_params(l_lstm, trainable=True)
	print 'Params in lstm: ', lasagne.layers.count_params(l_lstm)
	return out, params

def build_lstm(posts, post_masks, max_len, word_embeddings, word_emb_dim, hidden_dim, len_voc, batch_size):

	l_in = lasagne.layers.InputLayer(shape=(batch_size, max_len), input_var=posts)
	l_mask = lasagne.layers.InputLayer(shape=(batch_size, max_len), input_var=post_masks)
	l_emb = lasagne.layers.EmbeddingLayer(l_in, len_voc, word_emb_dim, W=word_embeddings)
	l_lstm = lasagne.layers.LSTMLayer(l_emb, hidden_dim, mask_input=l_mask, )
	out = lasagne.layers.get_output(l_lstm)
	out = T.mean(out * post_masks[:,:,None], axis=1)
	l_emb.params[l_emb.W].remove('trainable')
	params = lasagne.layers.get_all_params(l_lstm, trainable=True)
	print 'Params in post_lstm: ', lasagne.layers.count_params(l_lstm)
	return out, params

def build_lstm_bi(posts, post_masks, max_len, word_embeddings, word_emb_dim, hidden_dim, len_voc, batch_size):

	l_in = lasagne.layers.InputLayer(shape=(batch_size, max_len), input_var=posts)
	l_mask = lasagne.layers.InputLayer(shape=(batch_size, max_len), input_var=post_masks)
	l_emb = lasagne.layers.EmbeddingLayer(l_in, len_voc, word_emb_dim, W=word_embeddings)

	l_lstm_forward = lasagne.layers.LSTMLayer(l_emb, hidden_dim, mask_input=l_mask, )
	l_lstm_backward = lasagne.layers.LSTMLayer(l_emb, hidden_dim, mask_input=l_mask, backwards=True)

	l_lstm = lasagne.layers.ConcatLayer([l_lstm_forward, l_lstm_backward])

	out = lasagne.layers.get_output(l_lstm)
	out = T.mean(out * post_masks[:,:,None], axis=1)
	l_emb.params[l_emb.W].remove('trainable')
	params = lasagne.layers.get_all_params(l_lstm, trainable=True)
	print 'Params in post_lstm: ', lasagne.layers.count_params(l_lstm)
	return out, params

