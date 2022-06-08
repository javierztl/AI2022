import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('TkAgg');

if __name__ == '__main__':
	gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
	cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
	print(gpus, cpus)
	gpuid = 0
	tf.config.experimental.set_visible_devices(devices=gpus[gpuid], device_type='GPU')
	tf.config.experimental.set_virtual_device_configuration(
		gpus[gpuid],
		[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8000)]
	)
	class CircleLoss(tf.keras.losses.Loss):
		def __init__(self,
								 gamma: int = 64,
								 margin: float = 0.25,
								 batch_size: int = None,
								 reduction='auto',
								 name=None):
			super().__init__(reduction=reduction, name=name)
			self.gamma = gamma
			self.margin = margin
			self.O_p = 1 + self.margin
			self.O_n = -self.margin
			self.Delta_p = 1 - self.margin
			self.Delta_n = self.margin
			if batch_size:
				self.batch_size = batch_size
				self.batch_idxs = tf.expand_dims(
						tf.range(0, batch_size, dtype=tf.int32), 1)	# shape [batch,1]

		def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
			""" NOTE : y_pred must be cos similarity
			Args:
					y_true (tf.Tensor): shape [batch,ndim]
					y_pred (tf.Tensor): shape [batch,ndim]
			Returns:
					tf.Tensor: loss
			"""
			alpha_p = tf.nn.relu(self.O_p - tf.stop_gradient(y_pred))
			alpha_n = tf.nn.relu(tf.stop_gradient(y_pred) - self.O_n)
			# yapf: disable
			y_true = tf.cast(y_true, tf.float32)
			y_pred = (y_true * (alpha_p * (y_pred - self.Delta_p)) +
								(1 - y_true) * (alpha_n * (y_pred - self.Delta_n))) * self.gamma
			# yapf: enable
			return tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)
	
	def non_local_block(tensor_input, channel_denominator = 2):
		## theta-phi branch
		theta = tf.keras.layers.Conv2D(filters = int(tensor_input.shape[3] / channel_denominator), kernel_size = (1, 1))(tensor_input);
		theta = tf.keras.layers.Reshape((theta.shape[1] * theta.shape[2], theta.shape[3]))(theta);
		
		phi = tf.keras.layers.Conv2D(filters = int(tensor_input.shape[3] / channel_denominator), kernel_size = (1, 1))(tensor_input);
		phi = tf.keras.layers.Reshape((phi.shape[1] * phi.shape[2], phi.shape[3]))(phi);
		
		f = tf.keras.layers.dot([theta, phi], axes = 2);
		f = tf.keras.layers.Softmax(axis=1)(f);
		
		## g branch
		g = tf.keras.layers.Conv2D(filters = int(tensor_input.shape[3] / channel_denominator), kernel_size = (1, 1))(tensor_input);
		g = tf.keras.layers.Reshape((g.shape[1] * g.shape[2], g.shape[3]))(g);
		
		## weighted_matrix
		y = tf.keras.layers.dot([f, g], axes = [2, 1]);
		y = tf.keras.layers.Reshape((tensor_input.shape[1], tensor_input.shape[2], int(tensor_input.shape[3] / channel_denominator)))(y);
		y = tf.keras.layers.Conv2D(filters = tensor_input.shape[3], kernel_size = (1, 1))(y);
		y = tf.keras.layers.add([tensor_input, y]);
		
		return y;
	
	## MoSE Block
	def MoSE_Block(R_input):
		w = tf.keras.layers.GlobalAveragePooling2D()(R_input);
		w = tf.keras.layers.Reshape((1, w.shape[1], 1))(w);
		w = tf.keras.layers.Conv2D(filters = 16, kernel_size = (1, 1))(w);
		w = tf.keras.layers.Activation(activation = 'relu')(w);
		w = tf.keras.layers.Conv2D(filters = 1, kernel_size = (1, 1))(w);
		w = tf.keras.layers.Flatten()(w);
		w = tf.keras.layers.Activation(activation = 'softmax')(w);
		w = tf.keras.layers.Reshape((1, 1, w.shape[1]))(w);
		reweighted_R = tf.keras.layers.Multiply()([w, R_input]);
		
		return reweighted_R;
	
	def model():
		input = tf.keras.layers.Input(shape=(224, 224, 3));
		
		base_model = tf.keras.applications.MobileNetV3Small(weights="imagenet", include_top=False, input_tensor=input).output;
		##base_model = MoSE_Block(base_model);
		
		mask_classifier = tf.keras.layers.AveragePooling2D(pool_size=(7, 7))(base_model);
		mask_classifier = tf.keras.layers.Flatten()(mask_classifier);
		mask_classifier = tf.keras.layers.Dense(96, activation="relu")(mask_classifier);
		mask_classifier = tf.keras.layers.BatchNormalization()(mask_classifier);
		mask_classifier = tf.keras.layers.Dense(2, activation="softmax")(mask_classifier);
		
		person_identifier = non_local_block(base_model);
		person_identifier = MoSE_Block(person_identifier);
		person_identifier = tf.keras.layers.AveragePooling2D(pool_size=(7, 7))(person_identifier);
		person_identifier = tf.keras.layers.Conv2D(filters = 1024, kernel_size = (1, 1), activation="relu")(person_identifier);
		person_identifier = tf.keras.layers.BatchNormalization()(person_identifier);
		person_identifier = tf.keras.layers.Conv2D(filters = 512, kernel_size = (1, 1), activation="relu")(person_identifier);
		person_identifier = tf.keras.layers.BatchNormalization()(person_identifier);
		person_identifier = tf.keras.layers.Flatten()(person_identifier);
		person_identifier = tf.keras.layers.Dense(429, activation="softmax", kernel_regularizer=tf.keras.regularizers.l2())(person_identifier);
		
		
		

		model = tf.keras.Model(inputs = input, outputs = [mask_classifier, person_identifier]);
		adam = tf.keras.optimizers.Adam(lr = 0.0001);
		model.compile(loss=['categorical_crossentropy', CircleLoss(gamma=64, margin=0.25, batch_size=64)], loss_weights=[1,8], optimizer = adam, metrics = ['accuracy']);
		#model.compile(loss=[tfa.losses.SigmoidFocalCrossEntropy(reduction=tf.keras.losses.Reduction.AUTO), tfa.losses.SigmoidFocalCrossEntropy(reduction=tf.keras.losses.Reduction.AUTO)], loss_weights=[1,8], optimizer = adam, metrics = ['accuracy']);
		return model	
		
		
	print('Loading data....');
	data = [];
	label_mask = [];
	label_id = [];


	path = './dataset/'
	dirs = os.listdir(path)
	for f in dirs:
		img_path = path + '/' + f;
		data.append(cv2.imread(img_path));
		label_mask.append(f.split('_')[1]);
		label_id.append(f.split('_')[0]);
	data = np.array(data);
	

	(image_train, image_val, label_mask_train, label_mask_val, label_id_train, label_id_val) = train_test_split(data, label_mask, label_id, test_size = 0.10);
		
	label_mask_train_onehot = tf.keras.utils.to_categorical(label_mask_train, num_classes = 2);
	label_mask_val_onehot = tf.keras.utils.to_categorical(label_mask_val, num_classes = 2);
	label_id_train_onehot = tf.keras.utils.to_categorical(label_id_train, num_classes = 429);
	label_id_val_onehot = tf.keras.utils.to_categorical(label_id_val, num_classes = 429);
	
	aug = tf.keras.preprocessing.image.ImageDataGenerator(
		rotation_range=20,
		zoom_range=0.15,
		width_shift_range=0.2,
		height_shift_range=0.2,
		shear_range=0.15,
		horizontal_flip=True,
		fill_mode="nearest");
	
	BS = 64
	
	model = model();
	model.summary();
	checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath='facemask_personid_classifier_best.h5', save_best_only=True);
	history = model.fit(
		image_train, [label_mask_train_onehot, label_id_train_onehot],
		validation_data=(image_val, [label_mask_val_onehot, label_id_val_onehot]), 
		shuffle=True,
		epochs=200, 
		verbose=1,
		batch_size=BS, 
		callbacks=[checkpointer]
	);
	model.save('facemask_personid_classifier.h5');

	# 绘制训练 & 验证的准确率值
	plt.plot(history.history['accuracy']);
	plt.plot(history.history['val_accuracy']);
	plt.title('Accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Val'], loc='upper left')
	plt.show()

	# 绘制训练 & 验证的损失值
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Val'], loc='upper left')
	plt.show()

