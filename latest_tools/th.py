## code for tensorboard
## using tensorboard for visualization instead of X forward to save time
import cv2
import os
import numpy as np
import argparse
from io import BytesIO
import scipy

class TensorboardHelper:
	def __init__(self, filename, mode='train'):
		import tensorflow as tf
		self.tf = tf
		self.filename = filename
		self.writer = tf.summary.FileWriter(filename)
		self.init(0, mode)
	
	def init(self, step, mode):
		self.step = step
		self.mode = mode
	
	def name(self, name):
		return '{}'.format(name)
	
	def add_scalar(self, tag, value):
		"""Log a scalar variable."""
		summary = self.tf.Summary(value=[self.tf.Summary.Value(tag=self.name(tag), simple_value=value)])
		self.writer.add_summary(summary, self.step)
	
	def add_image(self, tag, images):
		"""Log a list of images."""
		
		img_summaries = []
		for i, img in enumerate(images):
			# Write the image to a string
			try:
				s = StringIO()
			except:
				s = BytesIO()
			scipy.misc.toimage(img).save(s, format="png")
		
			# Create an Image object
			img_sum = self.tf.Summary.Image(encoded_image_string=s.getvalue(),
				height=img.shape[0],
				width=img.shape[1])
			# Create a Summary value
			img_summaries.append(self.tf.Summary.Value(tag=self.name('%s/%d' % (tag, i)), image=img_sum))
		
		# Create and write Summary
		summary = self.tf.Summary(value=img_summaries)
		self.writer.add_summary(summary, self.step) 
	
	def tick(self):
		self.step += 1
	
	def flush(self):
		self.writer.flush()
