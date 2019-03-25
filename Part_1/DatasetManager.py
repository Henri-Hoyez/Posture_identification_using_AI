import numpy as np
import os
import sys
import datetime

class DatasetManager():
	def __init__(self, folder):
		try:
			os.makedirs(folder)
		except:
			pass
		finally:
			self.folder = folder
		self.filename = "dataset_"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S")+".csv"

	def save(self,data):
		print(len(data))
		path = os.path.join(self.folder,self.filename)
		with open(path, "a+") as f_out:
			for line in data:
				f_out.write(line + '\n')