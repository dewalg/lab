import sys
import struct
import numpy
import tables
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
'''
Program based on the work of Adam Light who wrote original pycine.py
'''

class cine(object):

	def __init__(self, filename):
	
		with open(filename, "rb") as cineFile:
			file = struct.unpack("<2s", cineFile.read(2))[0];
			cineFile.seek(0)
			assert file == "CI", \
				"File is either damaged or not a .cine"
			
			self._readCine(cineFile)
			
	def _readCine(self, cineFile):
		
		#define error message
		errorMsg = 'Error: cannot read - file may be damaged'
		
		
		fileHeader_length = 44  
		#getting header info
		self.fileHeader = self._getFileHeader(cineFile.read(fileHeader_length))
		
		bitMapInfoHeader_length = 40
		#get bitmap info
		self.bitMapInfoHeader = self._getBitMapInfoHeader(cineFile.read(bitMapInfoHeader_length))
		
		deprecatedSkip = 140
		cineFile.seek(fileHeader_length+bitMapInfoHeader_length+deprecatedSkip)
				
		setupCheck = struct.unpack("<2s", cineFile.read(2))[0]
		if (setupCheck != 'ST'):
			print errorMsg
		
		setup_length = struct.unpack("<H", cineFile.read(2))[0]
		
		setupSkip = 597
		setupInitPosition = fileHeader_length+bitMapInfoHeader_length+deprecatedSkip+setupSkip
		cineFile.seek(setupInitPosition)
		
		setupZeros_length = 1212
		setupRead_length = setup_length - setupSkip - setupZeros_length - deprecatedSkip
		
		#get Setup info
		self.setup = self._getSetup(cineFile.read(setupRead_length))
		
		cineFile.seek(fileHeader_length+bitMapInfoHeader_length+setup_length)
		self.images = self._getImages(cineFile, self.fileHeader["OffImageOffsets"])
		
		for key,value in self.fileHeader.iteritems():
			setattr(self,key,value)
		for key,value in self.bitMapInfoHeader.iteritems():
			setattr(self,key,value)
		for key,value in self.setup.iteritems():
			setattr(self,key,value)
	
		
	def _getFileHeader(self, bitString): 
		
		filetuple = struct.unpack('<2s 3H l I l 6I',bitString)
		header_dict = {}
		# assign dictionary key/value pairs and class attributes exactly
		# as laid out in cine640.pdf
		header_dict["Type"] = filetuple[0]
		header_dict["Headersize"] = filetuple[1]
		header_dict["Compression"] = filetuple[2]
		header_dict["Version"] = filetuple[3]
		header_dict["FirstMovieImage"] = filetuple[4]
		header_dict["TotalImageCount"] = filetuple[5]
		header_dict["FirstImageNo"] = filetuple[6]
		header_dict["ImageCount"] = filetuple[7]
		header_dict["OffImageHeader"] = filetuple[8]
		header_dict["OffSetup"] = filetuple[9]
		header_dict["OffImageOffsets"] = filetuple[10]
		header_dict["TriggerTime"] = (filetuple[11],filetuple[12])
		return header_dict		
		
	def _getBitMapInfoHeader(self, bitString):
		
		filetuple = struct.unpack('<I 2l 2H 2I 2l 2I',bitString)
		
		bitmap_dict = {}
		# assign dictionary key/value pairs and class attributes exactly
		# as laid out in cine640.pdf
		bitmap_dict["biSize"] = filetuple[0]
		bitmap_dict["biWidth"] = filetuple[1]
		bitmap_dict["biHeight"] = filetuple[2]
		bitmap_dict["biPlanes"] = filetuple[3]
		bitmap_dict["biBitCount"] = filetuple[4] 
		# note that this may be different from Setup.RealBPP
		# Pixels are stored only at 8 bits or 16 bits
		#     even if they are recorded at different depth
		bitmap_dict["biCompression"] = filetuple[5]
		bitmap_dict["biSizeImage"] = filetuple[6]
		bitmap_dict["biXPelsPerMeter"] = filetuple[7]
		bitmap_dict["biYPelsPerMeter"] = filetuple[8]
		bitmap_dict["biClrUsed"] = filetuple[9]
		bitmap_dict["biClrImportant"] = filetuple[10]
		
		return bitmap_dict	
		
	def _getSetup(self, bitString):
		format_ImWidth_RealBPP_inclusive ="2H H I l B I ???? ????  I I 2I I I"\
		    + "???? 3I l I 3l 3I 4I 8f l 2f I" 
		# the next chunk is skipped stuff
		format_Conv8min_MCPercent_inclusive = "2I 30l 3I 4? 2I 16l 32I l 64f" 
		format_CICalib_Description_end_inclusive = "7I 8I 4I 4I 4096s"
		format_string = ("<" + format_ImWidth_RealBPP_inclusive 
				 + format_Conv8min_MCPercent_inclusive 
				 + format_CICalib_Description_end_inclusive)
		filetuple = struct.unpack(format_string,bitString)
		setup_dict = {}
		# setup_dict["filetuple"]=filetuple  #Could include entire tuple
		
		# assign dictionary key/value pairs and class attributes exactly
		# as laid out in cine640.pdf
		setup_dict["ImWidth"] = filetuple[0]
		setup_dict["ImHeight"] = filetuple[1]
		# skip filetuple[2] = EDRShutter16
		setup_dict["Serial"] = filetuple[3]
		# skip filetuple[4:15] = saturation, autoexposure, and PCC setup vars
		setup_dict["FrameRate"] = filetuple[16]
		# skip [17:18] - deprecated shutter vars
		setup_dict["PostTrigger"] = filetuple[19]
		# skip [20:24], deprecated and color vars
		setup_dict["CameraVersion"] = filetuple[25]
		setup_dict["FirmwareVersion"] = filetuple[26]
		setup_dict["SoftwareVersion"] = filetuple[27]
		# skip [28:50] - timezone and PCC setup vars 
		setup_dict["RealBPP"] = filetuple[51]
		# note that this may be different from BitmapInfoHeader.biBitCount.
		# Pixels are stored only at 8 bits or 16 bits
		#     even if they are recorded at different depth
		
		# skip [52:205] - PCC image processing, 
		#     8-bit memory conversion (N/A for 12 bit recordings),
		setup_dict["CICalib"] = filetuple[206]
		setup_dict["CalibWidth"] = filetuple[207]
		setup_dict["CalibHeight"] = filetuple[208]
		setup_dict["CalibRate"] = filetuple[209]
		setup_dict["CalibExp"] = filetuple[210]
		setup_dict["CalibEDR"] = filetuple[211]
		setup_dict["CalibTemp"] = filetuple[212]
		# skip [213:220] - unused options        
		setup_dict["Sensor"] = filetuple[221]
		setup_dict["ShutterNs"] = filetuple[222]
		setup_dict["EDRShutterNs"] = filetuple[223]
		setup_dict["FrameDelayNs"] = filetuple[224]
		# skip [225:228] - sidestamped image offsets
		# strip trailing zeros from description and encode
		desc_string = filetuple[229].rstrip('\x00')
		return setup_dict
		
	def _getImages(self, cineFile, start_pos):
		
		nx = self.setup["ImWidth"]
		ny = self.setup["ImHeight"]
		nframes = self.fileHeader["ImageCount"]
		pointer_array = numpy.ndarray((nframes), numpy.int64)
		
		cineFile.seek(start_pos)
		
		pointer_array = struct.unpack("<"+str(nframes)+"Q", cineFile.read(nframes*8))
		image_array = numpy.ndarray((ny,nx,nframes),float)
		
		#go to first frame
		cineFile.seek(pointer_array[0])
		
		for frame in range(nframes):
			
			if (frame != 0 and numpy.mod(frame,500) == 0):
				print "Read" + str(frame) + " frames."
			
			AnnotationSize = struct.unpack('<I', cineFile.read(4))[0]
			stringSize = AnnotationSize - 8
			
			Annotation, ImageSize = struct.unpack("<"+str(stringSize)+"s I", cineFile.read(AnnotationSize-4))
			
			if self.bitMapInfoHeader["biBitCount"] == 8:
				image_bits = struct.unpack("<"+str(ImageSize)+"B", cineFile.read(ImageSize)) 
			else:
				image_bits = struct.unpack("<"+str(ImageSize/2)+"H", cineFile.read(ImageSize))	
			
			image_array[..., frame] = numpy.reshape(image_bits,(ny, nx))
			
		print "Read " + str(nframes) + " frames."
		
		'''
		plt.imshow(image_array[...,50], cmap=cm.Greys_r)
		plt.show()
		'''
		
		#self._dispImages(image_array, nframes)
		return image_array
	
	'''
	def _dispImages(self,imageArr, nframes):
		fig = plt.figure()
		new_imArr = numpy.ndarray(imageArr[:,:,1].shape)
		
		for n in range(nframes):
			new_imArr = imageArr[...,n]
		
		img = None
		for frame in new_imArr:
			'''	
							
def main(filename):

	if filename.endswith('.cine'):
		# Read into Cine object
		mov = cine(filename)

		return 1
	else:
		print "File extension '%s' not recognized."%filename.rsplit('.',1)[1]
		print "Current version only converts raw '.cine' files."
		return 0


if __name__ == "__main__":
	try:
		filename = sys.argv[1]
		main(filename)
	except KeyboardInterrupt:
		print >> sys.stderr, '\nExiting by user request.\n'
		sys.exit(0)