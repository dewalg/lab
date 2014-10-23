import sys
import struct
import numpy
import tables

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
	
		fileHeader_length = 44 #pre-defined value 
		#getting header info
		self.fileHeader = self._getFileHeader(cineFile.read(header_length))
		
		bitMapInfoHeader_length = 40
		#get bitmap info
		self.bitMapInfoHeader = self._getBitMapInfoHeader(cineFile.read(bitMapInfoHeader_length))
		
		'''
		add the stuff after bitmap info
		'''
		
		#self.images = self._getImages(cineFile, self.cineFileHeader["OffImageOffsets"];
		
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
		
		filetuple = struct.unpack('<I 2l 2H 2I 2l 2I',filestring)
		
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