import pathlib
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np

hu = {}
hu['and'] = cv2.imread("orig/and_orig.jpg", cv2.IMREAD_GRAYSCALE)
hu['nand'] = cv2.imread("orig/nand_orig.jpg", cv2.IMREAD_GRAYSCALE)
hu['nor'] = cv2.imread("orig/nor_orig.jpg", cv2.IMREAD_GRAYSCALE)
hu['or'] = cv2.imread("orig/or_orig.jpg", cv2.IMREAD_GRAYSCALE)
hu['buffer'] = cv2.imread("orig/buffer_orig.jpg", cv2.IMREAD_GRAYSCALE)
hu['not'] = cv2.imread("orig/not_orig.jpg", cv2.IMREAD_GRAYSCALE)
hu['xnor'] = cv2.imread("orig/xnor_orig.jpg", cv2.IMREAD_GRAYSCALE)
hu['xor'] = cv2.imread("orig/xor_orig.jpg", cv2.IMREAD_GRAYSCALE)
'''
R: 231, 25, 29
G: <40, >225, <10
B: <10, <90, >225
Y: 250, 225, 10

broken connection
output goes to output
disjoint circuit		done
indeterminant output	done

FEATURES:
indeterminant output
Disjoint/broken circuit
'''

gate_input = None

GATE_DIR = "./gates/"

RED = [(200, 0, 0), (255, 50, 50)]
GREEN = [(0, 100, 0), (40, 255, 10)]
BLUE = [(0,0,225), (20,100,255)]
YELLOW = [(230, 200, 0), (255, 255, 50)]

THRESH = 220
GATE_THRESH = 0.8
#GATES_THRESH = {'and': , 'nand': ,'or': , 'nor': , 'xor': , 'xnor': , 'not': , 'buffer': , }
MAX_ITERATIONS = 300
COLORS = {'R':RED, 'G':GREEN, 'B':BLUE, 'Y':YELLOW}

class Gate:

	'''
	inp_orienation: -1: left, 1:right, -2:top, 2:bottom
	'''
	def __init__(self, _id, operation, rect, inp_orientation=-1,):
		self.operation = operation
		self.rect = rect
		self.id = _id
		self.inp_orientation = inp_orientation

		if(operation not in ['not', 'buffer']):
			self.num_inputs = 2
		else:
			self.num_inputs = 1

	#return true if all inputs are non white
	def get_inputs(self, im):
		orient = self.inp_orientation

		rect = self.rect

		mask = np.zeros(im.shape, dtype=np.uint8)
		leftx = rect[0][0]
		top = rect[0][1]
		bottom = rect[1][1]


		mask[top+5:bottom-5,leftx-10:leftx-2] = 1


		mask2 = np.zeros(mask.shape)
		if(abs(orient) == 1):
			mask2[:,:orient] = mask[:, -orient:]
		else:
			orient /= 2
			mask2[:orient, :] = mask[-orient:, :]



		mask = np.array((1-mask)*mask2, dtype=np.uint8)

		inps = np.where((mask > 0) & (im != 255))


		#presentation
		global gate_input

		GATE_INPUT_COLOR = (0,0,0)
		GATE_OUTPUT_COLOR = (0,0, 255)


		for inp in np.array(inps).T:
			if(im[tuple(inp)]):
				gate_input = cv2.circle(gate_input, tuple(inp[::-1]), 5, GATE_INPUT_COLOR, -1)
		gate_input = cv2.circle(gate_input, (self.rect[1][0], (self.rect[0][1] + self.rect[1][1])//2), 5, GATE_OUTPUT_COLOR, -1)
		#end

		unique = np.unique(im[np.where(mask)].ravel())[1:]
		print(unique)

		if(len(unique) == self.num_inputs):
			self.inputs = list(unique)
			return True
		else:
			print('bad inputs')
			return False

	def paint_output(self, im):

		'''
		leftmost = (rect[0][0]-1, (rect[0][1] + rect[1][1])/2)
		rightmost = cnt[cnt[:,:,0].argmax()][0]
		topmost = cnt[cnt[:,:,1].argmin()][0]
		bottommost = cnt[cnt[:,:,1].argmax()][0]

		seed = {1: leftmost + np.array([-1,0]),
				-1:rightmost + np.array([1,0]),
				-2:bottommost + np.array([0,1]),
				2:topmost + np.array([0,-1])}[self.inp_orientation]
		'''

		rect = self.rect
		seed = (rect[1][0]+1, int((rect[0][1] + rect[1][1])/2))

		h, w = im.shape
		mask = np.zeros((h+2,w+2), dtype=np.uint8)
		im = cv2.floodFill(im, mask, tuple(seed), self.id)[1]

		return im

	def __compute_func(self, a, b=None):
		op = self.operation

		inp = self.inputs

		if(op == 'and'):
			ans = a and b
		elif(op == 'nand'):
			ans = not (a and b)
		elif(op == 'or'):
			ans = a or b
		elif(op == 'xor'):
			ans = (a and not b) or (b and not a)
		elif(op == 'xnor'):
			ans = (a and not b) or (b and not a)
			ans = not ans
		elif(op == 'nor'):
			ans = a or b
			ans = not ans
		elif(op == 'buffer'):
			ans = a
		elif(op == 'not'):
			ans = not a

		else:
			raise Exception("BAD OPERATION " + str(op))

		return int(ans)

	#cur_values = {1:None, 2:True, 3:False, 4:True, ...}
	def compute(self, cur_values):
		all_inputs_not_none = True

		is_changed = False

		for inp in self.inputs:
			if cur_values[inp] == None:
				all_inputs_not_none = False

		if(all_inputs_not_none and cur_values[self.id] == None):
			if(self.num_inputs == 2):
				cur_values[self.id] = self.__compute_func(cur_values[self.inputs[0]], cur_values[self.inputs[1]])
			else:
			  cur_values[self.id] = self.__compute_func(cur_values[self.inputs[0]])

			is_changed = True

		return is_changed, cur_values

	def __str__(self):
		return f'{self.id}-{self.operation}'

	def __repr__(self):
		return f'{self.id}-{self.operation}'


def print_tt(gate_objects_list, input_points_dict, image, output_point):
	image = np.array(image, dtype=np.uint8)

	for label in input_points_dict:
		for point in input_points_dict[label]:
			h, w = image.shape
			mask = np.zeros((h+2, w+2), dtype=np.uint8)
			#print(f'label: {label}')
			# print(label)
			image = cv2.floodFill(image, mask, point, label)[1]
	
	#mshow(image, win='flooded')

	for gate in gate_objects_list:
		image = gate.paint_output(image)

	for gate in gate_objects_list:
		gate.get_inputs(image)

	output = image[output_point[1], output_point[0]]

	if (np.any(image.ravel() == 255)):
		#temp = np.zeros(im.shape, dtype=np.uint8)
		#temp[np.where(image == 255)] = 1
		#mshow(temp, win='temp')
		raise Exception("ERROR: Broken connection/ disjoint circuit")
		exit()

	print('B\tG\tR\tY\n====================================')
	for i in range(2):
		for j in range(2):
			for k in range(2):
				values = [0, i, j, k,]
				for _ in range(len(gate_objects_list)):
					values.append(None)

				isChanging = True
				iterations = 0
				while isChanging:
					isChanging = False
					for gate in gate_objects_list:
						change, values = gate.compute(values)
						isChanging = change or isChanging
						iterations += 1
					if (iterations > MAX_ITERATIONS):
						raise Exception("ERROR: Indeterminant output")
						exit()
				s = ''

				for x in values:
					if (x == None):
						raise Exception("ERROR: Can't determine output of gate")
						exit()

				for gate in gate_objects_list:
					if (len(gate.inputs) != gate.num_inputs):
						raise Exception("ERROR: insufficient inputs")

				for x in values[1:4]:
					s += (f'{x}\t')

				print(s + str(values[output]))

	#mshow(image)

def mshow(im, cmap='gray', win=None):
	plt.clf()
	plt.imshow(im, cmap=cmap)
	if(win):
		plt.title(win)
	plt.show()

def show(im, win=''):
	cv2.imshow(win, im)
	cv2.waitKey(-1)
	cv2.destroyAllWindows()

def get_io(im):

	ios = {}

	for color in COLORS:
		ios[color] = []
		low = COLORS[color][0]
		high = COLORS[color][1]
		mask = cv2.inRange(im, low, high)
		mask = cv2.medianBlur(mask, 3)
		conts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

		for cont in conts:
			M = cv2.moments(cont)
			try:
				cx = int(M['m10']/M['m00'])
				cy = int(M['m01']/M['m00'])
			except ZeroDivisionError:
				print(cont)
			ios[color].append((cx, cy))

	return ios

def clean_image(im):
	io = get_io(im)

	gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
	#threshold
	thresh = np.array(gray < THRESH, dtype=np.uint8)*255

	try:
		seed = io['Y'][0]
	except IndexError:
		print('\n======\nERROR: No output found')

	counter = 0
	cleaned = thresh
	for color in io:
		for seed in io[color]:
			counter += 1
			cleaned = cv2.floodFill(thresh, None, seed, counter)[1]

	final = np.array(cleaned == counter, dtype=np.uint8)*255

	if(np.any((cleaned.ravel() < counter) & (cleaned.ravel() > 0))):
		print('\n=====\nERROR: Disjoint Circuit/ Broken connection\n=====')
		show(np.array((cleaned < counter) & (cleaned > 0), dtype=np.uint8)*255, win='broken')

	return final

def guess_the_gate(img):

	# Morphology

	#print(img.shape)

	_, img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
	img = cv2.morphologyEx(img, cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
	
	# Contours ( comparing number and shape of contours in next statements)
	contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	contours = contours[1:]

	# Will fail the test :=(
	assert len(contours) <= 3

	# Find valid shapes (is there a dot in front, doubled bar in rear)
	valid = hu.keys()
	negated = False
	cmplx = False
	for c in contours:
		x, y, h, w = cv2.boundingRect(c)
		if h < 10 and w < 10:
			negated = True
		if h < 10 and w > 20:
			cmplx = True
	if cmplx and negated:
		return 'xnor'
	if cmplx and not negated:
		return 'xor'
	if negated:
		valid = ['not', 'nand', 'nor']
	else:
		valid = ['buffer', 'and', 'or']

	# The simplest matching algorithm ( Hu invariants)
	mini = 1
	guess = 'undefined'
	#print(valid)
	for key in valid:
		# Match shapes automatically calculates contours and their 7 Hu invariants
		# So there is no need to do it ourselves ::yay::
		curr = cv2.matchShapes(img, hu[key], cv2.CONTOURS_MATCH_I1, 0)
		if mini > curr:
			mini = curr
			guess = key

	#print(str(guess))
	return guess

def paint_gates(im):
	gatedir = pathlib.Path(GATE_DIR)
	gatepaths = gatedir.glob('gate_*.png')
	gatepaths = list(gatepaths)
	names = [item.name[5:-4] for item in gatepaths]
	gatepaths.sort()
	plt.clf()

	final_image = np.array(im, dtype=np.uint8)

	gates = []


	id_counter = 4
	for gatepath in gatepaths:

		gate = gatepath.name[5:-4]
		template = cv2.imread(str(gatepath))
		template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
		res = cv2.matchTemplate(im,template,cv2.TM_CCOEFF_NORMED) > GATE_THRESH

		dilated = cv2.dilate(np.array(res, dtype=np.uint8), np.ones((10,10)))
		contours = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

		newres = np.zeros(res.shape)

		for cont in contours:
			M = cv2.moments(cont)
			cx = int(M['m10']/M['m00'])
			cy = int(M['m01']/M['m00'])
			newres[cy,cx]=1

		res = newres

		#remove nearby matches from res todo

		pts = tuple(np.where(res))
		xs, ys = pts
		for i in range(len(xs)):
			start = (ys[i], xs[i])
			end = (ys[i]+template.shape[1], xs[i]+template.shape[0])
			roi = im[start[1]-10:end[1]+10, start[0]:end[0]+10]
			gate_guess = guess_the_gate(roi)
			print(gate_guess)
			mshow(roi)

			'''
			array([[[ 5,  3]],

			[[ 5,  9]],

			[[10,  9]],

			[[10,  3]]], dtype=int32)
			'''

			#import random
			#cv2.imwrite(f'{random.randint(1,100000000000)}.png', roi)

			if(gate_guess != 'undefined'):
				gate = gate_guess
				gates.append(Gate(id_counter, gate, rect=(start, end)))
				#cv2.rectangle(im, (start[1]-10,start[0]), (end[1]+10, end[0]+10), id_counter, -1)
				im[start[1]:end[1], start[0]:end[0]] = id_counter
				im[start[1]:end[1], end[0]+1] = 255
				#cv2.drawContours(im, [contour], 0, id_counter)
				id_counter += 1

	return im, gates

if __name__ == '__main__':
	if len(sys.argv) != 2:
		print("No image file", file=sys.stderr)
		exit()
	plt.figure()
	# file_name = input('Enter filename: ')
	im = cv2.imread(sys.argv[1])[3:-3, 3:-3]
	im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
	#mshow(im)
	gate_input = np.array(im)

	#get the inputsorig/
	io = get_io(im)

	print(io)

	#print('io: ' ,io)
	new_io = {}
	mp = {'B':1, 'G':2, 'R':3, 'Y':'Y'}
	for k in io:
		if k != 'Y':
			new_io[mp[k]] = io[k]

	#remove the labels and get only wires and inputs and gatez
	clean = clean_image(im)
	mshow(clean)
	# cv2.imwrite('clean.png', clean)

	painted, gates = paint_gates(clean)
	#mshow(painted)
	out_gates = list(map(lambda x: x.operation, gates))
	out_gates = {x: out_gates.count(x) for x in out_gates}
	if out_gates.keys() == 0:
		print("No gates found")
		exit()
	else:
		for k, v in out_gates.items():
			print(f"Gate={k},\tCount={v}")

	#mshow(painted)

	#print_tt(gate_objects_list, input_points_dict, image, output_point)

	print_tt(gates, new_io, painted, io['Y'][0])

	mshow(gate_input)
