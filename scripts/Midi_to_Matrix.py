import mido
from mido import MidiTrack

lowerbound = 1
upperbound = 150

def calculate_timeleft(length, pattern):
	timeleft = []
	for i,track in enumerate(pattern):
		for message in track:
			timeleft.append(length - message.time)
			# print timeleft[i]
			i = i+1
	print 'The length of the array of time is: ' + str(len(timeleft))
	return timeleft

def midi_to_sequence(mid):

	file_name = 'test.txt'
	pattern = mid.tracks
	# length_of_track = mid.length
	# print mid.filename
	try:
		file = open(file_name, 'a+')
	except:
		print "Something went wrong"

	for i in range(len(list(pattern))):
		myList = list(pattern[i])
		myString = ",".join(map(str, myList))
		myString = myString.replace(",","\n")
		file.write(myString)
		# print len(list(pattern[i]))
	file.write('\n$$$\n')
	file.close()

