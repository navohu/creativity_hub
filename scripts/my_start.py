# import mido
import midi
import glob, os
import re
from mido import MidiFile
from mido import Message
from Midi_to_Matrix import *
from mido import MidiTrack

path = os.path.join(os.getcwd(), 'data')
os.chdir(path)

# Read all MIDI files in directory
# file = open("test.txt", 'w')
# file.write("")
# file.close()
# for file in glob.glob("*.mid"):
# 	myFile = mido.MidiFile(file)
# 	midi_to_sequence(myFile)

# Creates a new midi file from output of RNN
output = list(open("output.txt", 'r'))
output = map(lambda s: s.strip(), output) #strips the new lines

def return_number(message):
	return int(re.search(r'\d+', message).group())

# Reads the input txt file and transforms it into MIDI messages
def read_textfile(input, track):
	for i in range(len(output)):
		splitty = input[i].split(" ")
		if splitty[0] == 'control_change':
			pass

		elif splitty[0] == 'note_on':
			on = midi.NoteOnEvent(time = return_number(splitty[4]), velocity = return_number(splitty[3]), note = return_number(splitty[2]))
			track.append(on)

		elif splitty[0] == 'note_off':
			off = midi.NoteOnEvent(time = return_number(splitty[4]), velocity = return_number(splitty[3]), note = return_number(splitty[2]))
			track.append(off)

os.chdir('..')
pattern = midi.Pattern()
track = midi.Track()
pattern.append(track)
read_textfile(output, track)
midi.write_midifile("example.mid", pattern)
print pattern


