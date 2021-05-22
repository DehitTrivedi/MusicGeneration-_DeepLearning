"""
Machine Learning Special Assigment

Dehit Trivedi

Revant Lala

Creating music usin RNN LSTM network 

"""

from music21 import *
import glob
import pickle
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

notes_to_parse = None

notes = []  # contains elements only
offsets = []
pitches = [] # contains elements + their offset

test = []


multiple_output = True

offset = True
rest = False
offset_normalization = True
offset_rounding = 2

sequence_length = 50 

for file in glob.glob("songs/*.mid"):

    midi = converter.parse(file)
    print("Parsing %s" % file)

    highest_offset = 0.0

    try:
        inst = instrument.partitionByInstrument(midi)
        print("Number of instrument parts: " + str(len(inst.parts)))

        notes_to_parse = inst.parts[0].recurse()
    except:
        notes_to_parse = midi.flat.notes

    previous_offset_temp = 0.0
    temp = ""

    for element in notes_to_parse:

        if element.offset == previous_offset_temp:
            if isinstance(element, note.Note):
                temp += "_" + str(element.pitch)
            elif isinstance(element, chord.Chord):
                temp += "_" + '.'.join(str(n) for n in element.normalOrder)
            elif isinstance(element, note.Rest) and rest:
                temp += "_" + "rest"
            previous_offset_temp = element.offset
        else:
            test.append(temp)
            previous_offset_temp = element.offset


with open('data/multi_notes', 'wb') as filepath:
    pickle.dump(test, filepath)


pitchnames = sorted(set(item for item in test))
note_to_int = dict((test, number) for number, test in enumerate(pitchnames))
print("Dictionary size: %f" % len(note_to_int))

network_input = []
network_output = []

sequence_in = []
sequence_out = []

n_vocab = len(set(test))

print("Create input sequences and the corresponding outputs")
for i in range(0, len(test) - sequence_length, 1):
    sequence_in = test[i:i + sequence_length]
    sequence_out = test[i + sequence_length]
    network_input.append([note_to_int[char] for char in sequence_in])
    network_output.append(note_to_int[sequence_out])


print(network_input)
print(network_output)
n_patterns = len(network_input)


print("Reshape the input into a format compatible with LSTM layers")
network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))

print("Normalize input")
network_input = network_input / float(n_vocab)

network_output = np_utils.to_categorical(network_output)

# Creating model
print("Creating model")
model = Sequential()
model.add(LSTM(
    512,
    input_shape=(network_input.shape[1], network_input.shape[2]),
    return_sequences=True
))
model.add(Dropout(0.3))
model.add(LSTM(512, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(512))
model.add(Dense(256))
model.add(Dropout(0.3))
model.add(Dense(n_vocab))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

# Training model
print("Training model")
filepath = "weights-improvement-multi-{epoch:02d}-{loss:.4f}-bigger.hdf5"
checkpoint = ModelCheckpoint(
    filepath,
    monitor='loss',
    verbose=0,
    save_best_only=True,
    mode='min'
)
callbacks_list = [checkpoint]

model.fit(network_input, network_output, epochs=200, batch_size=64, callbacks=callbacks_list)
