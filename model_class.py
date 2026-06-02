import csv

from datetime import datetime
from pathlib import Path

class MarkovTrainingLessNaive(object):
  # This one should hopefully perform a little better by not initializing a giant "matrix" that will likely mostly be empty

  # params:
  #   training_data_path - a string that contains the entire filepath for the CSV file containing the training data
  #   vocab_data_path - a string that contains the entire filepath for the JSON file containing the mappings from event ID to their respective template text
  def __init__(self, training_data_path, vocab_data_path):
    self.training_data_path = training_data_path
    self.vocab_data_path = vocab_data_path

    # tracks total number of log lines iterated over during training
    self.training_data_length = 0

    # Build the map that acts as the matrix for storing transition probabilities
    #
    # first layer, a Dictionary{}
    # key = first event ID as a string (e.g. "1246"), serving as the first, or t - 2 state in the 2nd order Markov chain model
    # value = a Dictionary{}
    #
    # second layer, a Dictionary{}:
    # key = second event ID as a string (e.g. "1247"), serving as the second, or t - 1 state in the 2nd order Markov chain model
    # val = an array of length 2, like this: [number of occurrences of {state_1 -> state 2} sequence, Dictionary{}]
    #
    # third layer, a Dictionary{}:
    # key = third event ID as a string (e.g. "1248"), serving as the third, or t state in the 2nd order Markov chain model
    # val = an array of length 2, like this: [number of occurrences of {state_1 -> state_2 -> state_3} sequence, (probability as a Float)]
    #
    # The eventual goal will be to access the transition probabilities like so, using 3 layers of keys that map to one 2-element list:
    #
    # transition_probabilities[first_event_ID][second_event_ID][1][third_event_ID] = [number of 3-state occurrences, probability of occurring given the prior 2 states]
    #
    # In effect, it would be like a 1299 x 1299 x 1299 matrix of length=2 arrays
    self.transition_probabilities = {}

  # Train the model by simply measuring occurrences and of all the different state sequences to estimate a transition probability
  def train(self):
    total_lines_counter = 0

    with open(self.training_data_path, 'r') as f:
      f.readline() # get past header

      state_1_split = f.readline().split(',')
      state_2_split = f.readline().split(',')
      state_3_split = f.readline().split(',')

      # loop down line by line with a sliding window of size 3
      while state_3_split[0]:
        # get the event ID key, remove the '\n'
        state_1_eventID = state_1_split[3][:-1]
        state_2_eventID = state_2_split[3][:-1]
        state_3_eventID = state_3_split[3][:-1]

        total_lines_counter = total_lines_counter + 1

        # all these checks are used to avoid having to initialize all 1299^3 entities, which would take up > 32GB of RAM and make
        # proccessing time enormous, especially once we loop through them

        # update occurcance count for state-prior {S_t = state_1}
        if state_1_eventID not in self.transition_probabilities:
          self.transition_probabilities[state_1_eventID] = [0, {}]
        self.transition_probabilities[state_1_eventID][0] += 1

        # update occurrence count for 2-state sequence {S_t-2 = state_1, S_t-1 = state_2}
        if state_2_eventID not in self.transition_probabilities[state_1_eventID][1]:
          self.transition_probabilities[state_1_eventID][1][state_2_eventID] = [0, {}]
        self.transition_probabilities[state_1_eventID][1][state_2_eventID][0] += 1

        # update occurrence count for 3-state sequence {S_t-2 = state_1, S_t-1 = state_2, S_t = state_3}
        if state_3_eventID not in self.transition_probabilities[state_1_eventID][1][state_2_eventID][1]:
          self.transition_probabilities[state_1_eventID][1][state_2_eventID][1][state_3_eventID] = [0, 0]
        self.transition_probabilities[state_1_eventID][1][state_2_eventID][1][state_3_eventID][0] += 1

        # slide window down by 1
        state_1_split = state_2_split
        state_2_split = state_3_split
        state_3_split = f.readline().split(',')

    # Normalize using the occurrence counts to calculate the transition probabilities
    self.training_data_length = total_lines_counter
    for key in self.transition_probabilities:
      self.transition_probabilities[key][0] /= total_lines_counter
      for key_2 in self.transition_probabilities[key][1]:
        self.transition_probabilities[key][1][key_2][0] /= total_lines_counter
        for key_3 in self.transition_probabilities[key][1][key_2][1]:
          self.transition_probabilities[key][1][key_2][1][key_3][0] /= total_lines_counter

  # just a helper function to make sure the total probabilities add up properly
  # and that the total number of occurrences roughly matches the total number of log lines
  #
  # prob_margin_of_error = margin of error we will allow for the probability summation since it won't exactly be 1.0 due to rounding
  def verify_transition_matrix(self, prob_margin_of_error):
    total = 0
    for key in self.transition_probabilities:
      for key_2 in self.transition_probabilities[key][1]:
        for key_3 in self.transition_probabilities[key][1][key_2][1]:
          total += self.transition_probabilities[key][1][key_2][1][key_3][0]
    if abs(total - 1) > prob_margin_of_error:
      print(f"ERROR Total Probability outside acceptiable error: {total} (del={prob_margin_of_error})")

  # write the transition matrix to a CSV file output
  def save_matrices(self, output_dir):
    output_dir = Path(output_dir)
    if output_dir.exists():
      output_dir.rename(str(output_dir) + ".old-" + datetime.now().strftime("%y%m%d-%H%M%S"))
    output_dir.mkdir()
    with (open(output_dir / 'S0.csv', 'w')     as f_s0,
          open(output_dir / 'S0S1.csv', 'w')   as f_s01,
          open(output_dir / 'S0S1S2.csv', 'w') as f_s012):
      f_s0.write("state_0,probability\n")
      f_s01.write("state_0,state_1,probability\n")
      f_s012.write("state_0,state_1,state_2,probability\n")

      writer_s0 = csv.writer(f_s0)
      writer_s01 = csv.writer(f_s01)
      writer_s012 = csv.writer(f_s012)

      for key in self.transition_probabilities:
        writer_s0.writerow([key, self.transition_probabilities[key][0]])
        for key_2 in self.transition_probabilities[key][1]:
          writer_s01.writerow([key, key_2, self.transition_probabilities[key][1][key_2][0]])
          for key_3 in self.transition_probabilities[key][1][key_2][1]:
            writer_s012.writerow([key, key_2, key_3, self.transition_probabilities[key][1][key_2][1][key_3][0]])

