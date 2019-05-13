
import numpy as np
import time


def generate_data(b,seq_length,num_seq):
	'''
	Parameters

	b: Emission probablities for generated data (num_states x alphabet)
	seq_length: Length of each sequence generated
	num_seq: number of sequences generated

	'''
	sequences = np.zeros((num_seq,seq_length)) # array containing all generated sequences
	start_points = []

	for i in range(num_seq):

		start = np.random.randint(0,seq_length-b.shape[0]+1) # random start index for motif in sequence
		start_points.append(start)
		sequences[i,:start] = np.random.randint(0,b.shape[1],start)  # random sequence before motif
		sequences[i,start+b.shape[0]:]= np.random.randint(0,b.shape[1],seq_length-start-b.shape[0]) # random sequence after motif

		for j in range(b.shape[0]): # motif whose sequences depends on emission probabilities b
			sequences[i,start+j] = int(np.random.choice(b.shape[1],p=b[j,:]))
	
	return np.array(start_points), sequences.astype(int) # Return the start point of the motif in each generated seq, and generated sequences


def viterbi(a,b,s,seq):
	'''
	Parameters

	a:	Transition probabilities (num_states x num_states)
	b: 	Emission probabilities (num_states x alphabet)
	s:  Starting probabilities (num_states,)
	seq: Sequence of observations 

	Variables

	v: DP Matrix (len(seq) x num_states)
	bt: Backtrace Matrix (len(seq) x num_states)

	Retvals:

	prob: P(O,Q), the joint probability for list of states and list of observations
	state_list: The list of predicted states in sequence
	'''

	v = np.zeros((len(seq),len(s)))
	bt = np.zeros_like(v)

	#FILL DP TABLE

	for t in range(len(seq)):
		for j in range(len(s)):

			#Base Case
			if t == 0:
				v[t,j] = s[j]*b[j,seq[t]]
				bt[t,j] = -1					#use -1 to denote start state, 0...num_states-1 to denote other states

			#Recursive Case
			else:
				v[t,j] = 0
				bt[t,j] = -1
				for i in range(len(s)):
					if v[t-1,i]*a[i,j]*b[j,seq[t]] > v[t,j]:
						v[t,j] = v[t-1,i]*a[i,j]*b[j,seq[t]]
						bt[t,j] = i


	#BACKTRACE

	prob = np.amax(v[-1,:])
	state_list = [int(np.argmax(v[-1,:]))]

	for t in range(len(seq)-1,-1,-1):
		state = state_list[-1]
		state_list.append(int(bt[t,state]))

	state_list.reverse()

	return prob, state_list[1:]
		

def test(alg):

	print("Testing " + alg+":")
	print("")
	print("")

	test_vals = np.arange(1,0.6,-0.04)
	results = np.zeros(len(test_vals))
	motif_length = 6
	alphabet = 4

	num_sequences = 500
	seq_length = 200

	#Calculate transition matrix based on motif finding problem
	#States 1 to motif_length correspond to locations on motif
	#State 0 corresponds to nucleotide that is not in motif

	trans_prob = np.zeros((motif_length+2,motif_length+2))
	for i in range(1,motif_length+1):
		trans_prob[i,i+1] = 1
	trans_prob[0,1] = float(motif_length)/float(seq_length)
	trans_prob[0,0] = 1 - trans_prob[0,1]
	trans_prob[motif_length+1,motif_length+1] = 1

	
	for test,n in enumerate(test_vals):
		print("Test %d:" % test)
		print("")

		b = np.zeros((motif_length+2,alphabet))
		for i in range(1,motif_length+1):
			for j in range(alphabet):
				if i%alphabet == j:
					b[i,j] = n
				else: 
					b[i,j] = (1-n)/(float(alphabet-1))

		b[0,:] = np.array([1.0/(float(alphabet))]*alphabet)
		b[motif_length+1,:] = np.array([1.0/(float(alphabet))]*alphabet)
		start_points, sequences = generate_data(b[1:motif_length+1], seq_length, num_sequences)
	
		correct = 0

		print("Transition Probabilities:")
		print()
		print(trans_prob)
		print()
		print("Emission Probabilities:")
		print(b)
		print()
		total_time = 0
		for i in range(num_sequences):
			if alg == "Viterbi":
				t1 = time.time()
				pred = viterbi(trans_prob,b,trans_prob[0,:],sequences[i,:])[1]
				t2 = time.time()
			else:
				t1 = time.time()
				pred = forward_backward(trans_prob,b,trans_prob[0,:],sequences[i,:])[1]
				t2 = time.time()

			start = np.where(np.array(pred)==1)[0]
			total_time += (t2-t1)

			if start == start_points[i]:
				correct+=1


		accuracy = float(correct)/float(num_sequences)
		avg_time = total_time/float(num_sequences)
		print("Accuracy for test %d: %f" % (test,accuracy))
		print()
		print("Average runtime for test %d: %f" % (test,avg_time))
		print()

		results[test] = accuracy


	return results

	'''
Parameters
	a:	Transition probabilities (num_states x num_states)
	b: 	Emission probabilities (num_states x alphabet)
	pi:  Starting probabilities (num_states,)
	seq: Sequence of observations
'''

#the forward algorithm
def forward(a, b, pi, seq):
	T = len(seq)
	N = a.shape[0]
	alpha = np.zeros((T, N))
	alpha[0] = pi*b[:,seq[0]]
	for t in range(1, T):
		alpha[t] = alpha[t-1].dot(a) * b[:, seq[t]]
	return alpha

#uses the forward algorithm
def likelihood(a, b, pi, seq):
	# returns log P(Y  \mid  model)
	# using the forward part of the forward-backward algorithm
	return  forward(a, b, pi, seq)[-1].sum()

#the backward algorithm
def backward(a, b, pi, seq):
	N = a.shape[0]
	T = len(seq)

	beta = np.zeros((N,T))
	beta[:,-1:] = 1
	for t in reversed(range(T-1)):
		for n in range(N):
			beta[n,t] = np.sum(beta[:,t+1] * a[n,:] * b[:, seq[t+1]])
	return beta

#combining all the previous algorithms
def forward_backward(a, b, pi, seq):
	alpha = forward(a, b,pi, seq)
	beta  = backward(a, b, pi, seq)
	obs_prob = likelihood(a, b, pi, seq)
	posterior = (np.multiply(alpha,beta.T) / obs_prob)
	states = [0]*len(posterior)
	for i in range(len(posterior)):
		states[i] = np.argmax(posterior[i])
	return posterior, states

test("Forward-Backward")
test("Viterbi")





