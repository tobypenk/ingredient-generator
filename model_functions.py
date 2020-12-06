from utils import *


def clip(gradients, maxValue):
    '''
    Clips the gradients' values between minimum and maximum.
    
    Arguments:
    gradients -- a dictionary containing the gradients "dWaa", "dWax", "dWya", "db", "dby"
    maxValue -- everything above this number is set to this number, and everything less than -maxValue is set to -maxValue
    
    Returns: 
    gradients -- a dictionary with the clipped gradients.
    '''
    
    dWaa, dWax, dWya, db, dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['db'], gradients['dby']

    for gradient in [dWaa, dWax, dWya, db, dby]:
        gradient = np.clip(gradient,-maxValue,maxValue,out = gradient)
    
    gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "db": db, "dby": dby}
    
    return gradients
    
    
    
def sample(parameters, char_to_ix):
    """
    Sample a sequence of characters according to a sequence of probability distributions output of the RNN

    Arguments:
    parameters -- python dictionary containing the parameters Waa, Wax, Wya, by, and b. 
    char_to_ix -- python dictionary mapping each character to an index.

    Returns:
    indices -- a list of length n containing the indices of the sampled characters.
    """

    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    vocab_size = by.shape[0]
    n_a = Waa.shape[1]
    
    x = np.zeros(( vocab_size, 1))
    a_prev = np.zeros(( n_a, 1))

    indices = []
    
    idx = -1 

    counter = 0
    newline_character = char_to_ix['\n']
    
    while (idx != newline_character and counter != 50):
        
        # forward propagation
        a = np.tanh( np.dot(Waa, a_prev) + np.dot(Wax, x) + b)
        z = np.dot(Wya, a) + by
        y = softmax(z)
        
        # Sample a character
        idx = np.random.choice(range(vocab_size), p = y.ravel())
        indices.append(idx)
        x = np.zeros(( vocab_size, 1))
        x[idx,:] = 1
        a_prev = a
        counter +=1
 
    if (counter == 50):
        indices.append(char_to_ix['\n'])
    
    return indices
    
    
    
def optimize(X, Y, a_prev, parameters, learning_rate = 0.01,vocab_size=27):
    """
    Execute one step of the optimization to train the model.
    
    Arguments:
    X -- list of integers, where each integer is a number that maps to a character in the vocabulary.
    Y -- list of integers, exactly the same as X but shifted one index to the left.
    a_prev -- previous hidden state.
    parameters -- python dictionary containing:
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        b --  Bias, numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
    learning_rate -- learning rate for the model.
    
    Returns:
    loss -- value of the loss function (cross-entropy)
    gradients -- python dictionary containing:
                        dWax -- Gradients of input-to-hidden weights, of shape (n_a, n_x)
                        dWaa -- Gradients of hidden-to-hidden weights, of shape (n_a, n_a)
                        dWya -- Gradients of hidden-to-output weights, of shape (n_y, n_a)
                        db -- Gradients of bias vector, of shape (n_a, 1)
                        dby -- Gradients of output bias vector, of shape (n_y, 1)
    a[len(X)-1] -- the last hidden state, of shape (n_a, 1)
    """
    
    # forward propagate
    loss, cache = rnn_forward(X, Y, a_prev, parameters,vocab_size)
    
    # backpropagate
    gradients, a = rnn_backward(X, Y, parameters, cache)
    
    # clip
    gradients = clip(gradients, 5)
    
    # update   
    parameters = update_parameters(parameters, gradients, learning_rate)
    
    return loss, gradients, a[len(X)-1]
    

def model(data, ix_to_char, char_to_ix, num_iterations = 35000, n_a = 50, cross_section_depth = 10, vocab_size = 48, verbose = False):
    """
    Arguments:
    data -- text corpus
    ix_to_char -- dictionary that maps the index to a character
    char_to_ix -- dictionary that maps a character to an index
    num_iterations -- number of iterations to train the model for
    n_a -- number of units of the RNN cell
    cross_section_depth -- number of samples at each iteration. 
    vocab_size -- number of unique characters found in the text (size of the vocabulary)
    
    Returns:
    parameters -- learned parameters
    """
    
    n_x, n_y = vocab_size, vocab_size
    parameters = initialize_parameters(n_a, n_x, n_y)
    loss = get_initial_loss(vocab_size, cross_section_depth)
    
    with open("ings.txt") as f:
        examples = f.readlines()
    examples = [x.lower().strip() for x in examples]
    np.random.shuffle(examples)
    
    a_prev = np.zeros((n_a, 1))
    
    for j in range(num_iterations):
    
        idx = j % len(examples)
        
        single_example = examples[idx]
        single_example_chars = [c for c in single_example]
        single_example_ix = [char_to_ix[c] for c in single_example_chars]
        X = [None]+ [single_example_ix]
        X = [None] + [char_to_ix[ch] for ch in examples[idx]]; 

        ix_newline = char_to_ix["\n"]
        Y = X[1:]+ [ix_newline]

        curr_loss, gradients, a_prev = optimize(X, Y, a_prev, parameters, learning_rate=0.01, vocab_size=vocab_size)
        
        loss = smooth(loss, np.mean(curr_loss))

        if j % 2000 == 0:
            
            print('Iteration: %d, Loss: %f' % (j, loss) + '\n')
            
            for name in range(cross_section_depth):
                
                sampled_indices = sample(parameters, char_to_ix)
                print_sample(sampled_indices, ix_to_char)
                
            print('\n')
        
    return parameters

