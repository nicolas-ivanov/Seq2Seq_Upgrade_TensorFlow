# Seq2Seq_Upgrade_TensorFlow
Additional Sequence to Sequence Features for TensorFlow

Hey Guys, TensorFlow is great, but there are some seq2seq features that could be added. The goal is to add them as research in this field unfolds.

That's why there's this additional python package. This is meant to work in conjunction with TensorFlow. Simply clone this and import as:

```python
sys.path.insert(0, os.environ['HOME']) #add the dir that you cloned to
from Seq2Seq_Upgrade_TensorFlow.seq2seq_upgrade import seq2seq_enhanced, rnn_cell_enhanced
```

####Main Features Include:

- [Different RNN layers on different GPU's](https://github.com/LeavesBreathe/Seq2Seq_Upgrade_TensorFlow#different-rnn-layers-on-multiple-gpus)
- [Regularizing RNNs by Stabilizing Activations](https://github.com/LeavesBreathe/Seq2Seq_Upgrade_TensorFlow#norm-regularize-hidden-states-and-outputs) -- [Krueger's paper](http://arxiv.org/pdf/1511.08400.pdf)
- [GRU Mutants](#gru-mutants) -- [Jozefowicz's paper](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
- Identity RNN's -- [Le's paper](http://arxiv.org/pdf/1504.00941v2.pdf)
- Averaging Hidden States During Decoding
- Temperature Sampling Within Each Time-Step in Decoding --  [Explanation Here](https://www.reddit.com/r/MachineLearning/comments/3vzlzz/reproducing_a_neural_conversational_model_in_torch/)

####Currently Working On:

- Unitary RNN (will take at least 2 weeks) --[Arjovsky's paper](http://arxiv.org/abs/1511.06464v2.pdf)
- Adversarial Seq2Seq Training (will take at least 4 or 5 weeks)
- Diversity-Promoting Objective (1 week) -- [Li's paper](http://arxiv.org/pdf/1510.03055v1.pdf)


####Other Features To Come If There's Time:

- Async Loading Data during training 
- Decorrelating Representations -- [Cogswell's Paper](http://arxiv.org/pdf/1511.06068v1.pdf)
- Curriculum Learning 
- Removing Biases From GRU and LSTM (some reports that it improves performance)
- Relaxation Parameter from Neural GPU [Kaiser's Paper](http://arxiv.org/pdf/1511.08228v1.pdf)

You will find many, *many* comments in the code. Just like you guys, I'm still very much learning and it helps me to comment as much as possible. 

Lastly, I'm still have much to learn, so I apologize for mistakes in advance. I welcome pull requests and any help. 



##Using Seq2Seq Upgrade

Import All RNN Materials from Seq2Seq_Upgrade_TensorFlow_Tensorflow Only!

If you are using Seq2Seq_Upgrade_Tensorflow, please do not import:
- rnn.py
- rnn_cell.py
- seq2seq.py

from the regular tensorflow. Instead import:

- rnn_enhanced.py
- rnn_cell_enhanced.py
- seq2seq_enhanced.py

from Seq2Seq_Upgrade_Tensorflow. Otherwise class inheritance will be thrown off, and you will get an `isinstance` error!

You can also try to install seq2seq_upgrade as a python package

    sudo pip install git+ssh://github.com/LeavesBreathe/Seq2Seq_Upgrade_TensorFlow.git

or a bit longer version in case the previous one didn't work

    git clone git@github.com:LeavesBreathe/Seq2Seq_Upgrade_TensorFlow.git
    cd Seq2Seq_Upgrade_TensorFlow
    sudo python setup.py build & sudo python setup.py install
    
After that you hopefully be able to simply write `import seq2seq_upgrade`

------
Some Features are being tested while others are tested and functional. They are labelled:

Status | Meaning
------------- | -------------
Feature Working  | Tested and Should Work as Documented
Under Testing  | May Not Work or Produce Desired Result


##Different RNN Layers on Multiple GPU's
####Feature Working

To call in GRU for gpu 0, simply call in the class

`rnn_cell_enhanced.GRUCell(size, gpu_number = 0)`


```python      
from Seq2Seq_Upgrade_TensorFlow.seq2seq_upgrade import rnn_cell_enhanced

#assuming you're using two gpu's
first_layer = rnn_cell_enhanced.GRUCell(size, gpu_for_layer = 0)
second_layer = rnn_cell_enhanced.GRUCell(size, gpu_for_layer = 1)
cell = rnn_cell.MultiRNNCell(([first_layer]*(num_layers/2)) + ([second_layer]*(num_layers/2)))

#notice that we put consecutive layers on the same gpu. Also notice that you need to use an even number of layers.
```

You can apply dropout on a specific GPU as well:

```python
from Seq2Seq_Upgrade_TensorFlow.seq2seq_upgrade import rnn_cell_enhanced

first_layer = rnn_cell_enhanced.GRUCell(size, gpu_for_layer = 1)
dropout_first_layer = rnn_cell_enhanced.DropoutWrapper(first_layer, output_keep_prob = 0.80, gpu_for_layer = 1)
```


##Norm Regularize Hidden States And Outputs
####Under Testing

[Krueger's paper](http://arxiv.org/pdf/1511.08400.pdf)

Adds an additional cost to regularize hidden state activations and/or output activations (logits).

By default this feature is set to off. 

To use:

```python      
from Seq2Seq_Upgrade_TensorFlow.seq2seq_upgrade import seq2seq_enhanced
#to regularize both
seq2seq_enhanced.model_with_buckets(...norm_regularize_hidden_states = True, 
										norm_regularize_logits = True, norm_regularize_factor = 50)
#to regularize one
seq2seq_enhanced.model_with_buckets(...norm_regularize_hidden_states = True, 
										norm_regularize_logits = False, norm_regularize_factor = 50)
```

`norm_regularizer_factor`: The factor required to apply norm stabilization. Keep 
in mind that a larger factor will allow you to achieve a lower loss, but it will take
many more epochs to do so!


##GRU Mutants
####Feature Working

[Jozefowicz's paper](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)

These mutants do better in some seq2seq tasks. Memory wise, they approximately the same as GRU.

To use this simply:

```python
from Seq2Seq_Upgrade_TensorFlow.seq2seq_upgrade import rnn_cell_enhanced

first_layer = rnn_cell_enhanced.JZS1Cell(size)
```
Mutants are called in by:

1. `rnn_cell_enhanced.JZS1Cell(size, gpu_for_layer = 0)`
2. `rnn_cell_enhanced.JZS2Cell(size, gpu for layer = 1)`
3. `rnn_cell_enhanced.JZS3Cell(size, gpu_for_layer = 2)`

*Gpu arguments are not necessary. 


##Identity RNN
####Under Testing

Allows you to run an IRNN (on specified gpu of your choice). To call in:

```python
from Seq2Seq_Upgrade_TensorFlow.seq2seq_upgrade import rnn_cell_enhanced

first_layer = rnn_cell_enhanced.IdentityRNNCell(size, gpu_for_layer = 0)
```


##Averaging Hidden States
####Under Testing

Allows you to set ratio of last hidden state to mean hidden state

Simply call in:

```python
from Seq2Seq_Upgrade_TensorFlow.seq2seq_upgrade import seq2seq_enhanced as sse

seq2seq_model = sse.embedding_attention_seq2seq(....,average_states = True, average_hidden_state_influence = 0.5)
```

Note that `average_hidden_state_influence` should be any real number between 0 and 1. The higher the number, the higher the percentage that the inputted hidden state will be influenced by the average hidden state.

For example, an `average_hidden_state_influence = .75` means 75% of the hidden state will be the average hidden state and the remainder 25% of the hidden state will be the *last* hidden state. This allows you to choose how much you want the inputted hidden state to be affected by the previous timestep. 



##Temperature Sampling During Decoding
####Under Testing

Allows you to use a temperature to alter the output of your softmax. 

Higher temperature will result in more diversity in output, but more errors. **Do not use this during training!** Use this only while decoding after a model has been trained. 

This affects the internal decoding process. Your decoder will produce a distribution. It will then "roll a dice"
and depending on what words are most probable, it will select *one* of them. The selected word will then be put into your decoder
for the next timestep. Then the decoder will roll the dice again, and repeat the process.

Go Symbol --> Decoder Timestep 1 --> Dice Roll on Distribution --> One Word Selected --> Decoder Timestep 2

To use:

```python
from Seq2Seq_Upgrade_TensorFlow.seq2seq_upgrade import seq2seq_enhanced as sse

seq2seq_model = sse.embedding_attention_seq2seq(....,temperature_decode = True, temperature = 1.0)
```

**Note**: Right now, you have to recompile the model each time you want to test a different temperature. Currently,
I'm working on a way to allow you to test multiple temperatures without re-compiling. 



