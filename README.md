# Seq2Seq_Upgrade_TensorFlow
Additional Sequence to Sequence Features for TensorFlow

Hey Guys, TensorFlow is great, but there are some seq2seq features that could be added. The goal is to add them as research in this field unfolds.

That's why there's this additional python package. This is meant to work in conjunction with TensorFlow. Simply clone this and import as:

```python
sys.path.insert(0, os.environ['HOME']) #add the dir that you cloned to
from Seq2Seq_Upgrade_TensorFlow.seq2seq_upgrade import seq2seq_enhanced as sse
```

Main Features Include:

- Averaging Hidden States During Decoding, Allows you to set ratio of last hidden state to mean hidden state
- Different RNN layers on different GPU's
- GRU Mutants from [Jozefowicz et al. paper](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)

Currently Working On:
- Curriculum Learning (In Progress)
- Temperature Integration (Almost Done)
- uRNN (Ummm...gonna take a while)
- Async Loading Data during training (In Progress)

You will find many, *many* comments in the code. Just like you guys, I'm still very much learning and it helps me to comment as much as possible. 

Lastly, I'm still have much to learn, so I apologize for mistakes in advance. I welcome pull requests and any help. 


##Warning -- Import All RNN Materials from Seq2Seq_Upgrade_Tensorflow Only!
If you are using Seq2Seq_Upgrade_Tensorflow, please do not import:
- rnn.py
- rnn_cell.py
- seq2seq.py

from the regular tensor flow! instead import:

- rnn_enhanced.py
- rnn_cell_enhanced.py
- seq2seq_enhanced.py

from Seq2Seq_Upgrade_Tensorflow. Otherwise class inheritance will be thrown off, and you will get an `isinstance` error!


##Averaging Hidden States -- Working

Simply call in:

```python
from Seq2Seq_Upgrade_TensorFlow.seq2seq_upgrade import seq2seq_enhanced as sse

seq2seq_model = sse.embedding_attention_seq2seq(....,average_states = True, average_hidden_state_influence = 0.5)
```

Note that `average_hidden_state_influence` should be any real number between 0 and 1. The higher the number, the higher the percentage that the inputted hidden state will be influenced by the average hidden state.

For example, an `average_hidden_state_influence = .75` means 75% of the hidden state will be the average hidden state and the remainder 25% of the hidden state will be the *last* hidden state. This allows you to choose how much you want the inputted hidden state to be affected by the previous timestep. 



##Different RNN Layers on Multiple GPU's -- Working

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


##GRU Mutants (from Jozefowicz et al.) -- Working

These mutants do better in some seq2seq tasks. Memory wise, they approximately the same as GRU.

To use this simply:

```python
from Seq2Seq_Upgrade_TensorFlow.seq2seq_upgrade import rnn_cell_enhanced```

first_layer = rnn_cell_enhanced.JZS1Cell(size)
```
Mutants are called in by:

1. `rnn_cell_enhanced.JZS1Cell(size)`
2. `rnn_cell_enhanced.JZS2Cell(size)`
3. `rnn_cell_enhanced.JZS3Cell(size)`

