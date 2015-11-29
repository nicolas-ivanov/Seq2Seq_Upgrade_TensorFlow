# Seq2Seq_Upgrade_TensorFlow
Additional Sequence to Sequence Features for TensorFlow

Hey Guys, TensorFlow is great, but there are some seq2seq features that could be added. The goal is to add them as research in this field unfolds.

That's why there's this additional python package. This is meant to work in conjunction with TensorFlow. Simply clone this and import as:

```python
sys.path.insert(0, os.environ['HOME']) #add the dir that you cloned to
from Seq2Seq_Upgrade_TensorFlow.seq2seq_upgrade import seq2seq_enhanced as sse
```

Main Features Include:

- Averaging Hidden States During Decoding
- Different GRU's and LSTM's layers on different GPU's
- Curriculum Learning (In Progress)
- Temperature Integration (In Progress)
- uRNN (In Progress)
- Async Loading Data during training (In Progress)

You will find many, *many* comments in the code. Just like you guys, I'm still very much learning and it helps me to comment as much as possible. 

Lastly, I'm still have much to learn, so I apologize for mistakes in advance. I welcome pull requests and any help. 



##Averaging Hidden States

Simply call in:

```python
from Seq2Seq_Upgrade_TensorFlow.seq2seq_upgrade import seq2seq_enhanced as sse

seq2seq_model = sse.embedding_attention_decoder_average_states(....)
```


##Different RNN Layers on different GPU's

To call in GRU for gpu 0, simply call in the class

GRUCell_GPU0 instead of GRUCell

For GPU 1,

GRUCell_GPU1 instead of GRUCell, etc.

```python      
from Seq2Seq_Upgrade_TensorFlow.seq2seq_upgrade import rnn_cell_enhanced```


first_layer = rnn_cell_enhanced.GRUCell_GPU0(size) #this is saying for gpu 0
second_layer = rnn_cell_enhanced.GRUCell_GPU1(size)
cell = rnn_cell.MultiRNNCell(([first_layer]*(num_layers/2)) + ([second_layer]*(num_layers/2)))

#notice that we put consecutive layers on the same gpu. Also notice that you need to use an even number of layers.
```



