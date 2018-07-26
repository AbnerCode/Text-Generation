# -*-coding:utf-8 -*-
import collections
import io
import mxnet as mx

from mxnet import autograd, gluon, init, nd
from mxnet.contrib import text
from mxnet.gluon import data as gdata, loss as gloss, nn, rnn

PAD = '<pad>'
BOS = '<bos>'
EOS = '<eos>'


num_epochs = 50
eval_interval = 10
lr = 0.005
batch_size = 1
max_seq_len = 60
max_test_output_len = 60
encoder_num_layers = 1
decoder_num_layers = 1
encoder_drop_prob = 0.1
decoder_drop_prob = 0.1
encoder_embed_size = 256
encoder_num_hiddens = 256
decoder_num_hiddens = 256
alignment_size = 25
ctx = mx.cpu(0)

def read_data(max_seq_len):
    input_tokens = []
    input_seqs = []
    
    with io.open('demo.csv') as f_1:
        lines_1 = f_1.readlines()
        for line_1 in lines_1:
            input_seq = line_1
            cur_input_tokens = input_seq.split(' ')
            if len(cur_input_tokens) < max_seq_len:
                input_tokens.extend(cur_input_tokens)
                cur_input_tokens.append(EOS)
                while len(cur_input_tokens) < max_seq_len:
                    cur_input_tokens.append(PAD)
                input_seqs.append(cur_input_tokens)
    fr_vocab = text.vocab.Vocabulary(collections.Counter(input_tokens),
                                         reserved_tokens=[PAD, BOS, EOS])
    return fr_vocab,input_seqs

def out_data(max_seq_len):
    input_tokens = []

    input_seqs = []
    
    with io.open('initial_demo.csv') as f_1:
        lines_1 = f_1.readlines()
        for line_1 in lines_1:
            input_seq = line_1
            cur_input_tokens = input_seq.split(' ')
            if len(cur_input_tokens) < max_seq_len:
                input_tokens.extend(cur_input_tokens)
                cur_input_tokens.append(EOS)
                while len(cur_input_tokens) < max_seq_len:
                    cur_input_tokens.append(PAD)
                input_seqs.append(cur_input_tokens)
    fr_vocab = text.vocab.Vocabulary(collections.Counter(input_tokens),
                                         reserved_tokens=[PAD, BOS, EOS])
    return fr_vocab,input_seqs

input_vocab, input_seqs = read_data(max_seq_len)
output_vocab,output_seqs = out_data(max_seq_len)

fr = nd.zeros((len(input_seqs), max_seq_len), ctx=ctx)
en = nd.zeros((len(output_seqs), max_seq_len), ctx=ctx)
for i in range(len(output_seqs)):
    fr[i] = nd.array(input_vocab.to_indices(input_seqs[i]), ctx=ctx)
    en[i] = nd.array(output_vocab.to_indices(output_seqs[i]), ctx=ctx)
dataset = gdata.ArrayDataset(fr, en)

class Encoder(nn.Block):
    def __init__(self, num_inputs, embed_size, num_hiddens, num_layers,
                 drop_prob, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        with self.name_scope():
            #有问题 自己写embedding
            self.embedding = nn.Embedding(num_inputs, embed_size)
            self.dropout = nn.Dropout(drop_prob)
            self.rnn = rnn.LSTM(num_hiddens, num_layers, dropout=drop_prob,
                               input_size=embed_size)

    def forward(self, inputs, state):
        embedding = self.embedding(inputs).swapaxes(0, 1)
        embedding = self.dropout(embedding)
        output, state = self.rnn(embedding, state)
        return output, state

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)

class Decoder(nn.Block):
    def __init__(self, num_hiddens, num_outputs, num_layers, max_seq_len,
                 drop_prob, alignment_size, encoder_num_hiddens, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.max_seq_len = max_seq_len
        self.encoder_num_hiddens = encoder_num_hiddens
        self.hidden_size = num_hiddens
        self.num_layers = num_layers
        with self.name_scope():
            self.embedding = nn.Embedding(num_outputs, num_hiddens)
            self.dropout = nn.Dropout(drop_prob)
            # 注意力机制。
            self.attention = nn.Sequential()
            with self.attention.name_scope():
                self.attention.add(
                    nn.Dense(alignment_size,
                             in_units=num_hiddens + encoder_num_hiddens,
                             activation="tanh", flatten=False))
                self.attention.add(nn.Dense(1, in_units=alignment_size,
                                            flatten=False))

            self.rnn = rnn.LSTM(num_hiddens, num_layers, dropout=drop_prob,
                               input_size=num_hiddens)

            self.out = nn.Dense(num_outputs, in_units=num_hiddens,
                                flatten=False)
            self.rnn_concat_input = nn.Dense(
                num_hiddens, in_units=num_hiddens + encoder_num_hiddens,
                flatten=False)


    def forward(self, cur_input, state, encoder_outputs):
        # 当循环神经网络有多个隐藏层时，取靠近输出层的单层隐藏状态
        # 将memory cell拿出来之后在concat过去
        test = []
        test = state[0][:,:,256:]
        single_layer_state = [state[0][-1].expand_dims(0)]
        
        
        encoder_outputs = encoder_outputs.reshape((self.max_seq_len, -1,
                                                   self.encoder_num_hiddens))

        hidden_broadcast = nd.broadcast_axis(single_layer_state[0], axis=0,
                                             size=self.max_seq_len)
        encoder_outputs_and_hiddens = nd.concat(encoder_outputs,
                                                hidden_broadcast, dim=2)

        energy = self.attention(encoder_outputs_and_hiddens)

        batch_attention = nd.softmax(energy, axis=0).transpose((1, 2, 0))
        batch_encoder_outputs = encoder_outputs.swapaxes(0, 1)
        decoder_context = nd.batch_dot(batch_attention, batch_encoder_outputs)
        #改这里
        input_and_context = nd.concat(nd.expand_dims(self.embedding(cur_input), axis=1),
            decoder_context, dim=2)
        concat_input = self.rnn_concat_input(input_and_context).reshape((1, -1, 0))

        concat_input = self.dropout(concat_input)

        state = [nd.broadcast_axis(single_layer_state[0], axis=0,size=self.num_layers)]
        new = []
        new = nd.concat(state[0],test,dim = 2)
        state = new

        output, state = self.rnn(concat_input, state)

        output = self.dropout(output)
        output = self.out(output).reshape((-3, -1))
        return output, state

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)


class DecoderInitState(nn.Block):
    def __init__(self, encoder_num_hiddens, decoder_num_hiddens, **kwargs):
        super(DecoderInitState, self).__init__(**kwargs)
        with self.name_scope():
            #decoder_num_hiddens 乘了2
            self.dense = nn.Dense(decoder_num_hiddens*2,
                                  in_units=encoder_num_hiddens,
                                  activation="tanh", flatten=False)

    def forward(self, encoder_state):
        return [self.dense(encoder_state)]
        


def translate(encoder, decoder, decoder_init_state, fr_ens, ctx, max_seq_len):
    for fr_en in fr_ens:
        print('[input] ', fr_en[0])
        input_tokens = fr_en[0].split(' ') + [EOS]
        # 添加 PAD 符号使每个序列等长（长度为 max_seq_len）。
        while len(input_tokens) < max_seq_len:
            input_tokens.append(PAD)
        inputs = nd.array(input_vocab.to_indices(input_tokens), ctx=ctx)
        encoder_state = encoder.begin_state(func=nd.zeros, batch_size=1,
                                            ctx=ctx)
        encoder_outputs, encoder_state = encoder(inputs.expand_dims(0),
                                                 encoder_state)
        encoder_outputs = encoder_outputs.flatten()
        # 解码器的第一个输入为 BOS 符号。
        decoder_input = nd.array([output_vocab.token_to_idx[BOS]], ctx=ctx)
        decoder_state = decoder_init_state(encoder_state[0])
        output_tokens = []

        for _ in range(max_test_output_len):
            decoder_output, decoder_state = decoder(
                decoder_input, decoder_state, encoder_outputs)
            pred_i = int(decoder_output.argmax(axis=1).asnumpy()[0])
            # 当任一时间步搜索出 EOS 符号时，输出序列即完成。
            if pred_i == output_vocab.token_to_idx[EOS]:
                break
            else:
                output_tokens.append(output_vocab.idx_to_token[pred_i])
            decoder_input = nd.array([pred_i], ctx=ctx)
        print('[output]', ' '.join(output_tokens)) 
        print('[expect]', fr_en[1], '\n')


loss = gloss.SoftmaxCrossEntropyLoss()
eos_id = output_vocab.token_to_idx[EOS]


def train(encoder, decoder, decoder_init_state, max_seq_len, ctx,eval_fr_ens):
    encoder.initialize(init.Xavier(), ctx=ctx)
    decoder.initialize(init.Xavier(), ctx=ctx)
    decoder_init_state.initialize(init.Xavier(), ctx=ctx)
    encoder_optimizer = gluon.Trainer(encoder.collect_params(), 'adam',{'learning_rate': lr})
    decoder_optimizer = gluon.Trainer(decoder.collect_params(), 'adam',{'learning_rate': lr})
    decoder_init_state_optimizer = gluon.Trainer(
        decoder_init_state.collect_params(), 'adam', {'learning_rate': lr})
    data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)
    l_sum = 0
    for epoch in range(1, num_epochs + 1):
        for x, y in data_iter:
            cur_batch_size = x.shape[0]
            with autograd.record():
                l = nd.array([0], ctx=ctx)
                valid_length = nd.array([0], ctx=ctx)
                encoder_state = encoder.begin_state(
                    func=nd.zeros, batch_size=cur_batch_size, ctx=ctx)
                encoder_outputs, encoder_state = encoder(x, encoder_state)
                encoder_outputs = encoder_outputs.flatten()
                # 解码器的第一个输入为 BOS 符号。
                decoder_input = nd.array(
                    [output_vocab.token_to_idx[BOS]] * cur_batch_size,
                    ctx=ctx)
                mask = nd.ones(shape=(cur_batch_size,), ctx=ctx)
                decoder_state = decoder_init_state(encoder_state[0])
                for i in range(max_seq_len):
                    decoder_output, decoder_state = decoder(decoder_input, decoder_state, encoder_outputs)
                    # 解码器使用当前时间步的预测词作为下一时间步的输入。
                    decoder_input = decoder_output.argmax(axis=1)
                    valid_length = valid_length + mask.sum()
                    l = l + (mask * loss(decoder_output, y[:, i])).sum()
                    mask = mask * (y[:, i] != eos_id)
                l = l / valid_length
            l.backward()
            encoder_optimizer.step(1)
            decoder_optimizer.step(1)
            decoder_init_state_optimizer.step(1)
            l_sum += l.asscalar() / max_seq_len

        if epoch % eval_interval == 0 or epoch == 1:
            if epoch == 1:
                print('epoch %d, loss %f, ' % (epoch, l_sum / len(data_iter)))
            else:
                print('epoch %d, loss %f, '
                      % (epoch, l_sum / eval_interval / len(data_iter)))
            if epoch != 1:
                l_sum = 0
            translate(encoder, decoder, decoder_init_state, eval_fr_ens, ctx,max_seq_len)


encoder = Encoder(len(input_vocab), encoder_embed_size, encoder_num_hiddens,
                  encoder_num_layers, encoder_drop_prob)
decoder = Decoder(decoder_num_hiddens, len(output_vocab),
                  decoder_num_layers, max_seq_len, decoder_drop_prob,
                  alignment_size, encoder_num_hiddens)
decoder_init_state = DecoderInitState(encoder_num_hiddens,decoder_num_hiddens)


eval_fr_ens =[['kyle mom had been military was found pet watermelon corner garden painted eyes mouth pet marker grandmother gave yarn glue hair hugged went have dinner', 'kyle s mom had been in the military and was very proud of it.,She found her pet watermelon Mel in the corner of the garden.,She painted eyes and a mouth onto her pet with the marker.,Her grandmother gave her yarn to glue on for hair.,"Lucy hugged Me goodnight, and went inside to have dinner."'],
              ['lamp broke decided get lamp went store found lamp liked bought lamp', 'Jans lamp broke.,Jan decided to get a new lamp.,He went to the store.,He found a lamp he liked.,He bought the lamp.'],
              ]

train(encoder, decoder, decoder_init_state, max_seq_len, ctx, eval_fr_ens)