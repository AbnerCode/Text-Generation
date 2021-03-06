# -*-coding:utf-8 -*-
import collections
import io
import csv
import mxnet as mx

from mxnet import autograd, gluon, init, nd
from mxnet.contrib import text
from mxnet.gluon import data as gdata, loss as gloss, nn, rnn

PAD = '<pad>'
BOS = '<bos>'
EOS = '<eos>'


num_epochs = 100
eval_interval = 10
lr = 0.005
batch_size = 1
max_seq_len = 120
max_test_output_len = 60
encoder_num_layers = 1
decoder_num_layers = 2
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
    
    with io.open('word_30.csv') as f_1:
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
    
    with io.open('story_30.csv') as f_1:
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
        #print(inputs.shape)
        #inputs的尺寸：(batch_size,num_steps)   
        embedding = self.embedding(inputs).swapaxes(0,1)
        #print(embedding.shape)
        #print(embedding.shape)  embed尺寸:(num_steps,batch_size,256)
        #swapaxes后为(1,num_steps,256)
        embedding = self.dropout(embedding)
        #print(embedding.shape)  
        output, state = self.rnn(embedding, state)
        #print(output.shape)  
        #print(state[1]shape) (1,1,256)
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

            self.rnn = rnn.GRU(num_hiddens, num_layers, dropout=drop_prob,
                               input_size=num_hiddens)

            self.out = nn.Dense(num_outputs, in_units=num_hiddens,
                                flatten=False)
            self.rnn_concat_input = nn.Dense(
                num_hiddens, in_units=num_hiddens + encoder_num_hiddens,
                flatten=False)


    def forward(self, cur_input, state, encoder_outputs):
        # 当循环神经网络有多个隐藏层时，取靠近输出层的单层隐藏状态
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
            #encoder_num_hiddens = 256
            self.dense = nn.Dense(decoder_num_hiddens,
                                  in_units=encoder_num_hiddens,
                                  activation="tanh", flatten=False)

    def forward(self, encoder_state):
        #print(self.dense(encoder_state).shape)    (1,2,256) -> (1,1,256) why?
        return [self.dense(encoder_state)]
        #(X1,X2,x3...xn,in_units)
        


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
            
        with open('result.txt','a',encoding = "utf-8") as f:
            #f.write('epoch '+epoch+'\n')
            f.write('[input]'+fr_en[0]+'\n')
            f.write('[output]'+' '.join(output_tokens)+'\n')

        with open('result.txt','a',encoding = "utf-8") as f:
            f.write('[expect]'+fr_en[1]+'\n')
            #f.write('next epoch\n')
            f.write('\n')

        #print('[output]', ' '.join(output_tokens)) 
        #print('[expect]', fr_en[1], '\n')
        #print(' '.join(output_tokens))


loss = gloss.SoftmaxCrossEntropyLoss()

eos_id = output_vocab.token_to_idx[EOS]

def train(encoder, decoder, decoder_init_state, max_seq_len, ctx,
          eval_fr_ens):
    encoder.initialize(init.Xavier(), ctx=ctx)
    decoder.initialize(init.Xavier(), ctx=ctx)
    decoder_init_state.initialize(init.Xavier(), ctx=ctx)
    encoder_optimizer = gluon.Trainer(encoder.collect_params(), 'adam',
                                      {'learning_rate': lr})
    decoder_optimizer = gluon.Trainer(decoder.collect_params(), 'adam',
                                      {'learning_rate': lr})
    decoder_init_state_optimizer = gluon.Trainer(
        decoder_init_state.collect_params(), 'adam', {'learning_rate': lr})

    data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)

    l_sum = 0     #loss？
    for epoch in range(1, num_epochs + 1):
        for x, y in data_iter:
            cur_batch_size = x.shape[0]
            with autograd.record():
                l = nd.array([0], ctx=ctx)

                valid_length = nd.array([0], ctx=ctx)

                encoder_state = encoder.begin_state(
                    func=nd.zeros, batch_size=cur_batch_size, ctx=ctx)
                # encoder_outputs 包含了编码器在每个时间步的隐藏状态。
                encoder_outputs, encoder_state = encoder(x, encoder_state)
                #print(encoder_outputs)
                #print(encoder_outputs.shape)
                encoder_outputs = encoder_outputs.flatten()
                #print(encoder_outputs)
                #print(encoder_outputs.shape)
                # 解码器的第一个输入为 BOS 符号。
                decoder_input = nd.array(
                    [output_vocab.token_to_idx[BOS]] * cur_batch_size,
                    ctx=ctx)

                mask = nd.ones(shape=(cur_batch_size,), ctx=ctx)    #用处

                decoder_state = decoder_init_state(encoder_state[0])  #采用encoder_state的outputs
                #print(decoder_state)

                for i in range(max_seq_len):
                    decoder_output, decoder_state = decoder(
                        decoder_input, decoder_state, encoder_outputs)
                    # 解码器使用当前时间步的预测词作为下一时间步的输入。
                    #或者采用teacher forcing
                    #decoder_input = decoder_output.argmax(axis=1)
                    if (i<(max_seq_len/5)):
                        decoder_input = y[:, i]
                    else:
                        decoder_input = decoder_output.argmax(axis=1)
                     #采用Teacher Forcing

                    #print('teacher')
                    #print(decoder_input.shape)

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
            #在dev集上的训练
            translate(encoder, decoder, decoder_init_state, eval_fr_ens, ctx,
                      max_seq_len)


encoder = Encoder(len(input_vocab), encoder_embed_size, encoder_num_hiddens,
                  encoder_num_layers, encoder_drop_prob)
decoder = Decoder(decoder_num_hiddens, len(output_vocab),
                  decoder_num_layers, max_seq_len, decoder_drop_prob,
                  alignment_size, encoder_num_hiddens)
decoder_init_state = DecoderInitState(encoder_num_hiddens,decoder_num_hiddens)


eval_fr_ens =[['kyle mom had been military was found pet watermelon corner garden painted eyes mouth pet marker grandmother gave yarn glue hair hugged went have dinner', 'kyle\'s mom had been in the military and was very proud of it.,She found her pet watermelon Mel in the corner of the garden.,She painted eyes and a mouth onto her pet with the marker.,Her grandmother gave her yarn to glue on for hair.,"Lucy hugged Me goodnight, and went inside to have dinner."'],
              ['lamp broke decided get lamp went store found lamp liked bought lamp', 'Jan\'s lamp broke.,Jan decided to get a new lamp.,He went to the store.,He found a lamp he liked.,He bought the lamp.'],
              ['year watched dunk contest friends dunks were put show fans!I have seen kind dunks life!Zach beat finals','This year I watched the NBA dunk contest with my friends.,The dunks were out of this world!,Aaron Gordon and Zach LaVine put on a great show for the fans!,I have never seen those kind of dunks before in my life!,Zach LaVine barely beat Aaron Gordon in the finals.'],
              ['was day was clung mother leg left had stay while started take part class end day was had gone','It was Tyler\'s first day of Kindergarten.,He was very nervous.,"He clung to his mother\'s leg, but she finally left and he had to stay.","After a little while, he started to take part in the class.","By the end of the day, he was glad he had gone!"'],
              ['wanted find dress anniversary dinner ordered dress looked catalog dress arrived was try dress was zipper broke was returned dress refund','']]
train(encoder, decoder, decoder_init_state, max_seq_len, ctx, eval_fr_ens)