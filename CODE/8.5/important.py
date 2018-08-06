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


num_epochs = 30
eval_interval = 1
lr = 0.01
batch_size = 20
max_seq_len = 75
max_test_output_len = 60
encoder_num_layers = 1
decoder_num_layers = 2
encoder_drop_prob = 0.1
decoder_drop_prob = 0.1
encoder_embed_size = 256
encoder_num_hiddens = 256
decoder_num_hiddens = 256
alignment_size = 25
ctx = mx.gpu()

def read_data(max_seq_len):
    input_tokens = []
    input_seqs = []
    
    with io.open('data/word_100.csv') as f_1:
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
    
    with io.open('data/story_100.csv') as f_1:
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
            self.rnn_concat_input = nn.Dense(num_hiddens, in_units=num_hiddens + encoder_num_hiddens,flatten=False)


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
        

#dev 的word词典是否拥有
def translate(encoder, decoder, decoder_init_state, fr_ens, ctx, max_seq_len):
    for fr_en in fr_ens:
        #print('[input] ', fr_en[0])
        input_tokens = fr_en[0].split(' ') + [EOS]

        # 添加 PAD 符号使每个序列等长（长度为 max_seq_len）。
        while len(input_tokens) < max_seq_len:
            input_tokens.append(PAD)
        inputs = nd.array(input_vocab.to_indices(input_tokens), ctx=ctx)
        encoder_state = encoder.begin_state(func=nd.zeros, batch_size=1,
                                            ctx=ctx)
        encoder_outputs, encoder_state = encoder(inputs.expand_dims(0),encoder_state)
        
        encoder_outputs = encoder_outputs.flatten()
        # 解码器的第一个输入为 BOS 符号。
        decoder_input = nd.array([output_vocab.token_to_idx[BOS]], ctx=ctx)
        decoder_state = decoder_init_state(encoder_state[-1])
        output_tokens = []
#写dev的loss

        for i in range(max_test_output_len):
            decoder_output, decoder_state = decoder(decoder_input, decoder_state, encoder_outputs)
            #
            decoder_output = decoder_output/10
            
            #print(decoder_output)

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
                #print(encoder_outputs.shape)
                encoder_outputs = encoder_outputs.flatten()
                #print(encoder_outputs.shape)

                # 解码器的第一个输入为 BOS 符号。
                decoder_input = nd.array(
                    [output_vocab.token_to_idx[BOS]] * cur_batch_size,
                    ctx=ctx)


                mask = nd.ones(shape=(cur_batch_size,), ctx=ctx)    #用处

                decoder_state = decoder_init_state(encoder_state[-1])  

                for i in range(max_seq_len):
                    decoder_output, decoder_state = decoder(decoder_input, decoder_state, encoder_outputs)

                    decoder_input = y[:, i]

                    valid_length = valid_length + mask.sum()

                    l = l + (mask * loss(decoder_output, y[:, i])).sum()

                    mask = mask * (y[:, i] != eos_id)

                l = l / valid_length
                #with open('train_loss.txt','a',encoding = "utf-8") as f:
                    #f.write('batch_size_loss'+l+'\n')
                    

            l.backward()
            encoder_optimizer.step(1)
            decoder_optimizer.step(1)
            decoder_init_state_optimizer.step(1)

            l_sum += l.asscalar()

        if epoch % eval_interval == 0 or epoch == 1:
            if epoch == 1:
                with open('result.txt','a',encoding = "utf-8") as f:
                    f.write('epoch: '+str(epoch)+'\n')
         
                print('epoch %d, loss %f, ' % (epoch, l_sum / len(data_iter)))
            else:
                print('epoch %d, loss %f, '
                      % (epoch, l_sum / eval_interval / len(data_iter)))
            if epoch != 1:
                l_sum = 0
            with open('result.txt','a',encoding = "utf-8") as f:
                f.write('epoch: '+str(epoch)+'\n')
            #在dev集上的训练
            translate(encoder, decoder, decoder_init_state, eval_fr_ens, ctx,max_seq_len)


encoder = Encoder(len(input_vocab), encoder_embed_size, encoder_num_hiddens,
                  encoder_num_layers, encoder_drop_prob)
decoder = Decoder(decoder_num_hiddens, len(output_vocab),
                  decoder_num_layers, max_seq_len, decoder_drop_prob,
                  alignment_size, encoder_num_hiddens)
decoder_init_state = DecoderInitState(encoder_num_hiddens,decoder_num_hiddens)


eval_fr_ens =[['parents were was doctors told parents was parents understood decided make change got diet', ' Dan\'s parents were overweight. Dan was overweight as well. The doctors told his parents it was unhealthy. His parents understood and decided to make a change. They got themselves and Dan on a diet.'],
              ['had learned ride bike did have bike would sneak rides sister bike got hill crashed wall bike frame bent got gash leg', 'Carrie had just learned how to ride a bike. She didn\'t have a bike of her own. Carrie would sneak rides on her sister\'s bike. She got nervous on a hill and crashed into a wall. The bike frame bent and Carrie got a deep gash on her leg. '],
              ['enjoyed walks beach boyfriend decided go walk walking mile something happened decided propose boyfriend boyfriend was upset did propose',' Morgan enjoyed long walks on the beach. She and her boyfriend decided to go for a long walk. After walking for over a mile, something happened. Morgan decided to propose to her boyfriend. Her boyfriend was upset he didn\'t propose to her first.'],
              ['was working diner customer barged counter began yelling food was taking did know react coworker intervened calmed man','Jane was working at a diner. Suddenly, a customer barged up to the counter. He began yelling about how long his food was taking. Jane didn\'t know how to react. Luckily, her coworker intervened and calmed the man down.'],
              ['was talking crush today continued complain guys flirting decided agree what says listened got got text asked can hang tomorrow','I was talking to my crush today. She continued to complain about guys flirting with her. I decided to agree with what she says and listened to her patiently. After I got home, I got a text from her. She asked if we can hang out tomorrow.']]
train(encoder, decoder, decoder_init_state, max_seq_len, ctx, eval_fr_ens)