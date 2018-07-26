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
lr = 0.001
batch_size = 128
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
    
    with io.open('word_20000.csv') as f_1:
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
    output_tokens = []

    output_seqs = []
    
    with io.open('story_20000.csv') as f_1:
        lines_1 = f_1.readlines()
        for line_1 in lines_1:
            input_seq = line_1
            cur_input_tokens = input_seq.split(' ')
            if len(cur_input_tokens) < max_seq_len:
                output_tokens.extend(cur_input_tokens)
                cur_input_tokens.append(EOS)
                while len(cur_input_tokens) < max_seq_len:
                    cur_input_tokens.append(PAD)
                output_seqs.append(cur_input_tokens)
    en_vocab = text.vocab.Vocabulary(collections.Counter(output_tokens),reserved_tokens=[PAD, BOS, EOS])
    
    return en_vocab,output_seqs

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
            self.embedding = nn.Embedding(num_inputs, embed_size)
            self.dropout = nn.Dropout(drop_prob)
            self.rnn = rnn.LSTM(num_hiddens, num_layers, dropout=drop_prob,
                               input_size=embed_size)

    def forward(self, inputs, state):
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
        
        #encoder_output的shape是(max_seq_len,-1,encoder_num_hiddens)
        encoder_outputs = encoder_outputs.reshape((self.max_seq_len, -1,
                                                   self.encoder_num_hiddens))


        hidden_broadcast = nd.broadcast_axis(single_layer_state[0], axis=0,
                                             size=self.max_seq_len)
        encoder_outputs_and_hiddens = nd.concat(encoder_outputs,
                                                hidden_broadcast, dim=2)

        energy = self.attention(encoder_outputs_and_hiddens)
        
        batch_attention = nd.softmax(energy, axis=0)

        batch_attention = nd.softmax(energy, axis=0).transpose((1, 2, 0))
        #print(batch_attention.shape)
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
        #print('output.shape:\n')
        #print(output.shape)
        output = self.out(output)
        #print('dense shape:\n')
        #print(output.shape)
        output = output.reshape((-3,-1))
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
def translate(encoder, decoder, fr_ens, ctx, max_seq_len):
    for fr_en in fr_ens:
        print('[input] ', fr_en[0])
        input_tokens = fr_en[0].split(' ') + [EOS]

        # 添加 PAD 符号使每个序列等长(长度为 max_seq_len)
        while len(input_tokens) < max_seq_len:
            input_tokens.append(PAD)
        inputs = nd.array(input_vocab.to_indices(input_tokens), ctx=ctx)
        encoder_state = encoder.begin_state(func=nd.zeros, batch_size=1,
                                            ctx=ctx)
        encoder_outputs, encoder_state = encoder(inputs.expand_dims(0),encoder_state)
        
        encoder_outputs = encoder_outputs.flatten()
        # 解码器的第一个输入为 BOS 符号。
        decoder_input = nd.array([output_vocab.token_to_idx[BOS]], ctx=ctx)
        decoder_state = decoder.begin_state(func=nd.zeros, batch_size=1,ctx=ctx)
        output_tokens = []
        #写个dev loss
        for i in range(max_test_output_len):
            decoder_output, decoder_state = decoder(decoder_input, decoder_state, encoder_outputs)
            
            if(i<(max_test_output_len/5)):
                decoder_output = decoder_output/10
            else:
                decoder_output = decoder_output
                           
            
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



loss = gloss.SoftmaxCrossEntropyLoss()

eos_id = output_vocab.token_to_idx[EOS]

def train(encoder, decoder, max_seq_len, ctx,
          eval_fr_ens):
    encoder.initialize(init.Xavier(), ctx=ctx)
    decoder.initialize(init.Xavier(), ctx=ctx)
    encoder_optimizer = gluon.Trainer(encoder.collect_params(), 'SGD',
                                      {'learning_rate': lr})
    decoder_optimizer = gluon.Trainer(decoder.collect_params(), 'SGD',
                                      {'learning_rate': lr})

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
                encoder_outputs = encoder_outputs.flatten()

                # 解码器的第一个输入为 BOS 符号。
                decoder_input = nd.array(
                    [output_vocab.token_to_idx[BOS]] * cur_batch_size,
                    ctx=ctx)
                mask = nd.ones(shape=(cur_batch_size,), ctx=ctx)    #用处
                
                decoder_state = decoder.begin_state(func=nd.zeros, batch_size=cur_batch_size, ctx=ctx)  

                for i in range(max_seq_len):
                    decoder_output, decoder_state = decoder(decoder_input, decoder_state, encoder_outputs)

                    decoder_input = y[:, i]

                    valid_length = valid_length + mask.sum()

                    l = l + (mask * loss(decoder_output, y[:, i])).sum()
                    with open('train_story_loss.txt','a',encoding = "utf-8") as f:
                        f.write('epoch: '+str(epoch)+'batch_size_loss'+str(l)+'\n')
                    

                    mask = mask * (y[:, i] != eos_id)

                l = l / valid_length
                with open('train_token_loss.txt','a',encoding = "utf-8") as f:
                    f.write('epoch: '+str(epoch)+'batch_size_loss'+str(l)+'\n')
                    
            l.backward()
            encoder_optimizer.step(1)
            decoder_optimizer.step(1)

            l_sum += l.asscalar() / max_seq_len

        if epoch % eval_interval == 0 or epoch == 1:
            if epoch == 1:
                print('epoch %d, loss %f, ' % (epoch, l_sum / len(data_iter)))
            else:
                print('epoch %d, loss %f, '
                      % (epoch, l_sum / eval_interval / len(data_iter)))
            if epoch != 1:
                l_sum = 0
            with open('result.txt','a',encoding = "utf-8") as f:
                f.write('epoch: '+str(epoch)+'\n')
            translate(encoder, decoder, eval_fr_ens, ctx,max_seq_len)


encoder = Encoder(len(input_vocab), encoder_embed_size, encoder_num_hiddens,
                  encoder_num_layers, encoder_drop_prob)
decoder = Decoder(decoder_num_hiddens, len(output_vocab),
                  decoder_num_layers, max_seq_len, decoder_drop_prob,
                  alignment_size, encoder_num_hiddens)


eval_fr_ens =[['mother wanted pierce baby ears was tradition pierce girl babies ears birth was idea wanted wait tried talk mother who did understand decided pierce baby ears', ' '],
              ['got spot face did worry got school spots had spread spots itched was sent chickenpox', ' '],
              ['had exam decided go grade was could skip exam did read syllabus failed class was thought',' '],
              ['Reggie was living apartment sister came took look cats Reggie fell love cat named was could be cats Reggie adopted live',''],
              ['was riding bike school someone pushed bike stole arrived school saw thief beat thief tried get bike was suspended saw bike',''],
              ['was student who heard mturk campus could believe was website let people make money signed mturk account started making money while was being hits suspended mturk account months',''],
              ['had seen father years obtained phone number operator called left message called arranged meeting ate laughed days',''],
              ['strove be dancer focused perfection practiced master movement ballet audition danced precision was outshone dancer filled passion saw emotion is artistry mechanics',''],
              ['kept waking work week wanted change sleeping schedule started exercise hour bed would get exercising was fix sleeping schedule',''],
              ['had job interview tomorrow morning stayed updating resume arrived interview minutes interviewer noted arrival time clipboard gained confidence compliment given interviewer',''],
              ['year go cherry picking wake beat crowd arrive get buckets head cherry trees pick hours end bring 20lbs cherries','']]
    
train(encoder, decoder, max_seq_len, ctx, eval_fr_ens)