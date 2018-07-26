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
lr = 0.005
batch_size = 4
max_seq_len = 60
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
    output_tokens = []

    output_seqs = []
    
    with io.open('story_30.csv') as f_1:
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
        # [16,1,0], got [60,16,256]
        last_outputs = nn.Dense(1,in_units=encoder_outputs,flatten=True)
        last_outputs = last_outputs.reshape((4,256))
        print(last_outputs.shape)
       # last_outputs = encoder_outputs[-1, :, :] #  [16,256]
        #last_outputs = nd.expand_dims(last_outputs, axis=1)
        #print(last_outputs.shape)
        
#         last_outputs.swapaxes(0,1) # [16, 1, 256]
#         hidden_broadcast = nd.broadcast_axis(single_layer_state[0], axis=0,
#                                              size=self.max_seq_len)
#         encoder_outputs_and_hiddens = nd.concat(encoder_outputs,
#                                                 hidden_broadcast, dim=2)
        #print("after swap: " , last_outputs.shape)
        #print(nd.expand_dims(self.embedding(cur_input), axis=1).shape)
        input_and_context = nd.concat(nd.expand_dims(self.embedding(cur_input), axis=1), last_outputs,dim=2)
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
print(eos_id)


def train(encoder, decoder, max_seq_len, ctx,
          eval_fr_ens):
    encoder.initialize(init.Xavier(), ctx=ctx)
    decoder.initialize(init.Xavier(), ctx=ctx)
    encoder_optimizer = gluon.Trainer(encoder.collect_params(), 'adam',
                                      {'learning_rate': lr})
    decoder_optimizer = gluon.Trainer(decoder.collect_params(), 'adam',
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
                    #print(i)
                    decoder_output, decoder_state = decoder(decoder_input, decoder_state, encoder_outputs)
                  
                    decoder_input = y[:, i]

                    valid_length = valid_length + mask.sum()

                    l = l + (mask * loss(decoder_output, y[:, i])).sum()
                    #print(l)

                    mask = mask * (y[:, i] != eos_id)
                    #print(y[:,i])
                    #print(mask)
                    
                l = l / valid_length
                print(l)
                with open('train_loss.txt','a',encoding = "utf-8") as f:
                    f.write('epoch:'+str(epoch)+'batch_size_loss'+str(l)+'\n')
                    
            l.backward()
            encoder_optimizer.step(1)
            decoder_optimizer.step(1)

            l_sum += l.asscalar() / max_seq_len
'''
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
            translate(encoder, decoder, eval_fr_ens, ctx,max_seq_len)
'''

encoder = Encoder(len(input_vocab), encoder_embed_size, encoder_num_hiddens,
                  encoder_num_layers, encoder_drop_prob)
decoder = Decoder(decoder_num_hiddens, len(output_vocab),
                  decoder_num_layers, max_seq_len, decoder_drop_prob,
                  alignment_size, encoder_num_hiddens)


eval_fr_ens =[['kyle mom had been military was found pet watermelon corner garden painted eyes mouth pet marker grandmother gave yarn glue hair hugged went have dinner', 'kyle\'s mom had been in the military and was very proud of it.,She found her pet watermelon Mel in the corner of the garden.,She painted eyes and a mouth onto her pet with the marker.,Her grandmother gave her yarn to glue on for hair.,"Lucy hugged Me goodnight, and went inside to have dinner."'],
              ['lamp broke decided get lamp went store found lamp liked bought lamp', 'Jan\'s lamp broke.,Jan decided to get a new lamp.,He went to the store.,He found a lamp he liked.,He bought the lamp.'],
              ['year watched dunk contest friends dunks were put show fans!I have seen kind dunks life!Zach beat finals','This year I watched the NBA dunk contest with my friends.,The dunks were out of this world!,Aaron Gordon and Zach LaVine put on a great show for the fans!,I have never seen those kind of dunks before in my life!,Zach LaVine barely beat Aaron Gordon in the finals.'],
              ['was day was clung mother leg left had stay while started take part class end day was had gone','It was Tyler\'s first day of Kindergarten.,He was very nervous.,"He clung to his mother\'s leg, but she finally left and he had to stay.","After a little while, he started to take part in the class.","By the end of the day, he was glad he had gone!"'],
              ['wanted find dress anniversary dinner ordered dress looked catalog dress arrived was try dress was zipper broke was returned dress refund',''],
             ['mother wanted pierce baby ears was tradition pierce girl babies ears birth was idea wanted wait tried talk mother who did understand decided pierce baby ears', ' '],
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