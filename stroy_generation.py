import tensorflow as tf
import numpy as np
import tqdm


# 句子最大长度
SENTENCE_LIMIT_SIZE = 20
EMBEDDING_SIZE = 50

HIDDEN_SIZE = 512
LEARNING_RATE = 0.001
KEEP_PROB = 0.5
EPOCHES = 50
BATCH_SIZE = 256



with open("demo.csv","r") as f:
	initial_seg = f.read()

# 分句
initial_sentences = initial_seg.lower().split("\n")



vocab =initial_seg.split()
vocab.append("<pad>")
vocab.append("<unk>")

# 单词到编码的映射，例如machine -> 10283
word_to_token = {word: token for token, word in enumerate(vocab)}
# 编码到单词的映射，例如10283 -> machine
token_to_word = {token: word for word, token in word_to_token.items()}


def convert_text_to_token(sentence, word_to_token_map=word_to_token, limit_size=SENTENCE_LIMIT_SIZE):
 

    unk_id = word_to_token_map["<unk>"]
    pad_id = word_to_token_map["<pad>"]
    
    # 对句子进行token转换，对于未在词典中出现过的词用unk的token填充
    tokens = [word_to_token_map.get(word, unk_id) for word in sentence.lower().split()]
    
    # Pad
    if len(tokens) < limit_size:
        tokens.extend([0] * (limit_size - len(tokens)))
    # Trunc
    else:
        tokens = tokens[:limit_size]
    
    return tokens



#print(word_to_token.get('age'))


initial_tokens = []

for sentence in tqdm.tqdm(initial_sentences):
    tokens = convert_text_to_token(sentence)
    initial_tokens.append(tokens)

initial_tokens = np.array(initial_tokens)

#print(initial_tokens.shape)


with open("glove.6B.50d.txt", 'r') as f:
    words = set()
    word_to_vec = {}
    for line in f:
        line = line.strip().split()
        # 当前单词
        curr_word = line[0]
        words.add(curr_word)
        # 当前词向量
        word_to_vec[curr_word] = np.array(line[1:], dtype=np.float32)

VOCAB_SIZE = len(vocab)  

static_embeddings = np.zeros([VOCAB_SIZE, EMBEDDING_SIZE])
for word, token in tqdm.tqdm(word_to_token.items()):
    # 用glove词向量填充，如果没有对应的词向量，则用随机数填充
    word_vector = word_to_vec.get(word, 0.2 * np.random.random(EMBEDDING_SIZE) - 0.1)
    static_embeddings[token, :] = word_vector

# 重置PAD为0向量
pad_id = word_to_token["<pad>"]
static_embeddings[pad_id, :] = np.zeros(EMBEDDING_SIZE)

static_embeddings = static_embeddings.astype(np.float32)


#print(static_embeddings.shape) 所有的词

def lstm_model():
    # placeholders
    with tf.name_scope("placeholders"):
        inputs = tf.placeholder(dtype=tf.int32, shape=(None, 20), name="inputs")
        targets = tf.placeholder(dtype=tf.float32, shape=(None, 1), name="targets")
    
    # embeddings
    with tf.name_scope("embeddings"):
        embedding_matrix = tf.Variable(initial_value=static_embeddings, trainable=False, name="embedding_matrix")
        embed = tf.nn.embedding_lookup(embedding_matrix, inputs, name="embed")
    
    # model
    with tf.name_scope("model"):
        # 构造lstm单元
        lstm = tf.contrib.rnn.LSTMCell(HIDDEN_SIZE, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123))
        # 添加dropout
        drop_lstm = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=KEEP_PROB)
        _, lstm_state = tf.nn.dynamic_rnn(drop_lstm, embed, dtype=tf.float32)
        
        # 输出层权重
        W = tf.Variable(tf.truncated_normal((HIDDEN_SIZE, 1), mean=0.0, stddev=0.1), name="W")
        b = tf.Variable(tf.zeros(1), name="b")
        
        logits = tf.add(tf.matmul(lstm_state.h, W), b)
        outputs = tf.nn.sigmoid(logits, name="outputs")
        
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=targets, logits=logits))
    
    # optimizer
    with tf.name_scope("optimizer"):
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(EPOCHES):





